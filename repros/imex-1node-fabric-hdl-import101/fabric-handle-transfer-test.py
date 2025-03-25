import logging
import uuid
import time
import sys
import os
import threading
import base64

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pprint import pformat

import requests

# NVIDIA's python bindings for the CUDA API.
# https://github.com/NVIDIA/cuda-python
# import cuda
from cuda.bindings import driver, runtime, nvrtc

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%y%m%d-%H:%M:%S",
)

LEADER_HTTPD_BASE_URL = (
    "http://"
    + os.environ.get("LEADER_HTTPD_DNSNAME")
    + ":"
    + os.environ.get("LEADER_HTTPD_PORT")
)

CUDA_FABRIC_HANDLE_DATA = {}

SHUTDOWN_EVENT = threading.Event()


def main():
    log.info("cuDriverGetVersion(): %s", checkCudaErrors(driver.cuDriverGetVersion()))
    log.info(
        "getLocalRuntimeVersion(): %s",
        checkCudaErrors(runtime.getLocalRuntimeVersion()),
    )

    # Log some environment variables for debug info.
    for k, v in os.environ.items():
        if "CUDA" in k or "NVIDIA" in k:
            log.info("env: %s: %s", k, v)

    # Inspect IMEX /dev and /proc state.
    try:
        log.info(
            "listdir(/dev/nvidia-caps-imex-channels): %s",
            os.listdir("/dev/nvidia-caps-imex-channels"),
        )
    except Exception as exc:
        log.info("cannot enumerate /dev/nvidia-caps-imex-channels: %s", exc)
    try:
        devs = open("/proc/devices", "rb").read().decode("utf-8").splitlines()
        log.info(
            "/proc/devices contains IMEX devices: %s",
            [d for d in devs if "imex" in d.lower()],
        )
    except Exception as exc:
        log.info("cannot inspect /proc/devices: %s", exc)

    # Expect to be run as part of an indexed, parallel k8s job.
    assert "JOB_COMPLETION_INDEX" in os.environ, "JOB_COMPLETION_INDEX not set in env"

    jci = int(os.environ["JOB_COMPLETION_INDEX"])

    # Assume two copies of this program running, one as leader (jci 0) and one
    # as follower (jci 1).
    log.info("k8s job completion index: %s", jci)

    # Expect precisely one injected CUDA device, make this explicit in program's
    # log output, and log relevant device properties.
    _devcount = checkCudaErrors(runtime.cudaGetDeviceCount())
    assert _devcount == 1, "precisely one CUDA device expected"
    log.info("getDeviceCount(): %s", _devcount)
    log_device_properties()

    # Leader code path
    if jci == 0:
        # Allocate memory on GPU and get exportable handle of type FABRIC.
        CUDA_FABRIC_HANDLE_DATA["handle"] = leader_raw_cuda_mem_share_over_imex()

        # Start HTTP server for exposing handle to consumers.
        run_httpd_in_thread()
        # Wait for follower to read exported handle, then shut down a few seconds
        # later.
        SHUTDOWN_EVENT.wait()
        log.info("shutdown soon")

    # Follower code path.
    else:
        fabric_handle_data = wait_for_fabric_handle_from_leader()
        follower_raw_cuda_mem_share_over_imex(fabric_handle_data)

    # Leave workload running for a little bit longer so that
    # ComputeDomain-related resource are inspectable for a brief amount of time
    # (such as IMEX daemon log). These resources get torn down upon job
    # completion.
    time.sleep(40)
    log.info("shutdown")


def leader_raw_cuda_mem_share_over_imex():
    """
    Note(JP): this is part of a test where I try to demonstrate if sharing
    memory between GPUs via IMEX works node-locally. The question can be looked
    at as: does the MNNVL-way of doing things _also_ work node-locally?

    In this minmal test scenario there are two containerized processes running
    either on the same node (machine), or spread across two nodes.

    In case of the single-node scneario: the two processes have access to a
    shared IMEX channel. Each process sees just one GPU device. In different and
    maybe more precise words: two containerized processes running on the same
    machine. Each process holds one CUDA context. Each process has cgroup access
    to precisely one GPU. The two GPUs are different physical GPUs in that same
    machine. Both processes have cgroup access to the same shared IMEX channel.

    The question is: does memory sharing work via a handle of type
    CU_MEM_HANDLE_TYPE_FABRIC? That is the handle type used for MNNVL-based
    memory sharing across machine boundaries. The hope is that it _also_ works
    within machine boundaries.

    References that inspired this code:

    - https://github.com/NVIDIA/cuda-samples/blob/v12.8/Samples/3_CUDA_Features/memMapIPCDrv/memMapIpc.cpp#L202
    """
    checkCudaErrors(driver.cuInit(0))
    cudev = checkCudaErrors(driver.cuDeviceGet(0))
    # log.info("driver.cuDeviceGet(0) returned: %s", cudev)
    log.info("cudev: %s type: %s", cudev, type(cudev))

    # Check that the selected device supports virtual address management
    vaddr_supported = checkCudaErrors(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
            cudev,
        )
    )

    if not vaddr_supported:
        raise Exception("VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED: false")

    # The physical location where the memory will be allocated via
    # cuMemCreate(), plus additional parameters about how this will be
    # used/exported.
    prop = driver.CUmemAllocationProp()
    prop.location = driver.CUmemLocation()

    # Device-pinned memory backed on the "backing" device, exportable with the
    # specified handle type.
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED

    # Type FABRIC: required for IMEX-based communication. Documented to work on
    # different nodes
    prop.requestedHandleTypes = (
        driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    )

    # Indicate that location is a device location, thus id is a device ordinal.
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE

    # Back all allocations on 'this' device -- I took inspiration from CPP CUDA
    # examples here and I think this is 'just' the GPU device ID. Which is 0
    # here because we assume that a single GPU is injected into the container.

    # prop.location.id = cudev also works
    prop.location.id = 0

    status, granny = driver.cuMemGetAllocationGranularity(
        prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    )

    if status != driver.CUresult.CUDA_SUCCESS:
        raise Exception(f"err getting granny: {status}")

    # log.info("granularity: %s", granny)

    handle_local = checkCudaErrors(driver.cuMemCreate(granny * 1, prop, 0))

    handle_exp = checkCudaErrors(
        driver.cuMemExportToShareableHandle(
            handle_local,
            # Request an exportable handle of fabric type (IMEX!).
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            0,
        )
    )

    log.info("exportable fabric handle object: %s", handle_exp)

    # https://docs.nvidia.com/cuda/cuda-driver-api/structCUmemFabricHandle__v1.html#structCUmemFabricHandle__v1
    # Fabric handle - An opaque handle representing a memory allocation that can
    # be exported to processes in same or different nodes. For IPC between
    # processes on different nodes they must be connected via the NVSwitch
    # fabric.

    log.info("exportable fabric handle data: %s", handle_exp.data.hex())

    return handle_exp.data


def follower_raw_cuda_mem_share_over_imex(fabric_handle_data: bytes):
    """
    ... Given a CUDA memory handle, create a shareable memory allocation handle that
    can be used to share the memory with other processes. The recipient process
    can convert the shareable handle back into a CUDA memory handle using
    cuMemImportFromShareableHandle and map it with cuMemMap ...
    """
    # Currently expected to error out in a single-node use case:
    # RuntimeError: CUDA error code=101(b'CUDA_ERROR_INVALID_DEVICE')
    handle_local = checkCudaErrors(
        driver.cuMemImportFromShareableHandle(
            fabric_handle_data,
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
        )
    )
    log.info("cuMemImportFromShareableHandle(): successful")


def wait_for_fabric_handle_from_leader() -> bytes:
    url = f"{LEADER_HTTPD_BASE_URL}/fabric-handle"

    log.info("enter loop: get fabric handle from leader, URL: %s", url)
    while True:
        try:
            # Short connect()/recv() timeout to keep logs flowing.
            resp = requests.get(url, timeout=(4, 6))
            if resp.status_code == 200:
                break
            log.info(
                "unexpected response, retry: %s, %s",
                resp.status_code,
                resp.content[:200],
            )
        except requests.exceptions.RequestException as exc:
            # dns err, connect or recv timeout, etc
            log.info("request failed, retry soon -- err was: %s", exc)
        time.sleep(3)

    log.info(
        "leader responded, response body (hex prefix): %s...", resp.content[:5].hex()
    )

    # Decode bytes into bytes :)
    fabric_handle_data = base64.urlsafe_b64decode(resp.content)

    log.info("fabric handle data: %s", fabric_handle_data)
    return fabric_handle_data


def log_device_properties():
    _attr_filter = ["name", "pci", "uuid", "multi", "minor", "major"]

    # Assume there is just one device injected / available, and get its id
    # Should always be zero the way this is orchestrated.
    devidx = checkCudaErrors(runtime.cudaGetDevice())
    props = checkCudaErrors(runtime.cudaGetDeviceProperties(devidx))

    printprops = {}
    for k in dir(props):
        v = getattr(props, k)
        for ss in _attr_filter:
            if k.startswith(ss):
                if k == "uuid":
                    try:
                        v = uuid.UUID(bytes=v.bytes)
                    except ValueError:
                        # For some devices in some cases this does not seem to
                        # report proper UUID raw data:
                        # raise ValueError('bytes is not a 16-char string')
                        log.warning("funky UUID bytes: %s", v.bytes)
                        # Proceed with that funky value.
                printprops[k] = v
                break

    log.info("device %s properties:\n%s", devidx, pformat(printprops))


class HTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Handle any GET request to any path

        message = b"unknown path"

        if "/fabric-handle" in self.path:
            log.info("HTTPD: follower requests fabric handle")

            # If there's a newline-resembling byte in the data then we can't
            # just write the byte sequence as-is into the response body. Be sure
            # that this works for all byte sequences. b64encode() returns a byte
            # sequence, too.
            message = base64.urlsafe_b64encode(CUDA_FABRIC_HANDLE_DATA["handle"])
            SHUTDOWN_EVENT.set()

        self.send_response(200)
        self.end_headers()
        self.wfile.write(message)
        log.info("served request to %s", self.path)
        return


def run_httpd_in_thread():
    def run():
        log.info("start HTTP server")

        # Handle each incoming request in its own thread.
        s = ThreadingHTTPServer(
            # Listen on all interfaces.
            ("0.0.0.0", int(os.environ.get("LEADER_HTTPD_PORT"))),
            HTTPHandler,
        )

        # Block forever.
        s.serve_forever()

    # Start main loop of this server in its own thread.
    t = threading.Thread(target=run)

    # Program exits when only daemon threads are left (no explicit join
    # required in this case).
    t.daemon = True
    t.start()


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={}({})".format(
                result[0].value, _cudaGetErrorEnum(result[0])
            )
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


if __name__ == "__main__":
    # Always exit with code 0 to show k8s job as completed.
    try:
        main()
    except Exception:
        log.exception("main() crashed, traceback: ")
        log.info("exit 0")
        sys.exit(0)
