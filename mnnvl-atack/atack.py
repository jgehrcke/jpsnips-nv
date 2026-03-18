"""
atack — All-to-All CUDA Kubernetes test.

Tests Multi-Node NVLink (MNNVL) memory sharing using CUDA fabric handles
exported/imported via IMEX. Each pod in a Kubernetes StatefulSet:
- Runs an HTTP server that allocates GPU memory chunks on demand
- Periodically discovers peers via DNS and verifies cross-GPU memory access
- Cleans up completed chunks via a GC thread
"""

import base64
import ctypes
import json
import logging
import os
import signal
import socket
import struct
import sys
import threading
import time
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pprint import pformat

import dns.resolver
import requests

from cuda.bindings import driver, runtime, nvrtc

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%y%m%d-%H:%M:%S",
)

# Configuration from environment.
HTTPD_PORT = int(os.environ.get("HTTPD_PORT", "1337"))
CHUNK_MIB = int(os.environ.get("CHUNK_MIB", "100"))
FLOAT_VALUE = float(os.environ.get("FLOAT_VALUE", "1.0"))
SVC_NAME = os.environ.get("SVC_NAME", "svc-atack")
POLL_INTERVAL_S = int(os.environ.get("POLL_INTERVAL_S", "3"))
K8S_PODNAME = socket.gethostname()
K8S_NAMESPACE = os.environ.get("POD_NAMESPACE", "default")
K8S_NODENAME = os.environ.get("NODE_NAME", "unknown")

# Pre-computed chunk metadata (JSON bytes), set once during startup.
# A single GPU memory chunk is allocated, filled, and exported at init time.
# The HTTP handler serves this same handle on every request — no per-request
# GPU work. Each peer's cuMemcpyDtoD still transfers the full data over NVLink
# every time (the handle is just an address reference, not a cached copy).
# This eliminates GPU memory controller contention between memset fills and
# concurrent NVLink reads from remote peers.
SHARED_CHUNK_META = None  # JSON bytes, set by prepare_shared_chunk()

# Shared CUDA allocation properties (set during init).
ALLOC_PROP = None
GRANULARITY = None
CUDEV = None
CHECKSUM_KERNEL = None  # Compiled CUfunction, set during cuda_init().

# Pre-allocated GPU buffers for verify_chunk_on_gpu(), set during cuda_init().
# Eliminates per-round cuMemAlloc/cuMemFree churn which may contribute to the
# bandwidth measurement variance we observed (intermittent drops from ~820 to
# ~430 GB/s). The local buffer receives the DtoD copy, the partial sums buffer
# holds the checksum kernel output. Both are allocated once and reused.
VERIFY_LOCAL_BUF = None       # CUdeviceptr, sized to CHUNK_MIB
VERIFY_LOCAL_BUF_SIZE = None  # Actual allocation size in bytes
VERIFY_PARTIALS_BUF = None    # CUdeviceptr, sized to CHECKSUM_NUM_BLOCKS * 8


def main():
    log.info("pod name: %s", K8S_PODNAME)
    log.info("config: HTTPD_PORT=%s CHUNK_MIB=%s FLOAT_VALUE=%s SVC_NAME=%s POLL_INTERVAL_S=%s",
             HTTPD_PORT, CHUNK_MIB, FLOAT_VALUE, SVC_NAME, POLL_INTERVAL_S)

    log.info("cuDriverGetVersion(): %s", checkCudaErrors(driver.cuDriverGetVersion()))
    log.info(
        "getLocalRuntimeVersion(): %s",
        checkCudaErrors(runtime.getLocalRuntimeVersion()),
    )

    for k, v in os.environ.items():
        if "CUDA" in k or "NVIDIA" in k:
            log.info("env: %s: %s", k, v)

    log_imex_state()
    cuda_init()
    log_device_properties()
    prepare_shared_chunk()

    # Start threads.
    run_httpd_in_thread()
    start_peer_poll_thread()

    shutdown = threading.Event()
    signal.signal(signal.SIGTERM, lambda sig, frame: shutdown.set())
    signal.signal(signal.SIGINT, lambda sig, frame: shutdown.set())

    log.info("all threads started, main thread waiting")
    shutdown.wait()
    log.info("received shutdown signal, exiting")


def cuda_init():
    """Initialize CUDA, validate single device, set up shared allocation properties."""
    global ALLOC_PROP, GRANULARITY, CUDEV, CHECKSUM_NUM_BLOCKS

    checkCudaErrors(driver.cuInit(0))

    _devcount = checkCudaErrors(runtime.cudaGetDeviceCount())
    assert _devcount == 1, f"precisely one CUDA device expected, got {_devcount}"
    log.info("cudaGetDeviceCount(): %s", _devcount)

    cudev = checkCudaErrors(driver.cuDeviceGet(0))
    CUDEV = cudev
    log.info("cudev: %s type: %s", cudev, type(cudev))

    # Use the primary context so all threads can share it via
    # cuDevicePrimaryCtxRetain + cuCtxPushCurrent.
    ctx = checkCudaErrors(driver.cuDevicePrimaryCtxRetain(cudev))
    checkCudaErrors(driver.cuCtxPushCurrent(ctx))
    log.info("retained and pushed primary CUDA context: %s", ctx)

    vaddr_supported = checkCudaErrors(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
            cudev,
        )
    )
    if not vaddr_supported:
        raise Exception("VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED: false")

    sm_count = checkCudaErrors(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            cudev,
        )
    )
    CHECKSUM_NUM_BLOCKS = sm_count * 4
    log.info("SM count: %s, checksum kernel grid size: %s blocks", sm_count, CHECKSUM_NUM_BLOCKS)

    # Build shared allocation properties.
    prop = driver.CUmemAllocationProp()
    prop.location = driver.CUmemLocation()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = (
        driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    )
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = 0
    ALLOC_PROP = prop

    GRANULARITY = checkCudaErrors(
        driver.cuMemGetAllocationGranularity(
            prop,
            driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
    )
    log.info("allocation granularity: %s bytes", GRANULARITY)

    compile_checksum_kernel()
    preallocate_verify_buffers()


# CUDA kernel that sums all float32 values in the input array.
#
# Design choices optimized for maximizing NVLink read bandwidth:
#
# - Plain double accumulation instead of Kahan compensated summation.
#   Kahan adds 3 extra ALU ops per load (subtract, add, subtract) which
#   throttles the rate at which threads can issue new memory requests.
#   Double precision (~15 decimal digits) is more than sufficient for
#   summing float32 values without compensation.
#
# - __ldg() intrinsic to route loads through the read-only texture cache
#   path, which can improve throughput for read-only access patterns.
#
# - 4x loop unrolling to increase instruction-level parallelism and keep
#   more memory requests in flight per thread.
#
# - Launched with 4 blocks per SM (set at runtime) to maximize occupancy
#   and give the warp scheduler enough warps to hide NVLink latency.
#
# Each block reduces to a partial sum in shared memory. The host sums the
# partial results (a few hundred doubles — trivial).
CHECKSUM_KERNEL_SRC = r"""
extern "C" __global__
void checksum(const float* __restrict__ data, int n, double* out) {
    __shared__ double ssum[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    double sum = 0.0;

    // Unrolled loop: process 4 elements per iteration to increase
    // instruction-level parallelism and keep more loads in flight.
    int i = gid;
    for (; i + 3 * stride < n; i += 4 * stride) {
        sum += (double)__ldg(&data[i]);
        sum += (double)__ldg(&data[i + stride]);
        sum += (double)__ldg(&data[i + 2 * stride]);
        sum += (double)__ldg(&data[i + 3 * stride]);
    }
    // Handle remaining elements.
    for (; i < n; i += stride) {
        sum += (double)__ldg(&data[i]);
    }

    ssum[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        out[blockIdx.x] = ssum[0];
}
"""

# Number of blocks for the checksum kernel, set during cuda_init().
# 4 blocks per SM to maximize occupancy and hide NVLink latency.
CHECKSUM_NUM_BLOCKS = None


def compile_checksum_kernel():
    """Compile the checksum kernel with NVRTC and load it."""
    global CHECKSUM_KERNEL

    prog = check_nvrtc_errors(nvrtc.nvrtcCreateProgram(
        CHECKSUM_KERNEL_SRC.encode("utf-8"), b"checksum.cu", 0, [], [],
    ))
    try:
        check_nvrtc_errors(nvrtc.nvrtcCompileProgram(prog, 0, []))
    except RuntimeError:
        log_size = check_nvrtc_errors(nvrtc.nvrtcGetProgramLogSize(prog))
        log_buf = b" " * log_size
        check_nvrtc_errors(nvrtc.nvrtcGetProgramLog(prog, log_buf))
        raise RuntimeError(f"NVRTC compile failed:\n{log_buf.decode()}")

    ptx_size = check_nvrtc_errors(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ptx_size
    check_nvrtc_errors(nvrtc.nvrtcGetPTX(prog, ptx))

    module = checkCudaErrors(driver.cuModuleLoadData(ptx))
    CHECKSUM_KERNEL = checkCudaErrors(driver.cuModuleGetFunction(module, b"checksum"))
    log.info("compiled and loaded checksum kernel")


def preallocate_verify_buffers():
    """Pre-allocate GPU buffers used by verify_chunk_on_gpu()."""
    global VERIFY_LOCAL_BUF, VERIFY_LOCAL_BUF_SIZE, VERIFY_PARTIALS_BUF

    chunk_bytes = CHUNK_MIB * 1024 * 1024
    alloc_size = ((chunk_bytes + GRANULARITY - 1) // GRANULARITY) * GRANULARITY
    VERIFY_LOCAL_BUF = checkCudaErrors(driver.cuMemAlloc(alloc_size))
    VERIFY_LOCAL_BUF_SIZE = alloc_size

    partials_size = CHECKSUM_NUM_BLOCKS * ctypes.sizeof(ctypes.c_double)
    VERIFY_PARTIALS_BUF = checkCudaErrors(driver.cuMemAlloc(partials_size))

    log.info("pre-allocated verify buffers: local_buf=%d bytes, partials=%d bytes",
             alloc_size, partials_size)


def check_nvrtc_errors(result):
    """Check NVRTC return codes, similar to checkCudaErrors."""
    if result[0].value:
        raise RuntimeError(f"NVRTC error: {result[0]}")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def ensure_cuda_context():
    """Retain and push the primary CUDA context onto the calling thread's stack.

    Safe to call multiple times from the same thread — retaining an already-retained
    context just increments a refcount, and pushing it when it's already current is a
    no-op in practice.
    """
    ctx = checkCudaErrors(driver.cuDevicePrimaryCtxRetain(CUDEV))
    checkCudaErrors(driver.cuCtxPushCurrent(ctx))


def prepare_shared_chunk():
    """Allocate a single GPU memory chunk at startup, fill it, export the
    fabric handle, and store the JSON metadata in SHARED_CHUNK_META.

    Called once during init. The HTTP handler serves this same metadata on
    every /prepare-chunk request. No per-request GPU work means no memset
    can contend with concurrent NVLink reads from remote peers.
    """
    global SHARED_CHUNK_META

    chunk_bytes = CHUNK_MIB * 1024 * 1024
    alloc_size = ((chunk_bytes + GRANULARITY - 1) // GRANULARITY) * GRANULARITY
    num_floats = alloc_size // 4  # float32

    alloc_handle = checkCudaErrors(driver.cuMemCreate(alloc_size, ALLOC_PROP, 0))

    va_ptr = checkCudaErrors(driver.cuMemAddressReserve(alloc_size, GRANULARITY, 0, 0))
    checkCudaErrors(driver.cuMemMap(va_ptr, alloc_size, 0, alloc_handle, 0))

    access_desc = driver.CUmemAccessDesc()
    access_desc.location = ALLOC_PROP.location
    access_desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    checkCudaErrors(driver.cuMemSetAccess(va_ptr, alloc_size, [access_desc], 1))

    # Fill GPU memory with FLOAT_VALUE directly on the GPU using cuMemsetD32.
    # This avoids allocating a large host-side buffer and a PCIe host-to-device
    # copy (cuMemcpyHtoD). cuMemsetD32 sets N 32-bit values on the device.
    # It interprets the pattern as a uint32, so we reinterpret our float32's
    # bit pattern as uint32 via struct pack/unpack.
    float_as_uint32 = struct.unpack("I", struct.pack("f", FLOAT_VALUE))[0]
    checkCudaErrors(driver.cuMemsetD32(va_ptr, float_as_uint32, num_floats))
    # cuMemsetD32 is asynchronous — wait for it to complete before exporting
    # the handle, otherwise a peer could read partially-filled memory.
    checkCudaErrors(driver.cuCtxSynchronize())

    fabric_handle = checkCudaErrors(
        driver.cuMemExportToShareableHandle(
            alloc_handle,
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            0,
        )
    )

    handle_b64 = base64.urlsafe_b64encode(fabric_handle.data).decode("ascii")

    SHARED_CHUNK_META = json.dumps({
        "handle": handle_b64,
        "pod_name": K8S_PODNAME,
        "node_name": K8S_NODENAME,
        "num_floats": num_floats,
        "float_value": FLOAT_VALUE,
        "alloc_size": alloc_size,
    }).encode("utf-8")

    log.info("prepared shared chunk: %d MiB, %d floats, handle exported",
             alloc_size // (1024 * 1024), num_floats)


def fetch_chunk_meta(peer_host: str, port: int) -> dict:
    """GET /prepare-chunk from a peer. Returns parsed JSON metadata."""
    resp = requests.get(f"http://{peer_host}:{port}/prepare-chunk", timeout=(5, 30))
    if resp.status_code != 200:
        log.error("peer returned HTTP %s: %s", resp.status_code, resp.text)
        resp.raise_for_status()
    return resp.json()


def import_fabric_handle(handle_bytes: bytes):
    """Import a fabric handle and return the local allocation handle."""
    return checkCudaErrors(
        driver.cuMemImportFromShareableHandle(
            handle_bytes,
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
        )
    )


def map_imported_chunk(alloc_handle, alloc_size: int) -> int:
    """Map an imported allocation into local VA space. Returns the VA pointer."""
    va_ptr = checkCudaErrors(
        driver.cuMemAddressReserve(alloc_size, GRANULARITY, 0, 0)
    )
    checkCudaErrors(driver.cuMemMap(va_ptr, alloc_size, 0, alloc_handle, 0))

    access_desc = driver.CUmemAccessDesc()
    access_desc.location = ALLOC_PROP.location
    access_desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    checkCudaErrors(driver.cuMemSetAccess(va_ptr, alloc_size, [access_desc], 1))
    return va_ptr


CHECKSUM_REL_TOLERANCE = 1e-5


def verify_chunk_on_gpu(va_ptr: int, alloc_size: int, num_floats: int, expected_value: float) -> str:
    """Measure NVLink bandwidth via DtoD copy, then verify data via checksum.

    The measurement is split into two phases:

    Phase 1 — Bandwidth measurement: cuMemcpyDtoD from the remote-mapped VA
    to a local GPU buffer. This is a pure DMA transfer handled by the copy
    engine with zero SM involvement, giving the cleanest possible NVLink
    bandwidth number — no ALU overhead, no kernel launch latency, no
    instruction pipeline effects. This is what NVIDIA's own bandwidth test
    tools use.

    Phase 2 — Data integrity: run the checksum kernel on the local copy.
    Because the data is now in local GPU memory, this is a fast local read
    and does not affect the NVLink bandwidth measurement.

    Returns a string like '756.3 GB/s' on success, or a mismatch description.
    """
    assert alloc_size <= VERIFY_LOCAL_BUF_SIZE, \
        f"chunk {alloc_size} exceeds pre-allocated buffer {VERIFY_LOCAL_BUF_SIZE}"

    # Phase 1: time the DtoD copy (pure NVLink transfer).
    # Uses the pre-allocated local buffer — no per-round allocation.
    ev_start = checkCudaErrors(driver.cuEventCreate(0))
    ev_end = checkCudaErrors(driver.cuEventCreate(0))

    checkCudaErrors(driver.cuEventRecord(ev_start, 0))
    checkCudaErrors(driver.cuMemcpyDtoD(VERIFY_LOCAL_BUF, va_ptr, alloc_size))
    checkCudaErrors(driver.cuEventRecord(ev_end, 0))
    checkCudaErrors(driver.cuEventSynchronize(ev_end))

    elapsed_ms = checkCudaErrors(driver.cuEventElapsedTime(ev_start, ev_end))
    checkCudaErrors(driver.cuEventDestroy(ev_start))
    checkCudaErrors(driver.cuEventDestroy(ev_end))

    # Phase 2: checksum the local copy for data integrity.
    # Uses the pre-allocated partials buffer.
    num_blocks = CHECKSUM_NUM_BLOCKS
    out_size = num_blocks * ctypes.sizeof(ctypes.c_double)

    # cuda-python returns CUdeviceptr wrapper objects; extract the raw
    # integer address for ctypes kernel arg marshalling.
    n = ctypes.c_int(num_floats)
    data_ptr = ctypes.c_void_p(int(VERIFY_LOCAL_BUF))
    out_ptr_arg = ctypes.c_void_p(int(VERIFY_PARTIALS_BUF))
    args = (ctypes.c_void_p * 3)(
        ctypes.addressof(data_ptr),
        ctypes.addressof(n),
        ctypes.addressof(out_ptr_arg),
    )

    checkCudaErrors(driver.cuLaunchKernel(
        CHECKSUM_KERNEL,
        num_blocks, 1, 1,   # grid
        256, 1, 1,           # block
        0, 0,                # shared mem, stream
        args, 0,
    ))
    checkCudaErrors(driver.cuCtxSynchronize())

    # Read back partial sums and reduce on host.
    result_buf = bytearray(out_size)
    checkCudaErrors(driver.cuMemcpyDtoH(result_buf, VERIFY_PARTIALS_BUF, out_size))

    partial_sums = struct.unpack(f"{num_blocks}d", result_buf)
    gpu_sum = sum(partial_sums)
    expected_sum = float(num_floats) * expected_value

    if expected_sum == 0.0:
        diff = abs(gpu_sum)
        ok = diff < CHECKSUM_REL_TOLERANCE
    else:
        rel_err = abs(gpu_sum - expected_sum) / abs(expected_sum)
        ok = rel_err < CHECKSUM_REL_TOLERANCE

    if not ok:
        return f"CHECKSUM MISMATCH: gpu_sum={gpu_sum}, expected={expected_sum}"

    bw_gbs = alloc_size / (elapsed_ms / 1000.0) / 1e9
    return f"{bw_gbs:.1f} GB/s"


def unmap_imported_chunk(va_ptr, alloc_size: int, alloc_handle):
    """Unmap and release a locally-imported GPU memory chunk."""
    if va_ptr is not None:
        checkCudaErrors(driver.cuMemUnmap(va_ptr, alloc_size))
        checkCudaErrors(driver.cuMemAddressFree(va_ptr, alloc_size))
    if alloc_handle is not None:
        checkCudaErrors(driver.cuMemRelease(alloc_handle))



def import_and_verify_chunk(peer_name: str, peer_host: str, port: int) -> tuple[str, str]:
    """Request a chunk from a peer, import it, verify contents.

    Returns (result_str, peer_node_name). result_str is a bandwidth like
    '818.3 GB/s' on success, or an error string.
    """
    try:
        meta = fetch_chunk_meta(peer_host, port)
    except Exception:
        log.exception("failed to fetch chunk meta from %s:", peer_name)
        return ("req-err", "?")

    peer_node = meta.get("node_name", "?")
    alloc_size = meta["alloc_size"]
    handle_bytes = base64.urlsafe_b64decode(meta["handle"])

    imported_handle = None
    va_ptr = None
    result = "OK"

    try:
        imported_handle = import_fabric_handle(handle_bytes)
        va_ptr = map_imported_chunk(imported_handle, alloc_size)
        result = verify_chunk_on_gpu(va_ptr, alloc_size, meta["num_floats"], meta["float_value"])
    except CudaError:
        log.exception("CUDA error importing/verifying chunk from %s:", peer_name)
        result = "cuda-err"
    except Exception:
        log.exception("unexpected error importing/verifying chunk from %s:", peer_name)
        result = "err"
    finally:
        try:
            unmap_imported_chunk(va_ptr, alloc_size, imported_handle)
        except Exception:
            log.exception("cleanup error for chunk from %s:", peer_name)

    return (result, peer_node)


def discover_peers() -> list[tuple[str, str]]:
    """Discover peer pods via DNS SRV lookup on the headless service.

    Returns list of (pod_name, fqdn) for all peers except self.
    Kubernetes CoreDNS serves SRV records for headless services with named
    ports: _<port-name>._<proto>.<svc>.<namespace>.svc.cluster.local
    """
    srv_name = f"_http._tcp.{SVC_NAME}.{K8S_NAMESPACE}.svc.cluster.local"
    answers = dns.resolver.resolve(srv_name, "SRV", lifetime=5)

    peers = []
    for rdata in answers:
        target = str(rdata.target).rstrip(".")
        # Extract pod name from FQDN: atack-0.svc-atack.default.svc.cluster.local
        pod_name = target.split(".")[0]
        if pod_name != K8S_PODNAME:
            peers.append((pod_name, target))

    return peers


def peer_poll_loop():
    """Periodically discover peers and verify memory sharing with each."""
    ensure_cuda_context()
    # Give the cluster a moment to settle.
    time.sleep(POLL_INTERVAL_S)

    while True:
        try:
            peers = discover_peers()
            if not peers:
                log.info("peer poll: no peers discovered yet")
                time.sleep(POLL_INTERVAL_S)
                continue

            results = {}
            for pod_name, peer_host in peers:
                status, peer_node = import_and_verify_chunk(pod_name, peer_host, HTTPD_PORT)
                peer_idx = pod_name.rsplit("-", 1)[1]
                results[peer_idx] = (status, peer_node)

            my_idx = K8S_PODNAME.rsplit("-", 1)[1]
            parts = []
            for idx, (status, peer_node) in sorted(results.items()):
                parts.append(f"{idx}@{peer_node}:{status}")
            log.info("result(%s@%s): %s", my_idx, K8S_NODENAME, " ".join(parts))

        except dns.resolver.NXDOMAIN:
            log.info("peer poll: DNS name does not exist yet (expected at startup)")
        except Exception:
            log.exception("peer poll error:")

        time.sleep(POLL_INTERVAL_S)


def start_peer_poll_thread():
    t = threading.Thread(target=peer_poll_loop, daemon=True)
    t.start()



# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class HTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if "/healthz" in self.path:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if "/prepare-chunk" not in self.path:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"unknown path")
            return

        # Serve the pre-computed shared chunk metadata — no GPU work per request.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(SHARED_CHUNK_META)

    def log_message(self, format, *args):
        # Suppress default BaseHTTPRequestHandler logging.
        pass


def run_httpd_in_thread():
    def run():
        log.info("starting HTTP server on port %s", HTTPD_PORT)
        s = ThreadingHTTPServer(("0.0.0.0", HTTPD_PORT), HTTPHandler)
        s.serve_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# IMEX / device inspection
# ---------------------------------------------------------------------------

def log_imex_state():
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


def log_device_properties():
    _attr_filter = ["name", "pci", "uuid", "multi", "minor", "major"]

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
                        log.warning("funky UUID bytes: %s", v.bytes)
                printprops[k] = v
                break

    log.info("device %s properties:\n%s", devidx, pformat(printprops))


# ---------------------------------------------------------------------------
# CUDA error handling (kept from original)
# ---------------------------------------------------------------------------

class CudaError(RuntimeError):
    """Raised by checkCudaErrors() for CUDA API failures."""
    pass


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
        raise CudaError(
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
    try:
        main()
    except Exception:
        log.exception("main() crashed:")
        sys.exit(1)
