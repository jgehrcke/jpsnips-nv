"""
atack — All-to-All CUDA Kubernetes test.

Tests Multi-Node NVLink (MNNVL) memory sharing using CUDA fabric handles
exported/imported via IMEX. Each pod in a Kubernetes StatefulSet:
- Runs an HTTP server that serves pre-allocated GPU memory chunk handles
- Periodically discovers peers via DNS and verifies cross-GPU memory access
- Supports multiple GPUs per pod for full NxM bandwidth matrix measurement
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
from urllib.parse import urlparse, parse_qs

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
GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", "1"))
K8S_PODNAME = socket.gethostname()
K8S_NAMESPACE = os.environ.get("POD_NAMESPACE", "default")
K8S_NODENAME = os.environ.get("NODE_NAME", "unknown")

# Per-GPU state, keyed by gpu_idx (0..GPUS_PER_NODE-1). Set during cuda_init().
CUDEVS = {}              # {gpu_idx: CUdevice}
ALLOC_PROPS = {}         # {gpu_idx: CUmemAllocationProp}
GRANULARITIES = {}       # {gpu_idx: int}
CHECKSUM_KERNELS = {}    # {gpu_idx: CUfunction}
CHECKSUM_NUM_BLOCKS = {} # {gpu_idx: int}

# Pre-computed chunk metadata per GPU (JSON bytes), set once during startup.
# Each GPU gets its own chunk allocated, filled, and exported at init time.
# The HTTP handler serves these on GET /prepare-chunk?gpu_index=N — no
# per-request GPU work. Each peer's cuMemcpyDtoD still transfers the full
# data over NVLink every time (the handle is just an address reference).
SHARED_CHUNK_METAS = {}  # {gpu_idx: bytes (JSON)}

# Pre-allocated GPU buffers for verify_chunk_on_gpu(), per GPU.
# Eliminates per-round cuMemAlloc/cuMemFree churn which may contribute to
# bandwidth measurement variance we observed.
VERIFY_LOCAL_BUFS = {}       # {gpu_idx: CUdeviceptr}
VERIFY_LOCAL_BUF_SIZES = {}  # {gpu_idx: int}
VERIFY_PARTIALS_BUFS = {}    # {gpu_idx: CUdeviceptr}


def main():
    log.info("pod name: %s", K8S_PODNAME)
    log.info("config: HTTPD_PORT=%s CHUNK_MIB=%s FLOAT_VALUE=%s SVC_NAME=%s "
             "POLL_INTERVAL_S=%s GPUS_PER_NODE=%s",
             HTTPD_PORT, CHUNK_MIB, FLOAT_VALUE, SVC_NAME, POLL_INTERVAL_S,
             GPUS_PER_NODE)

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
    prepare_all_shared_chunks()

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
    """Initialize CUDA for all GPUs: contexts, alloc properties, kernels, buffers."""
    checkCudaErrors(driver.cuInit(0))

    devcount = checkCudaErrors(runtime.cudaGetDeviceCount())
    log.info("cudaGetDeviceCount(): %s, GPUS_PER_NODE: %s", devcount, GPUS_PER_NODE)
    assert devcount >= GPUS_PER_NODE, \
        f"GPUS_PER_NODE={GPUS_PER_NODE} but only {devcount} devices visible"

    for gpu_idx in range(GPUS_PER_NODE):
        cudev = checkCudaErrors(driver.cuDeviceGet(gpu_idx))
        CUDEVS[gpu_idx] = cudev
        log.info("GPU %d: cudev=%s", gpu_idx, cudev)

        # Each GPU has its own primary context.
        ctx = checkCudaErrors(driver.cuDevicePrimaryCtxRetain(cudev))
        checkCudaErrors(driver.cuCtxPushCurrent(ctx))
        log.info("GPU %d: retained and pushed primary context: %s", gpu_idx, ctx)

        vaddr_supported = checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                cudev,
            )
        )
        if not vaddr_supported:
            raise Exception(f"GPU {gpu_idx}: VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED: false")

        sm_count = checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                cudev,
            )
        )
        CHECKSUM_NUM_BLOCKS[gpu_idx] = sm_count * 4
        log.info("GPU %d: SM count: %s, checksum grid: %s blocks",
                 gpu_idx, sm_count, CHECKSUM_NUM_BLOCKS[gpu_idx])

        # Build allocation properties for this GPU.
        prop = driver.CUmemAllocationProp()
        prop.location = driver.CUmemLocation()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = (
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        )
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = gpu_idx
        ALLOC_PROPS[gpu_idx] = prop

        GRANULARITIES[gpu_idx] = checkCudaErrors(
            driver.cuMemGetAllocationGranularity(
                prop,
                driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        )
        log.info("GPU %d: allocation granularity: %s bytes", gpu_idx, GRANULARITIES[gpu_idx])

        # Compile checksum kernel for this GPU's context.
        compile_checksum_kernel(gpu_idx)

        # Pre-allocate verify buffers on this GPU.
        preallocate_verify_buffers(gpu_idx)

        # Pop context — we'll push per-GPU as needed.
        checkCudaErrors(driver.cuCtxPopCurrent())


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


def compile_checksum_kernel(gpu_idx):
    """Compile the checksum kernel with NVRTC and load it. Must be called
    with the GPU's context already pushed."""
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
    CHECKSUM_KERNELS[gpu_idx] = checkCudaErrors(
        driver.cuModuleGetFunction(module, b"checksum"))
    log.info("GPU %d: compiled and loaded checksum kernel", gpu_idx)


def preallocate_verify_buffers(gpu_idx):
    """Pre-allocate GPU buffers used by verify_chunk_on_gpu(). Must be called
    with the GPU's context already pushed."""
    chunk_bytes = CHUNK_MIB * 1024 * 1024
    granularity = GRANULARITIES[gpu_idx]
    alloc_size = ((chunk_bytes + granularity - 1) // granularity) * granularity
    VERIFY_LOCAL_BUFS[gpu_idx] = checkCudaErrors(driver.cuMemAlloc(alloc_size))
    VERIFY_LOCAL_BUF_SIZES[gpu_idx] = alloc_size

    partials_size = CHECKSUM_NUM_BLOCKS[gpu_idx] * ctypes.sizeof(ctypes.c_double)
    VERIFY_PARTIALS_BUFS[gpu_idx] = checkCudaErrors(driver.cuMemAlloc(partials_size))

    log.info("GPU %d: pre-allocated verify buffers: local_buf=%d bytes, partials=%d bytes",
             gpu_idx, alloc_size, partials_size)


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


def ensure_cuda_context(gpu_idx):
    """Retain and push the primary CUDA context for the given GPU onto the
    calling thread's stack.

    Safe to call multiple times — retaining an already-retained context just
    increments a refcount.
    """
    ctx = checkCudaErrors(driver.cuDevicePrimaryCtxRetain(CUDEVS[gpu_idx]))
    checkCudaErrors(driver.cuCtxPushCurrent(ctx))


def pop_cuda_context():
    """Pop the current CUDA context from the calling thread's stack."""
    checkCudaErrors(driver.cuCtxPopCurrent())


def prepare_all_shared_chunks():
    """Allocate, fill, and export a chunk on each GPU. Called once at startup."""
    for gpu_idx in range(GPUS_PER_NODE):
        ensure_cuda_context(gpu_idx)
        _prepare_shared_chunk_for_gpu(gpu_idx)
        pop_cuda_context()


def _prepare_shared_chunk_for_gpu(gpu_idx):
    """Prepare a single GPU's shared chunk. Context must already be pushed."""
    chunk_bytes = CHUNK_MIB * 1024 * 1024
    granularity = GRANULARITIES[gpu_idx]
    alloc_size = ((chunk_bytes + granularity - 1) // granularity) * granularity
    num_floats = alloc_size // 4

    alloc_handle = checkCudaErrors(
        driver.cuMemCreate(alloc_size, ALLOC_PROPS[gpu_idx], 0))

    va_ptr = checkCudaErrors(
        driver.cuMemAddressReserve(alloc_size, granularity, 0, 0))
    checkCudaErrors(driver.cuMemMap(va_ptr, alloc_size, 0, alloc_handle, 0))

    access_desc = driver.CUmemAccessDesc()
    access_desc.location = ALLOC_PROPS[gpu_idx].location
    access_desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    checkCudaErrors(driver.cuMemSetAccess(va_ptr, alloc_size, [access_desc], 1))

    # Fill GPU memory with FLOAT_VALUE directly on the GPU using cuMemsetD32.
    # cuMemsetD32 interprets the pattern as a uint32, so we reinterpret our
    # float32's bit pattern via struct pack/unpack.
    float_as_uint32 = struct.unpack("I", struct.pack("f", FLOAT_VALUE))[0]
    checkCudaErrors(driver.cuMemsetD32(va_ptr, float_as_uint32, num_floats))
    # cuMemsetD32 is asynchronous — wait for completion before exporting.
    checkCudaErrors(driver.cuCtxSynchronize())

    fabric_handle = checkCudaErrors(
        driver.cuMemExportToShareableHandle(
            alloc_handle,
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            0,
        )
    )

    handle_b64 = base64.urlsafe_b64encode(fabric_handle.data).decode("ascii")

    SHARED_CHUNK_METAS[gpu_idx] = json.dumps({
        "handle": handle_b64,
        "pod_name": K8S_PODNAME,
        "node_name": K8S_NODENAME,
        "gpu_index": gpu_idx,
        "num_floats": num_floats,
        "float_value": FLOAT_VALUE,
        "alloc_size": alloc_size,
    }).encode("utf-8")

    log.info("GPU %d: prepared shared chunk: %d MiB, %d floats",
             gpu_idx, alloc_size // (1024 * 1024), num_floats)


def fetch_chunk_meta(peer_host: str, port: int, gpu_index: int) -> dict:
    """GET /prepare-chunk?gpu_index=N from a peer. Returns parsed JSON metadata."""
    url = f"http://{peer_host}:{port}/prepare-chunk?gpu_index={gpu_index}"
    resp = requests.get(url, timeout=(5, 30))
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


def map_imported_chunk(alloc_handle, alloc_size: int, gpu_idx: int) -> int:
    """Map an imported allocation into local VA space. Returns the VA pointer."""
    granularity = GRANULARITIES[gpu_idx]
    va_ptr = checkCudaErrors(
        driver.cuMemAddressReserve(alloc_size, granularity, 0, 0)
    )
    checkCudaErrors(driver.cuMemMap(va_ptr, alloc_size, 0, alloc_handle, 0))

    access_desc = driver.CUmemAccessDesc()
    access_desc.location = ALLOC_PROPS[gpu_idx].location
    access_desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    checkCudaErrors(driver.cuMemSetAccess(va_ptr, alloc_size, [access_desc], 1))
    return va_ptr


CHECKSUM_REL_TOLERANCE = 1e-5


def verify_chunk_on_gpu(local_gpu_idx: int, va_ptr: int, alloc_size: int,
                        num_floats: int, expected_value: float) -> str:
    """Measure NVLink bandwidth via DtoD copy, then verify data via checksum.

    The measurement is split into two phases:

    Phase 1 — Bandwidth measurement: cuMemcpyDtoD from the remote-mapped VA
    to a local GPU buffer. This is a pure DMA transfer handled by the copy
    engine with zero SM involvement, giving the cleanest possible NVLink
    bandwidth number.

    Phase 2 — Data integrity: run the checksum kernel on the local copy.
    Because the data is now in local GPU memory, this is a fast local read
    and does not affect the NVLink bandwidth measurement.

    Context for local_gpu_idx must already be pushed.
    Returns a string like '756.3 GB/s' on success, or a mismatch description.
    """
    local_buf = VERIFY_LOCAL_BUFS[local_gpu_idx]
    assert alloc_size <= VERIFY_LOCAL_BUF_SIZES[local_gpu_idx], \
        f"chunk {alloc_size} exceeds pre-allocated buffer {VERIFY_LOCAL_BUF_SIZES[local_gpu_idx]}"

    # Phase 1: time the DtoD copy (pure NVLink transfer).
    ev_start = checkCudaErrors(driver.cuEventCreate(0))
    ev_end = checkCudaErrors(driver.cuEventCreate(0))

    checkCudaErrors(driver.cuEventRecord(ev_start, 0))
    checkCudaErrors(driver.cuMemcpyDtoD(local_buf, va_ptr, alloc_size))
    checkCudaErrors(driver.cuEventRecord(ev_end, 0))
    checkCudaErrors(driver.cuEventSynchronize(ev_end))

    elapsed_ms = checkCudaErrors(driver.cuEventElapsedTime(ev_start, ev_end))
    checkCudaErrors(driver.cuEventDestroy(ev_start))
    checkCudaErrors(driver.cuEventDestroy(ev_end))

    # Phase 2: checksum the local copy for data integrity.
    num_blocks = CHECKSUM_NUM_BLOCKS[local_gpu_idx]
    out_size = num_blocks * ctypes.sizeof(ctypes.c_double)
    partials_buf = VERIFY_PARTIALS_BUFS[local_gpu_idx]

    n = ctypes.c_int(num_floats)
    data_ptr = ctypes.c_void_p(int(local_buf))
    out_ptr_arg = ctypes.c_void_p(int(partials_buf))
    args = (ctypes.c_void_p * 3)(
        ctypes.addressof(data_ptr),
        ctypes.addressof(n),
        ctypes.addressof(out_ptr_arg),
    )

    checkCudaErrors(driver.cuLaunchKernel(
        CHECKSUM_KERNELS[local_gpu_idx],
        num_blocks, 1, 1,
        256, 1, 1,
        0, 0,
        args, 0,
    ))
    checkCudaErrors(driver.cuCtxSynchronize())

    result_buf = bytearray(out_size)
    checkCudaErrors(driver.cuMemcpyDtoH(result_buf, partials_buf, out_size))

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


def import_and_verify_chunk(peer_name: str, peer_host: str, port: int,
                            remote_gpu_idx: int, local_gpu_idx: int) -> tuple[str, str]:
    """Request a chunk from a specific remote GPU, import it on a specific local
    GPU, and verify contents.

    Returns (result_str, peer_node_name). result_str is a bandwidth like
    '818.3 GB/s' on success, or an error string.
    """
    try:
        meta = fetch_chunk_meta(peer_host, port, remote_gpu_idx)
    except Exception:
        log.exception("failed to fetch chunk meta from %s gpu %d:", peer_name, remote_gpu_idx)
        return ("req-err", "?")

    peer_node = meta.get("node_name", "?")
    alloc_size = meta["alloc_size"]
    handle_bytes = base64.urlsafe_b64decode(meta["handle"])

    imported_handle = None
    va_ptr = None
    result = "OK"

    ensure_cuda_context(local_gpu_idx)
    try:
        imported_handle = import_fabric_handle(handle_bytes)
        va_ptr = map_imported_chunk(imported_handle, alloc_size, local_gpu_idx)
        result = verify_chunk_on_gpu(local_gpu_idx, va_ptr, alloc_size,
                                     meta["num_floats"], meta["float_value"])
    except CudaError:
        log.exception("CUDA error importing/verifying chunk from %s g%d→g%d:",
                      peer_name, remote_gpu_idx, local_gpu_idx)
        result = "cuda-err"
    except Exception:
        log.exception("unexpected error importing/verifying chunk from %s g%d→g%d:",
                      peer_name, remote_gpu_idx, local_gpu_idx)
        result = "err"
    finally:
        try:
            unmap_imported_chunk(va_ptr, alloc_size, imported_handle)
        except Exception:
            log.exception("cleanup error for chunk from %s:", peer_name)
        pop_cuda_context()

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
        pod_name = target.split(".")[0]
        if pod_name != K8S_PODNAME:
            peers.append((pod_name, target))

    return peers


def peer_poll_loop():
    """Periodically discover peers and verify memory sharing with each.

    For each peer, iterates all remote GPU × local GPU pairs sequentially
    in a single thread, switching CUDA contexts as needed.
    """
    # Give the cluster a moment to settle.
    time.sleep(POLL_INTERVAL_S)

    while True:
        try:
            peers = discover_peers()
            if not peers:
                log.info("peer poll: no peers discovered yet")
                time.sleep(POLL_INTERVAL_S)
                continue

            # results: { "peer_idx@peer_node-gR-gL": status }
            results = {}
            peer_node_for_idx = {}

            for pod_name, peer_host in peers:
                peer_idx = pod_name.rsplit("-", 1)[1]

                for remote_gpu_idx in range(GPUS_PER_NODE):
                    for local_gpu_idx in range(GPUS_PER_NODE):
                        status, peer_node = import_and_verify_chunk(
                            pod_name, peer_host, HTTPD_PORT,
                            remote_gpu_idx, local_gpu_idx)
                        peer_node_for_idx[peer_idx] = peer_node
                        key = f"{peer_idx}@{peer_node}-g{remote_gpu_idx}-g{local_gpu_idx}"
                        results[key] = status

            my_idx = K8S_PODNAME.rsplit("-", 1)[1]
            parts = []
            for key, status in sorted(results.items()):
                parts.append(f"{key}:{status}")
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

        # Parse gpu_index query parameter.
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        gpu_index_vals = params.get("gpu_index")
        if not gpu_index_vals:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"missing gpu_index parameter")
            return

        try:
            gpu_idx = int(gpu_index_vals[0])
        except ValueError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"invalid gpu_index")
            return

        if gpu_idx not in SHARED_CHUNK_METAS:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(f"no chunk for gpu_index={gpu_idx}".encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(SHARED_CHUNK_METAS[gpu_idx])

    def log_message(self, format, *args):
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

    for gpu_idx in range(GPUS_PER_NODE):
        ensure_cuda_context(gpu_idx)
        props = checkCudaErrors(runtime.cudaGetDeviceProperties(gpu_idx))

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

        log.info("GPU %d properties:\n%s", gpu_idx, pformat(printprops))
        pop_cuda_context()


# ---------------------------------------------------------------------------
# CUDA error handling
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
