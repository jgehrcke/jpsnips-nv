"""
ATACK — All-To-All raw CUDA API-based MNNVL test runner for Kubernetes.

Measures NVLink bandwidth between all cross-node GPU pairs in a
Kubernetes StatefulSet. GPUs within the same node are not benchmarked
against each other — only pairs where source and destination are on
different nodes. Uses CU_MEM_HANDLE_TYPE_FABRIC handles exported via
IMEX for cross-node GPU memory access.

Each pod:
  - Allocates and fills GPU memory at startup per GPU, then refreshes
    every 30 seconds with a fresh allocation (old memory is freed). This
    exercises the full CUDA allocation + IMEX export path regularly.
  - On each HTTP request (GET /prepare-chunk?gpu_index=N), re-exports a
    fresh CU_MEM_HANDLE_TYPE_FABRIC handle via cuMemExportToShareableHandle.
    Both the periodic re-allocation and per-request re-export may help
    detect or recover from degraded IMEX daemon state.
  - Periodically benchmarks every (remote_gpu, local_gpu) pair across
    all peer pods: copies data from the remote GPU to the local GPU
    via cuMemcpyDtoD (device-to-device), timed with CUDA events.
  - Coordinates exclusive HBM access via per-GPU HTTP-based locks.
  - Reports bandwidth results for consumption by the dashboard.

Transfer duration is measured on-GPU via cuEventRecord before and after
cuMemcpyDtoD, and cuEventElapsedTime to compute the interval. This
excludes host-side overhead and reflects pure NVLink transfer time.

Recommended chunk size is 1000–4000 MiB. Larger allocations (e.g. 10 GB)
fail silently during cuMemcpyDtoD — likely a CUDA driver limitation
on the maximum size of a single fabric-handle-backed allocation or
transfer. At NVLink 5 speeds (~820 GB/s net unidirectional), a 4 GB
transfer takes approximately 4.9 ms per benchmark.

GPU locking: each benchmark transfers data from one GPU's HBM to
another's. A GPU's HBM bandwidth is finite — if two benchmarks use the
same GPU simultaneously (one reading from it, another writing to it),
they share HBM bandwidth and the measurements are distorted. To prevent
this, each GPU has a lock that must be held during the timed transfer.
A benchmark acquires locks on both the source GPU (remote, via HTTP)
and the destination GPU (local, in-process) before starting the copy.

Three mechanisms keep the locking robust:
  - Consistent ordering: locks are always acquired in ascending
    (pod_index, gpu_index) order to prevent deadlock between pods that
    would otherwise acquire each other's locks in opposite order.
  - Tokens: the lock-gpu HTTP endpoint returns a random token. The
    unlock-gpu endpoint only releases the lock if the token matches,
    preventing a late retry from releasing a lock that was since
    re-acquired by a different pod.
  - Auto-release watchdog: a background thread force-releases any
    remotely-acquired lock held longer than 30 seconds, guarding
    against crashed clients that acquired a lock but never released it.
    Local locks (held for ~5 ms) are not subject to the watchdog.
"""

import atexit
import base64
import ctypes
import json
import random
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
logging.Formatter.converter = time.gmtime
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03dZ %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
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
CHECKSUM_REL_TOLERANCE = 1e-5

# Per-GPU state, keyed by gpu_idx (0..GPUS_PER_NODE-1). Set during cuda_init().
CUDEVS = {}              # {gpu_idx: CUdevice}
ALLOC_PROPS = {}         # {gpu_idx: CUmemAllocationProp}
GRANULARITIES = {}       # {gpu_idx: int}
CHECKSUM_KERNELS = {}    # {gpu_idx: CUfunction}
CHECKSUM_NUM_BLOCKS = {} # {gpu_idx: int}

# Per-GPU allocation handles and static metadata, set once during startup.
# The allocation and fill happen once; the fabric handle is re-exported
# fresh on each HTTP request via cuMemExportToShareableHandle. This may
# help detect stale IMEX daemon state early.
SHARED_CHUNK_ALLOC_HANDLES = {}  # {gpu_idx: CUmemGenericAllocationHandle}
SHARED_CHUNK_STATIC_META = {}    # {gpu_idx: dict (without handle field)}
LAST_EXPORTED_HANDLE_BYTES = {}  # {gpu_idx: bytes} — for detecting identical re-exports

# Pre-allocated GPU buffers for verify_chunk_on_gpu(), per GPU.
# Eliminates per-round cuMemAlloc/cuMemFree churn which may contribute to
# bandwidth measurement variance we observed.
# Per-GPU locks to ensure exclusive HBM access during bandwidth measurement.
# A GPU's HBM is shared between local DtoD writes (receiving data) and remote
# DtoD reads (serving data to other pods). Without coordination, concurrent
# operations on the same GPU's HBM halve the measured bandwidth. Each lock
# ensures only one benchmark uses a given GPU's HBM at a time.
# The lock is acquired locally (for the DtoD destination GPU) and via HTTP
# (for the remote source GPU). Remote locks auto-expire after GPU_LOCK_TIMEOUT_S
# to handle crashed clients.
GPU_LOCK_TIMEOUT_S = 30
GPU_LOCK_ACQUIRE_HTTP_TIMEOUT_S = 25
GPU_LOCKS = {}           # {gpu_idx: threading.Lock}
GPU_LOCK_TIMESTAMPS = {} # {gpu_idx: monotonic time of last acquire}
GPU_LOCK_TOKENS = {}     # {gpu_idx: str} — token identifying current lock holder

VERIFY_LOCAL_BUFS = {}       # {gpu_idx: CUdeviceptr}
VERIFY_LOCAL_BUF_SIZES = {}  # {gpu_idx: int}
VERIFY_PARTIALS_BUFS = {}    # {gpu_idx: CUdeviceptr}

# Fatal CUDA error tracking. When set, the /healthz endpoint returns 500
# to fail the liveness probe, causing kubelet to replace the pod (not just
# restart the container). A new pod triggers fresh IMEX daemon resource
# claims. This is specifically for CUDA_ERROR_ILLEGAL_STATE which indicates
# unrecoverable GPU state corruption.
FATAL_CUDA_ERROR = None  # Set to error string on fatal error

# Shared chunk allocations tracked for cleanup on exit.
# {gpu_idx: (va_ptr, alloc_size, alloc_handle)}
SHARED_CHUNK_ALLOCS = {}


def cuda_cleanup():
    """Release all CUDA resources (shared chunks, verify buffers, contexts).

    Registered via atexit so that graceful exits (SIGTERM, SIGINT, normal
    return) clean up properly. SIGKILL cannot be caught — in that case the
    CUDA driver handles cleanup at process teardown.
    """
    log.info("cuda_cleanup: releasing GPU resources")
    for gpu_idx in list(SHARED_CHUNK_ALLOCS.keys()):
        va_ptr, alloc_size, alloc_handle = SHARED_CHUNK_ALLOCS[gpu_idx]
        try:
            ensure_cuda_context(gpu_idx)
            driver.cuMemUnmap(va_ptr, alloc_size)
            driver.cuMemRelease(alloc_handle)
            driver.cuMemAddressFree(va_ptr, alloc_size)
            pop_cuda_context()
        except Exception as exc:
            log.warning("cuda_cleanup: GPU %d shared chunk: %s", gpu_idx, exc)

    for gpu_idx in list(VERIFY_LOCAL_BUFS.keys()):
        try:
            ensure_cuda_context(gpu_idx)
            driver.cuMemFree(VERIFY_LOCAL_BUFS[gpu_idx])
            driver.cuMemFree(VERIFY_PARTIALS_BUFS[gpu_idx])
            pop_cuda_context()
        except Exception as exc:
            log.warning("cuda_cleanup: GPU %d verify buffers: %s", gpu_idx, exc)

    for gpu_idx in list(CUDEVS.keys()):
        try:
            driver.cuDevicePrimaryCtxRelease(CUDEVS[gpu_idx])
        except Exception as exc:
            log.warning("cuda_cleanup: GPU %d context release: %s", gpu_idx, exc)

    log.info("cuda_cleanup: done")


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
    atexit.register(cuda_cleanup)
    log_device_properties()
    prepare_all_shared_chunks()

    # Initialize per-GPU locks.
    for gpu_idx in range(GPUS_PER_NODE):
        GPU_LOCKS[gpu_idx] = threading.Lock()

    # Start threads.
    run_httpd_in_thread()
    start_gpu_lock_watchdog()
    threading.Thread(target=chunk_refresh_loop, daemon=True).start()
    start_peer_poll_thread()

    shutdown = threading.Event()
    signal.signal(signal.SIGTERM, lambda sig, frame: shutdown.set())
    signal.signal(signal.SIGINT, lambda sig, frame: shutdown.set())

    log.info("all threads started, main thread waiting")
    shutdown.wait()
    log.info("received shutdown signal, exiting")


def cuda_init():
    """Initialize CUDA runtime for all local GPUs.

    For each GPU: retains its primary context, validates capabilities,
    builds allocation properties, compiles the checksum kernel, and
    pre-allocates verification buffers.

    Raises:
        CudaError: On any CUDA API failure.
        AssertionError: If fewer devices are visible than GPUS_PER_NODE.
    """
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
    """Compile the checksum kernel via NVRTC and load it into the current context.

    Precondition: the GPU's CUDA context must be pushed on the calling thread.

    Raises:
        RuntimeError: On NVRTC compilation failure (includes compiler log).
        CudaError: On module load or function lookup failure.
    """
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
    """Pre-allocate GPU buffers used by verify_chunk_on_gpu().

    Allocates a DtoD destination buffer (CHUNK_MIB) and a checksum
    partials buffer on the given GPU. Stored in VERIFY_LOCAL_BUFS and
    VERIFY_PARTIALS_BUFS.

    Precondition: the GPU's CUDA context must be pushed on the calling thread.

    Raises:
        CudaError: On allocation failure.
    """
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
    """Unwrap an NVRTC API result tuple. Raises RuntimeError on failure."""
    if result[0].value:
        raise RuntimeError(f"NVRTC error: {result[0]}")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def ensure_cuda_context(gpu_idx):
    """Push the primary CUDA context for ``gpu_idx`` onto the calling thread.

    Safe to call repeatedly; retaining an already-retained context increments
    a refcount. Must be balanced with a pop_cuda_context() call.

    Raises:
        CudaError: If the context cannot be retained or pushed.
    """
    ctx = checkCudaErrors(driver.cuDevicePrimaryCtxRetain(CUDEVS[gpu_idx]))
    checkCudaErrors(driver.cuCtxPushCurrent(ctx))


def pop_cuda_context():
    """Pop the current CUDA context from the calling thread's stack.

    Raises:
        CudaError: If no context is active on this thread.
    """
    checkCudaErrors(driver.cuCtxPopCurrent())


def acquire_local_gpu_lock(gpu_idx) -> float:
    """Acquire the local GPU lock, blocking until available.

    Returns the wall-clock wait time in milliseconds.
    """
    log.debug("local lock GPU %d: acquiring", gpu_idx)
    t0 = time.monotonic()
    GPU_LOCKS[gpu_idx].acquire()
    GPU_LOCK_TIMESTAMPS[gpu_idx] = time.monotonic()
    wait_ms = (time.monotonic() - t0) * 1000
    log.debug("local lock GPU %d: acquired in %.1f ms", gpu_idx, wait_ms)
    return wait_ms


def release_local_gpu_lock(gpu_idx):
    """Release the local GPU lock. No-op if already released."""
    try:
        GPU_LOCKS[gpu_idx].release()
    except RuntimeError:
        pass  # Already released (e.g. by timeout watchdog).


def acquire_remote_gpu_lock(peer_host: str, port: int, gpu_index: int) -> tuple[str, float]:
    """Acquire a GPU lock on a remote pod via HTTP POST /lock-gpu.

    Blocks until the remote pod grants the lock or the HTTP timeout fires.

    Returns:
        (token, wait_ms): The lock token for release, and wall-clock wait.

    Raises:
        RuntimeError: If the remote pod returns a non-200 status.
        requests.exceptions.RequestException: On connection or timeout failure.
    """
    log.debug("remote lock %s GPU %d: acquiring", peer_host, gpu_index)
    t0 = time.monotonic()
    resp = requests.post(
        f"http://{peer_host}:{port}/lock-gpu?gpu_index={gpu_index}",
        timeout=(1, GPU_LOCK_ACQUIRE_HTTP_TIMEOUT_S),
    )
    if resp.status_code != 200:
        raise RuntimeError(f"lock-gpu failed: HTTP {resp.status_code}: {resp.text}")
    wait_ms = (time.monotonic() - t0) * 1000
    log.debug("remote lock %s GPU %d: acquired in %.1f ms",
              peer_host, gpu_index, wait_ms)
    return (resp.text, wait_ms)


def release_remote_gpu_lock(peer_host: str, port: int, gpu_index: int,
                            token: str):
    """Release a GPU lock on a remote pod via HTTP POST /unlock-gpu.

    Retries up to 7 times (0.5s apart) to avoid leaving a stale lock that
    blocks other pods until the watchdog force-releases it. The token
    prevents accidentally releasing a lock re-acquired by another pod.

    Never raises — failures are logged as warnings.
    """
    url = f"http://{peer_host}:{port}/unlock-gpu?gpu_index={gpu_index}&token={token}"
    for attempt in range(7):
        try:
            requests.post(url, timeout=(1, 2))
            return
        except Exception as exc:
            log.warning("failed to release remote GPU lock on %s gpu %d "
                        "(attempt %d/7): %s", peer_host, gpu_index,
                        attempt + 1, exc)
            time.sleep(0.5)


def start_gpu_lock_watchdog():
    """Start a daemon thread that force-releases GPU locks held longer than
    GPU_LOCK_TIMEOUT_S. Guards against crashed remote clients that acquired
    a lock via HTTP but never released it."""
    def watchdog():
        while True:
            time.sleep(1)
            now = time.monotonic()
            for gpu_idx in list(GPU_LOCK_TIMESTAMPS.keys()):
                ts = GPU_LOCK_TIMESTAMPS.get(gpu_idx)
                if ts is None:
                    continue
                if now - ts > GPU_LOCK_TIMEOUT_S:
                    if GPU_LOCKS[gpu_idx].locked():
                        log.warning("GPU lock watchdog: force-releasing GPU %d "
                                    "(held for %.0fs)", gpu_idx, now - ts)
                        GPU_LOCK_TOKENS.pop(gpu_idx, None)
                        release_local_gpu_lock(gpu_idx)
                    GPU_LOCK_TIMESTAMPS.pop(gpu_idx, None)

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()


CHUNK_REFRESH_INTERVAL_S = 30


def prepare_all_shared_chunks():
    """Allocate, fill, and export a fabric handle on each local GPU.

    Called once at startup. Populates SHARED_CHUNK_ALLOC_HANDLES and
    SHARED_CHUNK_STATIC_META.

    Raises:
        CudaError: On any GPU allocation, fill, or export failure.
    """
    for gpu_idx in range(GPUS_PER_NODE):
        ensure_cuda_context(gpu_idx)
        _prepare_shared_chunk_for_gpu(gpu_idx)
        pop_cuda_context()


def _free_shared_chunk_for_gpu(gpu_idx):
    """Free the old shared chunk allocation for one GPU.

    Precondition: the GPU's CUDA context must be pushed.
    """
    old = SHARED_CHUNK_ALLOCS.get(gpu_idx)
    if old is None:
        return
    va_ptr, alloc_size, alloc_handle = old
    # Order: unmap → release handle → free VA range.
    driver.cuMemUnmap(va_ptr, alloc_size)
    driver.cuMemRelease(alloc_handle)
    driver.cuMemAddressFree(va_ptr, alloc_size)


def _refresh_shared_chunk_for_gpu(gpu_idx):
    """Replace the shared chunk on one GPU with a fresh allocation.

    Acquires the GPU lock to prevent benchmarks from reading during the
    swap. The lock hold time is ~1-2 ms (allocate + fill + sync).

    Precondition: the GPU's CUDA context must be pushed.

    Raises:
        CudaError: On allocation or memset failure.
    """
    acquire_local_gpu_lock(gpu_idx)
    try:
        _free_shared_chunk_for_gpu(gpu_idx)
        _prepare_shared_chunk_for_gpu(gpu_idx)
    finally:
        release_local_gpu_lock(gpu_idx)


def chunk_refresh_loop():
    """Periodically replace shared chunks with fresh allocations on all GPUs.

    This exercises the full CUDA allocation + IMEX export path regularly,
    rather than relying on memory allocated once at startup. May help
    detect or recover from degraded IMEX daemon state.
    """
    while True:
        time.sleep(CHUNK_REFRESH_INTERVAL_S)
        for gpu_idx in range(GPUS_PER_NODE):
            t0 = time.monotonic()
            ensure_cuda_context(gpu_idx)
            try:
                _refresh_shared_chunk_for_gpu(gpu_idx)
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.info("refreshed shared chunk on GPU %d in %.1f ms", gpu_idx, elapsed_ms)
            except Exception:
                log.exception("failed to refresh shared chunk on GPU %d:", gpu_idx)
            finally:
                pop_cuda_context()


def _prepare_shared_chunk_for_gpu(gpu_idx):
    """Allocate and fill GPU memory for one GPU. Does not export a handle —
    that happens per HTTP request in export_fabric_handle_for_gpu().

    Precondition: the GPU's CUDA context must be pushed.

    Raises:
        CudaError: On allocation or memset failure.
    """
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
    # cuMemsetD32 is asynchronous — wait for completion.
    checkCudaErrors(driver.cuCtxSynchronize())

    # Track for cleanup on exit.
    SHARED_CHUNK_ALLOCS[gpu_idx] = (va_ptr, alloc_size, alloc_handle)
    SHARED_CHUNK_ALLOC_HANDLES[gpu_idx] = alloc_handle
    SHARED_CHUNK_STATIC_META[gpu_idx] = {
        "pod_name": K8S_PODNAME,
        "node_name": K8S_NODENAME,
        "gpu_index": gpu_idx,
        "num_floats": num_floats,
        "float_value": FLOAT_VALUE,
        "alloc_size": alloc_size,
    }

    log.info("GPU %d: prepared shared chunk: %d MiB, %d floats",
             gpu_idx, alloc_size // (1024 * 1024), num_floats)


def export_fabric_handle_for_gpu(gpu_idx) -> bytes:
    """Export a fresh CU_MEM_HANDLE_TYPE_FABRIC handle and return the
    complete chunk metadata as JSON bytes.

    Called on every /prepare-chunk request. Re-exporting the handle each
    time (rather than caching it) may help detect stale IMEX daemon state.
    Logs a warning if the export takes longer than 10 ms.

    Precondition: the GPU's CUDA context must be pushed.

    Raises:
        CudaError: If cuMemExportToShareableHandle fails.
    """
    t0 = time.monotonic()
    fabric_handle = checkCudaErrors(
        driver.cuMemExportToShareableHandle(
            SHARED_CHUNK_ALLOC_HANDLES[gpu_idx],
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            0,
        )
    )
    elapsed_ms = (time.monotonic() - t0) * 1000
    if elapsed_ms > 10:
        log.warning("cuMemExportToShareableHandle for GPU %d took %.1f ms",
                    gpu_idx, elapsed_ms)

    # Empirically verified: cuMemExportToShareableHandle returns the same
    # opaque byte sequence for the same underlying allocation across calls.
    # We re-export anyway (rather than caching) because it exercises the
    # IMEX daemon code path and may surface stale state earlier.
    handle_bytes = bytes(fabric_handle.data)
    prev = LAST_EXPORTED_HANDLE_BYTES.get(gpu_idx)
    if prev is not None and handle_bytes == prev:
        log.debug("GPU %d: re-exported handle is identical to previous", gpu_idx)
    LAST_EXPORTED_HANDLE_BYTES[gpu_idx] = handle_bytes

    handle_b64 = base64.urlsafe_b64encode(handle_bytes).decode("ascii")
    meta = dict(SHARED_CHUNK_STATIC_META[gpu_idx])
    meta["handle"] = handle_b64
    return json.dumps(meta).encode("utf-8")


def fetch_chunk_meta(peer_host: str, port: int, gpu_index: int) -> dict:
    """Fetch chunk metadata from a peer pod via GET /prepare-chunk?gpu_index=N.

    Raises:
        requests.exceptions.HTTPError: On non-200 response.
        requests.exceptions.RequestException: On connection or timeout failure.
    """
    url = f"http://{peer_host}:{port}/prepare-chunk?gpu_index={gpu_index}"
    resp = requests.get(url, timeout=(5, 30))
    if resp.status_code != 200:
        log.error("peer returned HTTP %s: %s", resp.status_code, resp.text)
        resp.raise_for_status()
    return resp.json()


def import_fabric_handle(handle_bytes: bytes):
    """Import a CUDA fabric handle from raw bytes.

    Raises:
        CudaError: If the handle cannot be imported.
    """
    return checkCudaErrors(
        driver.cuMemImportFromShareableHandle(
            handle_bytes,
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
        )
    )


def map_imported_chunk(alloc_handle, alloc_size: int, gpu_idx: int) -> int:
    """Map an imported allocation into the local GPU's VA space.

    Raises:
        CudaError: On address reservation, mapping, or access control failure.
    """
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



def verify_chunk_on_gpu(local_gpu_idx: int, va_ptr: int, alloc_size: int,
                        num_floats: int, expected_value: float) -> tuple[str, float]:
    """Measure NVLink bandwidth and verify data integrity for a mapped chunk.

    Phase 1 — cuMemcpyDtoD from the remote-mapped VA to a pre-allocated
    local buffer. Timed via CUDA events. This is a pure DMA copy engine
    transfer with no SM involvement, yielding a clean NVLink bandwidth
    measurement.

    Phase 2 — Checksum kernel on the local copy. Verifies data integrity
    without affecting the bandwidth measurement (local memory read only).

    Precondition: CUDA context for local_gpu_idx must be pushed.

    Returns:
        (result_str, elapsed_ms): Bandwidth string like '818.5 GB/s' or a
        CHECKSUM MISMATCH description, plus the raw DtoD transfer time.

    Raises:
        CudaError: On any CUDA operation failure.
        AssertionError: If alloc_size exceeds the pre-allocated buffer.
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
        return (f"CHECKSUM MISMATCH: gpu_sum={gpu_sum}, expected={expected_sum}",
                elapsed_ms)

    bw_gbs = alloc_size / (elapsed_ms / 1000.0) / 1e9
    return (f"{bw_gbs:.1f} GB/s", elapsed_ms)


def unmap_imported_chunk(va_ptr, alloc_size: int, alloc_handle):
    """Unmap and release a locally-imported GPU memory chunk.

    Safe to call with None arguments (no-ops for cleanup convenience).

    Raises:
        CudaError: On unmap, address free, or release failure.
    """
    # Order: unmap → release physical handle → free VA range.
    # Must unmap before release, per CUDA driver API contract.
    if va_ptr is not None:
        checkCudaErrors(driver.cuMemUnmap(va_ptr, alloc_size))
    if alloc_handle is not None:
        checkCudaErrors(driver.cuMemRelease(alloc_handle))
    if va_ptr is not None:
        checkCudaErrors(driver.cuMemAddressFree(va_ptr, alloc_size))


def acquire_gpu_lock_pair(peer_name, peer_host, port, remote_gpu_idx,
                          local_gpu_idx):
    """Acquire locks on both the remote source GPU and local destination GPU.

    Locks are always acquired in ascending (pod_idx, gpu_idx) order to
    prevent deadlock between pods that would otherwise acquire each
    other's locks in opposite order.

    Returns:
        (remote_token, lock_wait_ms): Token for remote unlock, max wait time.

    Raises:
        RuntimeError: If the remote lock HTTP request fails.
        requests.exceptions.RequestException: On connection/timeout failure.
    """
    my_idx = int(K8S_PODNAME.rsplit("-", 1)[1])
    peer_idx_int = int(peer_name.rsplit("-", 1)[1])
    local_key = (my_idx, local_gpu_idx)
    remote_key = (peer_idx_int, remote_gpu_idx)

    if local_key < remote_key:
        local_wait = acquire_local_gpu_lock(local_gpu_idx)
        try:
            remote_token, remote_wait = acquire_remote_gpu_lock(
                peer_host, port, remote_gpu_idx)
        except Exception:
            release_local_gpu_lock(local_gpu_idx)
            raise
    else:
        remote_token, remote_wait = acquire_remote_gpu_lock(
            peer_host, port, remote_gpu_idx)
        local_wait = acquire_local_gpu_lock(local_gpu_idx)

    return remote_token, max(local_wait, remote_wait)


def import_map_and_verify(peer_host, port, remote_gpu_idx, local_gpu_idx,
                          handle_bytes, alloc_size, num_floats, float_value):
    """Import a remote fabric handle, map it locally, benchmark, and clean up.

    Manages CUDA context push/pop, lock acquire/release, and VA
    unmap/release internally. Caller must not hold any GPU locks.

    Returns:
        (result_str, lock_wait_ms, benchmark_ms).

    Raises:
        CudaError: On import, mapping, or benchmark failure.
        RuntimeError: On lock acquisition failure.
    """
    peer_name = peer_host.split(".")[0]
    ensure_cuda_context(local_gpu_idx)

    imported_handle = import_fabric_handle(handle_bytes)
    va_ptr = map_imported_chunk(imported_handle, alloc_size, local_gpu_idx)

    try:
        remote_token, lock_wait_ms = acquire_gpu_lock_pair(
            peer_name, peer_host, port, remote_gpu_idx, local_gpu_idx)
        try:
            result, benchmark_ms = verify_chunk_on_gpu(
                local_gpu_idx, va_ptr, alloc_size, num_floats, float_value)
        finally:
            release_local_gpu_lock(local_gpu_idx)
            release_remote_gpu_lock(peer_host, port, remote_gpu_idx,
                                    remote_token)
    finally:
        unmap_imported_chunk(va_ptr, alloc_size, imported_handle)
        pop_cuda_context()

    return result, lock_wait_ms, benchmark_ms


def import_and_verify_chunk(peer_name: str, peer_host: str, port: int,
                            remote_gpu_idx: int,
                            local_gpu_idx: int) -> tuple[str, str, float, float]:
    """Run one benchmark: fetch remote chunk metadata, import, measure, verify.

    This is the top-level entry point for a single GPU-to-GPU measurement.
    Handles all errors internally — never raises.

    Returns:
        (result_str, peer_node_name, lock_wait_ms, benchmark_ms).
        On failure, result_str is an error tag and timing values are 0.0.
    """
    try:
        meta = fetch_chunk_meta(peer_host, port, remote_gpu_idx)
    except Exception:
        log.exception("fetch chunk meta from %s gpu %d failed:",
                      peer_name, remote_gpu_idx)
        return ("req-err", "?", 0.0, 0.0)

    peer_node = meta.get("node_name", "?")
    handle_bytes = base64.urlsafe_b64decode(meta["handle"])

    try:
        result, lock_wait_ms, benchmark_ms = import_map_and_verify(
            peer_host, port, remote_gpu_idx, local_gpu_idx,
            handle_bytes, meta["alloc_size"], meta["num_floats"],
            meta["float_value"])
    except CudaError:
        log.exception("CUDA error %s g%d→g%d:", peer_name, remote_gpu_idx,
                      local_gpu_idx)
        return ("cuda-err", peer_node, 0.0, 0.0)
    except Exception:
        log.exception("error %s g%d→g%d:", peer_name, remote_gpu_idx,
                      local_gpu_idx)
        return ("err", peer_node, 0.0, 0.0)

    log.debug("benchmark done %s-g%d -> local-g%d: %.1f ms, %s",
              peer_name, remote_gpu_idx, local_gpu_idx, benchmark_ms, result)
    return (result, peer_node, lock_wait_ms, benchmark_ms)


def discover_peers() -> list[tuple[str, str]]:
    """Discover peer pods via DNS SRV lookup on the headless Service.

    Returns:
        List of (pod_name, fqdn) for all peers except self.

    Raises:
        dns.resolver.NXDOMAIN: If the SRV record does not exist yet.
        dns.resolver.NoAnswer: If the DNS server has no SRV records.
        dns.resolver.LifetimeTimeout: If the DNS query times out.
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


def _run_one_poll_round():
    """Run one round of all-to-all benchmarks against discovered peers.

    Discovers peers, builds all (peer, remote_gpu, local_gpu) work items
    in randomized order (to spread lock contention), benchmarks each pair,
    and logs round stats + results.

    Raises:
        dns.resolver.NXDOMAIN: If peer DNS does not exist yet.
    """
    peers = discover_peers()
    if not peers:
        log.info("peer poll: no peers discovered yet")
        return

    # Randomize to spread lock contention — without this, all pods try
    # to lock the same GPUs at the same time.
    work_items = [
        (pod_name, peer_host, rg, lg)
        for pod_name, peer_host in peers
        for rg in range(GPUS_PER_NODE)
        for lg in range(GPUS_PER_NODE)
    ]
    random.shuffle(work_items)

    results = {}
    max_lock_wait_ms = 0.0
    benchmark_durations = []

    for pod_name, peer_host, remote_gpu_idx, local_gpu_idx in work_items:
        peer_idx = pod_name.rsplit("-", 1)[1]
        log.debug("benchmark %s-g%d -> local-g%d",
                  pod_name, remote_gpu_idx, local_gpu_idx)
        status, peer_node, lock_wait, bench_ms = import_and_verify_chunk(
            pod_name, peer_host, HTTPD_PORT,
            remote_gpu_idx, local_gpu_idx)
        max_lock_wait_ms = max(max_lock_wait_ms, lock_wait)
        if bench_ms > 0:
            benchmark_durations.append(bench_ms)
        key = f"{peer_idx}@{peer_node}-g{remote_gpu_idx}-g{local_gpu_idx}"
        results[key] = status

    if benchmark_durations:
        bmin = min(benchmark_durations)
        bmax = max(benchmark_durations)
        bmean = sum(benchmark_durations) / len(benchmark_durations)
        log.info("round stats: DtoD min=%.1f max=%.1f mean=%.1f ms, "
                 "max_lock_wait=%.1f ms", bmin, bmax, bmean, max_lock_wait_ms)

    my_idx = K8S_PODNAME.rsplit("-", 1)[1]
    parts = [f"{k}:{v}" for k, v in sorted(results.items())]
    log.info("result(%s@%s): %s", my_idx, K8S_NODENAME, " ".join(parts))


def peer_poll_loop():
    """Deadline-based poll loop: starts a new round every POLL_INTERVAL_S
    seconds, measured from the start of the previous round."""
    time.sleep(POLL_INTERVAL_S)
    next_deadline = time.monotonic()

    while True:
        next_deadline += POLL_INTERVAL_S
        try:
            _run_one_poll_round()
        except dns.resolver.NXDOMAIN:
            log.info("peer poll: DNS name does not exist yet (expected at startup)")
        except Exception:
            log.exception("peer poll error:")
        remaining = next_deadline - time.monotonic()
        log.info("waiting %.1fs for next poll deadline", max(0, remaining))
        _wait_until(next_deadline)


def _wait_until(deadline):
    """Busy-wait until the given monotonic-clock deadline (10ms resolution)."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.01))


def start_peer_poll_thread():
    t = threading.Thread(target=peer_poll_loop, daemon=True)
    t.start()




class HTTPHandler(BaseHTTPRequestHandler):

    def _respond(self, code, body):
        """Send an HTTP response. Accepts str or bytes body."""
        self.send_response(code)
        self.end_headers()
        if isinstance(body, str):
            body = body.encode()
        self.wfile.write(body)

    def _parse_gpu_index(self):
        """Parse gpu_index from the query string.

        Returns (parsed_url, params, gpu_idx) on success, or sends a
        400 error response and returns None.
        """
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        gpu_index_vals = params.get("gpu_index")
        if not gpu_index_vals:
            self._respond(400, b"missing gpu_index parameter")
            return None
        try:
            gpu_idx = int(gpu_index_vals[0])
        except ValueError:
            self._respond(400, b"invalid gpu_index")
            return None
        return parsed, params, gpu_idx

    def do_GET(self):
        if "/healthz" in self.path:
            if FATAL_CUDA_ERROR:
                self._respond(500, f"fatal: {FATAL_CUDA_ERROR}")
            else:
                self._respond(200, b"ok")
            return

        if "/prepare-chunk" not in self.path:
            self._respond(404, b"unknown path")
            return

        result = self._parse_gpu_index()
        if result is None:
            return
        _, _, gpu_idx = result

        if gpu_idx not in SHARED_CHUNK_ALLOC_HANDLES:
            self._respond(404, f"no chunk for gpu_index={gpu_idx}")
            return

        ensure_cuda_context(gpu_idx)
        try:
            body = export_fabric_handle_for_gpu(gpu_idx)
        except CudaError as exc:
            pop_cuda_context()
            log.exception("export fabric handle for GPU %d failed:", gpu_idx)
            self._respond(500, str(exc))
            return
        pop_cuda_context()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        result = self._parse_gpu_index()
        if result is None:
            return
        parsed, params, gpu_idx = result

        if gpu_idx not in GPU_LOCKS:
            self._respond(404, f"no GPU {gpu_idx}")
            return

        if "/lock-gpu" in parsed.path:
            self._handle_lock_gpu(gpu_idx)
        elif "/unlock-gpu" in parsed.path:
            self._handle_unlock_gpu(gpu_idx, params)
        else:
            self._respond(404, b"unknown path")

    def _handle_lock_gpu(self, gpu_idx):
        """Block until GPU lock is acquired, respond with a lock token."""
        log.debug("HTTPD lock-gpu %d: request from %s, waiting",
                  gpu_idx, self.client_address[0])
        t0 = time.monotonic()
        acquired = GPU_LOCKS[gpu_idx].acquire(timeout=GPU_LOCK_TIMEOUT_S)
        wait_ms = (time.monotonic() - t0) * 1000

        if not acquired:
            self._respond(503, b"lock acquisition timed out")
            return

        log.debug("HTTPD lock-gpu %d: granted to %s after %.1f ms",
                  gpu_idx, self.client_address[0], wait_ms)
        token = uuid.uuid4().hex[:12]
        GPU_LOCK_TOKENS[gpu_idx] = token
        GPU_LOCK_TIMESTAMPS[gpu_idx] = time.monotonic()
        self._respond(200, token)

    def _handle_unlock_gpu(self, gpu_idx, params):
        """Release GPU lock if token matches. Idempotent; always returns 200."""
        token_vals = params.get("token")
        token = token_vals[0] if token_vals else None
        current_token = GPU_LOCK_TOKENS.get(gpu_idx)

        if current_token is not None and token != current_token:
            # Token mismatch: a different holder owns this lock now.
            # Return success — the caller's lock was already released
            # (by watchdog or timeout). Don't disturb the current holder.
            log.warning("unlock-gpu %d: token mismatch (got %s, current %s), "
                        "ignoring stale unlock", gpu_idx, token, current_token)
            self._respond(200, b"stale-unlock-ignored")
            return

        GPU_LOCK_TOKENS.pop(gpu_idx, None)
        release_local_gpu_lock(gpu_idx)
        self._respond(200, b"unlocked")

    def log_message(self, format, *args):
        pass


def run_httpd_in_thread():
    def run():
        log.info("starting HTTP server on port %s", HTTPD_PORT)
        s = ThreadingHTTPServer(("0.0.0.0", HTTPD_PORT), HTTPHandler)
        s.serve_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()




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
    """Unwrap a CUDA driver API result tuple.

    Raises CudaError if the status code indicates failure. On success,
    returns the payload (single value or tuple of values), or None if
    the result contains only the status code.
    """
    global FATAL_CUDA_ERROR
    if result[0].value:
        error_name = _cudaGetErrorEnum(result[0])
        error_msg = "CUDA error code={}({})".format(result[0].value, error_name)

        # CUDA_ERROR_ILLEGAL_STATE (code 401) indicates unrecoverable GPU
        # state corruption. Flag it so /healthz fails the liveness probe,
        # causing kubelet to replace this pod with a fresh one (new IMEX
        # daemon resource claims).
        if result[0] == driver.CUresult.CUDA_ERROR_ILLEGAL_STATE:
            FATAL_CUDA_ERROR = error_msg
            log.error("FATAL: %s — liveness probe will fail, pod will be replaced",
                      error_msg)

        raise CudaError(error_msg)
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
