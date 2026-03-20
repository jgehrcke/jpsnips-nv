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

Graceful shutdown (SIGTERM/SIGINT):
  1. SHUTTING_DOWN event is set. The HTTP handler immediately starts
     rejecting /prepare-chunk (503) and /lock-gpu (503), preventing
     new benchmarks from importing our handles. /evict-peer and
     /unlock-gpu remain accepted throughout shutdown.
  2. Wait for remotely-held local GPU locks to be released. A peer
     holding our lock is mid-DtoD from our memory — we must wait for
     that to complete before freeing.
  3. Broadcast POST /evict-peer to all peers. Each peer unmaps and
     releases its cached imports of our fabric handles. The HTTP
     response confirms the peer has completed cleanup, so when the
     broadcast returns no peer holds any reference to our GPU memory.
  4. cuda_cleanup releases all local CUDA resources: import cache
     entries (our cached imports of peer handles), shared chunks,
     verify buffers, and device contexts. Idempotent — also registered
     via atexit as a safety net.
  5. A hard-exit watchdog thread kills the process after 10s if atexit
     cleanup hangs (e.g. a CUDA driver call blocks indefinitely).
"""

import atexit
import base64
import collections
import concurrent.futures
import ctypes
import datetime
import itertools
import json
import logging
import os
import random
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
import orjson
import requests
from requests.adapters import HTTPAdapter

from cuda.bindings import driver, runtime, nvrtc


class AtackError(Exception):
    """Base exception for atack application errors."""


class CudaError(AtackError):
    """Raised by cucheck() for CUDA API failures."""


class LockError(AtackError):
    """Raised when GPU lock acquisition fails (timeout, connection error)."""


class PeerUnreachableError(AtackError):
    """Raised when a peer pod cannot be reached (DNS failure, connection refused).
    The caller should skip all remaining GPU pairs for this peer."""


log = logging.getLogger()
logging.Formatter.converter = time.gmtime
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03dZ %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# Fatal CUDA error codes: unrecoverable GPU/context corruption.
_FATAL_CUDA_CODES = (
    driver.CUresult.CUDA_ERROR_ILLEGAL_STATE,
    driver.CUresult.CUDA_ERROR_LAUNCH_FAILED,
)


def _cuda_get_error_name(error):
    """Get human-readable name for a CUDA or NVRTC error code."""
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def cucheck(result):
    """Unwrap a CUDA driver API result tuple.

    Raises CudaError if the status code indicates failure. Sets
    FATAL_CUDA_ERROR for unrecoverable errors (ILLEGAL_STATE,
    LAUNCH_FAILED) so /healthz returns 500.

    On success, returns the payload value(s) or None.
    """
    global FATAL_CUDA_ERROR
    if result[0].value:
        error_name = _cuda_get_error_name(result[0])
        error_msg = f"CUDA error code={result[0].value}({error_name})"

        if result[0] in _FATAL_CUDA_CODES:
            was_healthy = FATAL_CUDA_ERROR is None
            FATAL_CUDA_ERROR = error_msg
            if was_healthy:
                log.error("fatal CUDA error, setting FATAL_CUDA_ERROR: %s "
                          "(/healthz will return 500)", error_msg)
            else:
                log.error("fatal CUDA error (FATAL_CUDA_ERROR already set): "
                          "%s", error_msg)

        raise CudaError(error_msg)
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


# HTTP session with no internal retries. Retrying is done explicitly
# in the calling code where we can control timing and log level.
_http_session = requests.Session()
_http_session.mount("http://", HTTPAdapter(max_retries=0))
_http_session.mount("https://", HTTPAdapter(max_retries=0))

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
LAST_CHUNK_SERVED_TIME = {}     # {gpu_idx: monotonic} — when /prepare-chunk last served this GPU

# Per-GPU locks to ensure exclusive HBM access during bandwidth measurement.
# A GPU's HBM is shared between local DtoD writes (receiving data) and remote
# DtoD reads (serving data to other pods). Without coordination, concurrent
# operations on the same GPU's HBM halve the measured bandwidth. Each lock
# ensures only one benchmark uses a given GPU's HBM at a time.
# The lock is acquired locally (for the DtoD destination GPU) and via HTTP
# (for the remote source GPU). Remote locks auto-expire after GPU_LOCK_TIMEOUT_S
# to handle crashed clients.
GPU_LOCK_TIMEOUT_S = 5
GPU_LOCK_ACQUIRE_HTTP_TIMEOUT_S = 8
GPU_LOCKS = {}           # {gpu_idx: threading.Lock}
GPU_LOCK_TIMESTAMPS = {} # {gpu_idx: monotonic time of last acquire}
GPU_LOCK_TOKENS = {}     # {gpu_idx: str} — token identifying current lock holder
GPU_LOCK_HOLDERS = {}    # {gpu_idx: str} — who holds the lock (IP or "local")

# Pre-allocated GPU buffers for verify_chunk_on_gpu(), per GPU.
# Avoids per-round cuMemAlloc/cuMemFree churn.
VERIFY_LOCAL_BUFS = {}       # {gpu_idx: CUdeviceptr}
VERIFY_LOCAL_BUF_SIZES = {}  # {gpu_idx: int}
VERIFY_PARTIALS_BUFS = {}    # {gpu_idx: CUdeviceptr}

# Set to error string by cucheck() on fatal CUDA errors.
# When set, /healthz returns 500 to fail the liveness probe.
FATAL_CUDA_ERROR = None

# Timestamp of last successful result emission. The /healthz endpoint
# returns 500 if no result was produced within 3× the poll interval,
# indicating the pod is stuck (hung CUDA call, dead thread, etc.).
LAST_RESULT_TIME = None       # Set after ANY round (even partial).

# Recent round results for the /results endpoint.
# Deque of completed results, most recent last.
RESULTS_HISTORY_MAX = 5
RESULTS_HISTORY = collections.deque(maxlen=RESULTS_HISTORY_MAX)

# Graceful shutdown coordination. Set by SIGTERM/SIGINT handler.
# Components check this to wind down cleanly.
SHUTTING_DOWN = threading.Event()

# Cache of imported fabric handles and VA mappings. Avoids the expensive
# cuMemImportFromShareableHandle + cuMemMap + cuMemSetAccess on every
# benchmark (~35-100ms). The cache key is (local_gpu_idx, handle_bytes).
# Entries are evicted when the handle bytes change (remote chunk refresh),
# the remote peer disappears, or the peer requests eviction during shutdown.
# {(local_gpu_idx, handle_bytes): (imported_handle, va_ptr, alloc_size, peer_pod_name)}
IMPORT_CACHE = {}

# Shared chunk allocations tracked for cleanup on exit.
# {gpu_idx: (va_ptr, alloc_size, alloc_handle)}
SHARED_CHUNK_ALLOCS = {}

# Retired chunk allocations kept alive before freeing. A remote pod may have
# imported an old fabric handle and could still be doing a DtoD copy from it.
# Freeing immediately would cause CUDA_ERROR_INVALID_HANDLE or SIGSEGV.
# Chunks are freed only after CHUNK_RETIRE_S seconds (3 refresh cycles).
CHUNK_REFRESH_INTERVAL_S = 30
CHUNK_RETIRE_S = CHUNK_REFRESH_INTERVAL_S * 3
RETIRED_CHUNKS = []  # [(gpu_idx, va_ptr, alloc_size, alloc_handle, retire_time)]

# Track currently held remote GPU locks for best-effort release on exit.
# Set of (peer_host, port, gpu_index, token).
HELD_REMOTE_LOCKS = set()


_cuda_cleanup_done = False

def cuda_cleanup():
    """Release all CUDA resources (shared chunks, verify buffers, contexts).

    Called explicitly during graceful shutdown and registered via atexit as
    a safety net. Idempotent — second call is a no-op.
    """
    global _cuda_cleanup_done
    if _cuda_cleanup_done:
        return
    _cuda_cleanup_done = True
    t0 = time.monotonic()
    log.info("cuda_cleanup: releasing GPU resources")

    # Release cached imports.
    for key in list(IMPORT_CACHE.keys()):
        imported_handle, va_ptr, alloc_size, _peer = IMPORT_CACHE.pop(key)
        local_gpu_idx = key[0]
        try:
            ensure_cuda_context(local_gpu_idx)
            unmap_imported_chunk(va_ptr, alloc_size, imported_handle)
            pop_cuda_context()
        except Exception as exc:
            log.warning("cuda_cleanup: import cache entry: %s", exc)

    # Best-effort release of any remote GPU locks we still hold.
    # Use very short timeouts — we're in teardown and may be killed soon.
    for peer_host, port, gpu_index, token in list(HELD_REMOTE_LOCKS):
        try:
            url = (f"http://{peer_host}:{port}/unlock-gpu"
                   f"?gpu_index={gpu_index}&token={token}")
            _http_session.post(url, timeout=(0.3, 0.5))
            log.info("cuda_cleanup: released remote lock %s gpu %d",
                     peer_host, gpu_index)
        except Exception:
            log.warning("cuda_cleanup: failed to release remote lock %s gpu %d",
                        peer_host, gpu_index)
    HELD_REMOTE_LOCKS.clear()

    for gpu_idx in list(SHARED_CHUNK_ALLOCS.keys()):
        entry = SHARED_CHUNK_ALLOCS[gpu_idx]
        if entry is None:
            continue
        va_ptr, alloc_size, alloc_handle = entry
        try:
            ensure_cuda_context(gpu_idx)
            driver.cuMemUnmap(va_ptr, alloc_size)
            driver.cuMemRelease(alloc_handle)
            driver.cuMemAddressFree(va_ptr, alloc_size)
            pop_cuda_context()
        except Exception as exc:
            log.warning("cuda_cleanup: GPU %d shared chunk: %s", gpu_idx, exc)

    for r_gpu, va_ptr, alloc_size, alloc_handle, _ in RETIRED_CHUNKS:
        try:
            ensure_cuda_context(r_gpu)
            driver.cuMemUnmap(va_ptr, alloc_size)
            driver.cuMemRelease(alloc_handle)
            driver.cuMemAddressFree(va_ptr, alloc_size)
            pop_cuda_context()
        except Exception as exc:
            log.warning("cuda_cleanup: GPU %d retired chunk: %s", r_gpu, exc)

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

    log.info("cuda_cleanup: done in %.1f ms", (time.monotonic() - t0) * 1000)


def main():
    # Reduce GIL switch interval from default 5ms to 1ms. Our HTTP handler
    # threads need low-latency GIL access for serving lock requests and
    # healthz probes while the poll thread does CUDA work.
    sys.setswitchinterval(0.001)

    log.info("pod name: %s", K8S_PODNAME)
    log.info("config: HTTPD_PORT=%s CHUNK_MIB=%s FLOAT_VALUE=%s SVC_NAME=%s "
             "POLL_INTERVAL_S=%s GPUS_PER_NODE=%s",
             HTTPD_PORT, CHUNK_MIB, FLOAT_VALUE, SVC_NAME, POLL_INTERVAL_S,
             GPUS_PER_NODE)

    log.info("cuDriverGetVersion(): %s", cucheck(driver.cuDriverGetVersion()))
    log.info(
        "getLocalRuntimeVersion(): %s",
        cucheck(runtime.getLocalRuntimeVersion()),
    )

    for k, v in os.environ.items():
        if "CUDA" in k or "NVIDIA" in k:
            log.info("env: %s: %s", k, v)

    log_imex_state()
    cuda_init()
    def _atexit_handler():
        cuda_cleanup()
        log.info("atexit handler complete")
    atexit.register(_atexit_handler)
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

    def _signal_handler(signum, frame):
        signame = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        log.info("received %s — initiating graceful shutdown", signame)
        SHUTTING_DOWN.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    log.info("all threads started, main thread waiting")
    SHUTTING_DOWN.wait()

    # Graceful shutdown sequence:
    # 1. SHUTTING_DOWN is set — HTTP handler rejects new /prepare-chunk
    #    and /lock-gpu with 503. No new benchmarks will start using our
    #    handles.
    # 2. Wait for any remotely-held local GPU locks to be released.
    #    A peer holding our lock is mid-DtoD from our memory — we must
    #    wait for that to complete before evicting.
    _wait_for_local_locks_released()
    # 3. Now no peer is actively reading our memory. Tell all peers to
    #    evict their cached imports of our handles. The HTTP response
    #    confirms the peer has unmapped — so when this returns, no peer
    #    holds any reference to our GPU memory.
    _broadcast_evict_to_peers()
    # 4. The poll thread checks SHUTTING_DOWN and finishes its current
    #    benchmark before exiting (daemon thread, will die with process).
    # 5. Release CUDA resources (contexts, allocations). The atexit
    #    handler is a safety net if we crash before reaching this point.
    cuda_cleanup()
    log.info("graceful shutdown complete, proceeding to exit")

    # Schedule a hard kill if atexit cleanup hangs (e.g., CUDA driver
    # call blocks indefinitely). Give it 10s then force exit.
    def _hard_exit_watchdog():
        time.sleep(10)
        log.error("atexit cleanup hung for 10s, forcing exit")
        os._exit(1)
    threading.Thread(target=_hard_exit_watchdog, daemon=True).start()


def cuda_init():
    """Initialize CUDA runtime for all local GPUs.

    For each GPU: retains its primary context, validates capabilities,
    builds allocation properties, compiles the checksum kernel, and
    pre-allocates verification buffers.

    Raises:
        CudaError: On any CUDA API failure.
        AssertionError: If fewer devices are visible than GPUS_PER_NODE.
    """
    cucheck(driver.cuInit(0))

    devcount = cucheck(runtime.cudaGetDeviceCount())
    log.info("cudaGetDeviceCount(): %s, GPUS_PER_NODE: %s", devcount, GPUS_PER_NODE)
    assert devcount >= GPUS_PER_NODE, \
        f"GPUS_PER_NODE={GPUS_PER_NODE} but only {devcount} devices visible"

    for gpu_idx in range(GPUS_PER_NODE):
        cudev = cucheck(driver.cuDeviceGet(gpu_idx))
        CUDEVS[gpu_idx] = cudev
        log.info("GPU %d: cudev=%s", gpu_idx, cudev)

        # Each GPU has its own primary context.
        ctx = cucheck(driver.cuDevicePrimaryCtxRetain(cudev))
        cucheck(driver.cuCtxPushCurrent(ctx))
        log.info("GPU %d: retained and pushed primary context: %s", gpu_idx, ctx)

        vaddr_supported = cucheck(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                cudev,
            )
        )
        if not vaddr_supported:
            raise Exception(f"GPU {gpu_idx}: VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED: false")

        sm_count = cucheck(
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

        GRANULARITIES[gpu_idx] = cucheck(
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
        cucheck(driver.cuCtxPopCurrent())


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

    module = cucheck(driver.cuModuleLoadData(ptx))
    CHECKSUM_KERNELS[gpu_idx] = cucheck(
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
    VERIFY_LOCAL_BUFS[gpu_idx] = cucheck(driver.cuMemAlloc(alloc_size))
    VERIFY_LOCAL_BUF_SIZES[gpu_idx] = alloc_size

    partials_size = CHECKSUM_NUM_BLOCKS[gpu_idx] * ctypes.sizeof(ctypes.c_double)
    VERIFY_PARTIALS_BUFS[gpu_idx] = cucheck(driver.cuMemAlloc(partials_size))

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
    ctx = cucheck(driver.cuDevicePrimaryCtxRetain(CUDEVS[gpu_idx]))
    cucheck(driver.cuCtxPushCurrent(ctx))


def pop_cuda_context():
    """Pop the current CUDA context from the calling thread's stack.

    Raises:
        CudaError: If no context is active on this thread.
    """
    cucheck(driver.cuCtxPopCurrent())


def acquire_local_gpu_lock(gpu_idx) -> float:
    """Acquire the local GPU lock, blocking until available.

    Returns the wall-clock wait time in milliseconds.
    """
    log.debug("local lock GPU %d: acquiring", gpu_idx)
    t0 = time.monotonic()
    GPU_LOCKS[gpu_idx].acquire()
    GPU_LOCK_TIMESTAMPS[gpu_idx] = time.monotonic()
    GPU_LOCK_HOLDERS[gpu_idx] = "local"
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
    resp = _http_session.post(
        f"http://{peer_host}:{port}/lock-gpu?gpu_index={gpu_index}"
        f"&holder={K8S_PODNAME}",
        timeout=(1, GPU_LOCK_ACQUIRE_HTTP_TIMEOUT_S),
    )
    if resp.status_code != 200:
        raise LockError(f"lock-gpu failed: HTTP {resp.status_code}: {resp.text}")
    wait_ms = (time.monotonic() - t0) * 1000
    token = resp.text
    log.debug("remote lock %s GPU %d: acquired in %.1f ms",
              peer_host, gpu_index, wait_ms)
    HELD_REMOTE_LOCKS.add((peer_host, port, gpu_index, token))
    return (token, wait_ms)


def release_remote_gpu_lock(peer_host: str, port: int, gpu_index: int,
                            token: str):
    """Release a GPU lock on a remote pod via HTTP POST /unlock-gpu.

    Retries up to 7 times (0.5s apart) to avoid leaving a stale lock that
    blocks other pods until the watchdog force-releases it. The token
    prevents accidentally releasing a lock re-acquired by another pod.

    Never raises — failures are logged as warnings.
    """
    HELD_REMOTE_LOCKS.discard((peer_host, port, gpu_index, token))
    url = f"http://{peer_host}:{port}/unlock-gpu?gpu_index={gpu_index}&token={token}"
    for attempt in range(7):
        try:
            _http_session.post(url, timeout=(1, 2))
            return
        except Exception as exc:
            log.warning("failed to release remote GPU lock on %s gpu %d "
                        "(attempt %d/7): %s", peer_host, gpu_index,
                        attempt + 1, exc)
            time.sleep(0.5)


def _broadcast_evict_to_peers():
    """Tell all discoverable peers to evict their cached imports of our handles.

    Called during graceful shutdown, after all in-flight benchmarks have
    completed (local locks drained). Each peer is contacted in parallel
    with a 20s budget (15s first attempt + remaining for retry). The
    evict-peer endpoint is idempotent -- retries are safe. When a peer
    confirms (HTTP 200), it has unmapped all our fabric handles.

    Peers that don't respond keep holding stale references to our
    GPU memory, which become invalid once our process exits.
    """
    log.info("shutdown: broadcasting evict-peer to all peers")
    try:
        peers = discover_peers()
    except Exception as exc:
        log.warning("shutdown: could not discover peers for evict broadcast: %s", exc)
        return

    # Send evict requests in parallel — don't let one slow peer delay
    # the others. Each request has a 2s total budget (0.5s connect + 1.5s recv).
    EVICT_BUDGET_S = 20  # Total time budget per peer for both attempts.

    def _evict_one(pod_name, peer_host):
        url = (f"http://{peer_host}:{HTTPD_PORT}"
               f"/evict-peer?pod_name={K8S_PODNAME}")
        deadline = time.monotonic() + EVICT_BUDGET_S
        # First attempt: up to 15s recv.
        try:
            _http_session.post(url, timeout=(1, 15))
            log.info("shutdown: evict-peer confirmed by %s", pod_name)
            return
        except Exception as exc:
            log.warning("shutdown: evict-peer to %s attempt 1/2: %s",
                        pod_name, exc)
        # Second attempt: use whatever time remains.
        remaining = deadline - time.monotonic()
        if remaining < 0.5:
            log.warning("shutdown: evict-peer to %s: no time for retry", pod_name)
            return
        try:
            _http_session.post(url, timeout=(0.5, remaining))
            log.info("shutdown: evict-peer confirmed by %s (attempt 2)",
                     pod_name)
        except Exception as exc:
            log.warning("shutdown: evict-peer to %s failed (peer may hit "
                        "ILLEGAL_STATE): %s", pod_name, exc)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(peers)) as pool:
        futures = [pool.submit(_evict_one, pn, ph) for pn, ph in peers]
        concurrent.futures.wait(futures, timeout=EVICT_BUDGET_S + 2)

    log.info("shutdown: evict broadcast complete")


def _wait_for_local_locks_released():
    """Wait briefly for remote pods to release locks they hold on our GPUs.

    After SHUTTING_DOWN is set, the HTTP handler rejects new lock requests
    with 503. Existing lock holders should release within a few ms (their
    DtoD benchmark completes). We wait up to 3s for all locks to clear.
    """
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        held = [idx for idx in GPU_LOCKS if GPU_LOCKS[idx].locked()]
        if not held:
            log.info("shutdown: all local GPU locks released")
            return
        holders = [f"GPU {idx} by {GPU_LOCK_HOLDERS.get(idx, '?')}"
                   for idx in held]
        log.info("shutdown: waiting for local locks: %s", ", ".join(holders))
        time.sleep(0.1)
    held = [idx for idx in GPU_LOCKS if GPU_LOCKS[idx].locked()]
    if held:
        log.warning("shutdown: %d local locks still held after 3s, proceeding",
                    len(held))


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
                        holder = GPU_LOCK_HOLDERS.get(gpu_idx, "unknown")
                        log.warning("GPU lock watchdog: force-releasing GPU %d "
                                    "(held for %.0fs by %s)",
                                    gpu_idx, now - ts, holder)
                        GPU_LOCK_TOKENS.pop(gpu_idx, None)
                        GPU_LOCK_HOLDERS.pop(gpu_idx, None)
                        release_local_gpu_lock(gpu_idx)
                    GPU_LOCK_TIMESTAMPS.pop(gpu_idx, None)

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()


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


def _free_retired_chunks(gpu_idx):
    """Free chunks that have been retired for longer than CHUNK_RETIRE_S.

    Precondition: the GPU's CUDA context must be pushed.
    """
    now = time.monotonic()
    remaining = []
    for entry in RETIRED_CHUNKS:
        r_gpu, va_ptr, alloc_size, alloc_handle, retire_time = entry
        if r_gpu != gpu_idx:
            remaining.append(entry)
            continue
        if now - retire_time < CHUNK_RETIRE_S:
            remaining.append(entry)
            continue
        try:
            driver.cuMemUnmap(va_ptr, alloc_size)
            driver.cuMemRelease(alloc_handle)
            driver.cuMemAddressFree(va_ptr, alloc_size)
            log.info("GPU %d: freed retired chunk (age %.0fs)",
                     gpu_idx, now - retire_time)
        except Exception:
            log.warning("GPU %d: failed to free retired chunk (ignored)",
                        gpu_idx)
    RETIRED_CHUNKS[:] = remaining


def _refresh_shared_chunk_for_gpu(gpu_idx):
    """Replace the shared chunk on one GPU with a fresh allocation.

    The allocation + fill happens without holding the GPU lock.
    _prepare_shared_chunk_for_gpu writes to SHARED_CHUNK_ALLOCS,
    SHARED_CHUNK_ALLOC_HANDLES, and SHARED_CHUNK_STATIC_META. In CPython,
    dict mutations are thread-safe, so concurrent reads from the HTTP
    handler see consistent values.

    The old allocation is not freed immediately -- a remote pod may
    still be mid-DtoD from it. It is retired and freed after
    CHUNK_RETIRE_S seconds.

    Also frees any previously retired chunks that have aged out.

    Precondition: the GPU's CUDA context must be pushed.

    Raises:
        CudaError: On allocation or memset failure.
    """
    _free_retired_chunks(gpu_idx)

    # Stash old allocation before overwriting.
    old = SHARED_CHUNK_ALLOCS.get(gpu_idx)

    # Allocate and fill the new chunk OUTSIDE the lock. This is the
    # slow part (~1-30ms, sometimes longer) that must not block benchmarks.
    # _prepare_shared_chunk_for_gpu writes to SHARED_CHUNK_ALLOCS,
    # SHARED_CHUNK_ALLOC_HANDLES, and SHARED_CHUNK_STATIC_META.
    # The HTTP handler reads SHARED_CHUNK_ALLOC_HANDLES concurrently;
    # CPython dict mutations are thread-safe.
    _prepare_shared_chunk_for_gpu(gpu_idx)

    # Retire the old allocation.
    if old is not None:
        RETIRED_CHUNKS.append(
            (gpu_idx, old[0], old[1], old[2], time.monotonic()))
        last_served = LAST_CHUNK_SERVED_TIME.get(gpu_idx)
        if last_served is not None:
            ago = time.monotonic() - last_served
            log.info("GPU %d: retired old chunk (last served %.1fs ago)",
                     gpu_idx, ago)
        else:
            log.info("GPU %d: retired old chunk (never served)", gpu_idx)


def chunk_refresh_loop():
    """Periodically replace shared chunks with fresh allocations on all GPUs.

    This exercises the full CUDA allocation + IMEX export path regularly,
    rather than relying on memory allocated once at startup. May help
    detect or recover from degraded IMEX daemon state.
    """
    while not SHUTTING_DOWN.is_set():
        SHUTTING_DOWN.wait(CHUNK_REFRESH_INTERVAL_S)
        if SHUTTING_DOWN.is_set():
            break
        fatal_seen = False
        for gpu_idx in range(GPUS_PER_NODE):
            if fatal_seen:
                log.warning("skipping GPU %d refresh (ILLEGAL_STATE on earlier GPU)",
                            gpu_idx)
                continue
            t0 = time.monotonic()
            ensure_cuda_context(gpu_idx)
            try:
                _refresh_shared_chunk_for_gpu(gpu_idx)
                elapsed_s = time.monotonic() - t0
                if elapsed_s > 1:
                    log.warning("refreshed shared chunk on GPU %d in %.1fs (slow!)",
                                gpu_idx, elapsed_s)
                else:
                    log.info("refreshed shared chunk on GPU %d in %.0f ms",
                             gpu_idx, elapsed_s * 1000)
            except CudaError as exc:
                elapsed_s = time.monotonic() - t0
                log.exception("failed to refresh shared chunk on GPU %d "
                              "(after %.1fs):", gpu_idx, elapsed_s)
                if "ILLEGAL_STATE" in str(exc):
                    fatal_seen = True
                    log.warning("ILLEGAL_STATE during refresh — skipping "
                                "remaining GPUs this cycle")
            except Exception:
                elapsed_s = time.monotonic() - t0
                log.exception("failed to refresh shared chunk on GPU %d "
                              "(after %.1fs):", gpu_idx, elapsed_s)
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

    log.info("GPU %d: cuMemCreate(%d MiB) starting", gpu_idx,
             alloc_size // (1024 * 1024))
    t0 = time.monotonic()
    alloc_handle = cucheck(
        driver.cuMemCreate(alloc_size, ALLOC_PROPS[gpu_idx], 0))
    create_ms = (time.monotonic() - t0) * 1000
    if create_ms > 100:
        log.warning("GPU %d: cuMemCreate took %.1fs", gpu_idx, create_ms / 1000)

    t0 = time.monotonic()
    va_ptr = cucheck(
        driver.cuMemAddressReserve(alloc_size, granularity, 0, 0))
    cucheck(driver.cuMemMap(va_ptr, alloc_size, 0, alloc_handle, 0))

    access_desc = driver.CUmemAccessDesc()
    access_desc.location = ALLOC_PROPS[gpu_idx].location
    access_desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    cucheck(driver.cuMemSetAccess(va_ptr, alloc_size, [access_desc], 1))
    map_ms = (time.monotonic() - t0) * 1000
    if map_ms > 100:
        log.warning("GPU %d: reserve+map+access took %.1fs", gpu_idx, map_ms / 1000)

    # Fill GPU memory with FLOAT_VALUE directly on the GPU using cuMemsetD32.
    # cuMemsetD32 interprets the pattern as a uint32, so we reinterpret our
    # float32's bit pattern via struct pack/unpack.
    t0 = time.monotonic()
    float_as_uint32 = struct.unpack("I", struct.pack("f", FLOAT_VALUE))[0]
    cucheck(driver.cuMemsetD32(va_ptr, float_as_uint32, num_floats))
    # cuMemsetD32 is asynchronous — wait for completion.
    cucheck(driver.cuCtxSynchronize())
    fill_ms = (time.monotonic() - t0) * 1000
    if fill_ms > 100:
        log.warning("GPU %d: memset+sync took %.1fs", gpu_idx, fill_ms / 1000)

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
    """Export a CU_MEM_HANDLE_TYPE_FABRIC handle for SHARED_CHUNK_ALLOC_HANDLES[gpu_idx].

    Called on every /prepare-chunk request. The returned JSON contains
    the base64-encoded handle plus all fields from SHARED_CHUNK_STATIC_META
    (pod_name, node_name, gpu_index, num_floats, float_value, alloc_size).

    The underlying allocation changes periodically (chunk_refresh_loop).
    Between refreshes, the same allocation is re-exported and
    cuMemExportToShareableHandle returns identical handle bytes.
    Re-exporting on every request exercises the IMEX daemon code path.

    Logs a warning if the export takes longer than 10 ms.

    Precondition: the GPU's CUDA context must be pushed.

    Raises:
        CudaError: If cuMemExportToShareableHandle fails.
    """
    t0 = time.monotonic()
    fabric_handle = cucheck(
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
    resp = _http_session.get(url, timeout=(0.5, 1))
    if resp.status_code != 200:
        log.error("peer returned HTTP %s: %s", resp.status_code, resp.text)
        resp.raise_for_status()
    return resp.json()


# Connection errors that indicate the remote end is simply not reachable
# right now. These are expected during pod restarts and don't warrant a
# full stack trace.
_TRANSIENT_CONN_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def _is_dns_failure(exc):
    """Check if a requests exception is a DNS resolution failure."""
    msg = str(exc).lower()
    return "name or service not known" in msg or "name resolution" in msg


def _fetch_chunk_meta_with_retry(peer_host, port, gpu_index, peer_name):
    """Fetch chunk metadata with one explicit retry for transient connection
    errors. No retry for DNS failures — raises PeerUnreachableError
    immediately so the caller can skip the entire peer.

    Raises:
        PeerUnreachableError: If the peer's DNS name doesn't resolve.
    """
    for attempt in range(2):
        try:
            return fetch_chunk_meta(peer_host, port, gpu_index)
        except _TRANSIENT_CONN_ERRORS as exc:
            msg = str(exc)
            if hasattr(exc, "args") and exc.args:
                inner = exc.args[0]
                if hasattr(inner, "reason"):
                    msg = str(inner.reason)
            # DNS failure: skip entire peer, no retry.
            if _is_dns_failure(exc):
                log.warning("peer %s unreachable (DNS): %s", peer_name, msg)
                raise PeerUnreachableError(msg) from exc
            if attempt == 0:
                log.info("fetch chunk %s gpu %d: %s (retrying)", peer_name,
                         gpu_index, msg)
                time.sleep(0.05)
            else:
                log.warning("fetch chunk %s gpu %d: %s (giving up)", peer_name,
                            gpu_index, msg)
                raise
        except Exception:
            log.exception("fetch chunk %s gpu %d failed:", peer_name, gpu_index)
            raise


def import_fabric_handle(handle_bytes: bytes):
    """Import a CUDA fabric handle from raw bytes.

    Raises:
        CudaError: If the handle cannot be imported.
    """
    return cucheck(
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
    va_ptr = cucheck(
        driver.cuMemAddressReserve(alloc_size, granularity, 0, 0)
    )
    cucheck(driver.cuMemMap(va_ptr, alloc_size, 0, alloc_handle, 0))

    access_desc = driver.CUmemAccessDesc()
    access_desc.location = ALLOC_PROPS[gpu_idx].location
    access_desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    cucheck(driver.cuMemSetAccess(va_ptr, alloc_size, [access_desc], 1))
    return va_ptr


def verify_chunk_on_gpu(local_gpu_idx: int, va_ptr: int, alloc_size: int,
                        num_floats: int, expected_value: float) -> tuple[str, float]:
    """Copy remote-mapped chunk to local buffer via DtoD, measure bandwidth, verify data.

    Phase 1: cuMemcpyDtoD from remote-mapped VA to pre-allocated local
    buffer, timed with CUDA events.

    Phase 2: checksum kernel sums all float32 values in the local copy,
    compares against expected_value * num_floats with relative tolerance
    CHECKSUM_REL_TOLERANCE. This runs after the timed copy so it does
    not affect the bandwidth measurement.

    Precondition: CUDA context for local_gpu_idx must be pushed.

    Returns:
        (result_str, elapsed_ms): '{bw} GB/s' on success, or
        'CHECKSUM MISMATCH: ...' on verification failure.

    Raises:
        CudaError: On any CUDA operation failure.
        AssertionError: If alloc_size exceeds the pre-allocated buffer.
    """
    local_buf = VERIFY_LOCAL_BUFS[local_gpu_idx]
    assert alloc_size <= VERIFY_LOCAL_BUF_SIZES[local_gpu_idx], \
        f"chunk {alloc_size} exceeds pre-allocated buffer {VERIFY_LOCAL_BUF_SIZES[local_gpu_idx]}"

    # Phase 1: time the DtoD copy (pure NVLink transfer).
    ev_start = cucheck(driver.cuEventCreate(0))
    ev_end = cucheck(driver.cuEventCreate(0))

    cucheck(driver.cuEventRecord(ev_start, 0))
    cucheck(driver.cuMemcpyDtoD(local_buf, va_ptr, alloc_size))
    cucheck(driver.cuEventRecord(ev_end, 0))
    cucheck(driver.cuEventSynchronize(ev_end))

    elapsed_ms = cucheck(driver.cuEventElapsedTime(ev_start, ev_end))
    cucheck(driver.cuEventDestroy(ev_start))
    cucheck(driver.cuEventDestroy(ev_end))

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

    cucheck(driver.cuLaunchKernel(
        CHECKSUM_KERNELS[local_gpu_idx],
        num_blocks, 1, 1,
        256, 1, 1,
        0, 0,
        args, 0,
    ))
    cucheck(driver.cuCtxSynchronize())

    result_buf = bytearray(out_size)
    cucheck(driver.cuMemcpyDtoH(result_buf, partials_buf, out_size))

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
    # Order: unmap, release physical handle, free VA range.
    # Must unmap before release, per CUDA driver API contract.
    if va_ptr is not None:
        cucheck(driver.cuMemUnmap(va_ptr, alloc_size))
    if alloc_handle is not None:
        cucheck(driver.cuMemRelease(alloc_handle))
    if va_ptr is not None:
        cucheck(driver.cuMemAddressFree(va_ptr, alloc_size))


def evict_peer_imports(peer_pod_name):
    """Evict all cached imports originating from the given peer.

    Called when a peer announces it's shutting down, so we release its
    fabric handles before its IMEX daemon tears down.
    """
    evict_keys = [k for k, v in IMPORT_CACHE.items() if v[3] == peer_pod_name]
    for key in evict_keys:
        entry = IMPORT_CACHE.pop(key, None)
        if entry is None:
            continue  # Already evicted by another thread.
        imported_handle, va_ptr, alloc_size, _peer = entry
        local_gpu_idx = key[0]
        try:
            ensure_cuda_context(local_gpu_idx)
            unmap_imported_chunk(va_ptr, alloc_size, imported_handle)
            pop_cuda_context()
        except Exception:
            log.warning("evict peer %s: cleanup failed for GPU %d (ignored)",
                        peer_pod_name, local_gpu_idx)
    if evict_keys:
        log.info("evicted %d cached imports from peer %s",
                 len(evict_keys), peer_pod_name)


def acquire_gpu_lock_pair(peer_name, peer_host, port, remote_gpu_idx,
                          local_gpu_idx):
    """Acquire locks on both the remote source GPU and local destination GPU.

    Locks are always acquired in ascending (pod_idx, gpu_idx) order to
    prevent deadlock between pods that would otherwise acquire each
    other's locks in opposite order.

    Returns:
        (remote_token, lock_wait_ms): Token for remote unlock, max wait time.

    Raises:
        LockError: If the remote lock cannot be acquired.
    """
    my_idx = int(K8S_PODNAME.rsplit("-", 1)[1])
    peer_idx_int = int(peer_name.rsplit("-", 1)[1])
    local_key = (my_idx, local_gpu_idx)
    remote_key = (peer_idx_int, remote_gpu_idx)

    try:
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
    except LockError:
        raise
    except Exception as exc:
        raise LockError(str(exc)) from exc

    return remote_token, max(local_wait, remote_wait)


def _get_cached_import(local_gpu_idx, handle_bytes, alloc_size, peer_pod_name):
    """Get or create a cached fabric handle import + VA mapping.

    Returns (imported_handle, va_ptr, cache_hit). If the handle bytes
    match a cached entry, returns the existing mapping (fast path).
    Otherwise imports and maps fresh, caching the result.

    Precondition: CUDA context for local_gpu_idx must be pushed.

    Raises:
        CudaError: On import or mapping failure.
    """
    cache_key = (local_gpu_idx, handle_bytes)
    cached = IMPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], cached[1], True

    imported_handle = import_fabric_handle(handle_bytes)
    va_ptr = map_imported_chunk(imported_handle, alloc_size, local_gpu_idx)
    IMPORT_CACHE[cache_key] = (imported_handle, va_ptr, alloc_size, peer_pod_name)
    return imported_handle, va_ptr, False


def _evict_stale_cache_entries(active_handle_bytes):
    """Remove cache entries whose handle bytes are no longer in use.

    Called once per round with the set of handle bytes seen this round.
    Entries for handles that changed (chunk refresh) or disappeared
    (peer gone) are unmapped and freed.
    """
    stale_keys = [k for k in IMPORT_CACHE if k[1] not in active_handle_bytes]
    for key in stale_keys:
        entry = IMPORT_CACHE.pop(key, None)
        if entry is None:
            continue  # Already evicted by another thread.
        imported_handle, va_ptr, alloc_size, peer = entry
        local_gpu_idx = key[0]
        try:
            ensure_cuda_context(local_gpu_idx)
            unmap_imported_chunk(va_ptr, alloc_size, imported_handle)
            pop_cuda_context()
        except Exception:
            log.warning("evict cache: cleanup failed for GPU %d (ignored)",
                        local_gpu_idx)
    if stale_keys:
        log.info("evicted %d stale import cache entries", len(stale_keys))


def _evict_import_on_error(local_gpu_idx, handle_bytes):
    """Evict import cache entry on error -- the handle may be stale."""
    evicted = IMPORT_CACHE.pop((local_gpu_idx, handle_bytes), None)
    if evicted is None:
        return
    try:
        unmap_imported_chunk(evicted[1], evicted[2], evicted[0])
    except Exception:
        log.warning("failed to unmap evicted import for GPU %d", local_gpu_idx)


def import_map_and_verify(peer_host, port, remote_gpu_idx, local_gpu_idx,
                          handle_bytes, alloc_size, num_floats, float_value):
    """Import a remote fabric handle, map it, acquire GPU locks, run benchmark.

    Uses the import cache (_get_cached_import) so repeated calls with the
    same handle_bytes skip the expensive import+map. On any error, the
    cache entry for this handle is evicted (the handle may be stale).

    Manages CUDA context push/pop and lock acquire/release internally.
    Caller must not hold any GPU locks.

    Returns:
        (result_str, lock_wait_ms, benchmark_ms).

    Raises:
        CudaError: On import, mapping, or benchmark failure.
        LockError: On lock acquisition failure.
    """
    peer_name = peer_host.split(".")[0]
    t_total = time.monotonic()

    ensure_cuda_context(local_gpu_idx)
    locks_held = False

    try:
        t0 = time.monotonic()
        imported_handle, va_ptr, cache_hit = _get_cached_import(
            local_gpu_idx, handle_bytes, alloc_size, peer_name)
        import_map_ms = (time.monotonic() - t0) * 1000

        remote_token, lock_wait_ms = acquire_gpu_lock_pair(
            peer_name, peer_host, port, remote_gpu_idx, local_gpu_idx)
        locks_held = True

        result, benchmark_ms = verify_chunk_on_gpu(
            local_gpu_idx, va_ptr, alloc_size, num_floats, float_value)
    except Exception:
        _evict_import_on_error(local_gpu_idx, handle_bytes)
        raise
    finally:
        if locks_held:
            t0 = time.monotonic()
            release_local_gpu_lock(local_gpu_idx)
            release_remote_gpu_lock(peer_host, port, remote_gpu_idx,
                                    remote_token)
            unlock_ms = (time.monotonic() - t0) * 1000
        pop_cuda_context()

    total_ms = (time.monotonic() - t_total) * 1000
    log.debug("phases: import+map=%.1f%s lock_wait=%.1f DtoD=%.1f "
              "unlock=%.1f total=%.1f ms",
              import_map_ms, "(cached)" if cache_hit else "",
              lock_wait_ms, benchmark_ms, unlock_ms, total_ms)

    return result, lock_wait_ms, benchmark_ms


def _run_single_benchmark(peer_name: str, peer_host: str, port: int,
                          remote_gpu_idx: int, local_gpu_idx: int,
                          meta: dict) -> tuple[str, str, float, float]:
    """Run one benchmark using pre-fetched chunk metadata.

    Handles all errors internally — never raises.

    Returns:
        (result_str, peer_node_name, lock_wait_ms, benchmark_ms).
        On failure, result_str is an error tag and timing values are 0.0.
    """
    peer_node = meta.get("node_name", "?")
    handle_bytes = base64.urlsafe_b64decode(meta["handle"])

    try:
        result, lock_wait_ms, benchmark_ms = import_map_and_verify(
            peer_host, port, remote_gpu_idx, local_gpu_idx,
            handle_bytes, meta["alloc_size"], meta["num_floats"],
            meta["float_value"])
    except CudaError as exc:
        log.exception("CUDA error %s g%d->g%d:", peer_name, remote_gpu_idx,
                      local_gpu_idx)
        # Include specific CUDA error in result for dashboard display.
        # e.g. "INVALID_HANDLE" from "CUDA error code=400(b'CUDA_ERROR_INVALID_HANDLE')"
        tag = "cuda-err"
        exc_str = str(exc)
        if "INVALID_HANDLE" in exc_str:
            tag = "INVALID_HANDLE"
        elif "ILLEGAL_STATE" in exc_str:
            tag = "ILLEGAL_STATE"
        elif "LAUNCH_FAILED" in exc_str:
            tag = "LAUNCH_FAILED"
        return (tag, peer_node, 0.0, 0.0)
    except LockError as exc:
        log.warning("%s g%d->g%d: %s", peer_name, remote_gpu_idx,
                    local_gpu_idx, exc)
        return ("lock-err", peer_node, 0.0, 0.0)
    except Exception:
        log.exception("error %s g%d->g%d:", peer_name, remote_gpu_idx,
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


def _prefetch_peer_chunk_metas(peer_name, peer_host, port):
    """Fetch chunk metadata for all GPUs on a peer in one batch.

    Returns {gpu_index: meta_dict} or raises PeerUnreachableError.
    """
    metas = {}
    for gpu_idx in range(GPUS_PER_NODE):
        meta = _fetch_chunk_meta_with_retry(peer_host, port, gpu_idx,
                                            peer_name)
        metas[gpu_idx] = meta
    return metas


def _benchmark_one_peer(pod_name, peer_host, port, peer_metas):
    """Run all GPU-pair benchmarks against one peer. Called from a thread.

    Benchmarks all (remote_gpu, local_gpu) pairs in randomized order
    to reduce lock contention. Stops early if SHUTTING_DOWN is set.

    Returns (results_list, max_lock_wait_ms, benchmark_durations_list).
    Each result is a dict with peer_idx, peer_node, remote_gpu,
    local_gpu, value.
    """
    metas = peer_metas.get(pod_name)
    if metas is None:
        return [], 0.0, []

    peer_idx = pod_name.rsplit("-", 1)[1]
    gpu_range = range(GPUS_PER_NODE)
    work_items = list(itertools.product(gpu_range, gpu_range))
    random.shuffle(work_items)

    results = []
    max_lock_wait_ms = 0.0
    benchmark_durations = []

    for remote_gpu_idx, local_gpu_idx in work_items:
        if SHUTTING_DOWN.is_set():
            log.info("peer %s: shutdown requested, finishing after %d/%d benchmarks",
                     pod_name, len(results), len(work_items))
            break
        meta = metas[remote_gpu_idx]
        log.debug("benchmark %s-g%d -> local-g%d",
                  pod_name, remote_gpu_idx, local_gpu_idx)
        status, peer_node, lock_wait, bench_ms = _run_single_benchmark(
            peer_name=pod_name, peer_host=peer_host, port=port,
            remote_gpu_idx=remote_gpu_idx, local_gpu_idx=local_gpu_idx,
            meta=meta)
        max_lock_wait_ms = max(max_lock_wait_ms, lock_wait)
        if bench_ms > 0:
            benchmark_durations.append(bench_ms)
        results.append({
            "peer_idx": peer_idx,
            "peer_node": peer_node,
            "remote_gpu": remote_gpu_idx,
            "local_gpu": local_gpu_idx,
            "value": status,
        })

    return results, max_lock_wait_ms, benchmark_durations


def _run_one_poll_round():
    """Run one round of all-to-all benchmarks against discovered peers.

    Prefetches chunk handles for all peers, then benchmarks each peer
    in parallel (one thread per peer). Round time scales with the
    slowest peer, not the sum of all peers.

    Raises:
        dns.resolver.NXDOMAIN: If peer DNS does not exist yet.
    """
    global LAST_RESULT_TIME, FATAL_CUDA_ERROR
    round_t0 = time.monotonic()
    peers = discover_peers()
    if not peers:
        # No peers, but DNS worked — we're alive, just alone.
        LAST_RESULT_TIME = time.monotonic()
        log.info("peer poll: no peers discovered yet")
        return

    # Prefetch chunk metadata for all peers. Track unreachable peers
    # so we can emit explicit error entries for them in the results.
    peer_metas = {}
    unreachable_peers = {}  # {pod_name: error_tag}
    for pod_name, peer_host in peers:
        try:
            peer_metas[pod_name] = _prefetch_peer_chunk_metas(
                pod_name, peer_host, HTTPD_PORT)
        except PeerUnreachableError:
            log.warning("lost peer %s — DNS resolution failed, skipping for this round",
                        pod_name)
            unreachable_peers[pod_name] = "unreachable"
        except Exception:
            log.exception("failed to prefetch chunk metas from %s:", pod_name)
            unreachable_peers[pod_name] = "prefetch-err"

    if not peer_metas and not unreachable_peers:
        return

    # Evict import cache entries for handles that changed or disappeared.
    active_handles = set()
    for metas in peer_metas.values():
        for meta in metas.values():
            active_handles.add(base64.urlsafe_b64decode(meta["handle"]))
    _evict_stale_cache_entries(active_handles)

    # Benchmark all peers in parallel — one thread per peer.
    # Local GPU locks handle contention for shared local GPUs.
    results = []
    max_lock_wait_ms = 0.0
    benchmark_durations = []

    _error_tags = ("err", "ILLEGAL_STATE", "INVALID_HANDLE", "LAUNCH_FAILED",
                   "MISMATCH", "lock-err", "unreachable", "prefetch-err")

    round_ts = (datetime.datetime.now(datetime.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
    round_mono = time.monotonic()

    # Emit error entries for unreachable peers.
    gpu_range = range(GPUS_PER_NODE)
    gpu_pairs = list(itertools.product(gpu_range, gpu_range))
    for pod_name, error_tag in unreachable_peers.items():
        peer_idx = pod_name.rsplit("-", 1)[1]
        for rg, lg in gpu_pairs:
            results.append({
                "peer_idx": peer_idx,
                "peer_node": "?",
                "remote_gpu": rg,
                "local_gpu": lg,
                "value": error_tag,
            })

    reachable_peers = [(pn, ph) for pn, ph in peers if pn in peer_metas]
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(reachable_peers)) as pool:
        futures = {
            pool.submit(_benchmark_one_peer, pod_name, peer_host,
                        HTTPD_PORT, peer_metas): pod_name
            for pod_name, peer_host in reachable_peers
        }
        for future in concurrent.futures.as_completed(futures):
            pod_name = futures[future]
            try:
                peer_results, peer_lock_wait, peer_durations = future.result()
                results.extend(peer_results)
                max_lock_wait_ms = max(max_lock_wait_ms, peer_lock_wait)
                benchmark_durations.extend(peer_durations)
            except Exception:
                log.exception("peer %s benchmark thread failed:", pod_name)

    if benchmark_durations:
        bmin = min(benchmark_durations)
        bmax = max(benchmark_durations)
        bmean = sum(benchmark_durations) / len(benchmark_durations)
        log.info("round stats: DtoD min=%.1f max=%.1f mean=%.1f ms, "
                 "max_lock_wait=%.1f ms", bmin, bmax, bmean, max_lock_wait_ms)

    # If the entire round completed without any CUDA errors, clear the
    # fatal flag. This restores liveness after a transient ILLEGAL_STATE.
    has_errors = any(any(tag in b["value"] for tag in _error_tags)
                     for b in results)
    if not has_errors and FATAL_CUDA_ERROR is not None:
        log.info("round completed without errors, clearing FATAL_CUDA_ERROR "
                 "(was: %s)", FATAL_CUDA_ERROR)
        FATAL_CUDA_ERROR = None

    LAST_RESULT_TIME = time.monotonic()

    round_dur = time.monotonic() - round_t0

    # Finalize and move to history.
    for b in results:
        if "key" not in b:
            b["key"] = (f"{b['peer_idx']}@{b['peer_node']}"
                        f"-g{b['remote_gpu']}-g{b['local_gpu']}")
    ok_count = sum(1 for b in results
                   if not any(tag in b["value"] for tag in _error_tags))
    benchmarks_sorted = sorted(results, key=lambda b: b["key"])
    final_result = {
        "timestamp": round_ts,
        "_monotonic": round_mono,
        "round_time_s": round(round_dur, 2),
        "benchmarks": benchmarks_sorted,
        "total": len(results),
        "ok": ok_count,
        "errors": len(results) - ok_count,
    }
    RESULTS_HISTORY.append(final_result)

    my_idx = K8S_PODNAME.rsplit("-", 1)[1]
    parts = [f"{b['key']}:{b['value']}" for b in benchmarks_sorted]
    log.info("result(%s@%s): round_time=%.1fs %s",
             my_idx, K8S_NODENAME, round_dur, " ".join(parts))


def peer_poll_loop():
    """Deadline-based poll loop: starts a new round every POLL_INTERVAL_S
    seconds, measured from the start of the previous round."""
    time.sleep(POLL_INTERVAL_S)
    next_deadline = time.monotonic()

    while not SHUTTING_DOWN.is_set():
        next_deadline += POLL_INTERVAL_S
        try:
            _run_one_poll_round()
        except dns.resolver.NXDOMAIN:
            log.info("peer poll: DNS name does not exist yet (expected at startup)")
        except Exception:
            log.exception("peer poll error:")
        if SHUTTING_DOWN.is_set():
            log.info("peer poll: shutdown requested, exiting loop")
            break
        remaining = next_deadline - time.monotonic()
        log.info("waiting %.1fs for next poll deadline", max(0, remaining))
        _wait_until(next_deadline)


def _wait_until(deadline):
    """Sleep until the given monotonic-clock deadline (~10ms resolution)."""
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
        if "/results" in self.path:
            now_mono = time.monotonic()
            # Build results list: completed history + in-progress round.
            results = []
            for entry in list(RESULTS_HISTORY):
                e = {k: v for k, v in entry.items()
                     if not k.startswith("_")}
                e["age_s"] = round(now_mono - entry["_monotonic"], 1)
                results.append(e)
            body = orjson.dumps({
                "pod_name": K8S_PODNAME,
                "node_name": K8S_NODENAME,
                "gpus_per_node": GPUS_PER_NODE,
                "results": results,
                "fatal_cuda_error": FATAL_CUDA_ERROR,
                "shutting_down": SHUTTING_DOWN.is_set(),
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return

        if "/readyz" in self.path:
            # Readiness: just confirms the HTTP server is alive and can
            # respond. No state checks — a pod with CUDA errors is still
            # "ready" to receive requests (lock/unlock, evict, etc.).
            self._respond(200, b"ok")
            return

        if "/healthz" in self.path:
            if FATAL_CUDA_ERROR:
                self._respond(500, f"fatal: {FATAL_CUDA_ERROR}")
                return
            if LAST_RESULT_TIME is None:
                # No result yet (startup) -- healthy, give it time.
                self._respond(200, b"ok")
                return
            age = time.monotonic() - LAST_RESULT_TIME
            max_age = POLL_INTERVAL_S * 3
            if age > max_age:
                self._respond(500,
                    f"stale: no result for {age:.0f}s (max {max_age}s)")
            else:
                self._respond(200, b"ok")
            return

        if "/prepare-chunk" not in self.path:
            self._respond(404, b"unknown path")
            return

        if SHUTTING_DOWN.is_set():
            self._respond(503, b"shutting down")
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
        LAST_CHUNK_SERVED_TIME[gpu_idx] = time.monotonic()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)

        if "/evict-peer" in parsed.path:
            params = parse_qs(parsed.query)
            pod_name_vals = params.get("pod_name")
            if not pod_name_vals:
                self._respond(400, b"missing pod_name parameter")
                return
            pod_name = pod_name_vals[0]
            log.info("HTTPD: evict-peer request for %s from %s",
                     pod_name, self.client_address[0])
            t0 = time.monotonic()
            evict_peer_imports(pod_name)
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.info("HTTPD: evict-peer for %s completed in %.1f ms",
                     pod_name, elapsed_ms)
            self._respond(200, b"evicted")
            return

        result = self._parse_gpu_index()
        if result is None:
            return
        parsed, params, gpu_idx = result

        if gpu_idx not in GPU_LOCKS:
            self._respond(404, f"no GPU {gpu_idx}")
            return

        if "/lock-gpu" in parsed.path:
            if SHUTTING_DOWN.is_set():
                self._respond(503, b"shutting down")
                return
            holder_vals = params.get("holder")
            holder = holder_vals[0] if holder_vals else self.client_address[0]
            self._handle_lock_gpu(gpu_idx, holder)
        elif "/unlock-gpu" in parsed.path:
            self._handle_unlock_gpu(gpu_idx, params)
        else:
            self._respond(404, b"unknown path")

    def _handle_lock_gpu(self, gpu_idx, holder):
        """Block until GPU lock is acquired, respond with a lock token."""
        log.debug("HTTPD lock-gpu %d: request from %s, waiting",
                  gpu_idx, holder)
        t0 = time.monotonic()
        acquired = GPU_LOCKS[gpu_idx].acquire(timeout=GPU_LOCK_TIMEOUT_S)
        wait_ms = (time.monotonic() - t0) * 1000

        if not acquired:
            holder = GPU_LOCK_HOLDERS.get(gpu_idx, "unknown")
            self._respond(503, f"lock acquisition timed out (held by {holder})")
            return

        log.debug("HTTPD lock-gpu %d: granted to %s after %.1f ms",
                  gpu_idx, holder, wait_ms)
        GPU_LOCK_HOLDERS[gpu_idx] = holder
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


class _ATACKHTTPServer(ThreadingHTTPServer):
    # Allow many queued connections — with N pods × M GPUs, many lock
    # requests arrive concurrently. Default of 5 causes connection refused.
    request_queue_size = 128

    # Don't wait for handler threads on shutdown.
    daemon_threads = True

    # Disable Nagle's algorithm for lower latency on small responses.
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        super().server_bind()


def run_httpd_in_thread():
    def run():
        log.info("starting HTTP server on port %s", HTTPD_PORT)
        s = _ATACKHTTPServer(("0.0.0.0", HTTPD_PORT), HTTPHandler)
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
        props = cucheck(runtime.cudaGetDeviceProperties(gpu_idx))

        printprops = {}
        for k in dir(props):
            if not any(k.startswith(prefix) for prefix in _attr_filter):
                continue
            v = getattr(props, k)
            if k == "uuid":
                try:
                    v = uuid.UUID(bytes=v.bytes)
                except ValueError:
                    log.warning("unexpected UUID bytes: %s", v.bytes)
            printprops[k] = v

        log.info("GPU %d properties:\n%s", gpu_idx, pformat(printprops))
        pop_cuda_context()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("main() crashed:")
        sys.exit(1)
