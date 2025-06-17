# Note(JP): the -devel- image contains CUDA headers required for building nccl
# from source. The -devel- also image contains pre-built nccl, but not the
# newest release. My goal here is to take control of the nccl version used
# (built from source further below).
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS build

# We need to build cupy from source below. Cupy binary distributions (wheel,
# container images) for aarch64 are limited precisely in terms of nccl. Hence,
# build everything from source.

# Install Ubuntu 24's system Python (3.12) with all libraries built and also all
# dev/header files. That allows for building cupy from source. I also tried a
# more lightweight approach with a uv-provided Python build but that tripped up
# the cupy build tooling.
RUN <<EOT
    apt update -qy
    apt install -qyy \
        -o APT::Install-Recommends=false \
        -o APT::Install-Suggests=false \
        build-essential git \
        wget devscripts debhelper fakeroot \
        g++ python3.12 python3.12-dev python3-pip-whl python3-setuptools-whl python3.12-venv
    apt clean
    rm -rf /var/lib/apt/lists/*
    rm -rf /opt/nvidia/nsight-compute
EOT

# note: maybe use build stages approach, and after building copy artifacts into
# image derived from 12.9.0-runtime-ubuntu24.04 (which is ~2 GB)

# Note: when running on an arch that we didn't compile for then nccl errors
# might be rather wild, see https://github.com/NVIDIA/nccl/issues/1372 -- this
# is when I added 90 and 120 below. 90 for Hopper, 120 for Blackwell (CUDA 12.8)
# see
# https://github.com/NVIDIA/nccl/blob/f44ac759fee12ecb3cc6891e9e739a000f66fd70/makefiles/common.mk#L42C18-L42C53
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

# Note(JP): build nccl from source, follow README. First, build deb packages.
# Then install deb packages (this overwrites the pre-installed system packages
# for both, libnccl2 and libnccl-dev). Assume build on a machine with many cores
# (100+), use ~half of them for build job concurrency.
RUN <<EOT
    export _NJOBS=$(($(nproc) / 2)) && echo "NJOBS: ${_NJOBS}"
    mkdir /ncclbuild && cd /ncclbuild
    wget https://github.com/NVIDIA/nccl/archive/refs/tags/v2.27.3-1.tar.gz
    tar xzf v2.27.3-1.tar.gz && cd nccl*
    make -j${_NJOBS} src.build \
        NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120"
    make -j${_NJOBS} pkg.debian.build
    cd build/pkg/deb/ && ls *deb
    mkdir /nccl-debs && cp *deb /nccl-debs
    #apt install --allow-change-held-packages -y ./libnccl2*deb ./libnccl-dev*deb
    cd / && rm -rf /ncclbuild # this frees up about 1 GB.
EOT

RUN mkdir /thing
WORKDIR /thing

#  Construct venv over system Python.
RUN python3 -m venv .venv
ENV PATH="/thing/.venv/bin:${PATH}"

# Diagnostics output about Python environment
RUN <<EOT
    command -v python
    command -v pip
    python -V
    python -Im site
EOT

# Install just-built nccl packages (important: -dev pkg) so that the cupy build
# below builds against that!
RUN apt install --allow-change-held-packages -y /nccl-debs/libnccl2*deb /nccl-debs/libnccl-dev*deb

# Build cupy from source. Limit to same CUDA architectures as nccl build above.
# Ref for env vars affecting build:
# https://docs.cupy.dev/en/latest/reference/environment.html
# Used dev version of 14.x cupy, also because of https://github.com/cupy/cupy/issues/9128

RUN <<EOT
    export _NJOBS=$(($(nproc) / 2)) && echo "NJOBS: ${_NJOBS}"
    export CUPY_NUM_BUILD_JOBS="${_NJOBS}"
    export CUPY_NUM_NVCC_THREADS="4"
    export CUPY_NVCC_GENERATE_CODE="arch=compute_90,code=sm_90;arch=compute_100,code=sm_100;arch=compute_120,code=sm_120"
    pip install --verbose  git+https://github.com/cupy/cupy.git@020213b469f12f5f7dc2031f6b097d51bf7655f9\
        --log pip_cupy_build.log && \
        pip cache purge
EOT

# Note(JP): fail if we accidentally build against the nccl version that's
# shipped in the base image.
RUN python -c "import cupy.cuda.nccl; print(cupy.cuda.nccl.get_version())" | grep "22703"

# nickelpie dependency
RUN pip install requests

# Note(JP): fix runtime errors such as "cupy/carray.cuh(57): catastrophic error:
# cannot open source file "cuda_fp16.h"" Turns out: cupy JIT/runtime compilation
# needs CUDA header files that are "efficiently" bundled in the NVIDIA
# nvidia-cuda-runtime-cu12 package. Also see
# https://github.com/cupy/cupy/issues/865
RUN pip install nvidia-cuda-runtime-cu12==12.9.79

RUN find / -name "libcurand.so.10"
RUN find / -name "libnvrtc.so.12"

# With just the `12.9.0-base` image: runtime dependencies are missing such as
# curand.so.
FROM nvcr.io/nvidia/cuda:12.9.0-base-ubuntu24.04 AS prod

RUN <<EOT
    apt update -qy
    apt install -qyy \
        -o APT::Install-Recommends=false \
        -o APT::Install-Suggests=false \
        python3  python3-venv
    apt clean
    rm -rf /var/lib/apt/lists/*
    rm -rf /opt/nvidia/nsight-compute
EOT

#  Construct venv over system Python.
RUN python3 -m venv .venv
ENV PATH="/thing/.venv/bin:${PATH}"

# Contains python venv (cupy build)
COPY --from=build /thing /thing

# diagnostics output about Python environment
RUN <<EOT
    command -v python3
    python3 -V
    python3 -Im site
EOT

# Do not install ./libnccl-dev*deb for now.
RUN mkdir /nccl-debs
COPY --from=build /nccl-debs/libnccl2*deb /nccl-debs
RUN cd /nccl-debs && ls -1 && apt install --allow-change-held-packages -y ./libnccl2*deb && rm -rf /nccl-debs

# We don't need all CUDA math libraries (such as cuFFT and cuSPARSE). This list
# was assembled by trial and error, more libs added as errors popped up,
# example: "cupy.cuda.compiler.CompileException: nvrtc: error: failed to open
# libnvrtc-builtins.so.12.9"
COPY --from=build /usr/local/cuda-12.9/targets/sbsa-linux/lib/libcurand.so.10 /usr/local/cuda-12.9/targets/sbsa-linux/lib/libcurand.so.10
COPY --from=build /usr/local/cuda-12.9/targets/sbsa-linux/lib/libnvrtc.so.12 /usr/local/cuda-12.9/targets/sbsa-linux/lib/libnvrtc.so.12
COPY --from=build /usr/local/cuda-12.9/targets/sbsa-linux/lib/libnvrtc-builtins.so.12.9 /usr/local/cuda-12.9/targets/sbsa-linux/lib/libnvrtc-builtins.so.12.9

RUN du -ha / | sort -h  -r | head -n 100

RUN python3 -c "import requests"
RUN python3 -c "import cupy.cuda.nccl; print(cupy.cuda.nccl.get_version())"
RUN python3 -c "import cupy; print(cupy.cuda.runtime.driverGetVersion())"

# Lower-level bindings from an NVIDIA-maintained library (not cupy)
# provides e.g. https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html#cuda.bindings.driver.cuMemExportToShareableHandle
# RUN pip install cuda-python[all]

COPY ./nickelpie.py /thing

# Override original/NVIDIA entry point.
# Launch Python application e.g. via
# docker run -v $(pwd):/jp jpnccl python /jp/jpcoms.py
#
# I build the image with
# docker buildx build --progress plain . -t jpnccl -f nccl.Dockerfile

# ENV NCCL_MNNVL_ENABLE=0
ENTRYPOINT []
