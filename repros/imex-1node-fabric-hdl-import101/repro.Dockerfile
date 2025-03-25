FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# Install Ubuntu 24's system Python (3.12).
RUN <<EOT
    apt update -qy
    apt install -qyy python3.12 python3.12-venv
    apt clean
    rm -rf /var/lib/apt/lists/*
    rm -rf /opt/nvidia/nsight-compute
EOT

RUN mkdir /thing
WORKDIR /thing

RUN python3.12 -m venv .venv
ENV PATH="/thing/.venv/bin:${PATH}"
RUN pip install requests cuda-python[all]

COPY ./fabric-handle-transfer-test.py /thing

ENTRYPOINT []
