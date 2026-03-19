FROM nvidia/cuda:12.9.1-base-ubuntu24.04

# Install Ubuntu 24's system Python (3.12).
RUN <<EOT
    apt update -qy
    apt install -qyy python3.12 python3.12-venv
    apt clean
    rm -rf /var/lib/apt/lists/*

EOT

RUN mkdir /atack
WORKDIR /atack

RUN python3.12 -m venv .venv
ENV PATH="/atack/.venv/bin:${PATH}"
RUN pip install requests cuda-python[all]==12.9.6 dnspython orjson

COPY ./atack.py /atack

ENTRYPOINT []
