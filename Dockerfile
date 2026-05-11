FROM ubuntu:22.04

LABEL org.opencontainers.image.title="S4-FIFO Artifact" \
      org.opencontainers.image.description="OSDI 2026 Artifact: Learning-Augmented Heuristics for Cache Eviction"

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for libCacheSim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libglib2.0-dev \
    libgoogle-perftools-dev \
    libzstd-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /artifact

# Copy libCacheSim submodule source (dockerignore excludes .git)
COPY libCacheSim/ /artifact/libCacheSim/

# Build libCacheSim with Release configuration
RUN cd /artifact/libCacheSim && \
    mkdir -p build && \
    cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_TESTS=OFF \
      -DUSE_HUGEPAGE=OFF && \
    make -j$(nproc)

# Make cachesim binary available on PATH
RUN ln -s /artifact/libCacheSim/build/bin/cachesim /usr/local/bin/cachesim

# Copy analysis scripts and documentation
COPY analysis/ /artifact/analysis/
COPY doc/ /artifact/doc/
COPY requirements.txt /artifact/requirements.txt
COPY Makefile /artifact/Makefile
COPY README.md /artifact/README.md
COPY LICENSE /artifact/LICENSE

# Install Python dependencies for analysis scripts
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    xgboost \
    lightgbm \
    joblib \
    m2cgen \
    jupyter

# Install PyTorch CPU-only (separate to avoid pip index conflicts)
RUN pip3 install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Verify build
RUN cachesim --help 2>/dev/null || echo "cachesim binary built at /artifact/libCacheSim/build/bin/cachesim"

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'S4-FIFO Artifact - OSDI 2026' && echo '' && echo 'Usage:' && echo '  cachesim <trace_path> <trace_type> s4fifo <cache_size>' && echo '  Example: cachesim libCacheSim/data/cloudPhysicsIO.vscsi vscsi s4fifo 1GB' && echo '' && echo 'Python analysis scripts are in /artifact/analysis/' && echo 'Run bash to start an interactive shell'"]
