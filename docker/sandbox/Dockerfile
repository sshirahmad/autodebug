# AutoDebug sandbox image.
#
# The system builds a per-bug conda env at the bug's EXACT Python version on the
# volume at clone time (micromamba create -p /workspace/env python=<version>), so
# this image carries NO Python package stack — just git, a C toolchain + the
# system libs that source builds commonly need, and micromamba. Per-bug deps land
# in the env; old numpy/scipy/etc. install as matching wheels instead of failing
# to build against a single baked-in Python.
#
# Build:
#   docker build -t autodebug-sandbox:latest ./docker/sandbox

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
        curl \
        bzip2 \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# micromamba: a single static binary. Envs are created by explicit prefix
# (-p /workspace/env), so no base env or shell hook is needed.
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

# Shared conda package cache (mounted as a named volume at runtime) so each
# Python version + wheel is downloaded once across all bugs.
ENV MAMBA_ROOT_PREFIX=/opt/mamba

WORKDIR /workspace
CMD ["bash"]
