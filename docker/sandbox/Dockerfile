FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pre-install common test frameworks so repro scripts don't need to download them
RUN pip install --no-cache-dir pytest pytest-asyncio

CMD ["bash"]
