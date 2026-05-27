FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Test frameworks
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    tox

# Scientific / data stack (pandas, scikit-learn, matplotlib, scipy all need numpy)
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib

# Web / network frameworks common in BugsInPy
RUN pip install --no-cache-dir \
    requests \
    httpie \
    flask \
    django \
    scrapy \
    sanic \
    aiohttp

# DevOps / automation
RUN pip install --no-cache-dir \
    ansible \
    luigi

# Code-quality tools (black, mypy, flake8 are themselves BugsInPy subjects)
RUN pip install --no-cache-dir \
    black \
    mypy \
    flake8 \
    pylint

# Misc BugsInPy subjects
RUN pip install --no-cache-dir \
    thefuck \
    youtube-dl \
    spacy \
    tqdm \
    attrs \
    click \
    pyyaml \
    toml

CMD ["bash"]
