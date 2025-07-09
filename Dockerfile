FROM python:3.11-slim AS juman-build

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Juman++
COPY scripts/install_jumanpp.sh .
RUN bash install_jumanpp.sh


# Use official Python runtime as base image
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

# Install system dependencies and locales
RUN apt-get update && apt-get install -y \
    apt-utils \
    git \
    wget \
    curl \
    build-essential \
    locales \
    nano \
    sudo \
    cmake \
    automake \
    libtool \
    zlib1g-dev \
    libprotobuf-dev \
    protobuf-compiler \
    # Docker CLIをインストール（SWE-bench評価用）
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Configure Japanese locale properly
RUN echo "ja_JP.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=ja_JP.UTF-8

# Set locale environment variables
ENV LANG=ja_JP.UTF-8
ENV LANGUAGE=ja_JP:ja
ENV LC_ALL=ja_JP.UTF-8

# Set working directory and change ownership
WORKDIR /workspace

# Copy Juman++ from juman-build stage
COPY --from=juman-build /usr/local/bin/jumanpp /usr/local/bin/
COPY --from=juman-build /usr/local/libexec/jumanpp /usr/local/libexec/jumanpp

# Juman++ を評価スクリプトから確実に見えるように
ENV PATH="/usr/local/bin:${PATH}" \
    JUMANPP_COMMAND=/usr/local/bin/jumanpp \
    JUMANPP_DICDIR=/usr/local/share/jumandic

# Initialize uv project and sync dependencies
COPY pyproject.toml uv.lock .
RUN uv sync

# SWE-benchを公式リポジトリからインストール（ソースコードは保持）
RUN git clone https://github.com/princeton-nlp/SWE-bench.git /opt/SWE-bench && \
    cd /workspace && \
    uv pip install -e /opt/SWE-bench

RUN echo 'source /workspace/.venv/bin/activate' >> ~/.bashrc

# Clone the repository
#RUN git clone -b nejumi4-dev https://github.com/wandb/llm-leaderboard.git .
COPY scripts ./scripts

# Set permissions for scripts (if they exist)
RUN chmod +x scripts/*.py 2>/dev/null || true

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Dockerグループが存在する場合のための準備（実行時に適切なGIDで動作する）
# 注意: 実際のdockerグループGIDはホストに依存するため、実行時に調整が必要

# Expose port (if needed for web interface)
EXPOSE 8080

# Set entrypoint for docker permissions
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command
CMD ["uv", "run", "scripts/run_eval.py"]