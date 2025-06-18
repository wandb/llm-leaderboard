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

# Clone the repository
#RUN git clone -b nejumi4-dev https://github.com/wandb/llm-leaderboard.git .
COPY ./ /workspace

# Initialize uv project and sync dependencies
RUN uv sync
RUN echo 'source /workspace/.venv/bin/activate' >> ~/.bashrc
ENV PATH="/workspace/.venv/bin:$PATH"

# Juman++
RUN bash scripts/install_jumanpp.sh

# Juman++ を評価スクリプトから確実に見えるように
ENV PATH="/usr/local/bin:${PATH}" \
    JUMANPP_COMMAND=/usr/local/bin/jumanpp \
    JUMANPP_DICDIR=/usr/local/share/jumandic

# Set permissions for scripts (if they exist)
RUN chmod +x scripts/*.py 2>/dev/null || true

# Expose port (if needed for web interface)
EXPOSE 8080

# Default command
CMD ["uv", "run", "scripts/run_eval.py", "--help"]