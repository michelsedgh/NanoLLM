# Jetson-compatible PyTorch 1.13 / Python 3.8 base (JetPack r35.4.x)
FROM dustynv/pytorch:1.13-r35.4.1

ENV DEBIAN_FRONTEND=noninteractive

# Install core utilities not already provided by the base image
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        g++ \
        curl \
        unzip \
        git \
        tzdata \
        ninja-build \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Optional: install NodeJS + npx
RUN apt-get update && apt-get install -y nodejs npm && npm install -g npx

# Install Jupyter
RUN pip install notebook ipykernel

# Set working directory
WORKDIR /workspace

# Optionally copy your codebase into container
# COPY . .

# Expose port for Jupyter
EXPOSE 5018

