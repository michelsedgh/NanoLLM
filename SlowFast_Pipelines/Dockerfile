# Base JetPack 5 (r35) container with PyTorch 1.13 / Python 3.8
FROM dustynv/pytorch:1.13-r35.4.1

ENV DEBIAN_FRONTEND=noninteractive

# Install only additional system utilities needed for notebooks and demos
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
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

# Run Jupyter Notebook on container start
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=5018", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

