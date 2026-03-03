---

# Jetson PyTorch GPU Setup with Docker – Troubleshooting Guide

This document explains the steps to get PyTorch with CUDA working on a Jetson device using Docker containers. It also highlights common issues and solutions encountered on **JetPack 6.x / L4T R36.x** systems.

---

## Final Working Docker Command

The GPU-enabled PyTorch container can be launched with:

```bash
sudo docker run --gpus all -it --rm \
  --network host \
  --shm-size=8g \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /home/laposte/jetson-setup/jetson-containers/data:/data \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  dustynv/l4t-pytorch:r36.4.0 /bin/bash
```

This command ensures:

* Full GPU access (`--gpus all`)
* X11 forwarding (`DISPLAY` and `/tmp/.X11-unix`)
* Shared memory for CUDA workloads (`--shm-size=8g`)
* Access to Argus cameras and container data volumes

---

## 🔹 Common Issues Encountered

### 1. CPU-only PyTorch

```python
>>> import torch
>>> torch.version.cuda
None
>>> torch.cuda.is_available()
False
```

**Cause:** installed the CPU-only PyTorch wheel (`torch-2.10.0+cpu`) from PyPI. Jetson devices require the NVIDIA-provided PyTorch wheel compiled with the JetPack CUDA version.

---

### 2. Missing CUDA libraries

```bash
OSError: libcudart.so.12: cannot open shared object file: No such file or directory
ValueError: libcublas.so.*[0-9] not found in the system path
```

**Cause:** The system did not have `/usr/lib/aarch64-linux-gnu/tegra/libcudart.so` or `libcublas.so` because the CUDA dev packages were missing. JetPack 6.x installs CUDA runtime differently and does not provide `nvidia-l4t-cuda-dev` via `apt`.

---

### 3. Docker GPU runtime errors

```bash
docker: Error response from daemon: unknown or invalid runtime name: nvidia
docker: could not select device driver "" with capabilities: [[gpu]]
```

**Cause:** The system was trying to use `nvidia-docker2` (common on x86) which **does not exist for Jetson**. Jetson uses only `nvidia-container-toolkit`.

---

## 🔹 How the Issues Were Fixed

### Step 1: Install NVIDIA Container Toolkit

```bash
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

No need for `nvidia-docker2` on Jetson.

---

### Step 2: Configure Docker runtime

Edit `/etc/docker/daemon.json`:

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

Then restart Docker:

```bash
sudo systemctl restart docker
```

---

### Step 3: Use Jetson-compatible PyTorch container

```bash
docker pull dustynv/l4t-pytorch:r36.4.0
```

**Key:** Use the `dustynv/l4t-pytorch` image that matches the **JetPack / L4T version**. Jetson containers come precompiled with the correct CUDA and PyTorch versions.

---

### Step 4: Launch container with GPU support

Use the final working `docker run` command (see above) to ensure:

* CUDA libraries are found
* PyTorch sees the GPU
* X11, Argus, and shared volumes are accessible

Inside the container:

```python
>>> import torch
>>> torch.version.cuda
'12.6'
>>> torch.cuda.is_available()
True
```

---

## 🔹 Key Lessons

1. **CPU wheels from PyPI don’t work on Jetson** — always use NVIDIA-provided wheels or containers.
2. **No `nvidia-docker2` on Jetson** — use `nvidia-container-toolkit` instead.
3. **Jetson containers handle CUDA and PyTorch** — avoid manual wheel installation unless targeting a specific CUDA version.
4. **GPU runtime must be configured in Docker** (`default-runtime: nvidia`).

---

This setup now allows GPU-accelerated PyTorch on Jetson devices reliably.

---
