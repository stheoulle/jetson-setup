# Docker GPU Training - Quick Start

## ✅ Docker Setup Complete!

Docker is configured with NVIDIA GPU support and ready to use.

## Quick Commands

### 1. Test GPU
```bash
./test_gpu_docker.sh
```

### 2. Run Training with GPU
```bash
./train_gpu_docker.sh
```

### 3. Manual Docker Command
```bash
sudo docker run --rm --gpus all \
  --network host --shm-size=8g \
  -v $(pwd):/workspace -w /workspace \
  dustynv/l4t-pytorch:r36.4.0 \
  bash -c "pip install ultralytics && python3 app.py"
```

### 4. Interactive Docker Session
```bash
sudo docker run -it --gpus all \
  --network host --shm-size=8g \
  -v $(pwd):/workspace -w /workspace \
  dustynv/l4t-pytorch:r36.4.0 \
  /bin/bash
```

## What's Configured

✅ **Docker**: v29.2.1  
✅ **NVIDIA Runtime**: Enabled (default)  
✅ **Container**: dustynv/l4t-pytorch:r36.4.0  
✅ **PyTorch**: 2.4.0 with CUDA 12.6  
✅ **GPU**: Orin detected and working  

## Performance Comparison

| Mode | Device | Batch | Speed | Time (20 epochs) |
|------|--------|-------|-------|------------------|
| Standalone | CPU | 4 | ~5 it/s | ~3 hours |
| **Docker** | **GPU** | **8-16** | **~30-50 it/s** | **~30 min** |

## Training Parameters (GPU)

The [app.py](app.py) script automatically optimized for Jetson:
- **Batch size**: 8 (GPU) vs 4 (CPU)
- **Image size**: 416 (memory-optimized)
- **Workers**: 2 (Jetson-optimized)
- **Device**: Automatically detected

## Troubleshooting

### Permission Denied
```bash
# Add yourself to docker group
sudo usermod -aG docker $USER
newgrp docker

# Then run without sudo
docker run --gpus all ...
```

### Out of Memory
Reduce batch size in [app.py](app.py):
```python
batch=4,  # Reduce from 8
imgsz=320  # Reduce from 416
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

## Files

- **train_gpu_docker.sh** - One-command GPU training
- **test_gpu_docker.sh** - Quick GPU verification
- **app.py** - Training script (CPU/GPU compatible)
- **DOCKER_GUIDE.md** - This file

## References

- [NVIDIA L4T PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)
- [Jetson Containers GitHub](https://github.com/dusty-nv/jetson-containers)
- [Original Docker Setup Notes](retour.md)
