#!/bin/bash
# GPU Training with Docker - Jetson Orin

echo "Starting YOLO Training with GPU (Docker)"
echo "============================================"
echo "Container: dustynv/l4t-pytorch:r36.4.0"
echo "GPU: Enabled "
echo "GPU Memory: Optimized for Jetson Orin"
echo "============================================"
echo ""

# Run training in Docker with GPU support
sudo docker run --rm --gpus all \
  --dns 8.8.8.8 --dns 8.8.4.4 \
  --network bridge \
  --shm-size=8g \
  --memory=7G \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e PIP_INDEX_URL=https://pypi.org/simple \
  -e PIP_TRUSTED_HOST="" \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  dustynv/l4t-pytorch:r36.4.0 \
  bash -c "
    echo 'Installing Ultralytics YOLO with NumPy compatibility...'
    pip install 'numpy<2.0' ultralytics
    echo ''
    echo 'Starting training with GPU memory optimization...'
    echo ''
    python3 app.py
  "

echo ""
echo "============================================"
echo "Training completed!"
echo "============================================"
