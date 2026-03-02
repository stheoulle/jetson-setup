#!/bin/bash
# GPU Video Inference with Docker (Simple Mode) - Jetson Orin
# This version uses YOLO's built-in video processing for better memory efficiency

echo "🚀 YOLO Video Inference - Simple Mode (Docker)"
echo "=============================================="
echo "Container: dustynv/l4t-pytorch:r36.4.0"
echo "GPU: Enabled ✅"
echo "Model: train22"
echo "Memory: Optimized for Jetson Orin"
echo "=============================================="
echo ""

if [ $# -lt 1 ]; then
    echo "❌ Usage: $0 <video_file> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --conf FLOAT     Confidence threshold (default: 0.5)"
    echo "  --imgsz INT      Inference size (default: 320)"
    echo ""
    echo "Examples:"
    echo "  $0 video.mp4"
    echo "  $0 video.mp4 --conf 0.7 --imgsz 416"
    exit 1
fi

# Run inference in Docker with same memory config as training
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
    echo '📦 Installing dependencies...'
    pip install -q 'numpy<2.0' ultralytics
    echo ''
    echo '🎬 Starting video inference...'
    echo ''
    python3 app_video_simple.py $@
  "

echo ""
echo "=============================================="
echo "✅ Video inference completed!"
echo "=============================================="
