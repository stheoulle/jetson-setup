#!/bin/bash
# GPU Video Inference with Docker - Jetson Orin

echo "🚀 Starting YOLO Video Inference with GPU (Docker)"
echo "==================================================="
echo "Container: dustynv/l4t-pytorch:r36.4.0"
echo "GPU: Enabled ✅"
echo "Model: train22 (trained model)"
echo "==================================================="
echo ""

if [ $# -lt 1 ]; then
    echo "❌ Usage: $0 <video_file> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --output PATH    Output video path (default: input_detected.mp4)"
    echo "  --conf FLOAT     Confidence threshold (default: 0.5)"
    echo "  --imgsz INT      Inference size in pixels (default: 640, lower=faster)"
    echo "  --device DEVICE  Use 'cuda' or 'cpu' (default: cuda)"
    echo ""
    echo "Examples:"
    echo "  $0 video.mp4"
    echo "  $0 video.mp4 --output output.mp4 --conf 0.7"
    echo "  $0 video.mp4 --imgsz 416  # Faster, less memory"
    exit 1
fi

# Run inference in Docker with GPU support
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
    echo '📦 Installing Ultralytics YOLO with NumPy compatibility...'
    pip install -q 'numpy<2.0' ultralytics tqdm
    echo ''
    echo '🎬 Starting video inference with GPU memory optimization...'
    echo ''
    python3 app_video_inference.py $@
  "

echo ""
echo "==================================================="
echo "✅ Video inference completed!"
echo "==================================================="
