#!/bin/bash
# GPU Video Inference with Docker (Simple Mode) - Jetson Orin
# This version uses YOLO's built-in video processing for better memory efficiency
# Uses persistent docker-compose container for faster execution

echo "YOLO Video Inference - Simple Mode (Docker)"
echo "=============================================="
echo "Container: dustynv/l4t-pytorch:r36.4.0"
echo "GPU: Enabled "
echo "Model: train22"
echo "Memory: Optimized for Jetson Orin"
echo "=============================================="
echo ""

if [ $# -lt 1 ]; then
    echo "Usage: $0 <video_file> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --conf FLOAT     Confidence threshold (default: 0.5)"
    echo "  --imgsz INT      Inference size (default: 320)"
    echo ""
    echo "Examples:"
    echo "  $0 video.mp4"
    echo "  $0 video.mp4 --conf 0.7 --imgsz 416"
    echo ""
    echo "Note: Make sure to run 'docker compose up -d' first!"
    exit 1
fi

# Check if container is running
if ! docker compose ps | grep -q "yolo-inference.*running"; then
    echo "⚠️  Container not running. Starting it now..."
    echo "    This will take a moment for first-time setup..."
    docker compose up -d
    echo ""
    echo "⏳ Waiting for dependencies to install..."
    sleep 10
fi

echo "🎬 Starting video inference..."
echo ""

# Run inference in the persistent container
docker compose exec yolo-inference python3 app_video_simple.py "$@"

echo ""
echo "=============================================="
echo "Video inference completed"
echo "=============================================="
