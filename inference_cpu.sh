#!/bin/bash
# CPU Video Inference with Docker - Jetson Orin (Fallback)
# Use this if GPU inference fails due to memory constraints

echo "🚀 YOLO Video Inference - CPU Mode (Docker)"
echo "============================================"
echo "Container: dustynv/l4t-pytorch:r36.4.0"
echo "Device: CPU (no GPU memory issues)"
echo "Model: train22"
echo "⚠️  Note: CPU is slower but more reliable"
echo "============================================"
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
    echo "  $0 video.mp4 --conf 0.7"
    exit 1
fi

# Run inference in Docker using CPU
sudo docker run --rm \
  --dns 8.8.8.8 --dns 8.8.4.4 \
  --network bridge \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -e PIP_INDEX_URL=https://pypi.org/simple \
  -e PIP_TRUSTED_HOST="" \
  dustynv/l4t-pytorch:r36.4.0 \
  bash -c "
    echo '📦 Installing dependencies...'
    pip install -q 'numpy<2.0' ultralytics
    echo ''
    echo '🎬 Starting video inference on CPU...'
    echo ''
    python3 -c \"
import torch
from pathlib import Path
from ultralytics import YOLO
import sys

# Parse arguments
args = '$@'.split()
video_path = args[0] if args else None
conf = 0.5
imgsz = 320

i = 1
while i < len(args):
    if args[i] == '--conf' and i + 1 < len(args):
        conf = float(args[i + 1])
        i += 2
    elif args[i] == '--imgsz' and i + 1 < len(args):
        imgsz = int(args[i + 1])
        i += 2
    else:
        i += 1

if not video_path:
    print('Error: No video file specified')
    sys.exit(1)

video_path = Path(video_path)
model_path = Path('runs/detect/train22/weights/best.pt')

print(f'📹 Video: {video_path}')
print(f'🤖 Model: {model_path}')
print(f'📊 Confidence: {conf}')
print(f'🎯 Size: {imgsz}px')
print(f'💻 Device: CPU')
print()

model = YOLO(str(model_path))

results = model.predict(
    source=str(video_path),
    conf=conf,
    imgsz=imgsz,
    device='cpu',
    save=True,
    project=str(video_path.parent),
    name=f'{video_path.stem}_detected',
    exist_ok=True,
    stream=True,
    verbose=True,
    augment=False,
    batch=1,
)

frame_count = 0
for result in results:
    frame_count += 1

print(f'\\n✅ Processed {frame_count} frames')
\" $@
  "

echo ""
echo "============================================"
echo "✅ Video inference completed!"
echo "============================================"
