#!/usr/bin/env python3
"""
YOLO Video Inference - Simple version using YOLO's built-in video processing
Optimized for Jetson Orin with memory constraints
"""

import torch
from pathlib import Path
from ultralytics import YOLO
import sys
import os

print("=" * 70)
print("YOLO Video Inference - Jetson Orin Optimized (Simple Mode)")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("⚠️  GPU not detected - using CPU")
    device = "cpu"

print(f"Inference device: {device.upper()}")
print("=" * 70)

# Parse arguments
if len(sys.argv) < 2:
    print("\n❌ Usage: python3 app_video_simple.py <video_file> [options]")
    print("\nOptions:")
    print("  --conf FLOAT     Confidence threshold (default: 0.5)")
    print("  --imgsz INT      Inference size (default: 320)")
    print("  --save-path DIR  Output directory (default: same as input)")
    sys.exit(1)

video_path = sys.argv[1]
conf = 0.5
imgsz = 320
save_path = None

# Parse optional arguments
i = 2
while i < len(sys.argv):
    if sys.argv[i] == '--conf' and i + 1 < len(sys.argv):
        conf = float(sys.argv[i + 1])
        i += 2
    elif sys.argv[i] == '--imgsz' and i + 1 < len(sys.argv):
        imgsz = int(sys.argv[i + 1])
        i += 2
    elif sys.argv[i] == '--save-path' and i + 1 < len(sys.argv):
        save_path = sys.argv[i + 1]
        i += 2
    else:
        i += 1

video_path = Path(video_path)
if not video_path.exists():
    print(f"\n❌ Video not found: {video_path}")
    sys.exit(1)

model_path = Path("runs/detect/train22/weights/best.pt")
if not model_path.exists():
    print(f"\n❌ Model not found: {model_path}")
    sys.exit(1)

print(f"\n📹 Input video: {video_path}")
print(f"🤖 Model: {model_path}")
print(f"📊 Confidence: {conf}")
print(f"🎯 Image size: {imgsz}px")
print("=" * 70)

# Clear GPU cache
if device == "cuda":
    torch.cuda.empty_cache()
    print("\n✅ GPU cache cleared")

# Load model
print("🔄 Loading model...")
model = YOLO(str(model_path))

# Set output directory
if save_path:
    project = save_path
    name = ""
else:
    project = str(video_path.parent)
    name = f"{video_path.stem}_detected"

print(f"💾 Output will be saved to: {Path(project) / name}")
print("\n🎬 Starting video processing...\n")

# Process video using YOLO's built-in predict with streaming
# This is more memory-efficient than loading the entire video
try:
    results = model.predict(
        source=str(video_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=True,
        project=project,
        name=name,
        exist_ok=True,
        stream=True,  # Process frames one at a time (memory efficient)
        verbose=True,
        half=False,  # Disable FP16 for stability on Jetson
        augment=False,  # No augmentation for faster inference
        batch=1,  # Process one frame at a time
    )
except Exception as e:
    print(f"\n❌ Error during video processing: {e}")
    print("\n💡 Try using CPU instead:")
    print("   ./inference_cpu.sh vidéos/C0088.MP4")
    sys.exit(1)

# Process the stream (this actually runs the inference)
frame_count = 0
for result in results:
    frame_count += 1
    # Clear cache periodically
    if device == "cuda" and frame_count % 50 == 0:
        torch.cuda.empty_cache()

print("\n" + "=" * 70)
print(f"✅ Video inference completed!")
print(f"📊 Processed {frame_count} frames")
print(f"📁 Output saved in: {Path(project) / name}")
print("=" * 70)
