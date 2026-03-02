# train_yolo.py - Jetson Orin Optimized
import torch
from ultralytics import YOLO

print("=" * 60)
print("YOLO Training on Jetson Orin")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    device = "cuda"
else:
    print("\u26a0\ufe0f  GPU not detected - using CPU")
    print("For GPU training, use Docker: dustynv/l4t-pytorch:r36.4.0")
    device = "cpu"

print(f"Training device: {device.upper()}")
print("=" * 60)

model = YOLO("yolo12s.pt")

# Training parameters optimized for Jetson Orin
# Adjust batch size based on available memory
import os

# Detect if running in Docker (lower batch size due to memory constraints)
in_docker = os.path.exists('/.dockerenv')
batch_size_gpu = 4 if in_docker else 8  # Smaller batch in Docker

training_params = {
    "data": "dataset/data.yaml",
    "epochs": 20,
    "imgsz": 320,
    "batch": 2,
    "workers": 0,
    "device": 0,
    "amp": False,
}

print(f"Training configuration:")
for key, value in training_params.items():
    print(f"  {key}: {value}")
print("=" * 60)

try:
    results = model.train(**training_params)
    print("\n" + "=" * 60)
    print("\u2705 Training completed successfully!")
    print("=" * 60)
except RuntimeError as e:
    if "CUDA" in str(e) or "GPU" in str(e):
        print(f"\n\u274c GPU training failed: {e}")
        print("\nFalling back to CPU training...")
        training_params["device"] = "cpu"
        training_params["batch"] = 4
        results = model.train(**training_params)
        print("\n" + "=" * 60)
        print("\u2705 CPU training completed")
        print("=" * 60)
    else:
        raise
