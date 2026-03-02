# train_yolo.py
import torch
from ultralytics import YOLO

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

model = YOLO("yolo12s.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    results = model.train(
        data="dataset/data.yaml",
        epochs=20,
        device=device,
        batch=8,  # Reduced batch size for Jetson memory constraints
        imgsz=416,  # Reduced image size
        workers=2,  # Fewer workers for Jetson
        patience=10,
        save=True
    )
    print("Training completed successfully!")
except RuntimeError as e:
    print(f"GPU training failed with error: {e}")
    print("Falling back to CPU training...")
    results = model.train(
        data="dataset/data.yaml",
        epochs=20,
        device="cpu",
        batch=4,
        imgsz=416,
        workers=2,
        patience=10,
        save=True
    )
    print("CPU training completed")
