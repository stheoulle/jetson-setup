# train_yolo.py
import torch
from ultralytics import YOLO

model = YOLO("yolo12s.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

results = model.train(
    data="dataset/data.yaml",
    epochs=20
)

print("Training completed")
