import torch
from ultralytics import YOLO



# Load the YOLO model (this is for YOLOv8)
model = YOLO('yolov12s.yaml')

# Set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Using device: {device}")

# Start training
results = model.train(data='/home/laposte/Documents/immat/dataset/data.yaml', epochs=100)

print(f"Training completed! Results: {results}")
