from ultralytics import YOLO
from pathlib import Path

# Load trained model
weights_path = Path("/home/debian/jetson-setup/runs/detect/train4/weights/best.pt")
model = YOLO(weights_path)

# Evaluate on test set
data_yaml = "dataset/data.yaml"
results = model.val(data=data_yaml, split="test")  # returns DetMetrics

# Access the main results
print("Results Dictionary:")
print(results.results_dict)  # dictionary with precision, recall, mAP50, mAP50-95, fitness

# Alternatively, print class-wise summary
print("\nPer-class summary:")
for cls_summary in results.summary():  # returns list of dicts, one per class
    print(cls_summary)

# Access mAP50-95
print(f"\nmAP50-95: {results.fitness:.4f}")  # fitness is usually mAP50-95
