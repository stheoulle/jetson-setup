import cv2
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
import paddle

# ---------------------------
# Load TRAINED YOLO model
# ---------------------------
YOLO_WEIGHTS = "/home/debian/jetson-setup/runs/detect/train6/weights/best.pt"

# YOLO still uses PyTorch
device = "cuda" if paddle.is_compiled_with_cuda() else "cpu"
model = YOLO(YOLO_WEIGHTS)
model.to(device)
print(f"YOLO loaded on {device}")

# ---------------------------
# Load PaddleOCR
# ---------------------------
ocr = PaddleOCR(lang="en")

# ---------------------------
# Helper
# ---------------------------
def extract_digits(text):
    return "".join(re.findall(r"\d+", text))

# ---------------------------
# Load image
# ---------------------------
image_path = "runs/detect/train6/val_batch2_pred.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# ---------------------------
# YOLO inference
# ---------------------------
results = model(image, conf=0.5)

# ---------------------------
# OCR on detections
# ---------------------------
for r in results:
    for box in r.boxes:
        conf = float(box.conf[0])
        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Preprocess for OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # OCR
        ocr_result = ocr.ocr(gray, cls=True)

        if not ocr_result or not ocr_result[0]:
            continue

        for line in ocr_result[0]:
            raw_text = line[1][0]
            score = line[1][1]

            text = extract_digits(raw_text)
            if not text:
                continue

            print(f"Detected code: {text} (OCR score={score:.2f})")

            # Draw result
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

# ---------------------------
# Display
# ---------------------------
output_path = "output.jpg"
cv2.imwrite(output_path, image)
print(f"Result saved to {output_path}")
