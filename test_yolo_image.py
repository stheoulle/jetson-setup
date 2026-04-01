#!/usr/bin/env python3
"""
Test YOLO detection on image.png with OCR extraction
Generates 3 output images with detected boxes and numbers
"""

from ultralytics import YOLO
import cv2
import torch
import easyocr
import re
from pathlib import Path

def extract_4digit(text):
    """Extract exactly 4 consecutive digits from text"""
    text = text.replace(' ', '').replace(',', '').replace('.', '')
    digits = "".join(re.findall(r"\d+", text))
    
    if len(digits) == 4:
        return digits
    elif len(digits) > 4:
        match = re.search(r'\d{4}', text)
        if match:
            return match.group()
    return None

# Setup
print("Loading model...")
model = YOLO('runs/detect/train22/weights/best.pt')
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device.upper()}")

# Load OCR
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=(device == "cuda"), verbose=False)
print("EasyOCR loaded")

# Load image
image_path = Path("image.png")
if not image_path.exists():
    print(f"ERROR: {image_path} not found!")
    exit(1)

frame = cv2.imread(str(image_path))
if frame is None:
    print(f"ERROR: Could not load image!")
    exit(1)

frame_height, frame_width = frame.shape[:2]
print(f"Image loaded: {frame.shape}")

# Test YOLO with different confidence thresholds
confidence_levels = [0.5, 0.3, 0.1]

for conf_threshold in confidence_levels:
    print(f"\n{'='*70}")
    print(f"Testing with confidence={conf_threshold}")
    print(f"{'='*70}")
    
    # Run YOLO detection
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        imgsz=192,
        device=device,
        verbose=False
    )
    
    # Create output frame copy
    output_frame = frame.copy()
    
    num_detections = len(results[0].boxes)
    print(f"Objects detected: {num_detections}")
    
    detection_list = []
    
    if num_detections > 0:
        # Process each detection
        for i, box in enumerate(results[0].boxes):
            conf = float(box.conf[0])
            
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Add padding to crop
            padding = 5
            x1_crop = max(0, x1 - padding)
            y1_crop = max(0, y1 - padding)
            x2_crop = min(frame_width, x2 + padding)
            y2_crop = min(frame_height, y2 + padding)
            
            # Crop detected region
            crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            detected_number = None
            ocr_score = 0.0
            
            if crop.size > 0:
                try:
                    # Preprocess for OCR
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    # Upscale for better OCR
                    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    
                    # Apply OCR
                    ocr_results = reader.readtext(gray, detail=1)
                    
                    # Extract 4-digit number from OCR results
                    for detection in ocr_results:
                        bbox, raw_text, score = detection
                        number = extract_4digit(raw_text)
                        if number and score > 0.3:
                            detected_number = number
                            ocr_score = score
                            break
                except Exception as e:
                    pass
            
            # Draw box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            if detected_number:
                label = f"#{detected_number} (OCR:{ocr_score:.2f}, YOLO:{conf:.2f})"
                detection_list.append(f"  {i+1}. Number: {detected_number} | YOLO: {conf:.3f} | OCR: {ocr_score:.3f}")
            else:
                label = f"No number (YOLO:{conf:.2f})"
                detection_list.append(f"  {i+1}. No number detected | YOLO: {conf:.3f}")
            
            # Draw label on frame
            cv2.putText(output_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(label)
        
        print(f"\nSummary:")
        for item in detection_list:
            print(item)
    
    # Save output image
    output_filename = f"result_conf_{conf_threshold}.png"
    cv2.imwrite(output_filename, output_frame)
    print(f"\n✓ Saved: {output_filename}")

print(f"\n{'='*70}")
print("All tests completed!")
print(f"Generated files:")
print(f"  - result_conf_0.5.png")
print(f"  - result_conf_0.3.png")
print(f"  - result_conf_0.1.png")
print(f"{'='*70}")
