#!/usr/bin/env python3
"""
YOLO Video Inference with OCR - Frame-by-frame processing
Detects boxes, applies OCR to recognize 4-digit numbers, and saves counts to CSV
Optimized for Jetson Orin with memory constraints
Uses EasyOCR for better Jetson compatibility
"""

import torch
import cv2
import re
import csv
from pathlib import Path
from ultralytics import YOLO
import easyocr
import sys
from collections import defaultdict

try:
    import numpy as np
except Exception as e:
    print("Numpy is not available in this environment")
    print("Install inside container: pip3 install --no-cache-dir \"numpy==1.26.4\"")
    raise

try:
    torch.from_numpy(np.zeros((1,), dtype=np.float32))
except Exception as e:
    print("NumPy/Torch compatibility issue detected")
    print(f"   NumPy version: {np.__version__}")
    print("Fix inside container: pip3 install --no-cache-dir \"numpy==1.26.4\"")
    sys.exit(1)

print("=" * 70)
print("YOLO Video Inference with OCR - Jetson Orin Optimized (EasyOCR)")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("GPU not detected - using CPU")
    device = "cpu"

print(f"Inference device: {device.upper()}")
print("=" * 70)

# Parse arguments
if len(sys.argv) < 2:
    print("\nUsage: python3 app_video_ocr_easy.py <video_file> [options]")
    print("\nOptions:")
    print("  --conf FLOAT        Confidence threshold (default: 0.5)")
    print("  --imgsz INT         Inference size (default: 320)")
    print("  --output-csv FILE   Output CSV file (default: detections.csv)")
    print("  --save-video        Save annotated video with OCR results")
    print("  --frame-skip INT    Process every Nth frame (default: 1)")
    print("  --ocr-cpu           Force OCR to use CPU (default: use GPU)")
    sys.exit(1)

video_path = sys.argv[1]
conf = 0.5
imgsz = 320
output_csv = "detections.csv"
save_video = False
frame_skip = 1
ocr_gpu = True

# Parse optional arguments
i = 2
while i < len(sys.argv):
    if sys.argv[i] == '--conf' and i + 1 < len(sys.argv):
        conf = float(sys.argv[i + 1])
        i += 2
    elif sys.argv[i] == '--imgsz' and i + 1 < len(sys.argv):
        imgsz = int(sys.argv[i + 1])
        i += 2
    elif sys.argv[i] == '--output-csv' and i + 1 < len(sys.argv):
        output_csv = sys.argv[i + 1]
        i += 2
    elif sys.argv[i] == '--save-video':
        save_video = True
        i += 1
    elif sys.argv[i] == '--frame-skip' and i + 1 < len(sys.argv):
        frame_skip = int(sys.argv[i + 1])
        i += 2
    elif sys.argv[i] == '--ocr-cpu':
        ocr_gpu = False
        i += 1
    else:
        i += 1

video_path = Path(video_path)
if not video_path.exists():
    print(f"\nVideo not found: {video_path}")
    sys.exit(1)

model_path = Path("runs/detect/train22/weights/best.pt")
if not model_path.exists():
    print(f"\nModel not found: {model_path}")
    sys.exit(1)

print(f"\nInput video: {video_path}")
print(f"Model: {model_path}")
print(f"Confidence: {conf}")
print(f"Image size: {imgsz}px")
print(f"Output CSV: {output_csv}")
print(f"Save video: {save_video}")
print(f"Frame skip: {frame_skip}")
print(f"OCR device: {'GPU' if ocr_gpu else 'CPU'}")
print("=" * 70)

# Clear GPU cache
if device == "cuda":
    torch.cuda.empty_cache()
    print("\nGPU cache cleared")

# Load YOLO model
print("Loading YOLO model...")
model = YOLO(str(model_path))
model.to(device)

# Load EasyOCR
print("Loading EasyOCR...")
print("   (First run will download model files, this may take a moment)")
try:
    reader = easyocr.Reader(['en'], gpu=ocr_gpu, verbose=False)
    print("EasyOCR loaded successfully")
except Exception as e:
    print(f"Warning: Could not initialize EasyOCR with GPU, trying CPU...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("EasyOCR loaded on CPU")

# Helper function to extract 4-digit numbers
def extract_4digit(text):
    """Extract exactly 4 consecutive digits from text"""
    # Remove spaces and clean text
    text = text.replace(' ', '').replace(',', '').replace('.', '')
    digits = "".join(re.findall(r"\d+", text))
    
    if len(digits) == 4:
        return digits
    elif len(digits) > 4:
        # Try to extract first 4-digit sequence
        match = re.search(r'\d{4}', text)
        if match:
            return match.group()
    return None

# Dictionary to count detections
detection_counts = defaultdict(int)

# Open video
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"\nFailed to open video: {video_path}")
    sys.exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo info: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

# Setup video writer if needed
video_writer = None
if save_video:
    output_video = video_path.parent / f"{video_path.stem}_ocr.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (frame_width, frame_height))
    print(f"Output video: {output_video}")

print("\nStarting video processing with OCR...\n")

frame_count = 0
processed_count = 0
detection_count = 0
processing_failed = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames if needed
        if frame_count % frame_skip != 0:
            if save_video and video_writer:
                video_writer.write(frame)
            continue
        
        processed_count += 1
        
        # Run YOLO detection
        results = model(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
        
        # Process each detection
        for r in results:
            for box in r.boxes:
                box_conf = float(box.conf[0])
                if box_conf < conf:
                    continue
                
                # Extract box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Add padding to crop
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame_width, x2 + padding)
                y2 = min(frame_height, y2 + padding)
                
                # Crop detected region
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Preprocess for OCR
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # Upscale for better OCR
                scale = 2
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                
                # Apply OCR
                try:
                    ocr_results = reader.readtext(gray, detail=1)
                    
                    for detection in ocr_results:
                        bbox, raw_text, ocr_score = detection
                        
                        # Extract 4-digit number
                        number = extract_4digit(raw_text)
                        
                        if number and ocr_score > 0.3:  # Minimum OCR confidence
                            detection_counts[number] += 1
                            detection_count += 1
                            
                            # Draw on frame if saving video
                            if save_video:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(
                                    frame,
                                    number,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    (0, 255, 0),
                                    2
                                )
                            
                            if detection_count % 10 == 0:
                                print(f"Frame {frame_count}/{total_frames}: Detected {number} "
                                      f"(OCR: {ocr_score:.2f}, YOLO: {box_conf:.2f})")
                
                except Exception as e:
                    # OCR can fail on some crops, continue
                    pass
        
        # Write frame if saving video
        if save_video and video_writer:
            video_writer.write(frame)
        
        # Clear cache periodically
        if device == "cuda" and processed_count % 50 == 0:
            torch.cuda.empty_cache()
        
        # Progress update
        if frame_count % 100 == 0:
            print(f"Progress: {frame_count}/{total_frames} frames "
                  f"({100*frame_count/total_frames:.1f}%) - "
                  f"{len(detection_counts)} unique numbers detected")

except KeyboardInterrupt:
    print("\n\nProcessing interrupted by user")
    processing_failed = True
except Exception as e:
    print(f"\nError during processing: {e}")
    import traceback
    traceback.print_exc()
    processing_failed = True
finally:
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()

print("\n" + "=" * 70)
print(f"Video processing completed!")
print(f"Processed {processed_count} frames (total: {frame_count})")
print(f"Total detections: {detection_count}")
print(f"Unique numbers: {len(detection_counts)}")
print("=" * 70)

if processing_failed:
    print("Processing ended with errors. CSV is not saved.")
    sys.exit(1)

# Save results to CSV
print(f"\nSaving results to {output_csv}...")
output_path = Path(output_csv)
try:
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Number', 'Count'])
        
        # Sort by number
        for number in sorted(detection_counts.keys()):
            writer.writerow([number, detection_counts[number]])
    
    print(f"CSV saved successfully!")
    
    # Show top 10 detected numbers
    print("\nTop 10 most detected numbers:")
    sorted_detections = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (number, count) in enumerate(sorted_detections[:10], 1):
        print(f"  {i}. {number}: {count} times")
    
except Exception as e:
    print(f"Error saving CSV: {e}")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
