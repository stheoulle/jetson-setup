#!/usr/bin/env python3
"""
YOLO Live Stream Inference with OCR - Real-time processing
Detects boxes, applies OCR to recognize 4-digit numbers from live RTSP feed
Displays results in real-time with bounding boxes and OCR text
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
from datetime import datetime

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
print("YOLO Live Stream Inference with OCR - Jetson Orin Optimized")
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
    print("\nUsage: python3 app_live_ocr.py <stream_url> [options]")
    print("\nExample stream URLs:")
    print("  rtsp://10.149.73.176:8080/h264.sdp")
    print("  rtsp://10.149.73.176:8080/h264_aac.sdp")
    print("\nOptions:")
    print("  --conf FLOAT        Confidence threshold (default: 0.5)")
    print("  --imgsz INT         Inference size (default: 320)")
    print("  --output-csv FILE   Save detections to CSV (default: no save)")
    print("  --frame-skip INT    Process every Nth frame (default: 1)")
    print("  --ocr-cpu           Force OCR to use CPU (default: use GPU)")
    print("  --output-dir DIR    Save frames to directory (no display)")
    print("  --headless          Headless mode (no display, save inference stats)")
    print("\nControls (display mode only):")
    print("  q                   Quit the stream")
    print("  s                   Save current frame with detections")
    sys.exit(1)

stream_url = sys.argv[1]
conf = 0.5
imgsz = 320
output_csv = None
save_video = False
frame_skip = 1
ocr_gpu = True
output_dir = None
headless = False

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
    elif sys.argv[i] == '--output-dir' and i + 1 < len(sys.argv):
        output_dir = sys.argv[i + 1]
        i += 2
    elif sys.argv[i] == '--headless':
        headless = True
        i += 1
    else:
        i += 1

# Create output directory if specified
if output_dir:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    headless = True  # Force headless mode when saving frames

model_path = Path("runs/detect/train22/weights/best.pt")
if not model_path.exists():
    print(f"\nModel not found: {model_path}")
    sys.exit(1)

print(f"\nInput stream: {stream_url}")
print(f"Model: {model_path}")
print(f"Confidence: {conf}")
print(f"Image size: {imgsz}px")
print(f"Output CSV: {output_csv if output_csv else 'None (no save)'}")
print(f"Output frames: {output_dir if output_dir else 'None (display mode)'}")
print(f"Frame skip: {frame_skip}")
print(f"OCR device: {'GPU' if ocr_gpu else 'CPU'}")
print(f"Mode: {'Headless' if headless else 'Display'}")
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
    text = text.replace(' ', '').replace(',', '').replace('.', '')
    digits = "".join(re.findall(r"\d+", text))
    
    if len(digits) == 4:
        return digits
    elif len(digits) > 4:
        match = re.search(r'\d{4}', text)
        if match:
            return match.group()
    return None

# Dictionary to count detections
detection_counts = defaultdict(int)

# Open live stream
print("\nConnecting to stream...")
cap = cv2.VideoCapture(stream_url)

# Set buffer size to reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print(f"\nFailed to open stream: {stream_url}")
    sys.exit(1)

print("Stream connected!")

# Get stream properties if available
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Stream info: {frame_width}x{frame_height} @ {fps}fps")
print("\n" + "=" * 70)
print("Starting real-time processing with OCR...")
print("Press 'q' to quit, 's' to save frame")
print("=" * 70 + "\n")

frame_count = 0
processed_count = 0
detection_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream disconnected or ended")
            break
        
        frame_count += 1
        
        # Skip frames if needed
        if frame_count % frame_skip != 0:
            if not headless:
                try:
                    cv2.imshow('Live Stream OCR', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    pass
            continue
        
        processed_count += 1
        display_frame = frame.copy()
        
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
                            
                            # Draw on display frame
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                display_frame,
                                f"{number} ({ocr_score:.2f})",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2
                            )
                            
                            if detection_count % 10 == 0:
                                print(f"Frame {frame_count}: Detected {number} "
                                      f"(OCR: {ocr_score:.2f}, YOLO: {box_conf:.2f})")
                
                except Exception as e:
                    pass
        
        # Add frame info to display
        cv2.putText(display_frame, f"Frame: {frame_count} | Detections: {detection_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Unique #s: {len(detection_counts)}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display or save the frame
        if not headless:
            try:
                cv2.imshow('Live Stream OCR', display_frame)
            except cv2.error:
                print("Warning: Cannot display - no display support detected")
                print("Switch to headless mode: use --headless or --output-dir")
                headless = True
        
        # Save frame to directory if requested
        if output_dir:
            try:
                frame_file = Path(output_dir) / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_file), display_frame)
                if frame_count % 100 == 0:
                    print(f"Saved frame {frame_count} to {output_dir}")
            except Exception as e:
                print(f"Error saving frame: {e}")
        
        # Handle keyboard input (display mode only)
        if not headless:
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nClosing stream...")
                    break
                elif key == ord('s'):
                    filename = f"frame_{frame_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Frame saved: {filename}")
            except:
                pass
        
        # Clear cache periodically
        if device == "cuda" and processed_count % 50 == 0:
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\n\nProcessing interrupted by user")
except Exception as e:
    print(f"\nError during processing: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup
    cap.release()
    if not headless:
        try:
            cv2.destroyAllWindows()
        except:
            pass

print("\n" + "=" * 70)
print(f"Stream processing completed!")
print(f"Processed {processed_count} frames (total: {frame_count})")
print(f"Total detections: {detection_count}")
print(f"Unique numbers: {len(detection_counts)}")
print("=" * 70)

# Save results to CSV if requested
if output_csv and detection_counts:
    print(f"\nSaving results to {output_csv}...")
    output_path = Path(output_csv)
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Number', 'Count', 'Timestamp'])
            
            # Sort by number
            for number in sorted(detection_counts.keys()):
                writer.writerow([number, detection_counts[number], datetime.now().isoformat()])
        
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
