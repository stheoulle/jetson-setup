#!/usr/bin/env python3
"""
YOLO Video Inference with OCR - Frame-by-frame processing
Detects boxes, applies OCR to recognize 4-digit numbers, saves counts to CSV,
and can export an annotated video showing both YOLO boxes and OCR results.
Optimized for Jetson Orin with memory constraints.
Uses EasyOCR for better Jetson compatibility.
"""

import torch
import cv2
import re
import csv
from pathlib import Path
from ultralytics import YOLO
import easyocr
import sys
import os
import time
from collections import defaultdict
from lwm2m_phase1 import LwM2MSummaryReporter

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
    print("  --output-video FILE Output annotated video file")
    print("  --frame-skip INT    Process every Nth frame (default: 1)")
    print("  --ocr-cpu           Force OCR to use CPU (default: use GPU)")
    print("  --lwm2m-enable      Enable Phase 1 LwM2M summary reporting")
    print("  --lwm2m-server URI  CoAP endpoint URI (example: coap://127.0.0.1:5683/lwm2m/summary)")
    print("  --lwm2m-endpoint ID LwM2M endpoint name (default: jetson-ocr)")
    print("  --lwm2m-device-id ID Device ID for payloads (default: endpoint name)")
    print("  --lwm2m-threshold N Minimum count before reporting number (default: 5)")
    print("  --lwm2m-interval S  Summary publish interval in seconds (default: 5)")
    print("  --lwm2m-store FILE  Store-and-forward file path (default: lwm2m_pending.ndjson)")
    sys.exit(1)

video_path = sys.argv[1]
conf = 0.5
imgsz = 320
output_csv = "detections.csv"
save_video = False
output_video = None
frame_skip = 1
ocr_gpu = True
lwm2m_enable = os.getenv("LWM2M_ENABLE", "0") == "1"
lwm2m_server = os.getenv("LWM2M_SERVER_URI", "")
lwm2m_endpoint = os.getenv("LWM2M_ENDPOINT_NAME", "jetson-ocr")
lwm2m_device_id = os.getenv("LWM2M_DEVICE_ID", lwm2m_endpoint)
lwm2m_threshold = int(os.getenv("LWM2M_THRESHOLD", "5"))
lwm2m_interval = int(os.getenv("LWM2M_INTERVAL_SEC", "5"))
lwm2m_store_file = os.getenv("LWM2M_STORE_FILE", "lwm2m_pending.ndjson")

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
    elif sys.argv[i] == '--output-video' and i + 1 < len(sys.argv):
        output_video = sys.argv[i + 1]
        save_video = True
        i += 2
    elif sys.argv[i] == '--frame-skip' and i + 1 < len(sys.argv):
        frame_skip = int(sys.argv[i + 1])
        i += 2
    elif sys.argv[i] == '--ocr-cpu':
        ocr_gpu = False
        i += 1
    elif sys.argv[i] == '--lwm2m-enable':
        lwm2m_enable = True
        i += 1
    elif sys.argv[i] == '--lwm2m-server' and i + 1 < len(sys.argv):
        lwm2m_server = sys.argv[i + 1]
        i += 2
    elif sys.argv[i] == '--lwm2m-endpoint' and i + 1 < len(sys.argv):
        lwm2m_endpoint = sys.argv[i + 1]
        if not os.getenv("LWM2M_DEVICE_ID"):
            lwm2m_device_id = lwm2m_endpoint
        i += 2
    elif sys.argv[i] == '--lwm2m-device-id' and i + 1 < len(sys.argv):
        lwm2m_device_id = sys.argv[i + 1]
        i += 2
    elif sys.argv[i] == '--lwm2m-threshold' and i + 1 < len(sys.argv):
        lwm2m_threshold = max(1, int(sys.argv[i + 1]))
        i += 2
    elif sys.argv[i] == '--lwm2m-interval' and i + 1 < len(sys.argv):
        lwm2m_interval = max(1, int(sys.argv[i + 1]))
        i += 2
    elif sys.argv[i] == '--lwm2m-store' and i + 1 < len(sys.argv):
        lwm2m_store_file = sys.argv[i + 1]
        i += 2
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
print(f"Output video: {output_video if output_video else 'auto'}")
print(f"Frame skip: {frame_skip}")
print(f"OCR device: {'GPU' if ocr_gpu else 'CPU'}")
print(f"LwM2M enabled: {'yes' if lwm2m_enable else 'no'}")
if lwm2m_enable:
    print(f"LwM2M server: {lwm2m_server if lwm2m_server else 'None (disabled until set)'}")
    print(f"LwM2M endpoint: {lwm2m_endpoint}")
    print(f"LwM2M device id: {lwm2m_device_id}")
    print(f"LwM2M threshold: {lwm2m_threshold}")
    print(f"LwM2M interval: {lwm2m_interval}s")
    print(f"LwM2M store file: {lwm2m_store_file}")
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


def draw_label(frame, text, x, y, color=(0, 255, 0)):
    """Draw a readable label with a filled background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

    pad_x = 6
    pad_y = 6
    x1 = max(0, x)
    y1 = max(0, y - text_height - pad_y * 2)
    x2 = x1 + text_width + pad_x * 2
    y2 = y1 + text_height + baseline + pad_y * 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x1 + pad_x, y2 - pad_y - baseline),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

# Dictionary to count detections
detection_counts = defaultdict(int)

# Open video
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"\nFailed to open video: {video_path}")
    sys.exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo info: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

lwm2m_reporter = LwM2MSummaryReporter(
    enabled=lwm2m_enable,
    server_uri=lwm2m_server,
    endpoint_name=lwm2m_endpoint,
    device_id=lwm2m_device_id,
    source=str(video_path),
    threshold=lwm2m_threshold,
    interval_sec=lwm2m_interval,
    store_file=lwm2m_store_file,
)

if lwm2m_enable and not lwm2m_reporter.is_active():
    print("[LWM2M] Disabled: set --lwm2m-server and ensure aiocoap is installed")
elif lwm2m_reporter.is_active():
    print(f"[LWM2M] Phase 1 summary reporter enabled: {lwm2m_server}")
    print("[LWM2M] Reachability check uses ICMP ping when available, otherwise a UDP route probe")
    print(f"[LWM2M] Ping check: {'reachable' if lwm2m_reporter.ping_server() else 'unreachable'}")
    lwm2m_reporter.start()
    startup_test_payload = lwm2m_reporter.build_test_payload()
    if lwm2m_reporter.enqueue(startup_test_payload):
        print("[LWM2M] Startup test payload queued")
    else:
        print("[LWM2M] Startup test payload not queued")

# Setup video writer if needed
video_writer = None
if save_video:
    if output_video:
        output_video_path = Path(output_video)
    else:
        output_video_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    if not video_writer.isOpened():
        print(
            f"\nFailed to create output video: {output_video_path}\n"
            "Please check that the output path is writable and that the selected codec is available."
        )
        sys.exit(1)
    print(f"Output video: {output_video_path}")

print("\nStarting video processing with OCR...\n")

frame_count = 0
processed_count = 0
detection_count = 0
processing_failed = False
last_lwm2m_time = time.time()

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
        output_frame = frame.copy() if save_video else None
        
        # Run YOLO detection
        results = model(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
        
        # Process each detection
        for r in results:
            class_names = getattr(r, 'names', None) or getattr(model, 'names', {})
            for box in r.boxes:
                box_conf = float(box.conf[0])
                if box_conf < conf:
                    continue
                
                # Extract box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0]) if box.cls is not None and len(box.cls) else 0
                class_name = class_names.get(class_id, str(class_id))
                
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
                    best_number = None
                    best_ocr_score = 0.0
                    
                    for detection in ocr_results:
                        bbox, raw_text, ocr_score = detection
                        
                        # Extract 4-digit number
                        number = extract_4digit(raw_text)

                        if number and ocr_score > 0.3 and ocr_score >= best_ocr_score:
                            best_number = number
                            best_ocr_score = float(ocr_score)

                    if best_number:
                        detection_counts[best_number] += 1
                        detection_count += 1

                        if detection_count % 10 == 0:
                            print(
                                f"Frame {frame_count}/{total_frames}: Detected {best_number} "
                                f"(OCR: {best_ocr_score:.2f}, YOLO: {box_conf:.2f})"
                            )

                    if save_video and output_frame is not None:
                        box_color = (0, 255, 0) if best_number else (0, 165, 255)
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_color, 2)

                        if best_number:
                            label = f"{class_name} | YOLO {box_conf:.2f} | OCR {best_number} ({best_ocr_score:.2f})"
                        else:
                            label = f"{class_name} | YOLO {box_conf:.2f} | OCR none"

                        draw_label(output_frame, label, x1, y1, color=box_color)
                
                except Exception as e:
                    # OCR can fail on some crops, continue
                    pass
        
        # Write frame if saving video
        if save_video and video_writer:
            video_writer.write(output_frame if output_frame is not None else frame)
        
        # Clear cache periodically
        if device == "cuda" and processed_count % 50 == 0:
            torch.cuda.empty_cache()
        
        # Progress update
        if frame_count % 100 == 0:
            print(f"Progress: {frame_count}/{total_frames} frames "
                  f"({100*frame_count/total_frames:.1f}%) - "
                  f"{len(detection_counts)} unique numbers detected")

        if lwm2m_reporter.is_active():
            now = time.time()
            if now - last_lwm2m_time >= lwm2m_reporter.interval_sec:
                summary_payload = lwm2m_reporter.build_summary_payload(
                    counts=dict(detection_counts),
                    frame_count=frame_count,
                    processed_count=processed_count,
                    detection_count=detection_count,
                )
                if summary_payload is not None and lwm2m_reporter.enqueue(summary_payload):
                    lwm2m_stats = lwm2m_reporter.get_stats()
                    print(
                        f"[LWM2M] Summary queued | queue={lwm2m_stats['queue_depth']} "
                        f"pending_disk={lwm2m_stats['pending_disk']}"
                    )
                last_lwm2m_time = now

except KeyboardInterrupt:
    print("\n\nProcessing interrupted by user")
    processing_failed = True
except Exception as e:
    print(f"\nError during processing: {e}")
    import traceback
    traceback.print_exc()
    processing_failed = True
finally:
    if lwm2m_reporter.is_active():
        final_payload = lwm2m_reporter.build_summary_payload(
            counts=dict(detection_counts),
            frame_count=frame_count,
            processed_count=processed_count,
            detection_count=detection_count,
        )
        if final_payload is not None:
            lwm2m_reporter.enqueue(final_payload)

        print("[MAIN] Stopping LwM2M reporter...")
        lwm2m_reporter.stop()

    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()

print("\n" + "=" * 70)
print(f"Video processing completed!")
print(f"Processed {processed_count} frames (total: {frame_count})")
print(f"Total detections: {detection_count}")
print(f"Unique numbers: {len(detection_counts)}")
if lwm2m_reporter.is_active():
    lwm2m_stats = lwm2m_reporter.get_stats()
    print(
        f"LwM2M sent: {lwm2m_stats['sent']} | failed: {lwm2m_stats['failed']} | "
        f"queued: {lwm2m_stats['queued']} | pending_disk: {lwm2m_stats['pending_disk']}"
    )
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
