#!/usr/bin/env python3
"""
YOLO Live Stream Inference with OCR - Real-time processing with 3-thread pipeline
Thread 1: Capture frames from stream
Thread 2: YOLO object detection
Thread 3: OCR processing
Optimized for Jetson Orin with memory constraints
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
from urllib.parse import urlparse
import requests
import time
import threading
import queue

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
print("YOLO Live Stream with OCR - 3-Thread Pipeline - Jetson Optimized")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("=" * 70)

# Parse arguments
if len(sys.argv) < 2:
    print("\nUsage: python3 app_live_ocr.py <stream_url> [options]")
    print("\nExample stream URLs:")
    print("  http://172.20.10.9/              (ESP32-CAM, auto-uses /capture)")
    print("  rtsp://10.149.73.176:8080/h264.sdp")
    print("  rtsp://10.149.73.176:8080/h264_aac.sdp")
    print("\nOptions:")
    print("  --conf FLOAT        Confidence threshold (default: 0.5)")
    print("  --imgsz INT         Inference size (default: 320)")
    print("  --output-csv FILE   Save detections to CSV (default: no save)")
    print("  --frame-skip INT    Process every Nth frame (default: 1)")
    print("  --cpu               Force CPU mode (no GPU for YOLO or OCR)")
    print("  --ocr-cpu           Force OCR to use CPU (default: use GPU)")
    print("  --output-dir DIR    Save frames to directory (no display)")
    print("  --headless          Headless mode (no display, save inference stats)")
    print("\nControls (display mode only):")
    print("  q                   Quit the stream")
    print("  s                   Save current frame with detections")
    sys.exit(1)

stream_url = sys.argv[1]
conf = 0.5
imgsz = 256
output_csv = None
save_video = False
frame_skip = 1
ocr_gpu = True
output_dir = None
headless = False
force_cpu = False

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
    elif sys.argv[i] == '--cpu':
        force_cpu = True
        ocr_gpu = False
        i += 1
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

# Determine device after parsing arguments
if force_cpu:
    device = "cpu"
    print("\nForced CPU mode (--cpu flag)")
elif not torch.cuda.is_available():
    device = "cpu"
    print("\nGPU not available - using CPU")
else:
    # Check available GPU memory
    try:
        torch.cuda.empty_cache()
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024**3)  # GB
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free_memory = total_memory - reserved
        
        print(f"\nGPU Memory: {free_memory:.2f}GB free / {total_memory:.2f}GB total")
        
        # Need at least 1GB free for model + inference
        if free_memory < 1.0:
            print(f"Warning: Low GPU memory ({free_memory:.2f}GB free)")
            print("Falling back to CPU mode to avoid OOM errors")
            print("Tip: Use smaller --imgsz or restart container to free GPU memory")
            device = "cpu"
        else:
            device = "cuda"
            print("Using GPU for inference")
    except Exception as e:
        print(f"\nError checking GPU memory: {e}")
        print("Falling back to CPU mode")
        device = "cpu"

model_path = Path("runs/detect/train22/weights/best.pt")
if not model_path.exists():
    print(f"\nModel not found: {model_path}")
    sys.exit(1)

print(f"\nInput stream: {stream_url}")
print(f"Model: {model_path}")
print(f"Inference device: {device.upper()}")
print(f"Confidence: {conf}")
print(f"Image size: {imgsz}px")
print(f"Output CSV: {output_csv if output_csv else 'None (no save)'}")
print(f"Output frames: {output_dir if output_dir else 'None (display mode)'}")
print(f"Frame skip: {frame_skip}")
print(f"OCR device: {'GPU' if ocr_gpu else 'CPU'}")
print(f"Mode: {'Headless' if headless else 'Display'}")
print("=" * 70)

# Load YOLO model
print(f"\nLoading YOLO model on {device.upper()}...")
try:
    model = YOLO(str(model_path))
    model.to(device)
    print(f"Model loaded successfully on {device.upper()}")
except RuntimeError as e:
    if "out of memory" in str(e).lower() and device == "cuda":
        print(f"\nGPU out of memory error detected!")
        print("Falling back to CPU mode...")
        torch.cuda.empty_cache()
        device = "cpu"
        model = YOLO(str(model_path))
        model.to(device)
        if device == "cuda":
            model.model.half()
        ocr_gpu = False  # Also use CPU for OCR
        print(f"Model loaded on CPU")
    else:
        raise

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


class HTTPCaptureSource:
    """Capture frames from ESP32-CAM /capture endpoint using HTTP polling."""
    
    def __init__(self, base_url, timeout=2):
        self.session = requests.Session()
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.capture_url = f"{self.base_url}/capture"
        self.frame_count = 0
        self._test_connection()
    
    def _test_connection(self):
        """Test if the capture endpoint is accessible."""
        try:
            r = self.session.get(self.capture_url, timeout=self.timeout)
            r.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to {self.capture_url}: {e}")
    
    def read(self):
        """Read a frame from the camera (mimics cv2.VideoCapture.read())."""
        try:
            r = self.session.get(self.capture_url, timeout=self.timeout)
            if r.status_code != 200:
                return False, None
            
            img_array = np.frombuffer(r.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return False, None
            
            self.frame_count += 1
            return True, frame
            
        except Exception as e:
            return False, None
    
    def release(self):
        """Close the session."""
        self.session.close()
    
    def get(self, prop):
        """Stub for compatibility with cv2.VideoCapture interface."""
        if prop == cv2.CAP_PROP_FPS:
            return 10  # Typical ESP32-CAM FPS
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640  # Default, will be updated from first frame
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        return 0


def open_http_capture(base_url):
    """Try opening HTTP capture endpoint for ESP32-CAM."""
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    
    # Try /capture first (ESP32-CAM standard)
    try:
        print(f"Trying ESP32-CAM capture endpoint: {base}/capture")
        source = HTTPCaptureSource(base)
        return source, f"{base}/capture"
    except Exception as e:
        print(f"   Failed: {e}")
        return None, None


def open_rtsp_stream(stream_url):
    """Try opening RTSP stream with VideoCapture."""
    print(f"Trying RTSP stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap, stream_url
    
    cap.release()
    return None, None


def open_stream(input_url):
    """Open stream - tries ESP32-CAM /capture for HTTP, VideoCapture for RTSP."""
    parsed = urlparse(input_url)
    
    # For HTTP/HTTPS URLs, try ESP32-CAM /capture endpoint
    if parsed.scheme in ("http", "https"):
        cap, url = open_http_capture(input_url)
        if cap:
            return cap, url
        print("   HTTP capture failed, trying VideoCapture as fallback...")
    
    # For RTSP or if HTTP capture failed, use VideoCapture
    return open_rtsp_stream(input_url)


# Open live stream
print("\nConnecting to stream...")
cap, resolved_stream_url = open_stream(stream_url)

if cap is None:
    print(f"\nFailed to open stream from: {stream_url}")
    print("\nTroubleshooting:")
    print("  - For ESP32-CAM: Ensure camera is accessible at http://<ip>/capture")
    print("  - For RTSP: Check the stream URL format (rtsp://...)")
    print("  - Test connectivity: ping <camera-ip>")
    sys.exit(1)

if resolved_stream_url != stream_url:
    print(f"Connected using: {resolved_stream_url}")
else:
    print("Stream connected!")

# Get stream properties if available
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

# For HTTP sources, get actual dimensions from first frame
if isinstance(cap, HTTPCaptureSource):
    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        frame_height, frame_width = test_frame.shape[:2]
        print(f"Stream info (ESP32-CAM): {frame_width}x{frame_height} @ ~{fps}fps (HTTP polling)")
    else:
        print(f"Warning: Could not read test frame from HTTP source")
else:
    print(f"Stream info: {frame_width}x{frame_height} @ {fps}fps")

# =============================================================================
# GLOBAL SHARED STATE
# =============================================================================
capture_queue = queue.Queue(maxsize=10)
yolo_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()
detection_counts = defaultdict(int)
stats_lock = threading.Lock()

# Shared statistics
frame_count = 0
processed_count = 0
detection_count = 0


# =============================================================================
# THREAD 1: CAPTURE FRAMES
# =============================================================================
def capture_frames(cap_source, capture_queue, stop_event, frame_skip):
    """Capture frames from stream and push to queue with frame skipping."""
    global frame_count
    local_frame_id = 0
    
    print("[CAPTURE] Thread started")
    
    while not stop_event.is_set():
        ret, frame = cap_source.read()
        if not ret:
            print("[CAPTURE] Stream disconnected")
            stop_event.set()
            break
        
        local_frame_id += 1
        
        # Update global frame count
        with stats_lock:
            frame_count = local_frame_id
        
        # Apply frame skipping
        if local_frame_id % frame_skip != 0:
            continue
        
        # Try to put frame in queue (non-blocking with timeout)
        try:
            capture_queue.put((local_frame_id, frame), timeout=0.1)
        except queue.Full:
            # Drop frame if queue is full (backpressure)
            pass
    
    print("[CAPTURE] Thread stopped")


# =============================================================================
# THREAD 2: YOLO INFERENCE
# =============================================================================
def yolo_inference(model, device, capture_queue, yolo_queue, conf, imgsz, stop_event):
    """Process frames with YOLO detection."""
    global processed_count
    local_processed = 0
    current_device = device
    oom_fallback_done = False
    
    print("[YOLO] Thread started")
    
    while not stop_event.is_set():
        try:
            frame_id, frame = capture_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        # Run YOLO detection (YOLO handles preprocessing internally)
        try:
            results = model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                device=current_device,
                verbose=False,
                stream=False
            )
            
            local_processed += 1
            with stats_lock:
                processed_count = local_processed
            
            # Push results to OCR queue
            try:
                yolo_queue.put((frame_id, frame, results), timeout=0.1)
            except queue.Full:
                # Drop if OCR queue is full
                pass
            
            # Clear GPU cache periodically
            if current_device == "cuda" and local_processed % 50 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            # Handle CUDA OOM by falling back to CPU
            if "out of memory" in str(e).lower() and current_device == "cuda" and not oom_fallback_done:
                print(f"[YOLO] GPU out of memory! Falling back to CPU mode...")
                torch.cuda.empty_cache()
                current_device = "cpu"
                model.to("cpu")
                oom_fallback_done = True
                print(f"[YOLO] Now running on CPU")
                continue
            elif "cuda" in str(e).lower() and current_device == "cuda" and not oom_fallback_done:
                print(f"[YOLO] CUDA error detected! Falling back to CPU mode...")
                torch.cuda.empty_cache()
                current_device = "cpu"
                model.to("cpu")
                oom_fallback_done = True
                print(f"[YOLO] Now running on CPU")
                continue
            else:
                print(f"[YOLO] Error: {e}")
                continue
        except Exception as e:
            print(f"[YOLO] Error: {e}")
            continue
    
    print("[YOLO] Thread stopped")


# =============================================================================
# THREAD 3: OCR PROCESSING
# =============================================================================
def ocr_processing(reader, yolo_queue, detection_counts, stop_event, 
                   frame_width, frame_height, conf, output_dir, headless):
    """Apply OCR to detected boxes and update results."""
    global detection_count
    local_detection_count = 0
    
    print("[OCR] Thread started")
    
    while not stop_event.is_set():
        try:
            frame_id, frame, results = yolo_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        display_frame = frame.copy()
        frame_detections = 0
        
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
                try:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    # Upscale for better OCR
                    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    
                    # Apply OCR
                    ocr_results = reader.readtext(gray, detail=1)
                    
                    for detection in ocr_results:
                        bbox, raw_text, ocr_score = detection
                        
                        # Extract 4-digit number
                        number = extract_4digit(raw_text)
                        
                        if number and ocr_score > 0.3:  # Minimum OCR confidence
                            with stats_lock:
                                detection_counts[number] += 1
                                local_detection_count += 1
                                detection_count = local_detection_count
                            
                            frame_detections += 1
                            
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
                            
                            if local_detection_count % 10 == 0:
                                print(f"[OCR] Frame {frame_id}: Detected {number} "
                                      f"(OCR: {ocr_score:.2f}, YOLO: {box_conf:.2f})")
                
                except Exception as e:
                    pass
        
        # Add frame info to display
        with stats_lock:
            current_frame_count = frame_count
            current_detection_count = detection_count
            unique_count = len(detection_counts)
        
        cv2.putText(display_frame, f"Frame: {current_frame_count} | Detections: {current_detection_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Unique #s: {unique_count}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save frame to directory if requested
        if output_dir:
            try:
                frame_file = Path(output_dir) / f"frame_{frame_id:06d}.jpg"
                cv2.imwrite(str(frame_file), display_frame)
                if frame_id % 100 == 0:
                    print(f"[OCR] Saved frame {frame_id} to {output_dir}")
            except Exception as e:
                print(f"[OCR] Error saving frame: {e}")
        
        # Display frame if not headless
        if not headless:
            try:
                cv2.imshow('Live Stream OCR', display_frame)
                cv2.waitKey(1)
            except cv2.error:
                pass
    
    print("[OCR] Thread stopped")


# =============================================================================
# START THREADS
# =============================================================================
print("\n" + "=" * 70)
print("Starting 3-thread pipeline...")
print("  Thread 1: Capture frames")
print("  Thread 2: YOLO inference")
print("  Thread 3: OCR processing")
print("=" * 70)
print("Press Ctrl+C to stop")
print("=" * 70 + "\n")

capture_thread = threading.Thread(
    target=capture_frames, 
    args=(cap, capture_queue, stop_event, frame_skip), 
    daemon=True
)
yolo_thread = threading.Thread(
    target=yolo_inference, 
    args=(model, device, capture_queue, yolo_queue, conf, imgsz, stop_event), 
    daemon=True
)
ocr_thread = threading.Thread(
    target=ocr_processing, 
    args=(reader, yolo_queue, detection_counts, stop_event, frame_width, frame_height, conf, output_dir, headless), 
    daemon=True
)

capture_thread.start()
yolo_thread.start()
ocr_thread.start()

# =============================================================================
# MAIN THREAD: MONITOR AND HANDLE KEYBOARD INPUT
# =============================================================================
try:
    last_stats_time = time.time()
    
    while not stop_event.is_set():
        # Display periodic stats
        current_time = time.time()
        if current_time - last_stats_time >= 5.0:
            with stats_lock:
                print(f"[STATS] Frames: {frame_count} | Processed: {processed_count} | Detections: {detection_count} | Unique: {len(detection_counts)}")
            last_stats_time = current_time
        
        # Handle keyboard input in display mode
        if not headless:
            try:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    print("\n[MAIN] Quit signal received")
                    stop_event.set()
                    break
                elif key == ord('s'):
                    # Save current frame (simplified - saves last processed frame)
                    filename = f"frame_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    print(f"[MAIN] Manual save triggered: {filename}")
            except:
                pass
        else:
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n[MAIN] Processing interrupted by user (Ctrl+C)")
    stop_event.set()
except Exception as e:
    print(f"\n[MAIN] Error: {e}")
    import traceback
    traceback.print_exc()
    stop_event.set()

# =============================================================================
# CLEANUP
# =============================================================================
print("\n[MAIN] Stopping threads...")
stop_event.set()

# Wait for threads to finish (with timeout)
capture_thread.join(timeout=2)
yolo_thread.join(timeout=2)
ocr_thread.join(timeout=2)

# Release resources
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
