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
import subprocess
import os
from collections import defaultdict
from datetime import datetime
import time
import threading
import queue
import gc
import signal
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
print("YOLO Live Stream with OCR - 3-Thread Pipeline - Jetson Optimized")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("=" * 70)

# Parse arguments
if len(sys.argv) < 2:
    print("\nUsage: python3 app_live_ocr.py <input_source> [options]")
    print("\nExample input sources:")
    print("  imx477                           (Arducam IMX477 on CSI, sensor-id=0)")
    print("  rtsp://<ip>:<port>/stream")
    print("  /path/to/video.mp4")
    print("\nOptions:")
    print("  --conf FLOAT        Confidence threshold (default: 0.5)")
    print("  --imgsz INT         Inference size (default: 320)")
    print("  --sensor-id INT     IMX477 CSI sensor-id (default: 0)")
    print("  --capture-width INT IMX477 capture width (default: 1280)")
    print("  --capture-height INT IMX477 capture height (default: 720)")
    print("  --capture-fps INT   IMX477 capture fps (default: 30)")
    print("  --output-csv FILE   Save detections to CSV (default: no save)")
    print("  --frame-skip INT    Process every Nth frame (default: 1)")
    print("  --cpu               Force CPU mode (no GPU for YOLO or OCR)")
    print("  --ocr-cpu           Force OCR to use CPU (default: use GPU)")
    print("  --output-dir DIR    Save frames to directory (no display)")
    print("  --headless          Headless mode (no display, save inference stats)")
    print("  --lwm2m-enable      Enable Phase 1 LwM2M summary reporting")
    print("  --lwm2m-server URI  CoAP endpoint URI (example: coap://127.0.0.1:5683/lwm2m/summary)")
    print("  --lwm2m-endpoint ID LwM2M endpoint name (default: jetson-ocr)")
    print("  --lwm2m-device-id ID Device ID for payloads (default: endpoint name)")
    print("  --lwm2m-threshold N Minimum count before reporting number (default: 5)")
    print("  --lwm2m-interval S  Summary publish interval in seconds (default: 5)")
    print("  --lwm2m-store FILE  Store-and-forward file path (default: lwm2m_pending.ndjson)")
    print("\nControls (display mode only):")
    print("  q                   Quit the stream")
    print("  s                   Save current frame with detections")
    sys.exit(1)

input_source = sys.argv[1]
conf = 0.3
imgsz = 256
output_csv = None
save_video = False
frame_skip = 1
ocr_gpu = True
output_dir = None
headless = False
force_cpu = False
sensor_id = 0
capture_width = 1280
capture_height = 720
capture_fps = 30
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
    elif sys.argv[i] == '--sensor-id' and i + 1 < len(sys.argv):
        sensor_id = int(sys.argv[i + 1])
        i += 2
    elif sys.argv[i] == '--capture-width' and i + 1 < len(sys.argv):
        capture_width = int(sys.argv[i + 1])
        i += 2
    elif sys.argv[i] == '--capture-height' and i + 1 < len(sys.argv):
        capture_height = int(sys.argv[i + 1])
        i += 2
    elif sys.argv[i] == '--capture-fps' and i + 1 < len(sys.argv):
        capture_fps = int(sys.argv[i + 1])
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

print(f"Model: {model_path}")
print(f"Inference device: {device.upper()}")
print(f"Confidence: {conf}")
print(f"Image size: {imgsz}px")
print(f"Input source: {input_source}")
if input_source.lower() in ("imx477", "arducam", "arducam-imx477", "csi"):
    print(f"IMX477 sensor-id: {sensor_id}")
    print(f"IMX477 capture: {capture_width}x{capture_height} @ {capture_fps}fps")
print(f"Output CSV: {output_csv if output_csv else 'None (no save)'}")
print(f"Output frames: {output_dir if output_dir else 'None (display mode)'}")
print(f"Frame skip: {frame_skip}")
print(f"OCR device: {'GPU' if ocr_gpu else 'CPU'}")
print(f"Mode: {'Headless' if headless else 'Display'}")
print(f"LwM2M enabled: {'yes' if lwm2m_enable else 'no'}")
if lwm2m_enable:
    print(f"LwM2M server: {lwm2m_server if lwm2m_server else 'None (disabled until set)'}")
    print(f"LwM2M endpoint: {lwm2m_endpoint}")
    print(f"LwM2M device id: {lwm2m_device_id}")
    print(f"LwM2M threshold: {lwm2m_threshold}")
    print(f"LwM2M interval: {lwm2m_interval}s")
    print(f"LwM2M store file: {lwm2m_store_file}")
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


def build_imx477_pipeline(sensor_id, width, height, fps):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def check_gstreamer_available():
    """Check if GStreamer is available via gst-inspect."""
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", "nvarguscamerasrc"],
            capture_output=True, timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def list_v4l2_devices():
    """List available V4L2 camera devices."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def open_imx477_stream(sensor_id, width, height, fps):
    """Open Arducam IMX477 CSI stream via GStreamer."""
    pipeline = build_imx477_pipeline(sensor_id, width, height, fps)
    print(f"Trying IMX477 CSI camera (sensor-id={sensor_id})")

    # Check if GStreamer nvarguscamerasrc is available
    use_gstreamer = check_gstreamer_available()
    if not use_gstreamer:
        print("  ⚠ Warning: nvarguscamerasrc not found in GStreamer plugins")
        print("    Falling back to V4L2 device capture (/dev/videoN)")
    else:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                actual_height, actual_width = frame.shape[:2]
                print(f"  ✓ IMX477 opened via GStreamer, resolution: {actual_width}x{actual_height}")
                return cap, f"imx477(sensor-id={sensor_id},backend=gstreamer)"

        print("  ✗ GStreamer pipeline failed to open")
        print(f"    Pipeline string: {pipeline[:100]}...")
        cap.release()

    # Fallback path for containers where OpenCV lacks GStreamer support.
    # Sensor-id 0 -> /dev/video0, sensor-id 1 -> /dev/video1, etc.
    v4l2_device = f"/dev/video{sensor_id}"
    print(f"  Trying V4L2 fallback: {v4l2_device}")
    cap = cv2.VideoCapture(v4l2_device, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = cap.read()
        if ok and frame is not None:
            actual_height, actual_width = frame.shape[:2]
            print(f"  ✓ IMX477 opened via V4L2, resolution: {actual_width}x{actual_height}")
            return cap, f"imx477(sensor-id={sensor_id},backend=v4l2)"

    print("  ✗ V4L2 fallback failed")
    cap.release()

    # Try to get more diagnostic info
    v4l2_devices = list_v4l2_devices()
    if v4l2_devices:
        print(f"    Available V4L2 devices:\n{v4l2_devices}")

    print("  Troubleshooting:")
    print("    - Check camera: v4l2-ctl --list-devices")
    print("    - Verify device exists in container: ls -l /dev/video*")
    print("    - If using GStreamer path: gst-inspect-1.0 nvarguscamerasrc")
    print("    - Ensure argus_socket mounted in Docker: /tmp/argus_socket")
    return None, None


def open_generic_stream(source):
    """Open RTSP/HTTP/file stream using OpenCV VideoCapture."""
    print(f"Trying generic stream source: {source}")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap, source

    cap.release()
    return None, None


def open_stream(source, sensor_id, width, height, fps):
    """Open input source with IMX477 support and generic fallback."""
    if source.lower() in ("imx477", "arducam", "arducam-imx477", "csi"):
        return open_imx477_stream(sensor_id, width, height, fps)
    return open_generic_stream(source)


# Open live stream
print("\nConnecting to input source...")
cap, resolved_stream_url = open_stream(input_source, sensor_id, capture_width, capture_height, capture_fps)

if cap is None:
    print(f"\nFailed to open source: {input_source}")
    print("\nTroubleshooting:")
    print("  - For IMX477: verify cable seating/orientation and sensor-id (--sensor-id 0 or 1)")
    print("  - For IMX477: check camera with 'v4l2-ctl --list-devices'")
    print("  - For network streams: verify URL and connectivity")
    print("  - For files: confirm path exists and codec is supported by OpenCV")
    sys.exit(1)

if resolved_stream_url != input_source:
    print(f"Connected using: {resolved_stream_url}")
else:
    print("Input source connected!")

# Get stream properties if available
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

print(f"Stream info: {frame_width}x{frame_height} @ {fps}fps")

lwm2m_reporter = LwM2MSummaryReporter(
    enabled=lwm2m_enable,
    server_uri=lwm2m_server,
    endpoint_name=lwm2m_endpoint,
    device_id=lwm2m_device_id,
    source=resolved_stream_url,
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

# =============================================================================
# GLOBAL SHARED STATE
# =============================================================================
# Keep tiny queues to prioritize low latency over full-frame throughput.
capture_queue = queue.Queue(maxsize=2)
yolo_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()
detection_counts = defaultdict(int)
stats_lock = threading.Lock()
display_lock = threading.Lock()

# Shared statistics
frame_count = 0
processed_count = 0
detection_count = 0
capture_dropped = 0
yolo_dropped = 0
latest_display_frame = None


def queue_put_latest(q, item):
    """Insert item while discarding stale queued data to keep real-time behavior."""
    dropped = 0
    while True:
        try:
            q.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                q.get_nowait()
                dropped += 1
            except queue.Empty:
                return dropped


def queue_get_latest(q, timeout=1.0):
    """Get one item, then drain queue and keep only the newest available item."""
    item = q.get(timeout=timeout)
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return item


def signal_handler(sig, frame):
    print("\n[MAIN] Signal received, shutting down...")
    stop_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# THREAD 1: CAPTURE FRAMES
# =============================================================================
def capture_frames(cap_source, capture_queue, stop_event, frame_skip):
    """Capture frames from stream and push to queue with frame skipping."""
    global frame_count, capture_dropped
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
        
        dropped = queue_put_latest(capture_queue, (local_frame_id, frame))
        if dropped:
            with stats_lock:
                capture_dropped += dropped
    
    print("[CAPTURE] Thread stopped")


# =============================================================================
# THREAD 2: YOLO INFERENCE
# =============================================================================
def yolo_inference(model, device, capture_queue, yolo_queue, conf, imgsz, stop_event):
    """Process frames with YOLO detection."""
    global processed_count, yolo_dropped
    local_processed = 0
    current_device = device
    oom_fallback_done = False
    
    print("[YOLO] Thread started")
    
    while not stop_event.is_set():
        try:
            frame_id, frame = queue_get_latest(capture_queue, timeout=1)
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
            
            # Debug: Print detection count every 25 frames
            if local_processed % 25 == 0:
                num_detections = sum(len(r.boxes) for r in results)
                print(f"[YOLO] Processed {local_processed} frames | Latest batch: {num_detections} objects detected")
            
            dropped = queue_put_latest(yolo_queue, (frame_id, frame, results))
            if dropped:
                with stats_lock:
                    yolo_dropped += dropped
            
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
    global detection_count, latest_display_frame
    local_detection_count = 0
    
    print("[OCR] Thread started")
    
    while not stop_event.is_set():
        try:
            frame_id, frame, results = queue_get_latest(yolo_queue, timeout=1)
        except queue.Empty:
            continue
        
        need_display_frame = (not headless) or bool(output_dir)
        display_frame = frame.copy() if need_display_frame else frame
        
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
        
        # Save every 100th frame to directory if requested
        if output_dir and frame_id % 100 == 0:
            try:
                frame_file = Path(output_dir) / f"frame_{frame_id:06d}.jpg"
                cv2.imwrite(str(frame_file), display_frame)
                print(f"[OCR] Saved frame {frame_id} to {output_dir}")
            except Exception as e:
                print(f"[OCR] Error saving frame: {e}")
        
        if not headless:
            with display_lock:
                latest_display_frame = display_frame
    
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
    last_lwm2m_time = 0.0
    
    while not stop_event.is_set():
        # Display periodic stats
        current_time = time.time()
        if current_time - last_stats_time >= 5.0:
            with stats_lock:
                stats_frame_count = frame_count
                stats_processed_count = processed_count
                stats_detection_count = detection_count
                stats_unique_count = len(detection_counts)
                stats_capture_dropped = capture_dropped
                stats_yolo_dropped = yolo_dropped
                detection_snapshot = dict(detection_counts)

                print(
                    f"[STATS] Frames: {stats_frame_count} | Processed: {stats_processed_count} | "
                    f"Detections: {stats_detection_count} | Unique: {stats_unique_count} | "
                    f"Dropped(cap->yolo): {stats_capture_dropped} | Dropped(yolo->ocr): {stats_yolo_dropped}"
                )

            if lwm2m_reporter.is_active() and current_time - last_lwm2m_time >= lwm2m_reporter.interval_sec:
                summary_payload = lwm2m_reporter.build_summary_payload(
                    counts=detection_snapshot,
                    frame_count=stats_frame_count,
                    processed_count=stats_processed_count,
                    detection_count=stats_detection_count,
                )
                if summary_payload is not None:
                    queued = lwm2m_reporter.enqueue(summary_payload)
                    if queued:
                        lwm2m_stats = lwm2m_reporter.get_stats()
                        print(
                            f"[LWM2M] Summary queued | queue={lwm2m_stats['queue_depth']} "
                            f"pending_disk={lwm2m_stats['pending_disk']}"
                        )
                last_lwm2m_time = current_time

            last_stats_time = current_time
        
        # Handle keyboard input in display mode
        if not headless:
            try:
                with display_lock:
                    frame_to_show = None if latest_display_frame is None else latest_display_frame.copy()

                if frame_to_show is not None:
                    cv2.imshow('Live Stream OCR', frame_to_show)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[MAIN] Quit signal received")
                    stop_event.set()
                    break
                elif key == ord('s'):
                    filename = f"frame_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    if frame_to_show is not None:
                        cv2.imwrite(filename, frame_to_show)
                        print(f"[MAIN] Manual save triggered: {filename}")
                    else:
                        print("[MAIN] No frame available yet for manual save")

                time.sleep(0.005)
            except Exception:
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
finally:
    # =============================================================================
    # CLEANUP
    # =============================================================================
    print("\n[MAIN] Stopping threads...")
    stop_event.set()

    capture_thread.join(timeout=3)
    yolo_thread.join(timeout=3)
    ocr_thread.join(timeout=3)

    if lwm2m_reporter.is_active():
        print("[MAIN] Stopping LwM2M reporter...")
        lwm2m_reporter.stop()

    print("[MAIN] Releasing camera...")
    try:
        cap.release()
        cap = None
    except Exception:
        pass

    cv2.destroyAllWindows()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Let Argus recover
    time.sleep(2)

print("\n" + "=" * 70)
print(f"Stream processing completed!")
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
