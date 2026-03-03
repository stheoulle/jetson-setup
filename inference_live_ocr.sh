#!/bin/bash
# GPU Live Stream Inference with OCR - Jetson Orin
# Displays live camera feed with YOLO detection + OCR in real-time
# Recognizes 4-digit numbers in detected regions
# Uses EasyOCR for better Jetson compatibility

echo "YOLO Live Stream Inference with OCR - Real-time"
echo "=============================================="
echo "Container: dustynv/l4t-pytorch:r36.4.0"
echo "GPU: Enabled"
echo "Model: train22"
echo "OCR: EasyOCR (Jetson optimized)"
echo "=============================================="
echo ""

if [ $# -lt 1 ]; then
    echo "Usage: $0 <stream_url> [OPTIONS]"
    echo ""
    echo "Stream URLs:"
    echo "  rtsp://10.149.73.176:8080/h264.sdp"
    echo "  rtsp://10.149.73.176:8080/h264_aac.sdp"
    echo "  rtsp://10.149.73.176:8080/h264_opus.sdp"
    echo "  rtsp://10.149.73.176:8080/h264_ulaw.sdp"
    echo ""
    echo "Options:"
    echo "  --conf FLOAT        Confidence threshold (default: 0.5)"
    echo "  --imgsz INT         Inference size (default: 320)"
    echo "  --output-csv FILE   Save detections to CSV"
    echo "  --output-dir DIR    Save frames to directory (headless mode)"
    echo "  --headless          Headless mode (no display, only stats)"
    echo "  --frame-skip INT    Process every Nth frame (default: 1)"
    echo "  --ocr-cpu           Force OCR to use CPU"
    echo ""
    echo "Examples:"
    echo "  $0 rtsp://10.149.73.176:8080/h264.sdp"
    echo "  $0 rtsp://10.149.73.176:8080/h264.sdp --headless"
    echo "  $0 rtsp://10.149.73.176:8080/h264.sdp --output-dir ./frames"
    echo "  $0 rtsp://10.149.73.176:8080/h264.sdp --output-csv detections.csv --frame-skip 2"
    echo ""
    echo "Controls during streaming:"
    echo "  q   - Quit the stream (display mode only)"
    echo "  s   - Save current frame with detections (display mode only)"
    echo ""
    echo "Note: Make sure to run 'docker compose up -d' first!"
    exit 1
fi

# Check if container is running
if ! docker compose ps | grep -q "yolo-inference.*running"; then
    echo "Container not running. Starting it now..."
    echo "    This will take a moment for first-time setup..."
    docker compose up -d
    echo ""
    echo "Waiting for dependencies to install..."
    
    # Wait for the container to finish installing dependencies
    for i in {1..60}; do
        if docker compose logs yolo-inference 2>/dev/null | grep -q "already running"; then
            echo "Container is ready!"
            break
        fi
        sleep 1
        if [ $i -eq 60 ]; then
            echo "Timeout waiting for setup. Proceeding anyway..."
        fi
    done
    echo ""
fi

echo "Starting live stream processing with OCR..."
echo "   This will stream and process frames in real-time"
echo "   Detection results displayed on screen"
echo ""

echo "Checking Python dependencies in container..."
if ! docker compose exec yolo-inference python3 - <<'PY' >/dev/null 2>&1
import sys
import numpy as np
import torch
import easyocr

major = int(np.__version__.split('.')[0])
if major >= 2:
    raise RuntimeError(f"Incompatible numpy version for current torch build: {np.__version__}")

torch.from_numpy(np.zeros((1,), dtype=np.float32))
print("ok")
PY
then
    echo "Missing or incompatible OCR dependencies detected"
    echo "Installing compatible versions (numpy==1.26.4, easyocr)..."
    if ! docker compose exec yolo-inference pip3 install --no-cache-dir "numpy==1.26.4" easyocr; then
        echo "Failed to install dependencies in container"
        echo "Try manually: docker compose exec yolo-inference pip3 install --no-cache-dir \"numpy==1.26.4\" easyocr"
        exit 1
    fi

    echo "Verifying dependencies after install..."
    if ! docker compose exec yolo-inference python3 - <<'PY' >/dev/null 2>&1
import numpy as np
import torch
import easyocr
major = int(np.__version__.split('.')[0])
assert major < 2, np.__version__
torch.from_numpy(np.zeros((1,), dtype=np.float32))
print("ok")
PY
    then
        echo "Dependencies are still incompatible after reinstall"
        echo "Try recreating container: docker compose down && docker compose up -d"
        exit 1
    fi

    echo "Dependencies installed"
fi
echo ""

# Run the Python script inside the container with display support
# Note: For X11 forwarding, make sure to set up display properly
docker compose exec yolo-inference python3 app_live_ocr.py "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Stream closed. Processing complete!"
else
    echo ""
    echo "Processing failed with exit code $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "   - No display available? Use: $0 $1 --headless"
    echo "   - Want to save frames? Use: $0 $1 --output-dir ./frames"
    echo "   - Check stream URL: $1"
    echo "   - Verify network: ping 10.149.73.176"
    echo "   - Check numpy/easyocr: docker compose exec yolo-inference python3 -c 'import numpy, easyocr; print(numpy.__version__)'"
    echo "   - Reinstall deps: docker compose exec yolo-inference pip3 install --no-cache-dir \"numpy==1.26.4\" easyocr"
    echo "   - View container logs: docker compose logs yolo-inference"
fi

exit $EXIT_CODE
