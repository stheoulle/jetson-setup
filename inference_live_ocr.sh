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
    echo "Usage: $0 <input_source> [OPTIONS]"
    echo ""
    echo "Input sources:"
    echo "  imx477"
    echo "  rtsp://<ip>:<port>/stream"
    echo "  /path/to/video.mp4"
    echo ""
    echo "Options:"
    echo "  --conf FLOAT        Confidence threshold (default: 0.5)"
    echo "  --imgsz INT         Inference size (default: 320)"
    echo "  --sensor-id INT     IMX477 CSI sensor-id (default: 0)"
    echo "  --capture-width INT IMX477 capture width (default: 1280)"
    echo "  --capture-height INT IMX477 capture height (default: 720)"
    echo "  --capture-fps INT   IMX477 capture fps (default: 30)"
    echo "  --output-csv FILE   Save detections to CSV"
    echo "  --output-dir DIR    Save frames to directory (headless mode)"
    echo "  --headless          Headless mode (no display, only stats)"
    echo "  --frame-skip INT    Process every Nth frame (default: 1)"
    echo "  --cpu               Force CPU mode (recommended if GPU out of memory)"
    echo "  --ocr-cpu           Force OCR to use CPU"
    echo ""
    echo "Examples:"
    echo "  $0 imx477"
    echo "  $0 imx477 --sensor-id 0 --capture-width 1920 --capture-height 1080"
    echo "  $0 imx477 --headless --output-csv detections.csv"
    echo "  $0 rtsp://<ip>:<port>/stream --output-dir ./frames"
    echo ""
    echo "Controls during streaming:"
    echo "  q   - Quit the stream (display mode only)"
    echo "  s   - Save current frame with detections (display mode only)"
    echo ""
    echo "Note: Make sure to run 'docker compose up -d' first!"
    exit 1
fi

INPUT_SOURCE="$1"

PYTHON_CMD=(docker compose exec -T yolo-inference python3 app_live_ocr.py "$@")

cleanup() {
    local exit_code=$?

    if [ -n "${PYTHON_CMD_PID:-}" ] && kill -0 "$PYTHON_CMD_PID" >/dev/null 2>&1; then
        docker compose exec -T yolo-inference sh -lc 'pkill -INT -f "[p]ython3 app_live_ocr.py" >/dev/null 2>&1 || true' >/dev/null 2>&1 || true
        kill "$PYTHON_CMD_PID" >/dev/null 2>&1 || true
        wait "$PYTHON_CMD_PID" >/dev/null 2>&1 || true
    fi

    exit "$exit_code"
}

trap cleanup INT TERM

# Detect display/headless intent from CLI args (for X11 auth setup).
HEADLESS_MODE=0
for arg in "$@"; do
    if [ "$arg" = "--headless" ] || [ "$arg" = "--output-dir" ]; then
        HEADLESS_MODE=1
        break
    fi
done

# Check if container is running
if ! docker compose ps --status running --services | grep -q "^yolo-inference$"; then
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

# For IMX477 sources, ensure container has a camera-compatible backend.
if [[ "$INPUT_SOURCE" =~ ^(imx477|arducam|arducam-imx477|csi)$ ]]; then
    echo "Checking IMX477 camera backend in container..."

    # Argus/GStreamer path requires EGL display access in many Jetson container setups.
    # Re-grant local X access for container root to avoid:
    # "Authorization required" and "Failed to initialize EGLDisplay".
    if [ "$HEADLESS_MODE" -eq 0 ] && [ -n "$DISPLAY" ] && command -v xhost >/dev/null 2>&1; then
        if ! xhost +SI:localuser:root >/dev/null 2>&1; then
            echo "Warning: could not update X11 permissions with xhost"
            echo "If camera open fails with EGL auth errors, run: xhost +SI:localuser:root"
        fi
    fi

    if ! docker compose exec -T yolo-inference python3 - <<'PY' >/dev/null 2>&1
import cv2
info = cv2.getBuildInformation()
raise SystemExit(0 if "GStreamer:                   YES" in info else 1)
PY
    then
        echo "OpenCV in container has no GStreamer support. Attempting repair..."
        if ! docker compose exec -T yolo-inference bash -lc '
            set -e
            export DEBIAN_FRONTEND=noninteractive
            apt-get update
            apt-get install -y --no-install-recommends \
                python3-opencv \
                gstreamer1.0-tools \
                gstreamer1.0-plugins-base \
                gstreamer1.0-plugins-good \
                gstreamer1.0-plugins-bad
            pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >/dev/null 2>&1 || true
        '; then
            echo "Failed to install IMX477 camera dependencies in container"
            exit 1
        fi
    fi

    if ! docker compose exec -T yolo-inference bash -lc 'command -v gst-inspect-1.0 >/dev/null 2>&1 && gst-inspect-1.0 nvarguscamerasrc >/dev/null 2>&1'; then
        echo "IMX477 plugin still not available: nvarguscamerasrc"
        echo "Host check: gst-inspect-1.0 nvarguscamerasrc"
        echo "If host works but container does not, recreate container: docker compose down && docker compose up -d"
        exit 1
    fi

    echo "IMX477 backend looks ready"
fi

echo "Starting live stream processing with OCR..."
echo "   This will stream and process frames in real-time"
echo "   Detection results displayed on screen"
echo ""

echo "Checking Python dependencies in container..."
if ! docker compose exec -T yolo-inference python3 - <<'PY' >/dev/null 2>&1
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
    if ! docker compose exec -T yolo-inference pip3 install --no-cache-dir "numpy==1.26.4" easyocr; then
        echo "Failed to install dependencies in container"
        echo "Try manually: docker compose exec yolo-inference pip3 install --no-cache-dir \"numpy==1.26.4\" easyocr"
        exit 1
    fi

    echo "Verifying dependencies after install..."
    if ! docker compose exec -T yolo-inference python3 - <<'PY' >/dev/null 2>&1
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

# Run the Python script inside the container.
PYTHON_CMD_PID=""
"${PYTHON_CMD[@]}" &
PYTHON_CMD_PID=$!
wait "$PYTHON_CMD_PID"
EXIT_CODE=$?
trap - INT TERM

if [[ "$EXIT_CODE" =~ ^[0-9]+$ ]] && [ "$EXIT_CODE" -eq 0 ]; then
    echo ""
    echo "Stream closed. Processing complete!"
else
    if ! [[ "$EXIT_CODE" =~ ^[0-9]+$ ]]; then
        EXIT_CODE=1
    fi
    echo ""
    echo "Processing failed with exit code $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "   - GPU out of memory? Use: $0 $1 --cpu"
    echo "   - No display available? Use: $0 $1 --headless"
    echo "   - Want to save frames? Use: $0 $1 --output-dir ./frames"
    echo "   - IMX477 source example: $0 imx477 --sensor-id 0"
    echo "   - Check numpy/easyocr: docker compose exec yolo-inference python3 -c 'import numpy, easyocr; print(numpy.__version__)'"
    echo "   - Reinstall deps: docker compose exec yolo-inference pip3 install --no-cache-dir \"numpy==1.26.4\" easyocr"
    echo "   - View container logs: docker compose logs yolo-inference"
fi

exit $EXIT_CODE
