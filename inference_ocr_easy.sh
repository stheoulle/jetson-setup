#!/bin/bash
# GPU Video Inference with OCR (EasyOCR) - Jetson Orin
# Detects boxes, applies OCR to recognize 4-digit numbers, saves to CSV
# Uses EasyOCR for better Jetson compatibility

echo "YOLO Video Inference with OCR - EasyOCR (Docker)"
echo "=============================================="
echo "Container: dustynv/l4t-pytorch:r36.4.0"
echo "GPU: Enabled "
echo "Model: train22"
echo "OCR: EasyOCR (Jetson optimized)"
echo "=============================================="
echo ""

if [ $# -lt 1 ]; then
    echo "Usage: $0 <video_file> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --conf FLOAT        Confidence threshold (default: 0.5)"
    echo "  --imgsz INT         Inference size (default: 320)"
    echo "  --output-csv FILE   Output CSV file (default: detections.csv)"
    echo "  --save-video        Save annotated video with OCR results"
    echo "  --frame-skip INT    Process every Nth frame (default: 1)"
    echo "  --ocr-cpu           Force OCR to use CPU"
    echo ""
    echo "Examples:"
    echo "  $0 video.mp4"
    echo "  $0 video.mp4 --output-csv results.csv --save-video"
    echo "  $0 video.mp4 --conf 0.7 --frame-skip 5 --ocr-cpu"
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
    echo "⏳ Waiting for dependencies to install..."
    
    # Wait for the container to finish installing dependencies
    for i in {1..60}; do
        if docker compose logs yolo-inference 2>/dev/null | grep -q "Container ready"; then
            echo "Dependencies installed successfully!"
            break
        fi
        sleep 1
        if [ $i -eq 60 ]; then
            echo "Timeout waiting for setup. Proceeding anyway..."
        fi
    done
    echo ""
fi

echo "Starting OCR video processing with EasyOCR..."
echo "   This will process each frame with YOLO + OCR"
echo "   Results will be saved to CSV with detection counts"
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

# Run the Python script inside the container
docker compose exec yolo-inference python3 app_video_ocr_easy.py "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Processing complete! Check your CSV file for results."
else
    echo ""
    echo "Processing failed with exit code $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "   - Make sure the video file path is correct"
    echo "   - Check numpy/easyocr: docker compose exec yolo-inference python3 -c 'import numpy, easyocr; print(numpy.__version__)'"
    echo "   - Reinstall deps: docker compose exec yolo-inference pip3 install --no-cache-dir \"numpy==1.26.4\" easyocr"
    echo "   - Try CPU mode: $0 <video> --ocr-cpu"
    echo "   - View container logs: docker compose logs yolo-inference"
fi

exit $EXIT_CODE
