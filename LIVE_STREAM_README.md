# Live Stream YOLO + OCR Guide

This guide covers the new real-time live stream processing script that combines YOLO detection with EasyOCR text recognition.

## Files Created

- **app_live_ocr.py** - Python script for live stream YOLO + OCR processing
- **inference_live_ocr.sh** - Bash wrapper for easy Docker execution

## Quick Start

### 1. Basic Usage

Display live camera feed with YOLO detection and OCR:

```bash
./inference_live_ocr.sh imx477
```

### 2. Available Input Sources

Use one of these input modes:

```
imx477                                        # Arducam IMX477 on CSI (sensor-id=0)
imx477 --sensor-id 1                          # Use second IMX477 sensor if present
rtsp://<ip>:<port>/stream                     # Optional network stream source
/path/to/video.mp4                            # Optional file source for testing
```

## Advanced Options

### Headless Mode (No Display)

If you get display errors or don't have a display available, use headless mode:

```bash
./inference_live_ocr.sh imx477 --headless
```

This will process the stream and save statistics without trying to display video.

### Save Frames to Directory

Save every processed frame with detections to a directory:

```bash
./inference_live_ocr.sh imx477 --output-dir ./frames
```

Frames are saved as `frame_000001.jpg`, `frame_000002.jpg`, etc. (automatically enables headless mode)

Useful for:
- Post-processing with other tools
- Creating videos or GIFs
- Archiving detection results
- Analyzing detections offline

### Save Detections to CSV

Save all detected 4-digit numbers to a CSV file:

```bash
./inference_live_ocr.sh imx477 --output-csv detections.csv
```

### Adjust Confidence Threshold

Change detection confidence (0.0-1.0, default: 0.5):

```bash
./inference_live_ocr.sh imx477 --conf 0.7
```

### Frame Skipping

Process every Nth frame (reduce load on GPU):

```bash
./inference_live_ocr.sh imx477 --frame-skip 2
```

Process every 2nd frame:
```bash
./inference_live_ocr.sh imx477 --frame-skip 3
```

### Inference Size

Adjust model input size (default: 320, options: 320, 416, 640):

```bash
./inference_live_ocr.sh imx477 --imgsz 416
```

### IMX477 Camera Parameters

Tune CSI capture for Arducam IMX477:

```bash
./inference_live_ocr.sh imx477 \
   --sensor-id 0 \
   --capture-width 1920 \
   --capture-height 1080 \
   --capture-fps 30
```

### Force OCR to CPU

If GPU memory is constrained:

```bash
./inference_live_ocr.sh imx477 --ocr-cpu
```

## Keyboard Controls

While the stream is running (display mode only):

| Key | Action |
|-----|--------|
| `q` | Quit the stream |
| `s` | Save current frame with detections |
| `Ctrl+C` | Force stop |

Controls are not available in headless mode (`--headless` or `--output-dir`).

## Combined Example

Process every 2nd frame, save to CSV with 70% confidence, save frames to directory:

```bash
./inference_live_ocr.sh imx477 \
  --frame-skip 2 \
  --output-csv results.csv \
  --output-dir ./frames \
  --conf 0.7
```

Headless mode with all options:

```bash
./inference_live_ocr.sh imx477 \
  --headless \
  --output-csv detections.csv \
  --conf 0.7 \
  --ocr-cpu
```

## Direct Python Usage

You can also run the Python script directly without Docker:

```bash
python3 app_live_ocr.py imx477
```

Or inside the Docker container:

```bash
docker compose exec yolo-inference python3 app_live_ocr.py imx477 --sensor-id 0
```

## Troubleshooting

### IMX477 camera won't connect

**Error: "Failed to open source: imx477"**

Follow these diagnostic steps:

#### Step 1: Verify Camera Hardware
```bash
# Check if camera is detected at all
v4l2-ctl --list-devices

# For more detailed device info
ls -la /dev/video*
```

If no video devices appear → hardware issue (ribbon disconnected/loose)

#### Step 2: Verify GStreamer and nvarguscamerasrc
```bash
# Check if GStreamer can see the plugin
gst-inspect-1.0 nvarguscamerasrc

# Try simple test pipeline (requires display or use remote display)
gst-launch-1.0 -e nvarguscamerasrc sensor-id=0 ! \
  "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1" ! \
  nvvidconv ! xvimagesink sync=false
```

If GStreamer pipeline works → issue is with OpenCV/Docker integration

#### Step 3: Verify Docker Container Configuration  

Restart the container with updated device mounts and capabilities:
```bash
# Stop old container
docker compose down

# Start fresh container (docker-compose.yml must have these):
#   volumes: /dev/video0, /dev/video1, /dev/nvhost-*
#   cap_add: SYS_ADMIN, SYS_RAWIO
#   devices: /dev/video0, /dev/video1, /dev/nvhost-*

docker compose up -d

# Verify devices are accessible in container
docker compose exec yolo-inference ls -la /dev/video*
docker compose exec yolo-inference gst-inspect-1.0 nvarguscamerasrc
```

#### Step 4: Run with Debug Output
```bash
# Test the pipeline directly inside the container
docker compose exec yolo-inference python3 -c "
import cv2
pipeline = 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print(f'isOpened: {cap.isOpened()}')
if cap.isOpened():
    ret, frame = cap.read()
    print(f'read success: {ret}, shape: {frame.shape if ret else None}')
cap.release()
"
```

#### Hardware Issues Checklist

**Symptom: dmesg shows "imx477 probe error -121"**
- Meaning: I2C no-ack on that camera path (sensor not responding)
- Most common cause: loose or incorrectly oriented ribbon cable
- Fix: 
  1. Power off completely
  2. Re-seat ribbon cable carefully on both ends
  3. Boot and check: `sudo i2cdetect -y -r 9` and `sudo i2cdetect -y -r 10`

**Symptom: Camera detected on one port but not other**
- Camera is functional, but port/ribbon issue
- Try swapping to different CSI port and reboot

### Display not showing / GUI errors

**Error: "The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support"**

This means no display is available. Solutions:

1. Use headless mode (output to console only):
   ```bash
   ./inference_live_ocr.sh imx477 --headless
   ```

2. Save frames to directory instead:
   ```bash
   ./inference_live_ocr.sh imx477 --output-dir ./frames
   ```

3. If using remote connection, enable X11 forwarding:
   ```bash
   export DISPLAY=:0
   ssh -X user@host  # When SSHing to the Jetson
   ```

### Poor OCR results

- Increase inference size: `--imgsz 640`
- Reduce frame skip for better frame quality
- Adjust confidence threshold: `--conf 0.7` or higher

### GPU memory issues

- Use `--frame-skip` to reduce processing load
- Use `--imgsz 320` (smaller than default)
- Use `--ocr-cpu` to offload OCR to CPU

## Comparison with Video Inference

| Feature | Video Script | Live Stream Script |
|---------|-------------|--------------------|
| Input | Video file | IMX477 CSI (default) or stream/file source |
| Output Display | Optional save to file | Real-time display OR headless mode |
| Detections | Saved to CSV after completion | Can save to CSV during stream |
| Frame Rate | Fixed based on video | Dynamic based on stream |
| Latency | Not critical | Minimized |
| Frame Saving | Not available | Optional (`--output-dir`) |
| Docker Display | Not needed | Optional X11 forwarding |

## Display Modes

### Display Mode (Default)
- Shows live video with bounding boxes and OCR results
- Requires X11 forwarding or direct display
- Keyboard controls: `q` to quit, `s` to save frame
- Press Ctrl+C to stop

### Headless Mode (`--headless`)
- No display output
- Only prints statistics to console
- Useful in Docker containers without display
- Perfect for CI/CD pipelines or background processing

### Frame Saving Mode (`--output-dir`)
- Saves every processed frame as JPG
- Automatically enables headless mode
- Frames named: `frame_000001.jpg`, `frame_000002.jpg`, etc.
- Useful for post-processing or creating videos

## Model Info

- **Model**: YOLO trained on custom dataset (train22)
- **Model Path**: `runs/detect/train22/weights/best.pt`
- **Detection**: Bounding boxes with confidence scores
- **OCR**: EasyOCR, extracts 4-digit numbers
- **GPU**: NVIDIA Jetson Orin (CUDA enabled by default)

## CSV Output Format

When using `--output-csv`, results are saved as:

```csv
Number,Count,Timestamp
0001,5,2026-03-03T10:15:30.123456
0042,3,2026-03-03T10:15:30.123456
0100,2,2026-03-03T10:15:30.123456
```

Top 10 most detected numbers are also printed to console when stream ends.
