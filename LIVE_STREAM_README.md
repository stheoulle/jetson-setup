# Live Stream YOLO + OCR Guide

This guide covers the new real-time live stream processing script that combines YOLO detection with EasyOCR text recognition.

For ESP32-CAM setup (to replace phone live feed), see `ESP32_CAM_LIVE_FEED_README.md`.

## Files Created

- **app_live_ocr.py** - Python script for live stream YOLO + OCR processing
- **inference_live_ocr.sh** - Bash wrapper for easy Docker execution

## Quick Start

### 1. Basic Usage

Display live camera feed with YOLO detection and OCR:

```bash
./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp
```

### 2. Available Stream URLs

Choose one of these RTSP streams:

```
rtsp://10.149.73.176:8080/h264.sdp           # Basic H264 (recommended)
rtsp://10.149.73.176:8080/h264_aac.sdp       # H264 + AAC audio
rtsp://10.149.73.176:8080/h264_opus.sdp      # H264 + Opus audio
rtsp://10.149.73.176:8080/h264_ulaw.sdp      # H264 + µlaw audio
```

## Advanced Options

### Headless Mode (No Display)

If you get display errors or don't have a display available, use headless mode:

```bash
./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp --headless
```

This will process the stream and save statistics without trying to display video.

### Save Frames to Directory

Save every processed frame with detections to a directory:

```bash
./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp --output-dir ./frames
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
./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp --output-csv detections.csv
```

### Adjust Confidence Threshold

Change detection confidence (0.0-1.0, default: 0.5):

```bash
./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp --conf 0.7
```

### Frame Skipping

Process every Nth frame (reduce load on GPU):

```bash
./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp --frame-skip 2
```

Process every 2nd frame:
```bash
./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp --frame-skip 3
```

### Inference Size

Adjust model input size (default: 320, options: 320, 416, 640):

```bash
./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp --imgsz 416
```

### Force OCR to CPU

If GPU memory is constrained:

```bash
./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp --ocr-cpu
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
./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp \
  --frame-skip 2 \
  --output-csv results.csv \
  --output-dir ./frames \
  --conf 0.7
```

Headless mode with all options:

```bash
./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp \
  --headless \
  --output-csv detections.csv \
  --conf 0.7 \
  --ocr-cpu
```

## Direct Python Usage

You can also run the Python script directly without Docker:

```bash
python3 app_live_ocr.py rtsp://10.149.73.176:8080/h264.sdp
```

Or inside the Docker container:

```bash
docker compose exec yolo-inference python3 app_live_ocr.py rtsp://10.149.73.176:8080/h264.sdp
```

## Troubleshooting

### Stream won't connect

1. Check network connectivity:
   ```bash
   ping 10.149.73.176
   ```

2. Try a different stream URL (some cameras prefer certain codec variants)

3. Check if stream is active on camera

### Display not showing / GUI errors

**Error: "The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support"**

This means no display is available. Solutions:

1. Use headless mode (output to console only):
   ```bash
   ./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp --headless
   ```

2. Save frames to directory instead:
   ```bash
   ./inference_live_ocr.sh rtsp://10.149.73.176:8080/h264.sdp --output-dir ./frames
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
| Input | Video file | RTSP live stream |
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
