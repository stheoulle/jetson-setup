# Video OCR Number Detection

This script processes videos to detect boxes with 4-digit numbers, applies OCR to recognize them, and saves detection counts to CSV. Uses EasyOCR for better Jetson compatibility.

## Quick Start

```bash
# Basic usage with EasyOCR (recommended for Jetson)
./inference_ocr_easy.sh vidéos/C0088.MP4

# Custom CSV output with video
./inference_ocr_easy.sh vidéos/C0088.MP4 --output-csv my_results.csv --save-video

# Fast mode - process every 5th frame
./inference_ocr_easy.sh vidéos/C0088.MP4 --frame-skip 5

# Force OCR to use CPU (if GPU has issues)
./inference_ocr_easy.sh vidéos/C0088.MP4 --ocr-cpu
```

## How It Works

1. **YOLO Detection**: Detects boxes in each video frame
2. **OCR Processing**: Applies EasyOCR to each detected box
3. **Number Extraction**: Extracts 4-digit numbers (0000-9999)
4. **Counting**: Tracks how many times each number appears
5. **CSV Export**: Saves results with counts

## Output

The script generates a CSV file with two columns:
- `Number`: The recognized 4-digit number
- `Count`: How many times it was detected

Example output:
```csv
Number,Count
0123,45
0456,12
1234,67
```

## Options

- `--conf FLOAT`: YOLO confidence threshold (default: 0.5)
- `--imgsz INT`: Inference image size (default: 320)
- `--output-csv FILE`: Output CSV filename (default: detections.csv)
- `--save-video`: Save annotated video with OCR results
- `--frame-skip INT`: Process every Nth frame for speed (default: 1)
- `--ocr-cpu`: Force OCR to use CPU instead of GPU

## Performance Tips

For **faster processing** (especially on long videos):
- Use `--frame-skip 5` to process every 5th frame
- Lower `--imgsz` to 224 or 256
- Don't use `--save-video` if not needed

For **better accuracy**:
- Use `--frame-skip 1` (process every frame)
- Increase `--imgsz` to 416 or 640
- Adjust `--conf` threshold to filter false positives

Make sure EasyOCR is installed:
```bash
docker compose exec yolo-inference pip3 install easyocr
```

Or rebuild with updated requirements:
```bash
docker compose down
docker compose up -d
```

## Troubleshooting

**Container not running:**
```bash
docker compose up -d
```

**EasyOCR errors or crashes:**
```bash
# Try CPU mode
./inference_ocr_easy.sh vidéos/C0088.MP4 --ocr-cpu

# Or reinstall EasyOCR
docker compose exec yolo-inference pip3 install --upgrade easyocr
```

**Memory issues:**
- Use smaller `--imgsz` value (try 224)
- Use `--frame-skip` to process fewer frames
- Close other applications
