# Run your YOLO training with GPU:

Run train_gpu_docker.sh
Run the video inference on a big GPU

```bash
# Standard inference (uses train22 model, 0.5 confidence, GPU)
./inference_gpu_docker.sh video.mp4

# Custom confidence threshold
./inference_gpu_docker.sh video.mp4 --conf 0.7

# Specify output path
./inference_gpu_docker.sh video.mp4 --output my_output.mp4

# Use CPU instead
./inference_gpu_docker.sh video.mp4 --device cpu

# Combine options
./inference_gpu_docker.sh input_video.mp4 --output detected_output.mp4 --conf 0.6
```

Run the vidéo inference on a smaller GPU that fallsbak to cpu if GPU is still not available 

```bash
# Run with optimized settings (320px - same as training)
./inference_simple.sh vidéos/C0088.MP4

# Higher confidence threshold
./inference_simple.sh vidéos/C0088.MP4 --conf 0.7

# Larger inference size (if it works)
./inference_simple.sh vidéos/C0088.MP4 --imgsz 416

# Annotated output video with YOLO boxes and OCR labels
./inference_ocr_easy.sh vidéos/C0088.MP4 --save-video --output-video annotated.mp4
```

To send count to the receiving server

```bash
./inference_live_ocr.sh imx477 \
  --lwm2m-enable \
  --lwm2m-server coap://192.168.1.7:5683/lwm2m/summary \
  --lwm2m-endpoint jetson-ocr-01 \
  --lwm2m-device-id jetson-01 \
  --lwm2m-threshold 5 \
  --lwm2m-interval 5
```