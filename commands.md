Run your YOLO training with GPU:

```bash
cd /home/laposte/jetson-setup
mamba run -n pytorch-gpu python app.py
```

Or activate the environment and run manually:

```bash
eval "$(conda shell.bash hook)"
mamba activate pytorch-gpu
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH
python app.py
```

# new version

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
```


ESP32-CAM (http://172.20.10.9/capture)
   ↓ HTTP GET with session reuse
HTTPCaptureSource.read()
   ↓ JPEG decode
NumPy array → YOLO inference → OCR


pio device monitor on laposte laptop
all 3 connected to iPhone