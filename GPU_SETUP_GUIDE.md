# GPU Setup Guide for Jetson Orin - Standalone Setup

## Overview
This guide shows how to set up PyTorch with GPU acceleration on a Jetson Orin device **without Docker**.

## System Information
- **Device**: Jetson Orin (Compute Capability 8.7)
- **JetPack**: 6.0 (L4T R36.4.7)
- **CUDA**: 12.6
- **Driver**: NVIDIA 540.4.0

## What You Need
✓ PyTorch 2.12.0.dev (CUDA 12.6 compatible)
✓ CUDA 12.6 toolkit  
✓ CUDA runtime libraries  
✓ Python 3.10 environment via mamba

## Setup Steps Completed

### 1. **Created Python 3.10 Environment**
```bash
mamba create -y -n pytorch-gpu python=3.10
```
*Why Python 3.10?* PyTorch GPU wheels for aarch64 (ARM64) don't support Python 3.13 yet.

### 2. **Installed PyTorch with CUDA 12.6 Support**
```bash
mamba run -n pytorch-gpu pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
```
This installs PyTorch nightly builds with full CUDA 12.6 support and all required libraries.

### 3. **Installed CUDA Toolkit**
```bash
sudo apt install -y cuda-toolkit
```
This installed CUDA 12.6 to `/usr/local/cuda-12.6/` with all necessary libraries and development tools.

### 4. **Configured Library Paths**
CUDA libraries are located at:
- `/usr/local/cuda-12.6/targets/aarch64-linux/lib/`
- `/usr/lib/aarch64-linux-gnu/nvidia/`

## Usage

### Quick Start
```bash
cd /home/laposte/jetson-setup
mamba run -n pytorch-gpu python app.py
```

### Activate Environment Manually
```bash
eval "$(conda shell.bash hook)"
mamba activate pytorch-gpu
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH
```

### Verify GPU Works
```bash
mamba run -n pytorch-gpu python << 'EOF'
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA version: {torch.version.cuda}')
EOF
```

## Important Notes

### ⚠️ Compute Capability Warning
You may see this warning:
```
Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
```

This is informational - your Orin with CC 8.7 may not have optimized kernels, but PyTorch will still work using fallback kernels.

### 🔧 Jetson Memory Constraints
The `app.py` has been optimized for Jetson with:
- Reduced batch size: 8 (instead of 16)
- Reduced image size: 416 (instead of 640)
- Fewer workers: 2 (instead of 8)

Adjust these parameters if you have memory issues:
```python
results = model.train(
    data="dataset/data.yaml",
    epochs=20,
    batch=8,      # ← Reduce if OOM errors
    imgsz=416,    # ← Reduce for less VRAM
    workers=2,    # ← Reduce for less CPU load
)
```

### CPU Fallback
If GPU memory allocation fails, `app.py` automatically falls back to CPU training with:
- Batch size: 4
- Image size: 416

## Troubleshooting

### GPU Not Detected
**Symptom**: `CUDA available: False`

**Check**:
```bash
nvidia-smi  # Should show Orin GPU
ls /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcuda* # Check CUDA libs exist
```

**Fix**:
1. Ensure CUDA toolkit is installed: `sudo apt install -y cuda-toolkit`
2. Set LD_LIBRARY_PATH correctly before running Python
3. Check Python 3.10 environment is activated

### CUBLAS Memory Errors
**Symptom**: `RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED`

**Solutions**:
1. Reduce batch size in `app.py`
2. Reduce image size (imgsz parameter)
3. Close other applications to free GPU memory
4. The script will automatically fall back to CPU if this happens

### Missing Dependencies
If you see `ModuleNotFoundError`:
```bash
mamba run -n pytorch-gpu pip install ultralytics opencv-python pyyaml
```

## Installation Commands Summary

If you need to reinstall from scratch:

```bash
# 1. Create environment
mamba create -y -n pytorch-gpu python=3.10

# 2. Install PyTorch with GPU
mamba run -n pytorch-gpu pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

# 3. Install YOLO
mamba run -n pytorch-gpu pip install ultralytics

# 4. Install CUDA toolkit (if not already installed)
sudo apt install -y cuda-toolkit

# 5. Run your script
mamba run -n pytorch-gpu python app.py
```

## Performance Tips

1. **Monitor GPU**: Run in another terminal while training
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Optimize batch size**: Start with batch=4, increase until you hit memory limits
3. **Use smaller models**: Consider faster models like YOLOv8n instead of YOLOv12s
4. **Pin memory**: Add `pin_memory=True` in DataLoader if using custom training loops

## References
- [NVIDIA Jet son L4T Documentation](https://docs.nvidia.com/jetson/index.html)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [JetPack 6.x Release Notes](https://docs.nvidia.com/jetson/jetpack/release-notes/index.html)
