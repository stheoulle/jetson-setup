# GPU Setup Guide for Jetson Orin - Standalone Setup

## ⚠️ IMPORTANT: Current Status

**GPU training on Jetson Orin with standalone PyTorch is challenging due to:**
1. Compute Capability 8.7 not fully supported by standard PyTorch wheels
2. NVIDIA Jetson-specific wheels have many missing dependencies (cuDNN8, libcusparseLt)
3. Standard PyTorch wheels don't include Jetson GPU kernels

**RECOMMENDED**: Use Docker container (`dustynv/l4t-pytorch:r36.4.0`) for reliable GPU acceleration.

**CURRENT SETUP**: CPU-only PyTorch 2.0.1 - Works reliably but slower training.

---

## Overview
This guide documents attempts to set up PyTorch with GPU acceleration on Jetson Orin **without Docker**.

## System Information
- **Device**: Jetson Orin (Compute Capability 8.7)
- **JetPack**: 6.0 (L4T R36.4.7)
- **CUDA**: 12.6
- **Driver**: NVIDIA 540.4.0

## What You Need
✓ PyTorch 2.0.1 (CPU-only but stable)
✓ CUDA 12.6 toolkit  
✓ cuDNN 9
✓ Python 3.10 environment via mamba

## Setup Steps Completed

### 1. **Created Python 3.10 Environment**
```bash
mamba create -y -n pytorch-gpu python=3.10
```
*Why Python 3.10?* PyTorch GPU wheels for aarch64 (ARM64) don't support Python 3.13 yet.

### 2. **Installed PyTorch 2.0.1 (CPU-only for now)**
```bash
mamba run -n pytorch-gpu pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121
```
⚠️ **Note**: This wheel doesn't detect GPU on Jetson due to missing CC 8.7 kernels.

### 3. **Installed CUDA Toolkit**
```bash
sudo apt install -y cuda-toolkit
```
Installed CUDA 12.6 to `/usr/local/cuda-12.6/`.

### 4. **Installed cuDNN 9**
```bash
sudo apt install -y cudnn9-cuda-12-6 libcudnn9-cuda-12
sudo ln -sf /usr/lib/aarch64-linux-gnu/libcudnn.so.9 /usr/lib/aarch64-linux-gnu/libcudnn.so.8
```

### 5. **CUDA Library Locations**
- `/usr/local/cuda-12.6/targets/aarch64-linux/lib/`
- `/usr/lib/aarch64-linux-gnu/nvidia/`

## Usage

### Quick Start (CPU Training)
```bash
cd /home/laposte/jetson-setup
mamba run -n pytorch-gpu python app.py
```
*Currently runs on CPU. Expect ~3-5x slower than GPU.*

### GPU Training (Docker - Recommended)
```bash
sudo docker run --runtime nvidia --gpus all -it --rm \
  -v $(pwd):/workspace \
  dustynv/l4t-pytorch:r36.4.0 \
  python /workspace/app.py
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

## Tested PyTorch Versions

| Version | Source | CUDA Detected | GPU Works | Issues |
|---------|--------|---------------|-----------|--------|
| 2.0.1 | PyTorch Index | ❌ No | ❌ No | No Jetson kernels |
| 2.12.0.dev | PyTorch Nightly | ✅ Yes | ❌ No | `no kernel image for CC 8.7` |
| 2.4.0a0 | NVIDIA JP6.0 | - | ❌ No | Missing: libcudnn8, libcusparseLt.so.0 |

## Troubleshooting

### GPU Not Detected
**Symptom**: `CUDA available: False` or `no kernel image available for execution on the device`

**Root Cause**: Jetson Orin has Compute Capability 8.7, which is not included in standard PyTorch builds.

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

### Current Working Setup (CPU-only)

```bash
# 1. Create environment
mamba create -y -n pytorch-gpu python=3.10

# 2. Install PyTorch (CPU-only but stable)
mamba run -n pytorch-gpu pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install YOLO
mamba run -n pytorch-gpu pip install ultralytics

# 4. Install CUDA toolkit
sudo apt install -y cuda-toolkit cudnn9-cuda-12-6

# 5. Run CPU training
mamba run -n pytorch-gpu python app.py
```

### For GPU Support: Use Docker

```bash
# Tested and working GPU setup
sudo docker run --runtime nvidia --gpus all -it --rm \
  --network host --shm-size=8g \
  -v $(pwd):/workspace \
  -w /workspace \
  dustynv/l4t-pytorch:r36.4.0 \
  bash -c "pip install ultralytics && python app.py"
```

### Future: Build PyTorch from Source

```bash
# For advanced users: Build with CC 8.7 support
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export TORCH_CUDA_ARCH_LIST="8.7"
python setup.py install
# ⚠️ Takes 4-6 hours to compile on Jetson
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
