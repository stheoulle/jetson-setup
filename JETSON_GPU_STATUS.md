# Jetson Orin GPU Training Status

## Current Status: CPU Training Only ❌ → Use Docker for GPU ✅

### Quick Summary

**Problem**: PyTorch GPU support on Jetson Orin (Compute Capability 8.7) is not working standalone due to:
- Standard PyTorch wheels don't include Jetson GPU kernels
- Nightly builds: `no kernel image available for CC 8.7`  
- NVIDIA wheels: Missing dependencies (libcudnn8, libcusparseLt.so.0)

### ✅ Current Working Setup 

**CPU Training** (Slower but works)
```bash
mamba run -n pytorch-gpu python app.py
```
- PyTorch 2.0.1 (CPU-only)
- Python 3.10 environment
- Batch size: 4, Image size: 416
- ~3-5x slower than GPU

### 🚀 Recommended: GPU Training with Docker

```bash
# One-line GPU training (tested and working)
sudo docker run --runtime nvidia --gpus all -it --rm \
  --network host --shm-size=8g \
  -v $(pwd):/workspace -w /workspace \
  dustynv/l4t-pytorch:r36.4.0 \
  bash -c "pip install ultralytics && python app.py"
```

This uses NVIDIA's pre-built PyTorch with full Jetson GPU support.

## Files

- **[app.py](app.py)** - Jetson-optimized YOLO training script with CPU fallback
- **[GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md)** - Detailed troubleshooting and setup documentation
- **[setup_gpu.sh](setup_gpu.sh)** - Environment verification script
- **[retour.md](retour.md)** - Original Docker setup working notes

## What Was Tried

| Approach | Result | Issue |
|----------|--------|-------|
| PyTorch 2.0.1 from PyPI | ❌ CPU only | No Jetson kernels |
| PyTorch 2.12 nightly cu126 | ❌ Failed | `no kernel image for CC 8.7` |
| NVIDIA PyTorch 2.4.0 wheel | ❌ Failed | Missing cuDNN8, libcusparseLt |
| Docker container | ✅ Works | Requires Docker |

## Next Steps

### Option 1: Use Docker (Recommended)
See command above - works immediately with GPU.

### Option 2: Build from Source (Advanced)
```bash
# Clone and build PyTorch with CC 8.7 support
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export TORCH_CUDA_ARCH_LIST="8.7"
export USE_CUDA=1
export USE_CUDNN=1
python setup.py install
```
⚠️ Takes 4-6 hours on Jetson Orin

### Option 3: Wait for Official Support
PyTorch may add CC 8.7 support in future releases.

## Environment Details

- **Device**: Jetson Orin (CC 8.7)
- **JetPack**: 6.0 (L4T R36.4.7)
- **CUDA**: 12.6
- **cuDNN**: 9.3.0
- **Python**: 3.10.19 (via mamba)
- **PyTorch**: 2.0.1 (CPU-only)

## Training Performance

CPU Training on Jetson Orin:
- Batch 4, Image 416: ~5-10 it/s (estimated 2-3 hours for 20 epochs)
  
GPU Training (Docker):
- Batch 8-16, Image 640: ~30-50 it/s (estimated 20-40 min for 20 epochs)

---

**Bottom Line**: For GPU training now, use Docker. For standalone CPU training, current setup works.
