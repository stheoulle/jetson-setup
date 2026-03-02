#!/bin/bash
# GPU Setup Guide for Jetson Orin

echo "Jetson Orin GPU Setup"
echo "========================"

# Activate the pytorch-gpu environment
echo "Activating pytorch-gpu environment..."
eval "$(conda shell.bash hook)"

# Create or activate environment if it doesn't exist
if ! conda env list | grep -q pytorch-gpu; then
    echo "Creating pytorch-gpu environment..."
    mamba create -y -n pytorch-gpu python=3.10
fi

mamba activate pytorch-gpu

# Set up CUDA library paths for Jetson
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6

echo "Environment activated!"
echo ""
echo "To use GPU in Python, run:"
echo "  mamba activate pytorch-gpu"
echo "  export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu/nvidia:\$LD_LIBRARY_PATH"
echo ""
echo "Or simply use:"
echo "  mamba run -n pytorch-gpu python your_script.py"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
python << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("GPU not available")
PYEOF
