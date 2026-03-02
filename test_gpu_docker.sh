#!/bin/bash
# Quick GPU Test with Docker

echo "🔍 Testing GPU in Docker..."
echo ""

sudo docker run --rm --gpus all dustynv/l4t-pytorch:r36.4.0 \
  python3 -c "
import torch
print('=' * 60)
print('GPU TEST - Docker Container')
print('=' * 60)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'Device count: {torch.cuda.device_count()}')
    print('')
    print('Testing GPU computation...')
    try:
        import time
        x = torch.randn(1000, 1000).cuda()
        start = time.time()
        result = torch.matmul(x, x)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        print(f'✅ GPU computation successful! ({elapsed:.2f}ms)')
    except Exception as e:
        print(f'❌ GPU computation failed: {e}')
else:
    print('❌ GPU not available')
print('=' * 60)
"

echo ""
echo "✅ GPU test completed!"
