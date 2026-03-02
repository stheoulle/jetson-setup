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