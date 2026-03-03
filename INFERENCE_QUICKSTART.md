# Quick Start: Persistent Inference Container

This setup uses Docker Compose to maintain a persistent container, eliminating the need to reinstall dependencies on every inference run.

## 🚀 First Time Setup

Start the persistent container (installs dependencies once):

```bash
docker compose up -d
```

This will:
- Start the container in the background
- Install numpy and ultralytics (one time only)
- Keep the container ready for inference

## 🎬 Running Inference

Once the container is running, use the inference script as before:

```bash
./inference_simple.sh video.mp4
./inference_simple.sh video.mp4 --conf 0.7 --imgsz 416
```

The script will automatically:
- Check if the container is running
- Start it if needed (with auto-install)
- Execute inference immediately (no reinstall!)

## 📊 Container Management

Check container status:
```bash
 docker compose ps
```

View container logs:
```bash
 docker compose logs -f
```

Stop the container:
```bash
 docker compose down
```

Restart the container:
```bash
 docker compose restart
```

## 💡 Benefits

- ✅ No dependency reinstallation on each run
- ✅ Faster inference execution
- ✅ Same memory and GPU configuration
- ✅ Container stays ready for multiple inference runs
- ✅ Automatic startup if container is stopped

## 🔄 Updating Dependencies

If you need to update dependencies, restart the container:

```bash
 docker compose down
 docker compose up -d
```

Or manually install inside the running container:
```bash
 docker compose exec yolo-inference pip install <package>
```
