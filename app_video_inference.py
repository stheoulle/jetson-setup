#!/usr/bin/env python3
"""
YOLO Video Inference - Process videos with trained model
Optimized for Jetson Orin
"""

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import argparse
from tqdm import tqdm

print("=" * 70)
print("YOLO Video Inference - Jetson Orin Optimized")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("⚠️  GPU not detected - using CPU")
    device = "cpu"

print(f"Inference device: {device.upper()}")
print("=" * 70)

def detect_video(video_path, model_path, output_path=None, conf=0.5, device="cuda", imgsz=640):
    """
    Process video with YOLO detection
    
    Args:
        video_path: Path to input video
        model_path: Path to trained model weights
        output_path: Path for output video (default: adds _detected suffix)
        conf: Confidence threshold
        device: Use 'cuda' or 'cpu'
        imgsz: Inference image size (default: 640, lower = faster + less memory)
    """
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Set output path
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_detected.mp4"
    else:
        output_path = Path(output_path)
    
    print(f"\n📹 Input video: {video_path}")
    print(f"🤖 Model: {model_path}")
    print(f"📊 Confidence threshold: {conf}")
    print(f"🎯 Inference size: {imgsz}px")
    print(f"💾 Output video: {output_path}")
    print("=" * 70)
    
    # Load model with memory optimization
    print("\n🔄 Loading model...")
    model = YOLO(str(model_path), task='detect')
    
    # Explicitly disable model fusion to save memory
    if hasattr(model.model, 'fuse'):
        print("⚙️  Skipping model fusion for memory optimization")
    
    model.to(device)
    
    # Clear GPU cache before starting
    if device == "cuda":
        torch.cuda.empty_cache()
        print("✅ GPU cache cleared")
    
    # Warm up with a small dummy image to initialize model properly
    print("🔥 Warming up model...")
    import numpy as np
    dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    try:
        _ = model.predict(dummy_img, conf=conf, device=device, verbose=False, imgsz=imgsz)
        if device == "cuda":
            torch.cuda.empty_cache()
        print("✅ Model ready")
    except Exception as e:
        print(f"⚠️  Warmup failed (continuing anyway): {e}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print("=" * 70)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_path}")
    
    print("\n🎬 Processing frames...")
    frame_count = 0
    
    with tqdm(total=total_frames, desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference with memory optimization
            # fuse=False prevents layer fusion which saves memory
            results = model.predict(
                frame, 
                conf=conf, 
                device=device, 
                verbose=False,
                imgsz=imgsz,
                half=False,  # Disable FP16 for stability
                augment=False
            )
            
            # Draw detections on frame
            annotated_frame = results[0].plot()
            
            # Write frame
            out.write(annotated_frame)
            
            frame_count += 1
            pbar.update(1)
            
            # Clear cache periodically to prevent memory buildup
            if device == "cuda" and frame_count % 50 == 0:
                torch.cuda.empty_cache()
    
    # Release everything
    cap.release()
    out.release()
    
    print("\n" + "=" * 70)
    print(f"✅ Video inference completed!")
    print(f"📊 Processed {frame_count} frames")
    print(f"📁 Output saved: {output_path}")
    print("=" * 70)
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Video Inference - Detect objects in video frames"
    )
    parser.add_argument(
        "video",
        help="Path to input video file"
    )
    parser.add_argument(
        "--model",
        default="runs/detect/train22/weights/best.pt",
        help="Path to trained model (default: runs/detect/train22/weights/best.pt)"
    )
    parser.add_argument(
        "--output",
        help="Path for output video (default: input_video_detected.mp4)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use for inference (default: cuda)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size in pixels (default: 640, lower = faster)"
    )
    
    args = parser.parse_args()
    
    detect_video(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        conf=args.conf,
        device=args.device,
        imgsz=args.imgsz
    )


if __name__ == "__main__":
    main()
