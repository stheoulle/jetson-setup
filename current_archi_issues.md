# Current Architecture Issues

## Biggest Bottlenecks

- ESP32-CAM is usually the hard limit: low sensor quality + weak encoder + unstable Wi-Fi = low resolution and latency spikes.
- Pulling frames via HTTP (`/capture`) is slower than a continuous video stream (`RTSP`/`MJPEG`/`WebRTC`).
- OCR on every detected box, every frame, is expensive and adds lag.

## Performance Upgrades (Highest Impact First)

- Replace ESP32-CAM with an IP camera (1080p, `H.264`/`H.265`, `RTSP`) or a CSI camera on Jetson.
- Use a continuous stream + hardware decode (`nvdec`/`GStreamer`) instead of HTTP polling.
- Split processing rates: run YOLO every `N` frames, track objects between detections, run OCR only when track is stable/new.
- Use ROI logic: OCR only on relevant classes/regions, and only if box size/quality passes a threshold.
- Add queue policy tuning: smaller bounded queues + drop-oldest strategy to keep “live” behavior (low latency over completeness).

## Image Quality Improvements

- Increase source quality first (better lens/sensor, fixed focus, lighting) before increasing model size.
- Keep transport at higher resolution, then downscale only for YOLO; run OCR on higher-res crops from the original frame.
- Add simple pre-OCR enhancement on crops (contrast/denoise/sharpen) and confidence-based retry.
- Add temporal voting: same ID seen across 3–5 frames before accepting a number.

## Security Improvements

- Put camera + Jetson on a dedicated VLAN/SSID; block internet egress from camera.
- Enforce network allowlist/firewall: only Jetson can reach camera stream ports.
- Replace open stream URLs with authenticated access (reverse proxy token, short-lived credentials).
- Use encrypted transport where possible (`RTSPS`, `HTTPS`, or a `WireGuard` tunnel if camera cannot do TLS well).
- Add observability: auth/access logs, rate limiting, and watchdog alerts for disconnects/reconnect storms.

## Practical Target Architecture

- Camera (`RTSP H.264`) -> Jetson hardware decode -> capture queue (latest-frame policy) -> YOLO + tracker -> OCR on stable ROI -> result bus/UI.
- Control plane separated from data plane; management access only via VPN + SSH keys.

## Current camera

CPU: Tensilica LX6, 240 MHz → can’t encode high-res video fast.
RAM: ~520 KB → very little for buffering video.
Wi-Fi: Only 2.4 GHz, sometimes unstable; UDP buffer is tiny.
JPEG/RTSP: Works, but even VGA @ 5 FPS often saturates the network and overflows buffers.

## Thoughts

- Should restart auto with a delay if connexion to camera is lost
- SHould use a better camera
- Trying to use a rtsp server