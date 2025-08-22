# Realtime WebRTC Multi-Object Detection

This project demonstrates **real-time multi-object detection** on live video streamed from a phone via WebRTC. The detections are sent back to the browser, where bounding boxes + labels are overlaid in near real-time.  

## ✨ Features
- Phone camera stream → Browser via WebRTC
- Real-time object detection (Server mode + WASM mode)
- Bounding box + label overlay in browser
- Low-resource mode (WASM + quantized model + 320×240 downscale)
- Metrics collection (latency, FPS, bandwidth)
- Dockerized setup for reproducibility

---

## 📦 Installation & Run

### 1. Clone repo
```bash
git clone https://github.com/ashish-2106/Realtime-webrtc-object-detection.git
cd Realtime-webrtc-object-detection
```
###2. Start app
Default run (WASM mode if no GPU available):
```bash
./start.sh
```
or via Docker:
```bash
docker-compose up --build
```
3. Open browser:
- On laptop: open http://localhost:3000
- Scan the displayed QR code with your phone browser (Chrome/Safari).
- Allow camera permissions.
- You should now see your phone video mirrored with detection overlays in your laptop browser.

🔀 Modes
Server mode (CPU inference)
```bash
MODE=server ./start.sh
```
WASM mode (browser inference, low-resource laptops)
```bash
MODE=wasm ./start.sh
```
