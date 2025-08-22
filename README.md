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
# System Design Report

## 1. Design Choices
Our design focuses on **modularity, scalability, and robustness**:
- **Modularity**: Components (camera input, processing, storage, and UI) are decoupled, making it easy to maintain or upgrade.  
- **Scalability**: The system supports variable frame rates and adaptive quality settings, allowing it to run on both high-end and low-end devices.  
- **User Experience**: A simple, browser-based interface ensures accessibility across devices without requiring installations.  

## 2. Low-Resource Mode
When system resources (CPU, memory, or network bandwidth) are limited:
- **Frame rate is reduced** dynamically to lower computational load.  
- **Resolution is downscaled** (e.g., 720p → 480p) to conserve bandwidth.  
- **Optional features** (like advanced filters or live analytics) are disabled to prioritize core functionality.  
This ensures the application remains responsive even on entry-level hardware or poor network conditions.  

## 3. Backpressure Policy
To avoid system overload during high demand:
- **Buffer limits**: Incoming data is queued with a maximum buffer size. Once full, new frames are dropped instead of overwhelming the system.  
- **Adaptive throttling**: Processing speed is matched to system capacity, preventing uncontrolled memory growth.  
- **Graceful degradation**: Instead of failing, the system reduces quality or skips frames to maintain stability.  

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
## Demo Images
<img src="https://drive.google.com/uc?export=view&id=115lmaqdG-649Ic5SgRDbdqbaZwuKgySW" width="400"/>
<img src="https://drive.google.com/uc?export=view&id=1tJ5Huq4MIikSQe9l8xzU96MopXYEiG_7" width="400"/>
<img src="https://drive.google.com/uc?export=view&id=16ZxxTlJD8ukgLXw21W7Rre2xjI7IRTAt" width="400"/>

