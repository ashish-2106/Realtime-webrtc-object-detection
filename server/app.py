import io
import time
import json
import uuid
import numpy as np
import cv2

from PIL import Image

import onnxruntime as ort
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
logging.basicConfig(level=logging.DEBUG)

# -------------------------
# Load YOLOv5s ONNX Model
# -------------------------
MODEL_PATH = "models/yolov5s.onnx"   # make sure this file is mounted in /app
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# COCO classes for YOLOv5
COCO_CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
    "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"
]

# Map YOLOv5 class IDs to names
def coco_name(cls_id: int) -> str:
    if 0 <= cls_id < len(COCO_CLASSES):
        return COCO_CLASSES[cls_id]
    return f"class_{cls_id}"
# -------------------------
# Utils
# -------------------------
def xywh2xyxy(x):
    """Convert [x, y, w, h] to [xmin, ymin, xmax, ymax]."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.4, iou_thres=0.45):
    """NMS for YOLOv5 outputs"""
    boxes = []
    for det in prediction:  # per image
        det = det[det[:, 4] > conf_thres]
        if not det.shape[0]:
            continue
        det[:, 5:] *= det[:, 4:5]
        box = xywh2xyxy(det[:, :4])
        conf = det[:, 5:].max(1)
        j = det[:, 5:].argmax(1)
        det_out = np.concatenate((box, conf.reshape(-1,1), j.reshape(-1,1)), axis=1)
        # Simple NMS
        keep = []
        det_out = det_out[det_out[:,4].argsort()[::-1]]
        while det_out.shape[0]:
            keep.append(det_out[0])
            if det_out.shape[0] == 1:
                break
            ious = bbox_iou(det_out[0][:4], det_out[1:, :4])
            det_out = det_out[1:][ious < iou_thres]
        boxes.append(np.stack(keep))
    return boxes

def bbox_iou(box1, boxes2):
    """IoU calc for NMS"""
    x1 = np.maximum(box1[0], boxes2[:,0])
    y1 = np.maximum(box1[1], boxes2[:,1])
    x2 = np.minimum(box1[2], boxes2[:,2])
    y2 = np.minimum(box1[3], boxes2[:,3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (boxes2[:,2]-boxes2[:,0])*(boxes2[:,3]-boxes2[:,1]) - inter
    return inter / (union + 1e-6)

# -------------------------
# Inference
# -------------------------


def run_inference(jpg_bytes):
    t0 = int(time.time() * 1000)
    if session is None:
        # Dummy fallback
        dets = [{"label": "person", "score": 0.9,
                 "xmin": 0.1, "ymin": 0.1, "xmax": 0.4, "ymax": 0.5}]
        return dets, int(time.time() * 1000)

    # Decode JPEG → BGR image
    img = np.frombuffer(jpg_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    H0, W0 = img.shape[:2]

    # Resize → 640x640 (YOLOv5 input)
    img_resized = cv2.resize(img, (640, 640))
    img_resized = img_resized[:, :, ::-1]  # BGR → RGB
    x = img_resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # [3,640,640]
    x = np.expand_dims(x, axis=0)   # [1,3,640,640]

    # Inference
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: x})[0]  # usually (1, N, 85)
    preds = preds[0]

    boxes, scores, class_ids = [], [], []
    for det in preds:
        conf = det[4]
        if conf < 0.4:
            continue
        cls_conf = det[5:]
        cls_id = np.argmax(cls_conf)
        score = conf * cls_conf[cls_id]
        if score < 0.4:
            continue

        # YOLOv5 box format = [cx, cy, w, h]
        cx, cy, w, h = det[:4]
        xmin = int((cx - w / 2) * W0 / 640)
        ymin = int((cy - h / 2) * H0 / 640)
        xmax = int((cx + w / 2) * W0 / 640)
        ymax = int((cy + h / 2) * H0 / 640)

        boxes.append([xmin, ymin, xmax, ymax])
        scores.append(float(score))
        class_ids.append(int(cls_id))

    # Apply NMS
    final_dets = []
    if len(boxes) > 0:
        idxs = cv2.dnn.NMSBoxes(
            bboxes=[[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes],
            scores=scores,
            score_threshold=0.4,
            nms_threshold=0.5
        )
        if len(idxs) > 0:
            for i in idxs.flatten():
                xmin, ymin, xmax, ymax = boxes[i]
                final_dets.append({
                    "label": coco_name(class_ids[i]),
                    "score": scores[i],
                    # send normalized coords to frontend
                    "xmin": max(0, xmin) / W0,
                    "ymin": max(0, ymin) / H0,
                    "xmax": min(W0, xmax) / W0,
                    "ymax": min(H0, ymax) / H0,
                })

    t1 = int(time.time() * 1000)
    print("Detections:", final_dets)  # 🔎 Debug print in server logs
    return final_dets[:50], t1

# -------------------------
# FastAPI
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.disconnect":
                print("🔌 WebSocket disconnected")
                break

            if "bytes" in message:
                jpg_bytes = message["bytes"]
                print("📷 Frame received:", len(jpg_bytes), "bytes")

                detections, t_inf = run_inference(jpg_bytes)
                print("🟢 Detections:", detections)

                payload = {
                    "type": "detections",
                    "frame_id": str(uuid.uuid4()),
                    "capture_ts": int(time.time() * 1000),
                    "detections": detections,
                    "t_inf": t_inf
                }
                await ws.send_text(json.dumps(payload))

            elif "text" in message:
                print("📩 Received TEXT:", message["text"])
                # Optionally respond if needed
                await ws.send_text(json.dumps({
                    "type": "ack",
                    "message": "text received"
                }))

    except Exception as e:
        import traceback
        print("❌ WS exception:", e)
        traceback.print_exc()
        await ws.close()
# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
