// Minimal, dependency-free front-end with:
// - getUserMedia
// - WASM inference via onnxruntime-web (loaded dynamically)
// - Server mode via WebSocket (send frames -> receive detections JSON)
// - Canvas overlay
// - Rolling metrics + POST to server for metrics.json aggregation

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const resSel = document.getElementById('resSel');
const fpsSel = document.getElementById('fpsSel');
const modeSel = document.getElementById('modeSel');
const modeTag = document.getElementById('modeTag');
const statusTag = document.getElementById('statusTag');
const metricsBox = document.getElementById('metricsBox');
const hostUrl = document.getElementById('hostUrl');
const metricsUrl = document.getElementById('metricsUrl');
const btnQR = document.getElementById('btnQR');
const qrCanvas = document.getElementById('qr');


const ENV = {
  MODE: (getEnv('MODE') || 'wasm'),
  SERVER_WS_URL: getEnv('SERVER_WS_URL') || `ws://${location.hostname}:8000/ws`,
  METRICS_POST_URL: getEnv('METRICS_POST_URL') || `http://${location.hostname}:8000/metrics/update`,
  METRICS_GET_URL: getEnv('METRICS_GET_URL') || `http://${location.hostname}:8000/metrics`,
};

function getEnv(k){ try { return (window.__ENV && window.__ENV[k]) || (new URLSearchParams(location.search).get(k.toLowerCase())); } catch { return null; } }

hostUrl.textContent = location.href;
metricsUrl.textContent = ENV.METRICS_GET_URL;
modeSel.value = ENV.MODE;
modeTag.textContent = `MODE=${ENV.MODE}`;

let stream = null, anim = null, processing = false, lastFrameTs = 0;
let ws = null, ort = null, ortSession = null;
let desiredFPS = parseInt(fpsSel.value,10);

const metrics = {
  frames: [],
  bytesUp: 0,
  bytesDown: 0,
  start: null,
  e2e: [], // overlay_display_ts - capture_ts
};

btnStart.onclick = start;
btnStop.onclick = stop;
fpsSel.onchange = () => { desiredFPS = parseInt(fpsSel.value,10); };
modeSel.onchange = () => { modeTag.textContent = `MODE=${modeSel.value}`; };

btnQR.onclick = () => {
  const url = location.href;
  drawSimpleQR(qrCanvas, url);
  qrCanvas.style.display = qrCanvas.style.display === 'none' ? 'block' : 'none';
};

async function start(){
  const [w,h] = resSel.value.split('x').map(Number);
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: {ideal:w}, height: {ideal:h}, facingMode:'environment' },
    audio: false
  });
  video.srcObject = stream;
  await video.play();

  overlay.width = video.videoWidth || w;
  overlay.height = video.videoHeight || h;

  metrics.start = performance.now();
  metrics.frames = [];
  metrics.bytesUp = 0;
  metrics.bytesDown = 0;
  metrics.e2e = [];

  if (modeSel.value === 'server') {
    await connectWS();
  } else {
    await initWasm();
  }

  processing = true;
  loop();
}

async function stop(){
  processing = false;
  if (anim) cancelAnimationFrame(anim);
  if (stream) stream.getTracks().forEach(t=>t.stop());
  if (ws) { ws.close(); ws = null; }
  statusTag.textContent = 'stopped';
}

async function connectWS(){
  return new Promise((resolve, reject)=>{
    ws = new WebSocket(ENV.SERVER_WS_URL);
    ws.binaryType = 'arraybuffer';
    ws.onopen = ()=>{ statusTag.textContent = 'server: connected'; resolve(); };
    ws.onerror = (e)=>{ statusTag.textContent = 'server: error'; reject(e); };
    ws.onmessage = (ev)=>{
      metrics.bytesDown += (typeof ev.data === 'string') ? ev.data.length : ev.data.byteLength;
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'detections') {
          drawDetections(msg);
        }
      } catch(e){}
    };
  });
}

async function initWasm(){
  if (!ort) {
    // load onnxruntime-web from CDN (small)
    await import('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
    // eslint-disable-next-line no-undef
    ort = window.ort;
  }
  if (!ortSession) {
    // place a small detector model in /models/ (e.g., ssd_mobilenet_v1_12_int8.onnx)
    const model = '/models/yolov5s.onnx';
    ortSession = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
    statusTag.textContent = 'wasm: model loaded';
  }
}

// backpressure: process only latest frame ~ desiredFPS
function loop(){
  anim = requestAnimationFrame(loop);
  const now = performance.now();
  const shouldProcess = (now - lastFrameTs) > (1000/desiredFPS);
  if (!shouldProcess || processing === false) return;
  lastFrameTs = now;

  const capture_ts = Date.now();
  // draw current frame to an offscreen canvas for processing
  const w = overlay.width, h = overlay.height;
  const off = new OffscreenCanvas(w,h);
  const octx = off.getContext('2d');
  octx.drawImage(video, 0,0,w,h);

  if (modeSel.value === 'server') {
    off.convertToBlob({ type: 'image/jpeg', quality: 0.7 }).then(b=>{
      b.arrayBuffer().then(buf=>{
        metrics.bytesUp += buf.byteLength;
        // send frame + timestamp
        
        ws.send(buf);
      });
    });
    // drawing happens when server replies
  } else {
    // WASM path
    off.convertToBlob({type:'image/jpeg', quality:0.8}).then(async b=>{
      const imgBitmap = await createImageBitmap(b);
      // simple preprocessing to NHWC float32 normalized
      const input = await preprocessImage(imgBitmap, 320, 240); // downscale regardless of display size
      const feeds = { 'input': input.tensor };
      const t0 = performance.now();
      const out = await ortSession.run(feeds);
      const t1 = performance.now();

      const detections = postprocess(out, input.ratioW, input.ratioH);
      const msg = {
        frame_id: crypto.randomUUID(),
        capture_ts,
        recv_ts: capture_ts,           // local in wasm
        inference_ts: Date.now(),      // after run
        detections
      };
      drawDetections(msg);

      // record E2E
      const overlay_display_ts = Date.now();
      metrics.e2e.push(overlay_display_ts - capture_ts);
      metrics.frames.push({ t0, t1 });
      updateMetricsBox();
      postMetricsMaybe();
    });
  }
}

function drawDetections(msg){
  const w = overlay.width, h = overlay.height;
  ctx.clearRect(0,0,w,h);
  ctx.lineWidth = 2;
  ctx.font = '14px system-ui';

  for (const d of (msg.detections||[])) {
    const x = d.xmin * w, y = d.ymin * h;
    const bw = (d.xmax - d.xmin) * w;
    const bh = (d.ymax - d.ymin) * h;
    ctx.strokeStyle = '#7bd88f';
    ctx.strokeRect(x,y,bw,bh);
    ctx.fillStyle = '#0b0e14';
    const label = `${d.label} ${(d.score*100|0)}%`;
    const tw = ctx.measureText(label).width + 8;
    ctx.fillRect(x, y-18, tw, 18);
    ctx.fillStyle = '#7bd88f';
    ctx.fillText(label, x+4, y-4);
  }

  // metrics E2E (server path)
  if (modeSel.value === 'server') {
    const overlay_display_ts = Date.now();
    metrics.e2e.push(overlay_display_ts - (msg.capture_ts||overlay_display_ts));
    updateMetricsBox();
    postMetricsMaybe();
  }
}

function updateMetricsBox(){
  const durSec = (performance.now() - metrics.start) / 1000;
  const fps = (metrics.e2e.length / Math.max(1, durSec));
  const med = percentile(metrics.e2e, 50)|0;
  const p95 = percentile(metrics.e2e, 95)|0;
  const up = (metrics.bytesUp*8/1000/Math.max(1,durSec)).toFixed(1);
  const down = (metrics.bytesDown*8/1000/Math.max(1,durSec)).toFixed(1);
  metricsBox.textContent = JSON.stringify({
    processed_fps: +fps.toFixed(1),
    e2e_latency_ms_median: med,
    e2e_latency_ms_p95: p95,
    uplink_kbps: +up,
    downlink_kbps: +down
  }, null, 2);
}

let lastPost = 0;
function postMetricsMaybe(){
  const now = Date.now();
  if (now - lastPost < 1000) return;
  lastPost = now;
  fetch(ENV.METRICS_POST_URL, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: metricsBox.textContent
  }).catch(()=>{});
}

// --- Helpers ---

function percentile(arr, p){
  if (!arr.length) return 0;
  const a = [...arr].sort((x,y)=>x-y);
  const idx = Math.floor((p/100)*a.length);
  return a[Math.min(idx, a.length-1)];
}

async function preprocessImage(imgBitmap, targetW, targetH){
  const c = new OffscreenCanvas(targetW, targetH);
  const cctx = c.getContext('2d');
  cctx.drawImage(imgBitmap, 0,0,targetW,targetH);
  const { data } = cctx.getImageData(0,0,targetW,targetH);
  const tensor = new Float32Array(targetW*targetH*3);
  for (let i=0, j=0; i<data.length; i+=4, j+=3) {
    tensor[j]   = data[i]   / 255.0; // R
    tensor[j+1] = data[i+1] / 255.0; // G
    tensor[j+2] = data[i+2] / 255.0; // B
  }
  // If your model needs NCHW, swap later in feeds
  // Here, we assume NHWC float32 with name "input"
  return { tensor: new ort.Tensor('float32', tensor, [1, targetH, targetW, 3]), ratioW: 1, ratioH: 1 };
}

function postprocess(out) {
  // YOLOv5 ONNX: out is usually { output0: Tensor }
  const tensor = out[Object.keys(out)[0]];
  const data = tensor.data;
  const [N, D] = tensor.dims; // e.g., 25200 x 85

  const dets = [];
  for (let i = 0; i < N; i++) {
    const offset = i * D;
    const cx = data[offset + 0];
    const cy = data[offset + 1];
    const w  = data[offset + 2];
    const h  = data[offset + 3];
    const obj = data[offset + 4];

    if (obj < 0.3) continue; // objectness threshold

    // class scores
    let bestCls = -1, bestScore = 0;
    for (let c = 5; c < D; c++) {
      const score = data[offset + c];
      if (score > bestScore) {
        bestScore = score;
        bestCls = c - 5;
      }
    }

    const finalScore = obj * bestScore;
    if (finalScore < 0.4) continue;

    // Convert cx,cy,w,h → normalized xyxy
    const xmin = (cx - w/2) / 640;
    const ymin = (cy - h/2) / 640;
    const xmax = (cx + w/2) / 640;
    const ymax = (cy + h/2) / 640;

    dets.push({
      label: cocoName(bestCls),
      score: finalScore,
      xmin: clamp01(xmin),
      ymin: clamp01(ymin),
      xmax: clamp01(xmax),
      ymax: clamp01(ymax),
    });
  }

  // simple NMS on client (optional)
  return dets.slice(0, 50);
}


function pick(obj, keys){
  for (const k of keys){
    if (obj[k]) return obj[k];
  }
  return null;
}
function clamp01(x){ return Math.max(0, Math.min(1, x)); }

function cocoName(id){
  const map = {0:'person',1:'bicycle',2:'car',3:'motorcycle',5:'bus',7:'truck',15:'cat',16:'dog',17:'horse',56:'chair',57:'couch',58:'potted plant',59:'bed',62:'tv',63:'laptop',64:'mouse',67:'cell phone'};
  return map[id] || `class_${id}`;
}

// super tiny QR (not full spec; works for short URLs): draws blocks from URL char codes
function drawSimpleQR(c, text){
  const size = c.width;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff'; ctx.fillRect(0,0,size,size);
  const n = 25; const cell = Math.floor(size/n);
  ctx.fillStyle = '#000';
  const seed = [...text].reduce((a,ch)=>a+ch.charCodeAt(0),0);
  function rnd(i,j){ const x = Math.sin(i*57+j*131+seed)*10000; return x - Math.floor(x); }
  for (let i=0;i<n;i++){
    for (let j=0;j<n;j++){
      if (i<3&&j<3 || i<3&&j>n-4 || i>n-4&&j<3) { // finder-like boxes
        if ((i%2===0 && j%2===0)) ctx.fillRect(j*cell,i*cell,cell,cell);
      } else {
        if (rnd(i,j) > 0.6) ctx.fillRect(j*cell,i*cell,cell,cell);
      }
    }
  }
}

statusTag.textContent = 'ready';
