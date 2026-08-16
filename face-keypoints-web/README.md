# Face Key Points — Browser Demo (LiteRT.js)

Face keypoint detection (YOLOv8s-pose, WFLW dataset, 5 keypoints) running entirely in the browser via [LiteRT.js](https://developers.google.com/edge/litert/web) — no server inference. Built with React + Vite + [`@ultralytics/yolo`](https://www.npmjs.com/package/@ultralytics/yolo).

## How it works

```
best.pt → export(format="litert", quantize="w8a32") → best_w8a32.tflite (11.4MB)
    → hosted on HuggingFace Hub → fetched by the browser → run on WebGPU (WASM fallback)
```

## Local dev

1. `npm install`
2. Put the model URL in `.env.local` (copy from `.env.example`):
   ```
   VITE_MODEL_URL=https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/face-keypoints.tflite
   ```
   No env var set? The page shows a text box — paste any `.tflite` URL there instead.
3. `npm run dev` — opens on http://localhost:5173

The dev server sends `COOP`/`COEP` headers (see `vite.config.ts`) so LiteRT.js can use threaded WASM.

## Hosting the model

The `.tflite` lives at `../face-keypoints.tflite` (in the Face Key Points repo root, 11.4MB).

1. Create a repo at https://huggingface.co/new (e.g. `litert-models`)
2. Drag-drop `face-keypoints.tflite` onto the Files tab
3. Copy the URL: `https://huggingface.co/<user>/<repo>/resolve/main/face-keypoints.tflite` — HF serves it with `Access-Control-Allow-Origin: *`, which is COEP-compatible.

## Deploying to Vercel

1. Push this folder to a GitHub repo
2. Import it at https://vercel.com/new (framework auto-detected: Vite)
3. Add env var `VITE_MODEL_URL` (the HF URL above) under Settings → Environment Variables
4. `vercel.json` already sets COOP/COEP headers so threaded WASM works in production

## Model conversion (for reference)

```python
from ultralytics import YOLO
YOLO("pose/train/weights/best.pt").export(format="litert", imgsz=640, quantize="w8a32")
```

Verified: IoU-matched box/keypoint error vs the original `.pt` is < 2px on WFLW test images.

## Notes

- WebGPU needs a recent Chrome/Edge/Safari; otherwise it falls back to WASM (slower, still works).
- The model weights are public — inherent to any browser-side ML demo.