import { useCallback, useEffect, useRef, useState } from "react";
import { YOLO, annotate, type Results } from "@ultralytics/yolo";
import "./App.css";

const DEFAULT_MODEL_URL = import.meta.env.VITE_MODEL_URL ?? "";

type Status =
  | { kind: "idle" }
  | { kind: "loading"; message: string }
  | { kind: "ready"; info: string }
  | { kind: "error"; message: string };

export default function App() {
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL_URL);
  const [status, setStatus] = useState<Status>(
    DEFAULT_MODEL_URL ? { kind: "loading", message: "Loading model…" } : { kind: "idle" },
  );
  const [fileName, setFileName] = useState<string | null>(null);
  const modelRef = useRef<YOLO | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const loadModel = useCallback(async (url: string) => {
    if (!url.trim()) return;
    setStatus({ kind: "loading", message: "Downloading model (11MB)…" });
    try {
      const model = await YOLO.load(url.trim(), { litertWasmUrl: "/wasm/" });
      modelRef.current = model;
      setStatus({ kind: "ready", info: `${model.task} · ${model.device} · ${Object.keys(model.names).length} class(es)` });
    } catch (e) {
      modelRef.current = null;
      setStatus({ kind: "error", message: `Failed to load model: ${String(e)}` });
    }
  }, []);

  useEffect(() => {
    if (DEFAULT_MODEL_URL) void loadModel(DEFAULT_MODEL_URL);
  }, [loadModel]);

  const onFile = useCallback(async (file: File | undefined) => {
    if (!file || !modelRef.current) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
      const canvas = canvasRef.current!;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      imageRef.current = img;
      setFileName(file.name);
      setStatus({ kind: "loading", message: "Running inference…" });
      try {
        const results: Results = await modelRef.current!.predict(img);
        await annotate(canvas, img, results);
        const n = results.boxes.length;
        const kpts = results.keypoints.reduce((a, k) => a + k.points.filter((p) => p[2] > 0.25).length, 0);
        setStatus({
          kind: "ready",
          info: `${modelRef.current!.task} · ${modelRef.current!.device} · ${n} face(s), ${kpts} keypoints · ${Math.round(results.speed.preprocess + results.speed.inference + results.speed.postprocess)}ms`,
        });
      } catch (e) {
        setStatus({ kind: "error", message: `Inference failed: ${String(e)}` });
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    img.src = url;
  }, []);

  return (
    <main className="app">
      <header>
        <h1>Face Key Points — LiteRT.js</h1>
        <p>YOLOv8s-pose (WFLW, 5 keypoints) running 100% in your browser via WebGPU/WASM.</p>
      </header>

      {!DEFAULT_MODEL_URL && (
        <section className="model-form">
          <label>
            Model URL (.tflite, hosted on HuggingFace Hub or anywhere CORS-enabled)
            <input
              type="url"
              placeholder="https://huggingface.co/USERNAME/REPO/resolve/main/model.tflite"
              value={modelUrl}
              onChange={(e) => setModelUrl(e.target.value)}
            />
          </label>
          <button onClick={() => void loadModel(modelUrl)} disabled={!modelUrl.trim() || status.kind === "loading"}>
            Load model
          </button>
        </section>
      )}

      <section className="controls">
        <label className="file-btn">
          Choose image
          <input type="file" accept="image/*" hidden onChange={(e) => void onFile(e.target.files?.[0])} />
        </label>
        <span className="file-name">{fileName ?? "no image selected"}</span>
      </section>

      <p className={`status status-${status.kind}`}>
        {status.kind === "idle" && "Set the model URL above and press Load model."}
        {status.kind === "loading" && status.message}
        {status.kind === "ready" && status.info}
        {status.kind === "error" && status.message}
      </p>

      <canvas ref={canvasRef} className="output" style={{ display: fileName ? "block" : "none" }} />
    </main>
  );
}