import { useState, useEffect, useRef, useCallback } from "react";
import * as ort from "onnxruntime-web";

export default function SpikingSnnDemo() {
  const [resolution, setResolution] = useState(256);
  const [session, setSession] = useState(null);
  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(true);
  const [error, setError] = useState(null);
  const [hasInitialActivity, setHasInitialActivity] = useState(false);
  const [controlsVisible, setControlsVisible] = useState(false);

  // Grid size based on selected resolution
  const H = resolution, W = resolution;

  // Model URL based on resolution
  const modelUrl = resolution === 256 ? "./spiking_step_enhanced.onnx" :
                   resolution === 512 ? "./spiking_step_enhanced_512.onnx" :
                   "./spiking_step_enhanced_1024.onnx";

  // State tensors (Float32Array backing stores) - will be resized when resolution changes
  const V = useRef(new Float32Array(H * W));
  const S = useRef(new Float32Array(H * W));
  const U = useRef(new Float32Array(H * W)); // external stimulation (decays each frame)

  // Resize state tensors when resolution changes
  useEffect(() => {
    const size = H * W;
    V.current = new Float32Array(size);
    S.current = new Float32Array(size);
    U.current = new Float32Array(size);
  }, [H, W]);

  // Parameters from spiking_pars (exact defaults)
  const [decay, setDecay] = useState(0.99);
  const [firingThreshold, setFiringThreshold] = useState(0.9);
  const [resetPoint, setResetPoint] = useState(-1);
  const [dropProb, setDropProb] = useState(0.6);
  const [lowerThreshold, setLowerThreshold] = useState(-0.9);

  const [excLocalScale, setExcLocalScale] = useState(6.0);
  const [inhLocalScale, setInhLocalScale] = useState(-1.0);

  // Fixed values for removed controls
  const excGlobalScale = 0;
  const inhGlobalScale = 0;
  const inputSplit = 0.1;

  // Mouse stimulation controls
  const [brushSign, setBrushSign] = useState("exc"); // "exc" | "inh"
  const [brushRadius, setBrushRadius] = useState(8); // pixels
  const [brushStrength, setBrushStrength] = useState(1.0); // scalar added to U

  // Canvas
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);

  // Refs to capture current parameter values without triggering useEffect
  const paramsRef = useRef({});
  paramsRef.current = {
    decay, firingThreshold, resetPoint, inputSplit, excLocalScale,
    excGlobalScale, inhLocalScale, inhGlobalScale, dropProb, lowerThreshold
  };

  // FPS meter
  const lastT = useRef(performance.now());
  const frames = useRef(0);
  const [fps, setFps] = useState(0);

  // Load ORT session when model URL changes (which happens when resolution changes)
  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!modelUrl) return;

      // Don't stop simulation when loading initial model
      if (session) setRunning(false);
      setReady(false);
      setError(null);

      try {
        console.log(`Loading ONNX model from: ${modelUrl} (${resolution}x${resolution})`);
        const s = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ["webgpu", "wasm"],
          graphOptimizationLevel: "all",
        });
        if (cancelled) return;
        console.log(`ONNX model loaded successfully (${resolution}x${resolution})`);
        setSession(s);
        setReady(true);

        // Initialize activity after a short delay to let the session settle
        setTimeout(() => {
          initializeActivity();
        }, 100);
      } catch (e) {
        console.error("Failed to load ONNX:", e);
        setError(e.message);
        setSession(null);
        setReady(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [modelUrl, resolution]);

  // Canvas init - update when resolution changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    ctxRef.current = canvas.getContext("2d");
    const img = ctxRef.current.createImageData(W, H);
    ctxRef.current.putImageData(img, 0, 0);
  }, [W, H]);

  // Utility: write a circular brush into U
  const stampBrush = useCallback((cx, cy) => {
    const sign = brushSign === "exc" ? 1 : -1;
    const r = Math.max(1, Math.floor(brushRadius));
    const s = brushStrength * sign;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx*dx + dy*dy > r*r) continue;
        let x = (cx + dx + W) % W; // torus wrap
        let y = (cy + dy + H) % H;
        U.current[y * W + x] += s;
      }
    }
  }, [brushSign, brushRadius, brushStrength, W, H]);

  // Mouse handlers
  const dragging = useRef(false);
  const handlePointerDown = (e) => { dragging.current = true; stampFromEvent(e); };
  const handlePointerUp = () => { dragging.current = false; };
  const handlePointerMove = (e) => { if (dragging.current) stampFromEvent(e); };
  const stampFromEvent = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * W);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * H);
    stampBrush(x, y);
  };

  // Main animation loop
  useEffect(() => {
    let raf = 0;
    let isRunning = false; // Prevent concurrent inference calls
    let shouldStop = false; // Flag to cleanly stop the loop

    if (!session || !ready || !running) return;

    const ctx = ctxRef.current;
    const image = ctx.createImageData(W, H);

    const step = async () => {
      if (shouldStop) return; // Exit early if effect is cleaning up

      if (isRunning) {
        raf = requestAnimationFrame(step);
        return;
      }

      isRunning = true;

      try {
        const V0 = new ort.Tensor("float32", V.current, [1,1,H,W]);
        const S0 = new ort.Tensor("float32", S.current, [1,1,H,W]);
        const U0 = new ort.Tensor("float32", U.current, [1,1,H,W]);
        const params = paramsRef.current;
        const feeds = {
          V0, S0, U: U0,
          decay: new ort.Tensor("float32", new Float32Array([params.decay])),
          thr: new ort.Tensor("float32", new Float32Array([params.firingThreshold])),
          reset: new ort.Tensor("float32", new Float32Array([params.resetPoint])),
          input_split: new ort.Tensor("float32", new Float32Array([params.inputSplit])),
          exc_local: new ort.Tensor("float32", new Float32Array([params.excLocalScale])),
          exc_global: new ort.Tensor("float32", new Float32Array([params.excGlobalScale])),
          inh_local: new ort.Tensor("float32", new Float32Array([params.inhLocalScale])),
          inh_global: new ort.Tensor("float32", new Float32Array([params.inhGlobalScale])),
          drop_prob: new ort.Tensor("float32", new Float32Array([params.dropProb])),
          lower_thr: new ort.Tensor("float32", new Float32Array([params.lowerThreshold])),
        };

        const out = await session.run(feeds);

        // Check again after async operation
        if (shouldStop) return;

        const V1 = out.V1 ?? out.V ?? Object.values(out)[0];
        const S1 = out.S1 ?? out.S ?? Object.values(out)[1];
        const Y  = out.Y  ?? V1;

        // Update state
        V.current.set(V1.data);
        S.current.set(S1.data);

        // Visualize with color mapping: voltage in blue/cyan, spikes in green
        const len = H * W;
        const vdata = V1.data;
        const sdata = S1.data;
        for (let i = 0; i < len; i++) {
          const v = Math.max(0, Math.min(1, vdata[i] * 0.5 + 0.5)); // normalize voltage
          const s = sdata[i]; // spike intensity
          const j = i * 4;

          // Color mapping: blue for voltage, green for spikes
          image.data[j] = Math.floor(v * 100); // R: voltage only
          image.data[j+1] = Math.floor((s * 255 + v * 150)); // G: spikes + voltage
          image.data[j+2] = Math.floor((v * 255 + s * 100)); // B: voltage + some spikes
          image.data[j+3] = 255; // A
        }
        ctx.putImageData(image, 0, 0);

        // Decay external stimulation field U toward 0
        for (let i = 0; i < len; i++) U.current[i] *= 0.9;

        // FPS
        frames.current += 1;
        const now = performance.now();
        if (now - lastT.current >= 500) {
          setFps(Math.round((frames.current * 1000) / (now - lastT.current)));
          frames.current = 0;
          lastT.current = now;
        }

      } catch (e) {
        console.error("Inference error:", e);
        setRunning(false);
      } finally {
        isRunning = false;
      }

      if (running && !shouldStop) {
        raf = requestAnimationFrame(step);
      }
    };

    raf = requestAnimationFrame(step);

    return () => {
      shouldStop = true;
      cancelAnimationFrame(raf);
    };
  }, [session, ready, running, H, W]);

  // Initialize some activity for first run (256x256 only)
  const initializeActivity = useCallback(() => {
    if (resolution !== 256 || hasInitialActivity) return;

    // Create irregular pattern using deterministic "random" positions
    const positions = [
      [64, 64], [128, 128], [192, 192], [96, 160], [160, 96],
      [48, 180], [180, 48], [120, 80], [80, 200], [200, 120],
      [72, 144], [144, 72], [36, 216], [216, 36], [108, 36]
    ];

    positions.forEach(([x, y]) => {
      stampBrush(x, y);
    });

    setHasInitialActivity(true);
  }, [resolution, hasInitialActivity, stampBrush]);

  // Reset state
  const resetState = () => {
    V.current.fill(0);
    S.current.fill(0);
    U.current.fill(0);
    setHasInitialActivity(false);
  };

  return (
    <div style={{ minHeight: "100vh", padding: "16px", backgroundColor: "#1a1a1a", color: "#e0e0e0" }}>
      <div style={{ maxWidth: "800px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "16px" }}>
        {/* Visualization Panel */}
        <div style={{ backgroundColor: "#2a2a2a", borderRadius: "8px", padding: "16px", boxShadow: "0 2px 8px rgba(0,0,0,0.3)" }}>
          <canvas
            ref={canvasRef}
            width={W}
            height={H}
            style={{
              width: "100%",
              aspectRatio: "1",
              backgroundColor: "black",
              borderRadius: "8px",
              cursor: "crosshair"
            }}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
          />
        </div>

        {/* Controls Toggle Tab */}
        <div
          onClick={() => setControlsVisible(!controlsVisible)}
          style={{
            backgroundColor: "#2a2a2a",
            borderRadius: "8px 8px 0 0",
            padding: "8px 16px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px",
            userSelect: "none"
          }}
        >
          <span style={{ color: "#fff", fontSize: "14px", fontWeight: "500" }}>Sim Controls</span>
          <span style={{
            color: "#888",
            fontSize: "12px",
            transform: controlsVisible ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform 0.2s ease"
          }}>
            ▼
          </span>
        </div>

        {/* Controls Panel */}
        {controlsVisible && (
        <div style={{ backgroundColor: "#2a2a2a", borderRadius: "0 0 8px 8px", padding: "16px", boxShadow: "0 2px 8px rgba(0,0,0,0.3)", marginTop: "-1px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "16px" }}>
            <h1 style={{ fontSize: "20px", fontWeight: "bold", color: "#fff" }}>Spiking SNN (WebGPU)</h1>
            <div style={{ fontSize: "14px", color: "#888" }}>{fps ? `${fps} FPS` : ""}</div>
          </div>

          {error && (
            <div style={{ color: "#ff6b6b", backgroundColor: "#3a1a1a", padding: "8px", borderRadius: "4px", marginBottom: "16px", border: "1px solid #ff6b6b" }}>
              Error: {error}
            </div>
          )}

          <div style={{ marginBottom: "16px" }}>
            <div style={{ display: "flex", gap: "8px", marginBottom: "12px" }}>
              <input
                type="text"
                placeholder="./spiking_step_enhanced.onnx"
                value={modelUrl}
                readOnly
                style={{ flex: 1, padding: "8px", border: "1px solid #555", borderRadius: "4px", backgroundColor: "#333", color: "#e0e0e0" }}
              />
              <button
                onClick={() => setRunning(!running)}
                disabled={!ready}
                style={{
                  padding: "8px 16px",
                  backgroundColor: ready ? (running ? "#dc3545" : "#28a745") : "#6c757d",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: ready ? "pointer" : "not-allowed",
                  fontSize: "14px",
                  fontWeight: "500"
                }}
              >
                {!ready ? "⏳ Loading..." : running ? "⏸️ Pause" : "▶️ Run"}
              </button>
            </div>

            <div>
              <label style={{ display: "block", fontSize: "14px", fontWeight: "500", marginBottom: "4px", color: "#e0e0e0" }}>
                Number of neurons
              </label>
              <select
                value={resolution}
                onChange={(e) => setResolution(parseInt(e.target.value))}
                disabled={running}
                style={{
                  width: "100%",
                  padding: "8px",
                  border: "1px solid #555",
                  borderRadius: "4px",
                  fontSize: "14px",
                  backgroundColor: running ? "#2a2a2a" : "#333",
                  color: "#e0e0e0",
                  cursor: running ? "not-allowed" : "pointer"
                }}
              >
                <option value={256}>256 × 256 (65K neurons)</option>
                <option value={512}>512 × 512 (262K neurons)</option>
                <option value={1024}>1024 × 1024 (1M neurons)</option>
              </select>
              {running && (
                <div style={{ fontSize: "12px", color: "#666", marginTop: "4px", fontStyle: "italic" }}>
                  Stop simulation to change resolution
                </div>
              )}
            </div>
          </div>

          {/* Neural Parameter Controls */}
          <div style={{ marginBottom: "16px" }}>
            <h3 style={{ fontSize: "16px", fontWeight: "600", marginBottom: "12px", borderBottom: "1px solid #555", paddingBottom: "4px", color: "#fff" }}>
              Neural Dynamics
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "16px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <label style={{ fontSize: "11px", fontWeight: "500", minWidth: "80px", color: "#ccc" }}>Voltage Decay</label>
                <input type="number" step="0.001" value={decay} onChange={(e) => setDecay(parseFloat(e.target.value) || 0)} style={{ width: "80px", padding: "2px 4px", fontSize: "11px", border: "1px solid #555", borderRadius: "2px", backgroundColor: "#333", color: "#e0e0e0" }} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <label style={{ fontSize: "11px", fontWeight: "500", minWidth: "80px", color: "#ccc" }}>Spike Firing Threshold</label>
                <input type="number" step="0.01" value={firingThreshold} onChange={(e) => setFiringThreshold(parseFloat(e.target.value) || 0)} style={{ width: "80px", padding: "2px 4px", fontSize: "11px", border: "1px solid #555", borderRadius: "2px", backgroundColor: "#333", color: "#e0e0e0" }} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <label style={{ fontSize: "11px", fontWeight: "500", minWidth: "80px", color: "#ccc" }}>Voltage Reset Point</label>
                <input type="number" step="0.01" value={resetPoint} onChange={(e) => setResetPoint(parseFloat(e.target.value) || 0)} style={{ width: "80px", padding: "2px 4px", fontSize: "11px", border: "1px solid #555", borderRadius: "2px", backgroundColor: "#333", color: "#e0e0e0" }} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <label style={{ fontSize: "11px", fontWeight: "500", minWidth: "80px", color: "#ccc" }}>Spike Dropout Probability</label>
                <input type="number" step="0.01" value={dropProb} onChange={(e) => setDropProb(parseFloat(e.target.value) || 0)} style={{ width: "80px", padding: "2px 4px", fontSize: "11px", border: "1px solid #555", borderRadius: "2px", backgroundColor: "#333", color: "#e0e0e0" }} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <label style={{ fontSize: "11px", fontWeight: "500", minWidth: "80px", color: "#ccc" }}>Lower Voltage Threshold</label>
                <input type="number" step="0.001" value={lowerThreshold} onChange={(e) => setLowerThreshold(parseFloat(e.target.value) || 0)} style={{ width: "80px", padding: "2px 4px", fontSize: "11px", border: "1px solid #555", borderRadius: "2px", backgroundColor: "#333", color: "#e0e0e0" }} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <label style={{ fontSize: "11px", fontWeight: "500", minWidth: "80px", color: "#ccc" }}>Local Excitation</label>
                <input type="number" step="0.1" value={excLocalScale} onChange={(e) => setExcLocalScale(parseFloat(e.target.value) || 0)} style={{ width: "80px", padding: "2px 4px", fontSize: "11px", border: "1px solid #555", borderRadius: "2px", backgroundColor: "#333", color: "#e0e0e0" }} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <label style={{ fontSize: "11px", fontWeight: "500", minWidth: "80px", color: "#ccc" }}>Local Inhibition</label>
                <input type="number" step="0.1" value={inhLocalScale} onChange={(e) => setInhLocalScale(parseFloat(e.target.value) || 0)} style={{ width: "80px", padding: "2px 4px", fontSize: "11px", border: "1px solid #555", borderRadius: "2px", backgroundColor: "#333", color: "#e0e0e0" }} />
              </div>
            </div>
          </div>

          {/* Brush Controls */}
          <div style={{ marginBottom: "16px", padding: "12px", backgroundColor: "#333", borderRadius: "4px" }}>
            <h3 style={{ fontSize: "16px", fontWeight: "600", marginBottom: "12px", color: "#fff" }}>Mouse Brush</h3>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", marginBottom: "12px" }}>
              <div>
                <label style={{ display: "block", fontSize: "14px", fontWeight: "500", marginBottom: "4px", color: "#ccc" }}>
                  Strength
                </label>
                <input
                  type="number"
                  min="0.1"
                  max="10.0"
                  step="0.1"
                  value={brushStrength}
                  onChange={(e) => setBrushStrength(parseFloat(e.target.value) || 0.1)}
                  style={{
                    width: "100%",
                    padding: "4px 8px",
                    border: "1px solid #555",
                    borderRadius: "4px",
                    fontSize: "14px",
                    backgroundColor: "#2a2a2a",
                    color: "#e0e0e0"
                  }}
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "14px", fontWeight: "500", marginBottom: "4px", color: "#ccc" }}>
                  Radius (px)
                </label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  step="1"
                  value={brushRadius}
                  onChange={(e) => setBrushRadius(parseInt(e.target.value) || 1)}
                  style={{
                    width: "100%",
                    padding: "4px 8px",
                    border: "1px solid #555",
                    borderRadius: "4px",
                    fontSize: "14px",
                    backgroundColor: "#2a2a2a",
                    color: "#e0e0e0"
                  }}
                />
              </div>
            </div>

            <div>
              <label style={{ display: "block", fontSize: "14px", fontWeight: "500", marginBottom: "8px", color: "#ccc" }}>
                Mode
              </label>
              <div style={{ display: "flex", gap: "16px" }}>
                <label style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <input
                    type="radio"
                    name="brushSign"
                    value="exc"
                    checked={brushSign === "exc"}
                    onChange={(e) => setBrushSign(e.target.value)}
                  />
                  <span style={{ color: "#28a745", fontWeight: "500" }}>Excitation</span>
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <input
                    type="radio"
                    name="brushSign"
                    value="inh"
                    checked={brushSign === "inh"}
                    onChange={(e) => setBrushSign(e.target.value)}
                  />
                  <span style={{ color: "#007bff", fontWeight: "500" }}>Inhibition</span>
                </label>
              </div>
            </div>
          </div>

          <button
            onClick={resetState}
            style={{
              padding: "8px 16px",
              backgroundColor: "#555",
              color: "white",
              border: "1px solid #666",
              borderRadius: "4px",
              cursor: "pointer"
            }}
          >
            Reset State
          </button>

          <div style={{
            backgroundColor: "#333",
            border: "1px solid #555",
            padding: "12px",
            borderRadius: "4px",
            marginTop: "16px",
            fontSize: "14px",
            color: "#ccc"
          }}>
            <strong style={{ color: "#fff" }}>Status:</strong> {ready ? "Model loaded ✓" : "Loading model..."}<br/>
            <strong style={{ color: "#fff" }}>Controls:</strong><br/>
            • Click and drag to stimulate neurons<br/>
            • Green areas = excitation/spikes<br/>
            • Blue areas = membrane potential<br/>
            • Adjust brush strength for stronger effects<br/>
            • Lower threshold for easier spiking
          </div>
        </div>
        )}
      </div>
    </div>
  );
}