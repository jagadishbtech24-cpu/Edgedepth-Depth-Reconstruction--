import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  AudioLines,
  Box,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Eye,
  FileImage,
  Focus,
  Gauge,
  Layers3,
  LoaderCircle,
  Menu,
  Network,
  ScanLine,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Upload,
  Waves,
} from "lucide-react";
import {
  downloadDepth,
  getHealth,
  getSurface,
  listScenes,
  reconstructScene,
} from "./services/backend";

const METHODS = [
  "Nearest",
  "Bilinear",
  "Bicubic",
  "Joint bilateral",
  "Guided filter",
  "EdgeGuideNet",
  "DepthUNet",
  "EdgeDepth hybrid",
];
const NAV_ITEMS = [
  { id: "overview", label: "Overview", icon: Layers3 },
  { id: "metrics", label: "Evaluation", icon: Gauge },
  { id: "boundaries", label: "Boundary analysis", icon: ScanLine },
  { id: "surface", label: "3D inspection", icon: Box },
  { id: "guide", label: "Model notes", icon: CircleHelp },
];
const initialSettings = {
  scale: 4,
  noise_sigma: 0,
  sparsity: 0,
  lidar_pattern: "grid",
  simulate_holes: false,
  filter: "guided",
  anti_texture: true,
  sharpen: 0.25,
  seed: 42,
};

function formatNumber(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function csvForResult(result) {
  const rows = [
    ["Method", "RMSE (m)", "MAE (m)", "AbsRel", "SSIM", "Boundary F1", "Latency (ms)"],
  ];
  for (const method of result.methods) {
    const metric = result.metrics?.[method];
    const boundary = result.boundary_metrics?.[method];
    rows.push([
      method,
      metric?.RMSE,
      metric?.MAE,
      metric?.AbsRel,
      metric?.SSIM,
      boundary?.f1_score,
      result.timings_ms?.[method],
    ]);
  }
  return rows.map((row) => row.map((cell) => {
    const value = cell == null ? "" : String(cell);
    return `"${value.replaceAll('"', '""')}"`;
  }).join(",")).join("\n");
}

function Plot({ data, layout, className = "" }) {
  const host = useRef(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!host.current || !data) return undefined;
    let active = true;
    let plotly;
    const node = host.current;
    setError("");
    import("plotly.js-dist-min")
      .then(({ default: module }) => {
        if (!active) return;
        plotly = module;
        return module.react(node, data, layout, {
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ["lasso2d", "select2d"],
        });
      })
      .catch((cause) => active && setError(`Could not render chart: ${cause.message}`));
    return () => {
      active = false;
      if (plotly) plotly.purge(node);
    };
  }, [data, layout]);
  if (error) return <div className="inline-error">{error}</div>;
  return <div className={`plot-host ${className}`} ref={host} />;
}

function PlotSurface({ result, method }) {
  const host = useRef(null);
  const [surface, setSurface] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setSurface(null);
    setError("");
    getSurface(result.result_id, method)
      .then((value) => active && setSurface(value))
      .catch((cause) => active && setError(cause.message));
    return () => { active = false; };
  }, [result.result_id, method]);

  useEffect(() => {
    if (!host.current || !surface) return undefined;
    let active = true;
    let plotly;
    const node = host.current;
    const z = surface.z;
    const trace = {
      type: "surface",
      x: surface.x,
      y: surface.y,
      z,
      colorscale: "Turbo",
      showscale: true,
      colorbar: {
        title: { text: "Depth (m)", font: { color: "#b3c0c5", size: 11 } },
        tickfont: { color: "#93a2a8" },
        outlinecolor: "#263238",
        thickness: 10,
      },
      contours: {
        z: {
          show: true,
          usecolormap: true,
          highlightcolor: "#62ded5",
          project: { z: true },
        },
      },
      connectgaps: false,
      hovertemplate: "x %{x}<br>y %{y}<br>depth %{z:.3f} m<extra></extra>",
    };
    import("plotly.js-dist-min")
      .then(({ default: module }) => {
        if (!active) return;
        plotly = module;
        return module.react(node, [trace], {
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#aebbc0", family: "Inter, sans-serif", size: 11 },
          margin: { l: 0, r: 12, t: 12, b: 0 },
          scene: {
            bgcolor: "rgba(0,0,0,0)",
            xaxis: { title: "Image x (px)", gridcolor: "#263238", zerolinecolor: "#263238" },
            yaxis: { title: "Image y (px)", gridcolor: "#263238", zerolinecolor: "#263238" },
            zaxis: { title: "Depth (m)", gridcolor: "#263238", zerolinecolor: "#263238" },
            aspectmode: "data",
            camera: { eye: { x: 1.65, y: -1.65, z: 1.1 } },
          },
        }, { responsive: true, displaylogo: false });
      })
      .catch((cause) => active && setError(`Could not render 3D surface: ${cause.message}`));
    return () => {
      active = false;
      if (plotly) plotly.purge(node);
    };
  }, [surface]);

  if (error) return <div className="inline-error">{error}</div>;
  if (!surface) return <div className="plot-loading"><LoaderCircle className="spin" /> Preparing depth surface…</div>;
  return <div className="surface-host" ref={host} />;
}

function MetricChart({ result }) {
  const methods = result.methods.filter((method) => result.metrics?.[method]);
  const traces = ["RMSE", "MAE", "Edge_RMSE"].map((metric, index) => ({
    type: "bar",
    name: metric.replace("_", " "),
    x: methods,
    y: methods.map((method) => result.metrics[method][metric]),
    marker: { color: ["#49c8c0", "#7194a0", "#c79c64"][index], opacity: 0.9 },
    hovertemplate: "%{x}<br>%{y:.4f} m<extra>%{fullData.name}</extra>",
  }));
  const layout = {
    barmode: "group",
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#aebbc0", family: "Inter, sans-serif", size: 11 },
    margin: { l: 48, r: 18, t: 12, b: 100 },
    legend: { orientation: "h", y: 1.16, x: 0, font: { color: "#aebbc0" } },
    xaxis: { tickangle: -30, gridcolor: "rgba(0,0,0,0)" },
    yaxis: { title: "Error (m)", gridcolor: "#202a2f", zerolinecolor: "#39474d" },
  };
  return <Plot data={traces} layout={layout} />;
}

function App() {
  const [activePage, setActivePage] = useState("overview");
  const [source, setSource] = useState("presets");
  const [scenes, setScenes] = useState([]);
  const [sceneKey, setSceneKey] = useState("");
  const [settings, setSettings] = useState(initialSettings);
  const [sensorIsLowResolution, setSensorIsLowResolution] = useState(false);
  const [rgbFile, setRgbFile] = useState(null);
  const [depthFile, setDepthFile] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);
  const [selectedMethod, setSelectedMethod] = useState("EdgeDepth hybrid");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    if (source === "upload") {
      setScenes([]);
      setSceneKey("");
      setResult(null);
      return;
    }
    let active = true;
    setScenes([]);
    setSceneKey("");
    listScenes(source)
      .then((items) => {
        if (!active) return;
        setScenes(items);
        setSceneKey(items[0]?.key || "");
      })
      .catch((cause) => active && setError(cause.message));
    setResult(null);
    return () => { active = false; };
  }, [source]);

  const selectedScore = result?.metrics?.[selectedMethod];
  const bestMethod = useMemo(() => {
    if (!result?.metrics) return null;
    return Object.entries(result.metrics)
      .filter(([, values]) => Number.isFinite(values.RMSE))
      .sort((left, right) => left[1].RMSE - right[1].RMSE)[0]?.[0] || null;
  }, [result]);
  const updateSetting = (key, value) => {
    setResult(null);
    setSettings((current) => ({ ...current, [key]: value }));
  };

  async function runReconstruction(event) {
    event.preventDefault();
    setError("");
    if (source === "upload" && (!rgbFile || !depthFile)) {
      setError("Choose both the registered RGB image and depth map.");
      return;
    }
    if (source !== "upload" && !sceneKey) {
      setError("Choose a scene before running reconstruction.");
      return;
    }
    setLoading(true);
    try {
      const payload = await reconstructScene({
        source,
        sceneKey,
        sensorIsLowResolution,
        settings,
        rgbFile,
        depthFile,
        referenceFile,
      });
      setResult(payload);
      if (!payload.methods.includes(selectedMethod)) setSelectedMethod(payload.methods[0]);
      setActivePage("overview");
    } catch (cause) {
      setError(cause.message || "Reconstruction failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleDepthExport() {
    if (!result) return;
    setError("");
    try {
      await downloadDepth(result.result_id, selectedMethod);
    } catch (cause) {
      setError(cause.message || "Could not export the reconstructed depth map.");
    }
  }

  const pageTitle = NAV_ITEMS.find((item) => item.id === activePage)?.label || "Overview";

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Focus size={20} strokeWidth={2.2} /></div>
          <div>
            <div className="brand-name">EDGEDEPTH</div>
            <div className="brand-subtitle">SCIENTIFIC WORKSPACE</div>
          </div>
          <button className="icon-button mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Close navigation">×</button>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav className="primary-nav" aria-label="Main navigation">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              className={`nav-link ${activePage === id ? "active" : ""}`}
              key={id}
              onClick={() => { setActivePage(id); setSidebarOpen(false); }}
            >
              <Icon size={17} strokeWidth={1.8} />
              <span>{label}</span>
              {activePage === id && <span className="active-mark" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-divider" />
        <form className="experiment-form" onSubmit={runReconstruction}>
          <div className="nav-label">EXPERIMENT</div>
          <label className="field-label" htmlFor="source">Input source</label>
          <select id="source" value={source} onChange={(event) => setSource(event.target.value)}>
            <option value="presets">Curated RGB-D scenes</option>
            <option value="nyu">NYU validation</option>
            <option value="upload">Upload custom RGB-D</option>
          </select>

          {source !== "upload" ? (
            <>
              <label className="field-label" htmlFor="scene">Scene</label>
              <select id="scene" value={sceneKey} onChange={(event) => { setSceneKey(event.target.value); setResult(null); }} disabled={!scenes.length}>
                {scenes.length === 0 && <option value="">Loading scenes…</option>}
                {scenes.map((scene) => <option value={scene.key} key={scene.key}>{scene.name}</option>)}
              </select>
            </>
          ) : (
            <div className="upload-fields">
              <label className="field-label" htmlFor="rgb-upload">RGB image</label>
              <label className="file-pick" htmlFor="rgb-upload">
                <FileImage size={16} /><span>{rgbFile?.name || "Choose RGB image"}</span><Upload size={15} />
              </label>
              <input id="rgb-upload" type="file" accept=".png,.jpg,.jpeg" onChange={(event) => { setRgbFile(event.target.files?.[0] || null); setResult(null); }} />
              <label className="field-label" htmlFor="depth-upload">Depth map</label>
              <label className="file-pick" htmlFor="depth-upload">
                <Waves size={16} /><span>{depthFile?.name || "Choose depth map"}</span><Upload size={15} />
              </label>
              <input id="depth-upload" type="file" accept=".png,.tif,.tiff,.npy" onChange={(event) => { setDepthFile(event.target.files?.[0] || null); setResult(null); }} />
              <label className="toggle-row">
                <span>Depth is sensor input</span>
                <input type="checkbox" checked={sensorIsLowResolution} onChange={(event) => { setSensorIsLowResolution(event.target.checked); setResult(null); }} />
              </label>
              {sensorIsLowResolution && (
                <>
                  <label className="field-label" htmlFor="reference-upload">Optional reference</label>
                  <input id="reference-upload" className="file-native" type="file" accept=".png,.tif,.tiff,.npy" onChange={(event) => { setReferenceFile(event.target.files?.[0] || null); setResult(null); }} />
                </>
              )}
            </div>
          )}

          <div className="field-label scale-label">Sampling factor</div>
          <div className="segmented" role="group" aria-label="Sampling factor">
            {[2, 4, 8].map((scale) => (
              <button type="button" className={settings.scale === scale ? "selected" : ""} key={scale} onClick={() => updateSetting("scale", scale)}>
                {scale}×
              </button>
            ))}
          </div>

          <details className="advanced-settings">
            <summary><SlidersHorizontal size={15} /> Advanced settings <ChevronDown size={14} /></summary>
            <label className="field-label" htmlFor="filter">Hybrid refinement</label>
            <select id="filter" value={settings.filter} onChange={(event) => updateSetting("filter", event.target.value)}>
              <option value="guided">Guided filter</option>
              <option value="bilateral">Joint bilateral</option>
            </select>
            <label className="range-label" htmlFor="noise">
              <span>Relative noise</span><span>{settings.noise_sigma.toFixed(2)}</span>
            </label>
            <input id="noise" type="range" min="0" max="0.08" step="0.01" value={settings.noise_sigma} onChange={(event) => updateSetting("noise_sigma", Number(event.target.value))} />
            <label className="range-label" htmlFor="sparsity">
              <span>Random sparsity</span><span>{Math.round(settings.sparsity * 100)}%</span>
            </label>
            <input id="sparsity" type="range" min="0" max="0.2" step="0.01" value={settings.sparsity} onChange={(event) => updateSetting("sparsity", Number(event.target.value))} />
            {settings.sparsity > 0 && (
              <>
                <label className="field-label" htmlFor="pattern">Sampling pattern</label>
                <select id="pattern" value={settings.lidar_pattern} onChange={(event) => updateSetting("lidar_pattern", event.target.value)}>
                  <option value="grid">Grid</option><option value="random">Random</option><option value="scanline">Scanline</option>
                </select>
              </>
            )}
            <label className="toggle-row">
              <span>Simulate sensor holes</span>
              <input type="checkbox" checked={settings.simulate_holes} onChange={(event) => updateSetting("simulate_holes", event.target.checked)} />
            </label>
            <label className="toggle-row">
              <span>Suppress texture copying</span>
              <input type="checkbox" checked={settings.anti_texture} onChange={(event) => updateSetting("anti_texture", event.target.checked)} />
            </label>
            <label className="range-label" htmlFor="sharpen">
              <span>Boundary refinement</span><span>{settings.sharpen.toFixed(2)}</span>
            </label>
            <input id="sharpen" type="range" min="0" max="1" step="0.05" value={settings.sharpen} onChange={(event) => updateSetting("sharpen", Number(event.target.value))} />
            <label className="field-label" htmlFor="seed">Random seed</label>
            <input id="seed" type="number" min="0" max="2147483647" value={settings.seed} onChange={(event) => updateSetting("seed", Number(event.target.value))} />
          </details>

          <button className="run-button" type="submit" disabled={loading}>
            {loading ? <LoaderCircle size={16} className="spin" /> : <Activity size={16} />}
            {loading ? "Reconstructing…" : "Run reconstruction"}
          </button>
        </form>

        <div className="sidebar-footer">
          <span className={`status-dot ${health?.status === "ok" ? "online" : ""}`} />
          <span>Inference service</span>
          <span className="footer-status">{health?.device ? health.device.toUpperCase() : "CHECKING"}</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setSidebarOpen(true)} aria-label="Open navigation"><Menu size={19} /></button>
          <div className="breadcrumb">EdgeDepth <span>/</span> {pageTitle}</div>
          <div className="topbar-right">
            <span className="service-state"><span className={`status-dot ${health?.status === "ok" ? "online" : ""}`} /> API {health?.status === "ok" ? "CONNECTED" : "CHECKING"}</span>
            <div className="avatar">ED</div>
          </div>
        </header>

        <div className="page-wrap">
          <div className="page-heading">
            <div>
              <div className="eyebrow">SCIENTIFIC IMAGING <span>/</span> DEPTH RECONSTRUCTION</div>
              <h1>{activePage === "overview" ? "Reconstruction overview" : pageTitle}</h1>
              <p>{activePage === "overview"
                ? "Inspect registered RGB-D scenes, compare model outputs, and evaluate depth quality."
                : "Explore real model predictions and measurements from the selected RGB-D reconstruction."}</p>
            </div>
            {result && (
              <button className="button secondary export-button" onClick={handleDepthExport}>
                <ArrowDownToLine size={16} /> Export depth
              </button>
            )}
          </div>

          {error && <div className="alert-error" role="alert">{error}</div>}
          {result?.warnings?.map((warning) => (
            <div className="alert-warning" role="status" key={warning}>{warning}</div>
          ))}

          {!result ? (
            <section className="empty-state panel">
              <div className="empty-icon"><Network size={24} /></div>
              <div className="eyebrow">READY FOR ANALYSIS</div>
              <h2>Select a scene and run reconstruction</h2>
              <p>Choose an RGB-D source in the experiment panel. EdgeDepth will run its interpolation baselines, EdgeGuideNet, the trained DepthUNet checkpoint, and hybrid refinement on the actual input.</p>
              <div className="empty-models">
                <span><Check size={14} /> EdgeGuideNet</span>
                <span><Check size={14} /> Trained DepthUNet</span>
                <span><Check size={14} /> Classical baselines</span>
              </div>
            </section>
          ) : (
            <ResultPage
              activePage={activePage}
              result={result}
              selectedMethod={selectedMethod}
              setSelectedMethod={setSelectedMethod}
              bestMethod={bestMethod}
              onExportDepth={handleDepthExport}
            />
          )}

          <footer className="page-footer">
            <span>EDGEDEPTH <span className="footer-separator">/</span> Depth reconstruction & evaluation</span>
            <span>Model outputs are evaluated only against available reference depth.</span>
          </footer>
        </div>
      </main>
      {sidebarOpen && <button className="mobile-backdrop" onClick={() => setSidebarOpen(false)} aria-label="Close navigation" />}
    </div>
  );
}

function ResultPage({ activePage, result, selectedMethod, setSelectedMethod, bestMethod, onExportDepth }) {
  const [boundaryMethod, setBoundaryMethod] = useState("EdgeDepth hybrid");
  const [surfaceMethod, setSurfaceMethod] = useState("EdgeDepth hybrid");
  const hasMetrics = result.metrics && Object.keys(result.metrics).length > 0;
  const latency = result.timings_ms?.[selectedMethod];

  useEffect(() => {
    if (!result.methods.includes(boundaryMethod)) setBoundaryMethod(result.methods[0]);
    if (!result.methods.includes(surfaceMethod)) setSurfaceMethod(result.methods[0]);
  }, [result.result_id, result.methods, boundaryMethod, surfaceMethod]);

  if (activePage === "metrics") {
    return (
      <>
        <MetricSummary result={result} bestMethod={bestMethod} />
        <section className="panel section-panel">
          <PanelHeading eyebrow="METHOD COMPARISON" title="Depth error by reconstruction method" subtitle="Measurements are computed on finite, positive ground-truth pixels." />
          {hasMetrics ? <MetricChart result={result} /> : <NoReference />}
        </section>
        <MetricsTable result={result} />
      </>
    );
  }

  if (activePage === "boundaries") {
    return (
      <>
        <div className="toolbar">
          <div><div className="eyebrow">GEOMETRY REVIEW</div><h2>Boundary analysis</h2></div>
          <MethodSelect value={boundaryMethod} onChange={setBoundaryMethod} methods={result.methods} />
        </div>
        <div className="image-grid boundary-grid">
          <ImagePanel title="Detected depth boundaries" meta={boundaryMethod} image={result.boundary_previews[boundaryMethod]} />
          {result.affinity_preview && <ImagePanel title="Geometric affinity" meta="Hybrid anti-texture cue" image={result.affinity_preview} />}
          {result.reference_boundary_preview && <ImagePanel title="Reference boundaries" meta="Ground-truth depth" image={result.reference_boundary_preview} />}
          {result.error_previews?.[boundaryMethod] && <ImagePanel title="Absolute error" meta="Metres · valid reference pixels" image={result.error_previews[boundaryMethod]} />}
        </div>
        {hasMetrics && <BoundaryTable result={result} />}
      </>
    );
  }

  if (activePage === "surface") {
    return (
      <>
        <section className="panel section-panel">
          <div className="toolbar">
            <PanelHeading eyebrow="SPATIAL INSPECTION" title="Interactive depth surface" subtitle={`${result.scene_name} · ${result.width} × ${result.height} px · drag to rotate, scroll to zoom`} />
            <MethodSelect value={surfaceMethod} onChange={setSurfaceMethod} methods={result.methods} />
          </div>
          <PlotSurface result={result} method={surfaceMethod} />
        </section>
        <p className="quiet-note">The plot is generated from the selected model's prediction; depth and image registration are unchanged.</p>
      </>
    );
  }

  if (activePage === "guide") {
    return <MethodGuide health={result} />;
  }

  return (
    <>
      <MetricSummary result={result} bestMethod={bestMethod} />
      <section className="panel section-panel compare-section">
        <div className="section-head">
          <PanelHeading eyebrow="VISUAL COMPARISON" title="RGB guidance and reconstructed depth" subtitle="Select a method to inspect its actual prediction against the registered input." />
          <MethodSelect value={selectedMethod} onChange={setSelectedMethod} methods={result.methods} />
        </div>
        <div className="image-grid primary-compare">
          <ImagePanel title="RGB reference" meta="Registered color input" image={result.rgb_preview} />
          <ImagePanel title={selectedMethod} meta={`Depth map · ${formatNumber(latency, 1)} ms inference`} image={result.prediction_previews[selectedMethod]} />
        </div>
        <div className="method-strip">
          {result.methods.map((method) => (
            <button key={method} onClick={() => setSelectedMethod(method)} className={`method-chip ${selectedMethod === method ? "selected" : ""}`}>
              <span className="method-dot" />{method}
            </button>
          ))}
        </div>
      </section>

      <section className="image-grid secondary-compare">
        <ImagePanel title="Sensor input" meta={`${result.valid_input_samples.toLocaleString()} valid samples · ${result.scale}×`} image={result.input_depth_preview} compact />
        {result.reference_preview
          ? <ImagePanel title="Ground-truth depth" meta="Reference measurement" image={result.reference_preview} compact />
          : <div className="panel reference-empty"><div className="eyebrow">REFERENCE</div><h3>No reference supplied</h3><p>Metrics are omitted for this sensor-only input. Upload a registered reference to enable quantitative evaluation.</p></div>}
        {result.error_previews?.[selectedMethod]
          ? <ImagePanel title="Absolute error" meta="Valid pixels · metres" image={result.error_previews[selectedMethod]} compact />
          : <div className="panel reference-empty"><div className="eyebrow">EXPORT</div><h3>Prediction ready</h3><p>Download the selected model output as a metric-depth NumPy array.</p><button className="button secondary" onClick={onExportDepth}><ArrowDownToLine size={15} /> Export .npy</button></div>}
      </section>

      {hasMetrics && (
        <section className="panel section-panel">
          <PanelHeading eyebrow="QUANTITATIVE EVALUATION" title="Method performance" subtitle="Lower is better for depth error metrics. Results are specific to this scene and sampling configuration." />
          <MetricChart result={result} />
          <div className="table-actions">
            <span>{result.methods.length} reconstruction methods evaluated</span>
            <a className="button secondary" href={`data:text/csv;charset=utf-8,${encodeURIComponent(csvForResult(result))}`} download="edgedepth_metrics.csv">
              <ArrowDownToLine size={15} /> Export metrics
            </a>
          </div>
        </section>
      )}
    </>
  );
}

function MetricSummary({ result, bestMethod }) {
  const rmse = bestMethod ? result.metrics[bestMethod]?.RMSE : null;
  const primaryMethod = result.methods.includes("EdgeDepth hybrid")
    ? "EdgeDepth hybrid"
    : result.methods[0];
  const f1 = result.boundary_metrics?.[primaryMethod]?.f1_score;
  return (
    <div className="metric-grid">
      <MetricCard label="Best RMSE" value={rmse == null ? "—" : `${formatNumber(rmse, 4)} m`} note={bestMethod || "Reference required"} icon={Gauge} />
      <MetricCard label="Best method" value={bestMethod || "—"} note={bestMethod ? "Per-scene result" : "No ground truth"} icon={ShieldCheck} />
      <MetricCard label="Primary boundary F1" value={f1 == null ? "—" : formatNumber(f1, 3)} note={`${primaryMethod} · one-pixel tolerance`} icon={ScanLine} />
      <MetricCard label="Primary inference" value={`${formatNumber(result.timings_ms?.[primaryMethod], 0)} ms`} note={`${primaryMethod} · ${result.scale}× · ${result.width} × ${result.height}`} icon={Clock3} />
    </div>
  );
}

function MetricCard({ label, value, note, icon: Icon }) {
  return (
    <div className="metric-card panel">
      <div className="metric-top"><span>{label}</span><Icon size={16} /></div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </div>
  );
}

function PanelHeading({ eyebrow, title, subtitle }) {
  return (
    <div className="panel-heading">
      <div className="eyebrow">{eyebrow}</div>
      <h2>{title}</h2>
      {subtitle && <p>{subtitle}</p>}
    </div>
  );
}

function ImagePanel({ title, meta, image, compact = false }) {
  return (
    <article className={`image-panel panel ${compact ? "compact" : ""}`}>
      <div className="image-panel-heading">
        <div><h3>{title}</h3><span>{meta}</span></div>
        <Eye size={16} />
      </div>
      <div className="image-frame">
        {image ? <img src={image} alt={title} /> : <div className="image-placeholder">No image available</div>}
      </div>
    </article>
  );
}

function MethodSelect({ methods, value, onChange }) {
  return (
    <label className="method-select">
      <span>METHOD</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {methods.map((method) => <option key={method} value={method}>{method}</option>)}
      </select>
    </label>
  );
}

function NoReference() {
  return <div className="no-reference"><Gauge size={18} /><span>No reference depth is available for this input. Quantitative metrics are intentionally not shown.</span></div>;
}

function MetricsTable({ result }) {
  if (!result.metrics || !Object.keys(result.metrics).length) {
    return <section className="panel section-panel"><PanelHeading eyebrow="METRIC REGISTER" title="Per-method measurements" /><NoReference /></section>;
  }
  return (
    <section className="panel section-panel">
      <div className="table-title-row">
        <PanelHeading eyebrow="METRIC REGISTER" title="Per-method measurements" subtitle="All depth errors are reported in metres." />
        <a className="button secondary" href={`data:text/csv;charset=utf-8,${encodeURIComponent(csvForResult(result))}`} download="edgedepth_metrics.csv"><ArrowDownToLine size={15} /> Export CSV</a>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Method</th><th>RMSE (m)</th><th>MAE (m)</th><th>AbsRel</th><th>SSIM</th><th>Edge RMSE</th><th>Boundary F1</th><th>Latency</th></tr></thead>
          <tbody>{result.methods.map((method) => {
            const metric = result.metrics[method] || {};
            const boundary = result.boundary_metrics?.[method] || {};
            return <tr key={method}><td><span className="table-method-dot" />{method}</td><td>{formatNumber(metric.RMSE, 4)}</td><td>{formatNumber(metric.MAE, 4)}</td><td>{formatNumber(metric.AbsRel, 4)}</td><td>{formatNumber(metric.SSIM, 4)}</td><td>{formatNumber(metric.Edge_RMSE, 4)}</td><td>{formatNumber(boundary.f1_score, 3)}</td><td>{formatNumber(result.timings_ms?.[method], 1)} ms</td></tr>;
          })}</tbody>
        </table>
      </div>
      <p className="quiet-note">Metrics use finite, positive reference pixels; boundary matching uses one-pixel tolerance.</p>
    </section>
  );
}

function BoundaryTable({ result }) {
  return (
    <section className="panel section-panel">
      <PanelHeading eyebrow="BOUNDARY SCORES" title="Silhouette localization" subtitle="Depth discontinuities are compared against the reference with one-pixel matching tolerance." />
      <div className="boundary-stats">
        {["EdgeGuideNet", "DepthUNet", "EdgeDepth hybrid"].filter((method) => result.methods.includes(method)).map((method) => {
          const value = result.boundary_metrics?.[method];
          return <div className="boundary-stat" key={method}><span>{method}</span><strong>{value ? formatNumber(value.f1_score, 3) : "—"}</strong><small>Precision {value ? formatNumber(value.precision, 3) : "—"} · Recall {value ? formatNumber(value.recall, 3) : "—"}</small></div>;
        })}
      </div>
    </section>
  );
}

function MethodGuide() {
  const methods = [
    ["Interpolation", "Nearest, bilinear and bicubic baselines resize depth samples without RGB edge guidance."],
    ["Joint bilateral / guided filter", "Classical RGB-guided methods use local color structure; image texture may leak into geometry."],
    ["EdgeGuideNet", "The existing RGB/depth residual network. It remains independent and is not replaced by the Kaggle model."],
    ["DepthUNet", "A separately loaded five-channel checkpoint: RGB, depth in metres divided by 10, and binary mask at 240×320. Checkpoint-derived architecture, strict state loading."],
    ["EdgeDepth hybrid", "EdgeGuideNet prediction refined with a heuristic RGB/depth affinity, selected classical filter and optional boundary sharpening."],
  ];
  return (
    <div className="guide-grid">
      <section className="panel section-panel">
        <PanelHeading eyebrow="METHOD NOTES" title="Model and reconstruction reference" subtitle="Predictions come from the project checkpoints and existing algorithms." />
        <div className="guide-list">{methods.map(([title, body], index) => <article className="guide-item" key={title}><span className="guide-index">0{index + 1}</span><div><h3>{title}</h3><p>{body}</p></div></article>)}</div>
      </section>
      <section className="panel section-panel">
        <PanelHeading eyebrow="INTERPRETATION" title="Reading the evaluation" />
        <div className="guide-list">
          <article className="guide-item"><Activity className="guide-icon" size={17} /><div><h3>Depth errors</h3><p>RMSE and MAE are absolute errors in metres. AbsRel is relative error; lower is better.</p></div></article>
          <article className="guide-item"><ScanLine className="guide-icon" size={17} /><div><h3>Boundaries</h3><p>Edge RMSE and boundary F1 describe depth discontinuity behavior and may differ from overall pixel accuracy.</p></div></article>
          <article className="guide-item"><Clock3 className="guide-icon" size={17} /><div><h3>Latency</h3><p>Single reconstruction timings are measured in the active runtime and depend on device and input dimensions.</p></div></article>
        </div>
      </section>
    </div>
  );
}

export default App;
