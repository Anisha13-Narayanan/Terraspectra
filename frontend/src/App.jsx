import { useEffect, useState } from "react";
import DeckGL from "@deck.gl/react";
import { PolygonLayer } from "@deck.gl/layers";
import { Map } from "react-map-gl/mapbox";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;
const CLASS_COLORS = [
  [193, 91, 74],
  [212, 154, 58],
  [123, 98, 153],
  [61, 120, 144],
  [91, 143, 98],
];
const CLASS_NAMES = [
  "Alternaria alternata",
  "Alternaria solani",
  "Botrytis cinerea",
  "Fusarium oxysporum",
  "Healthy",
];

function FieldMap({ patches }) {
  const georeferenced = patches.some((patch) => patch.longitude !== undefined);
  const columns = Math.max(...patches.map((patch) => patch.column / 32), 0) + 1;
  const rows = Math.max(...patches.map((patch) => patch.row / 32), 0) + 1;
  const layers = [
    new PolygonLayer({
      id: "disease-patches",
      data: patches,
      getPolygon: (patch) => {
        if (georeferenced && patch.polygon) {
          return patch.polygon;
        }
        const x = patch.column / 32;
        const y = patch.row / 32;
        return [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]];
      },
      getFillColor: (patch) => [...CLASS_COLORS[patch.predicted_label], 100 + patch.confidence * 155],
      getLineColor: [255, 255, 255, 180],
      getLineWidth: 1,
      pickable: true,
    }),
  ];
  const first = patches[0];
  const viewState = georeferenced
    ? { target: [first.longitude, first.latitude, 0], zoom: 14 }
    : { target: [columns / 2, rows / 2, 0], zoom: Math.min(8, 8 - Math.log2(Math.max(columns, rows))) };

  return (
    <div className="field-map">
      <DeckGL initialViewState={viewState} controller={true} layers={layers} getTooltip={({ object }) => object && `${object.predicted_class} · ${(object.confidence * 100).toFixed(1)}%`}>
        {MAPBOX_TOKEN ? <Map mapStyle="mapbox://styles/mapbox/satellite-streets-v12" mapboxAccessToken={MAPBOX_TOKEN} /> : null}
      </DeckGL>
      <div className="map-note">{georeferenced ? "WGS84 raster coordinates" : "Local pixel-coordinate view"}{MAPBOX_TOKEN ? " · Mapbox basemap enabled" : ""}</div>
    </div>
  );
}

function Analytics({ result }) {
  const affected = result.patch_predictions.filter(
    (patch) => patch.predicted_label !== 4,
  ).length;
  const riskPercentage = (affected / result.patch_count) * 100;
  const metadata = result.geospatial_metadata || {};

  return (
    <div className="analytics">
      <div><strong>{result.patch_count}</strong><span>total patches</span></div>
      <div><strong>{affected}</strong><span>patches at risk</span></div>
      <div><strong>{riskPercentage.toFixed(1)}%</strong><span>estimated patch risk</span></div>
      <div><strong>n/a</strong><span>acreage requires spatial resolution</span></div>
      {metadata.crs && <p className="map-meta">CRS: {metadata.crs} · geospatial bounds preserved</p>}
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [online, setOnline] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/health`).then((response) => response.json()).then((data) => setOnline(data.model_available)).catch(() => setOnline(false));
  }, []);

  async function analyze() {
    if (!file) return setError("Select a .mat, .h5, .hdf5, .tif, or .tiff file first.");
    setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${API_URL}/predict-geospatial`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Analysis failed");
      setResult(data);
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  return (
    <main className="shell">
      <header className="topbar"><div><div className="kicker">AGRICULTURAL INTELLIGENCE / HYPERSPECTRAL ANALYSIS</div><h1>TerraSpectra</h1></div><span className={`status ${online ? "online" : "offline"}`}>{online ? "MODEL ONLINE" : "MODEL OFFLINE"}</span></header>
      <section className="hero"><div><p className="kicker">FIELD SIGNALS</p><h2>Read plant stress<br />at pixel scale.</h2><p className="lede">Upload a hyperspectral MATLAB, HDF5, or GeoTIFF cube to transform spectral signatures into a disease distribution map.</p></div><div className="metric"><strong>46.88%</strong><span>current held-out accuracy</span></div></section>
      <section className="workspace"><div className="control-panel"><p className="kicker">01 / INGEST</p><h3>Analyze a field cube</h3><label className="dropzone"><input type="file" accept=".mat,.h5,.hdf5,.tif,.tiff" onChange={(event) => setFile(event.target.files[0])} /><span>{file ? file.name : "Choose a geospatial cube"}</span><small>MAT / HDF5 / GeoTIFF · shared PCA · 32 × 32 tiling</small></label><button onClick={analyze}>Run field analysis</button>{error && <p className="error">{error}</p>}</div><div className="result-panel"><p className="kicker">02 / SIGNAL MAP</p>{result ? <><div className="result-head"><div><h3>{result.predicted_class}</h3><span>{result.patch_count} patches analyzed</span></div><strong>{result.confidence_percent.toFixed(1)}%</strong></div><div className="probabilities">{Object.entries(result.probabilities).map(([name, value]) => <div className="probability" key={name}><span>{name}</span><span>{(value * 100).toFixed(1)}%</span><i><b style={{ width: `${value * 100}%` }} /></i></div>)}</div><Analytics result={result} /><FieldMap patches={result.patch_predictions} /></> : <div className="empty">Upload a field cube to reveal its spatial disease pattern.</div>}</div></section>
      <footer><span>Shared-PCA 3D-CNN · calibrated inference</span><span>Pixel coordinates · human review required</span></footer>
    </main>
  );
}

export default App;