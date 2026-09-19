
import { useEffect, useState } from "react";
import DeckGL from "@deck.gl/react";
import { PolygonLayer } from "@deck.gl/layers";
import { Map } from "react-map-gl/mapbox";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

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

/* =========================================================
   FIELD MAP
========================================================= */

function FieldMap({ patches }) {
  if (!patches || patches.length === 0) {
    return (
      <div className="field-map">
        <div className="empty">
          No spatial prediction data available.
        </div>
      </div>
    );
  }

  const georeferenced = patches.some(
    (patch) => patch.longitude !== undefined
  );

  const columns =
    Math.max(...patches.map((patch) => patch.column / 32), 0) + 1;

  const rows =
    Math.max(...patches.map((patch) => patch.row / 32), 0) + 1;

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

        return [
          [x, y],
          [x + 1, y],
          [x + 1, y + 1],
          [x, y + 1],
        ];
      },

      getFillColor: (patch) => {
        const classIndex = patch.predicted_label ?? 4;
        const confidence = patch.confidence ?? 0;

        const baseColor =
          CLASS_COLORS[classIndex] || CLASS_COLORS[4];

        return [
          ...baseColor,
          100 + confidence * 155,
        ];
      },

      getLineColor: [255, 255, 255, 180],
      getLineWidth: 1,
      pickable: true,
    }),
  ];

  const first = patches[0];

  const viewState = georeferenced
    ? {
        target: [first.longitude, first.latitude, 0],
        zoom: 14,
      }
    : {
        target: [columns / 2, rows / 2, 0],
        zoom: Math.min(
          8,
          8 - Math.log2(Math.max(columns, rows))
        ),
      };

  return (
    <div className="field-map">
      <DeckGL
        initialViewState={viewState}
        controller={true}
        layers={layers}
        getTooltip={({ object }) =>
          object &&
          `${object.predicted_class} · ${(
            object.confidence * 100
          ).toFixed(1)}%`
        }
      >
        {MAPBOX_TOKEN ? (
          <Map
            mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
            mapboxAccessToken={MAPBOX_TOKEN}
          />
        ) : null}
      </DeckGL>

      <div className="map-note">
        {georeferenced
          ? "WGS84 raster coordinates"
          : "Local pixel-coordinate view"}
        {MAPBOX_TOKEN
          ? " · Mapbox basemap enabled"
          : ""}
      </div>
    </div>
  );
}

/* =========================================================
   ANALYTICS
========================================================= */

function Analytics({ result }) {
  const summary = result.risk_summary || {};
  const spectralAlerts = result.patch_predictions.filter(
    (patch) => patch.spectral_anomaly_level === "high"
  ).length;

  const affected =
    summary.patches_at_risk ??
    result.patch_predictions.filter(
      (patch) => patch.predicted_label !== 4
    ).length;

  const riskPercentage =
    summary.patch_risk_percent ??
    (affected / result.patch_count) * 100;

  const metadata =
    result.geospatial_metadata || {};

  return (
    <div className="analytics">
      <div>
        <strong>{result.patch_count}</strong>
        <span>total patches</span>
      </div>

      <div>
        <strong>{affected}</strong>
        <span>patches at risk</span>
      </div>

      <div>
        <strong>
          {riskPercentage.toFixed(1)}%
        </strong>
        <span>estimated patch risk</span>
      </div>

      <div>
        <strong>
          {summary.at_risk_acres === undefined
            ? "n/a"
            : `${summary.at_risk_acres.toFixed(2)} ac`}
        </strong>

        <span>
          {summary.at_risk_acres === undefined
            ? "acreage requires georeferencing"
            : `of ${summary.total_acres.toFixed(
                2
              )} total acres`}
        </span>
      </div>

      <div>
        <strong>{spectralAlerts}</strong>
        <span>high spectral deviations</span>
      </div>

      {metadata.crs && (
        <p className="map-meta">
          CRS: {metadata.crs} · geospatial bounds
          preserved
        </p>
      )}
    </div>
  );
}

/* =========================================================
   TEMPORAL TIMELINE
========================================================= */

function AnalysisTimeline({
  observations,
  activeIndex,
  onSelect,
}) {
  if (!observations || observations.length < 2) {
    return null;
  }

  const active =
    observations[activeIndex] || observations[0];

  return (
    <div className="timeline">
      <div>
        <span className="kicker">
          03 / HISTORY
        </span>

        <strong>
          {active.observationDate}
        </strong>
      </div>

      <input
        type="range"
        min="0"
        max={observations.length - 1}
        value={activeIndex}
        onChange={(event) =>
          onSelect(Number(event.target.value))
        }
      />

      <div className="timeline-labels">
        <span>
          {observations[0].observationDate}
        </span>

        <span>
          {
            observations[
              observations.length - 1
            ].observationDate
          }
        </span>
      </div>

      <div className="timeline-summary">
        <div>
          <strong>
            {active.predicted_class ||
              active.predicted_class_name ||
              "Unknown"}
          </strong>

          <span>
            predicted condition
          </span>
        </div>

        <div>
          <strong>
            {active.confidence_percent !==
            undefined
              ? `${active.confidence_percent.toFixed(
                  1
                )}%`
              : active.confidence !==
                undefined
              ? `${(
                  active.confidence * 100
                ).toFixed(1)}%`
              : "n/a"}
          </strong>

          <span>confidence</span>
        </div>

        <div>
          <strong>
            {active.patch_count ?? "n/a"}
          </strong>

          <span>patches analyzed</span>
        </div>
      </div>
    </div>
  );
}

function downloadAnalysisReport(result, observationDate) {
  const report = {
    generated_at: new Date().toISOString(),
    observation_date: observationDate,
    filename: result.filename,
    field_prediction: {
      predicted_class: result.predicted_class,
      confidence_percent: result.confidence_percent,
      probabilities: result.probabilities,
    },
    risk_summary: result.risk_summary,
    geospatial_metadata: result.geospatial_metadata,
    patch_predictions: result.patch_predictions,
    note: "Decision-support output only; human agronomic review is required.",
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `terraspectra-${observationDate || "analysis"}-report.json`;
  link.click();
  URL.revokeObjectURL(url);
}

/* =========================================================
   MAIN APP
========================================================= */

function App() {
  const [file, setFile] = useState(null);

  const [result, setResult] =
    useState(null);

  const [observationDate, setObservationDate] =
    useState(() =>
      new Date()
        .toISOString()
        .slice(0, 10)
    );

  const [observations, setObservations] =
    useState([]);

  const [activeObservation, setActiveObservation] =
    useState(0);

  const [temporalData, setTemporalData] =
    useState(null);

  const [temporalLoading, setTemporalLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [online, setOnline] =
    useState(false);

  /* =======================================================
     CHECK BACKEND + LOAD TEMPORAL ANALYSIS
  ======================================================= */

  useEffect(() => {
    checkBackend();

    loadTemporalAnalysis();
  }, []);

  async function checkBackend() {
    try {
      const response = await fetch(
        `${API_URL}/health`
      );

      const data = await response.json();

      setOnline(
        Boolean(data.model_available)
      );
    } catch (requestError) {
      console.error(
        "Backend connection error:",
        requestError
      );

      setOnline(false);
    }
  }

  /* =======================================================
     TEMPORAL API
  ======================================================= */

  async function loadTemporalAnalysis() {
    setTemporalLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/temporal-analysis`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Temporal analysis request failed"
        );
      }

      console.log(
        "Temporal analysis:",
        data
      );

      setTemporalData(data);
    } catch (requestError) {
      console.error(
        "Temporal analysis error:",
        requestError
      );

      setTemporalData(null);
    } finally {
      setTemporalLoading(false);
    }
  }

  /* =======================================================
     FIELD ANALYSIS
  ======================================================= */

  async function analyze() {
    if (!file) {
      setError(
        "Select a .mat, .h5, .hdf5, .tif, or .tiff file first."
      );

      return;
    }

    setError("");

    const body = new FormData();

    body.append("file", file);

    try {
      const response = await fetch(
        `${API_URL}/predict-geospatial`,
        {
          method: "POST",
          body,
        }
      );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Analysis failed"
        );
      }

      setResult(data);

      setObservations((previous) => {
        const next =
          previous.filter(
            (item) =>
              item.observationDate !==
              observationDate
          );

        next.push({
          ...data,
          observationDate,
        });

        next.sort((left, right) =>
          left.observationDate.localeCompare(
            right.observationDate
          )
        );

        const newIndex =
          next.findIndex(
            (item) =>
              item.observationDate ===
              observationDate
          );

        setActiveObservation(
          newIndex
        );

        return next;
      });

      /*
       * Refresh temporal API after a
       * successful prediction.
       */
      await loadTemporalAnalysis();
    } catch (requestError) {
      setError(
        requestError.message
      );
    }
  }

  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <main className="shell">

      {/* HEADER */}

      <header className="topbar">
        <div>
          <div className="kicker">
            AGRICULTURAL INTELLIGENCE /
            HYPERSPECTRAL ANALYSIS
          </div>

          <h1>
            TerraSpectra
          </h1>
        </div>

        <span
          className={`status ${
            online
              ? "online"
              : "offline"
          }`}
        >
          {online
            ? "MODEL ONLINE"
            : "MODEL OFFLINE"}
        </span>
      </header>

      {/* HERO */}

      <section className="hero">
        <div>
          <p className="kicker">
            FIELD SIGNALS
          </p>

          <h2>
            Read plant stress
            <br />
            at pixel scale.
          </h2>

          <p className="lede">
            Upload a hyperspectral
            MATLAB, HDF5, or GeoTIFF
            cube to transform spectral
            signatures into a disease
            distribution map.
          </p>
        </div>

        <div className="metric">
          <strong>
            46.88%
          </strong>

          <span>
            current held-out accuracy
          </span>
        </div>
      </section>

      {/* WORKSPACE */}

      <section className="workspace">

        {/* CONTROL PANEL */}

        <div className="control-panel">

          <p className="kicker">
            01 / INGEST
          </p>

          <h3>
            Analyze a field cube
          </h3>

          <label className="dropzone">

            <input
              type="file"
              accept=".mat,.h5,.hdf5,.tif,.tiff"
              onChange={(event) =>
                setFile(
                  event.target.files[0]
                )
              }
            />

            <span>
              {file
                ? file.name
                : "Choose a geospatial cube"}
            </span>

            <small>
              MAT / HDF5 / GeoTIFF ·
              shared PCA · 32 × 32 tiling
            </small>

          </label>

          <button
            onClick={analyze}
          >
            Run field analysis
          </button>

          {error && (
            <p className="error">
              {error}
            </p>
          )}

        </div>

        {/* RESULT PANEL */}

        <div className="result-panel">

          <p className="kicker">
            02 / SIGNAL MAP
          </p>

          {result ? (
            <>

              <div className="result-head">

                <div>

                  <h3>
                    {result.predicted_class}
                  </h3>

                  <span>
                    {result.patch_count}
                    {" "}
                    patches analyzed
                  </span>

                </div>

                <strong>
                  {result.confidence_percent.toFixed(
                    1
                  )}
                  %
                </strong>

              </div>

              {/* PROBABILITIES */}

              <div className="probabilities">

                {Object.entries(
                  result.probabilities
                ).map(
                  ([name, value]) => (
                    <div
                      className="probability"
                      key={name}
                    >

                      <span>
                        {name}
                      </span>

                      <span>
                        {(
                          value * 100
                        ).toFixed(1)}
                        %
                      </span>

                      <i>
                        <b
                          style={{
                            width: `${
                              value * 100
                            }%`,
                          }}
                        />
                      </i>

                    </div>
                  )
                )}

              </div>

              {/* ANALYTICS */}

              <Analytics
                result={result}
              />

              <button
                className="report-button"
                onClick={() =>
                  downloadAnalysisReport(
                    result,
                    observationDate
                  )
                }
              >
                Download analysis report
              </button>

              {/* FIELD MAP */}

              <FieldMap
                patches={
                  result.patch_predictions
                }
              />

            </>
          ) : (
            <div className="empty">
              Upload a field cube to
              reveal its spatial disease
              pattern.
            </div>
          )}

        </div>

      </section>

      {/* FOOTER */}

      <footer>
        <span>
          Shared-PCA 3D-CNN ·
          calibrated inference
        </span>

        <span>
          Pixel coordinates ·
          human review required
        </span>
      </footer>

      {/* =================================================
          TEMPORAL HISTORY
      ================================================= */}

      <section className="history-panel">

        <label className="date-input">

          <span>
            Observation date
          </span>

          <input
            type="date"
            value={observationDate}
            onChange={(event) =>
              setObservationDate(
                event.target.value
              )
            }
          />

        </label>

        {/* TEMPORAL API STATUS */}

<div className="temporal-status">

  <span className="kicker">
    TEMPORAL ANALYSIS API
  </span>

  {temporalLoading ? (
    <span>
      Loading temporal data...
    </span>
  ) : temporalData ? (
    <>
      <span>
        Temporal analysis connected
      </span>

      {temporalData.records &&
        temporalData.records.length > 0 && (
          <div className="temporal-records">

            <h3>
              Observation History
            </h3>

            <div className="temporal-record-list">

              {temporalData.records.map(
                (record, index) => (
                  <div
                    className="temporal-record"
                    key={index}
                  >

                    {Object.entries(record).map(
                      ([key, value]) => (
                        <div
                          className="temporal-field"
                          key={key}
                        >
                          <span>
                            {key.replace(
                              /_/g,
                              " "
                            )}
                          </span>

                          <strong>
                            {value ?? "n/a"}
                          </strong>
                        </div>
                      )
                    )}

                  </div>
                )
              )}

            </div>

          </div>
        )}

    </>
  ) : (
    <span>
      No temporal data available
    </span>
  )}

</div>

        {/* TIMELINE */}

        <AnalysisTimeline
          observations={
            observations
          }
          activeIndex={
            activeObservation
          }
          onSelect={(index) => {
            setActiveObservation(
              index
            );

            setResult(
              observations[index]
            );
          }}
        />

      </section>

    </main>
  );
}

export default App;
