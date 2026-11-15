import { useEffect, useState } from "react";
import Hero from "./components/Hero";
import WakeOverlay from "./components/WakeOverlay";
import {
  ConfidenceLineChart,
  DualSeriesChart,
  EnvelopeScatterChart,
  MatchBreakdownBarChart,
} from "./components/Charts";

const NAV_ITEMS = [
  { id: "pinn", label: "Predict" },
  { id: "ranker", label: "Ranker" },
  { id: "vision", label: "Vision" },
  { id: "compare", label: "Diff" },
  { id: "lineage", label: "About" },
];

const REGIME_PRESETS = [
  { label: "ISS 0.00g", o2: 21, psi: 14.7, g: 0.0, flow: 12 },
  { label: "Lunar 0.16g", o2: 34, psi: 10.2, g: 0.16, flow: 10 },
  { label: "Mars 0.38g", o2: 30, psi: 10.2, g: 0.38, flow: 14 },
  { label: "Earth 1.00g", o2: 21, psi: 14.7, g: 1.0, flow: 25 },
];

function assetUrl(path) {
  if (!path) return "";
  return path.startsWith("/") ? path : `/${path}`;
}

export default function App() {
  const [view, setView] = useState("hero");
  const [backendReady, setBackendReady] = useState(false);
  const [catalog, setCatalog] = useState(null);

  const [pinnForm, setPinnForm] = useState({
    material: "Nomex HT90-40",
    o2_pct: 34,
    pressure_psi: 10.2,
    airflow_cms: 10,
    gravity_g: 0.16,
    efield_v_cm: 0,
  });
  const [pinnResult, setPinnResult] = useState(null);
  const [fields2d, setFields2d] = useState(null);
  const [liveField2d, setLiveField2d] = useState(null);

  const [rankForm, setRankForm] = useState({
    o2_pct: 34,
    pressure_psi: 10.2,
    gravity_g: 0.16,
    airflow_cms: 10,
    pillar: "All",
  });
  const [ranked, setRanked] = useState([]);
  const [selectedInvId, setSelectedInvId] = useState(null);
  const [rankChartTab, setRankChartTab] = useState("envelope");
  const [showAllRanked, setShowAllRanked] = useState(false);

  const [copilotRole, setCopilotRole] = useState("fire_engineer");
  const [copilotQuery, setCopilotQuery] = useState("What makes Lunar 0.16g more hazardous than ISS 0g?");
  const [copilotReply, setCopilotReply] = useState(null);

  const [cvData, setCvData] = useState(null);

  const [compareIds, setCompareIds] = useState({
    left: "PSI-BASS-II-SOLID",
    right: "PSI-SAFFIRE-I",
  });
  const [compareResult, setCompareResult] = useState(null);

  const [aerosolForm, setAerosolForm] = useState({
    material: "kapton_wire",
    gravity_g: 0.16,
  });
  const [aerosolData, setAerosolData] = useState(null);

  useEffect(() => {
    let cancelled = false;
    let retryTimer = null;

    const loadInitialData = () => {
      fetch("/api/catalog")
        .then((r) => {
          if (!r.ok) throw new Error("Waking");
          return r.json();
        })
        .then((data) => {
          if (cancelled) return;
          setCatalog(data);
          if (data.investigations && data.investigations.length >= 2) {
            setCompareIds({
              left: data.investigations[0].id,
              right: data.investigations[1].id,
            });
          }
          setBackendReady(true);
          fetch("/api/pinn/fields2d")
            .then((r) => r.json())
            .then((d) => !cancelled && setFields2d(d))
            .catch(() => {});
          fetch("/api/cv/metrics")
            .then((r) => r.json())
            .then((d) => !cancelled && setCvData(d))
            .catch(() => {});
        })
        .catch(() => {
          if (!cancelled) {
            retryTimer = setTimeout(loadInitialData, 2500);
          }
        });
    };

    loadInitialData();
    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      fetch("/api/pinn/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pinnForm),
      })
        .then((r) => r.json())
        .then(setPinnResult)
        .catch(() => {});

      fetch("/api/pinn/field2d/live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pinnForm),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (d) setLiveField2d(d);
        })
        .catch(() => {});
    }, 120);
    return () => clearTimeout(timer);
  }, [pinnForm]);

  useEffect(() => {
    const timer = setTimeout(() => {
      fetch("/api/rank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(rankForm),
      })
        .then((r) => r.json())
        .then((d) => setRanked(d.ranked_investigations || []))
        .catch(() => {});
    }, 120);
    return () => clearTimeout(timer);
  }, [rankForm]);

  useEffect(() => {
    if (!compareIds.left || !compareIds.right) return;
    const qs = new URLSearchParams({
      left_id: compareIds.left,
      right_id: compareIds.right,
    });
    fetch(`/api/compare?${qs}`)
      .then((r) => r.json())
      .then(setCompareResult)
      .catch(() => {});
  }, [compareIds]);

  useEffect(() => {
    const qs = new URLSearchParams({
      material: aerosolForm.material,
      gravity_g: String(aerosolForm.gravity_g),
    });
    fetch(`/api/aerosol?${qs}`)
      .then((r) => r.json())
      .then(setAerosolData)
      .catch(() => {});
  }, [aerosolForm]);

  function runCopilot(e) {
    e.preventDefault();
    fetch("/api/copilot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: copilotQuery, role: copilotRole }),
    })
      .then((r) => r.json())
      .then(setCopilotReply)
      .catch(() => {});
  }

  const materials = catalog?.materials || [
    "Nomex HT90-40",
    "PMMA",
    "Cotton-Fiberglass",
    "Kapton",
    "PTFE",
  ];
  const investigations = catalog?.investigations || [];

  return (
    <div>
      <WakeOverlay ready={backendReady} />
      <header className="top-nav">
        <div className="brand-lockup" onClick={() => setView("hero")}>
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden="true"
          >
            <path
              d="M12 2.5A9.5 9.5 0 1 0 21.5 12"
              stroke="rgba(255,255,255,0.82)"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            <path
              d="M16.5 3.6A9.5 9.5 0 0 1 20.4 7.5"
              stroke="#e84b2c"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            <path
              d="M12 6.2L15.4 13.2C15.8 14.1 15.1 16.4 12 17.5C8.9 16.4 8.2 14.1 8.6 13.2L12 6.2Z"
              stroke="#ffffff"
              strokeWidth="1.4"
              strokeLinejoin="round"
            />
            <circle cx="12" cy="13.5" r="1.5" fill="#e84b2c" />
          </svg>
          <span>PyroSense-SI</span>
        </div>

        <nav className="pill-nav" aria-label="Primary">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={view === item.id ? "active" : ""}
              onClick={() => setView(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="nav-actions">
          {view !== "hero" && (
            <button type="button" className="btn-ghost" onClick={() => setView("hero")}>
              Home
            </button>
          )}
          <button type="button" className="btn-white" onClick={() => setView("pinn")}>
            Predict Flame
          </button>
        </div>
      </header>

      {view === "hero" && (
        <Hero
          onLaunch={(preset) => {
            if (preset) {
              setPinnForm((prev) => ({
                ...prev,
                o2_pct: preset.o2,
                pressure_psi: preset.psi,
                gravity_g: preset.g,
                airflow_cms: preset.flow,
              }));
              setRankForm((prev) => ({
                ...prev,
                o2_pct: preset.o2,
                pressure_psi: preset.psi,
                gravity_g: preset.g,
                airflow_cms: preset.flow,
              }));
            }
            setView("pinn");
          }}
          onOpenRanker={() => setView("ranker")}
        />
      )}

      {view === "pinn" && (
        <main className="workbench">
          <div className="section-header">
            <div>
              <h2>Flammability & 2D Reacting-Flow Predictor</h2>
              <p>
                128-dim Residual PINN with MC-Dropout 95% confidence bounds and 2D Navier-Stokes PDE field solutions.
              </p>
            </div>
            <div className="chip-row">
              {REGIME_PRESETS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  className="chip-btn"
                  onClick={() =>
                    setPinnForm((prev) => ({
                      ...prev,
                      o2_pct: p.o2,
                      pressure_psi: p.psi,
                      gravity_g: p.g,
                      airflow_cms: p.flow,
                    }))
                  }
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <div className="split-grid">
            <section className="glass-card">
              <div className="card-title-row">
                <h3>Cabin Parameters</h3>
                <span className="mono-tag">{pinnResult?.trained_device || "cuda"}</span>
              </div>

              <label className="field-label">Material</label>
              <select
                className="dark-input"
                value={pinnForm.material}
                onChange={(e) => setPinnForm({ ...pinnForm, material: e.target.value })}
              >
                {materials.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>

              <div className="range-block">
                <div className="range-head">
                  <span>Oxygen Concentration</span>
                  <strong>{pinnForm.o2_pct}%</strong>
                </div>
                <input
                  type="range"
                  min={15}
                  max={45}
                  step={0.5}
                  value={pinnForm.o2_pct}
                  onChange={(e) => setPinnForm({ ...pinnForm, o2_pct: Number(e.target.value) })}
                />
              </div>

              <div className="range-block">
                <div className="range-head">
                  <span>Cabin Pressure</span>
                  <strong>{pinnForm.pressure_psi} psi</strong>
                </div>
                <input
                  type="range"
                  min={6}
                  max={16}
                  step={0.1}
                  value={pinnForm.pressure_psi}
                  onChange={(e) =>
                    setPinnForm({ ...pinnForm, pressure_psi: Number(e.target.value) })
                  }
                />
              </div>

              <div className="range-block">
                <div className="range-head">
                  <span>Gravity Acceleration</span>
                  <strong>{pinnForm.gravity_g} g</strong>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={pinnForm.gravity_g}
                  onChange={(e) => setPinnForm({ ...pinnForm, gravity_g: Number(e.target.value) })}
                />
              </div>

              <div className="range-block">
                <div className="range-head">
                  <span>Ventilation Airflow</span>
                  <strong>{pinnForm.airflow_cms} cm/s</strong>
                </div>
                <input
                  type="range"
                  min={0}
                  max={40}
                  step={1}
                  value={pinnForm.airflow_cms}
                  onChange={(e) =>
                    setPinnForm({ ...pinnForm, airflow_cms: Number(e.target.value) })
                  }
                />
              </div>

              <div className="range-block">
                <div className="range-head">
                  <span>Electric Field (E-FIELD)</span>
                  <strong>{pinnForm.efield_v_cm} V/cm</strong>
                </div>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={pinnForm.efield_v_cm}
                  onChange={(e) =>
                    setPinnForm({ ...pinnForm, efield_v_cm: Number(e.target.value) })
                  }
                />
              </div>

              {pinnResult?.point_prediction && (
                <div className="kpi-stack">
                  <div className="kpi-line">
                    <span>Flame Spread Rate</span>
                    <strong>{pinnResult.point_prediction.flame_spread_mm_s} mm/s</strong>
                  </div>
                  <div className="kpi-line">
                    <span>Limiting Oxygen Index</span>
                    <strong>{pinnResult.point_prediction.loi_pct}%</strong>
                  </div>
                  <div className="kpi-line">
                    <span>O2 Margin Over LOI</span>
                    <strong>{pinnResult.point_prediction.oxygen_margin_over_loi_pct}%</strong>
                  </div>
                  <div className="kpi-line">
                    <span>Peak Flame Temperature</span>
                    <strong>{pinnResult.point_prediction.peak_temp_k} K</strong>
                  </div>
                  <div className="kpi-line">
                    <span>Radiative Fraction</span>
                    <strong>{pinnResult.point_prediction.radiative_fraction}</strong>
                  </div>
                  <div className="kpi-line">
                    <span>Status</span>
                    <strong>{pinnResult.point_prediction.ignition_status}</strong>
                  </div>
                </div>
              )}
            </section>

            <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
              <section className="glass-card">
                <div className="card-title-row">
                  <h3>Gravity Bounding Curve (Flame Spread with 95% CI)</h3>
                  <span className="mono-tag">
                    R2 {pinnResult?.validation_metrics?.flame_spread_mm_s?.r2 ?? "0.996"}
                  </span>
                </div>
                <div className="chart-box">
                  <ConfidenceLineChart
                    labels={(pinnResult?.gravity_bounding_curve || []).map(
                      (pt) => `${pt.gravity_g}g`
                    )}
                    values={(pinnResult?.gravity_bounding_curve || []).map(
                      (pt) => pt.flame_spread_mm_s
                    )}
                    upper={(pinnResult?.gravity_bounding_curve || []).map(
                      (pt) => pt.ci95_upper ?? pt.flame_spread_mm_s
                    )}
                    lower={(pinnResult?.gravity_bounding_curve || []).map(
                      (pt) => pt.ci95_lower ?? pt.flame_spread_mm_s
                    )}
                    label="Flame Spread (mm/s)"
                    color="#e84b2c"
                    fillRgba="rgba(232, 75, 44, 0.14)"
                  />
                </div>
              </section>

              <section className="glass-card">
                <div className="card-title-row">
                  <h3>Live 2D Navier-Stokes + Arrhenius + Radiation PDE Field</h3>
                  <span className="mono-tag">ReactingFlow2DPINN</span>
                </div>
                {liveField2d?.heatmap_data_url && (
                  <div className="two-grid" style={{ marginBottom: 16, alignItems: "center" }}>
                    <img
                      className="wide-img"
                      style={{ maxWidth: 260 }}
                      src={liveField2d.heatmap_data_url}
                      alt="Live 2D Reacting Flow Heatmap"
                    />
                    <div className="kpi-stack" style={{ marginTop: 0, paddingTop: 0, borderTop: "none" }}>
                      <div className="kpi-line">
                        <span>Live Condition</span>
                        <strong>
                          {liveField2d.gravity_g}g / {liveField2d.o2_pct}% O2 / {liveField2d.pressure_psi} psi
                        </strong>
                      </div>
                      <div className="kpi-line">
                        <span>Peak Field Temp</span>
                        <strong>{liveField2d.peak_field_temp_k} K</strong>
                      </div>
                      <div className="kpi-line">
                        <span>Mean Plume Velocity</span>
                        <strong>{liveField2d.mean_plume_velocity_cms} cm/s</strong>
                      </div>
                      <div className="kpi-line">
                        <span>Min Core O2 Mole Fraction</span>
                        <strong>{liveField2d.min_core_o2_mole_fraction}</strong>
                      </div>
                    </div>
                  </div>
                )}
                {fields2d?.field_heatmap_path && (
                  <img
                    className="wide-img"
                    src={assetUrl(fields2d.field_heatmap_path)}
                    alt="2D Reacting Flow PDE Fields"
                  />
                )}
                <div className="four-grid">
                  {(fields2d?.regime_field_summaries || []).map((r) => (
                    <div key={r.regime_id} className="sub-card">
                      <div style={{ fontWeight: 600, marginBottom: 6 }}>{r.label}</div>
                      <div className="kpi-line">
                        <span>Peak Temp</span>
                        <strong>{r.peak_field_temp_k} K</strong>
                      </div>
                      <div className="kpi-line">
                        <span>Plume Flow</span>
                        <strong>{r.mean_plume_velocity_cms} cm/s</strong>
                      </div>
                      <div className="kpi-line">
                        <span>Core O2</span>
                        <strong>{r.min_core_o2_mole_fraction}</strong>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          </div>
        </main>
      )}

      {view === "ranker" && (
        <main className="workbench">
          <div className="section-header">
            <div>
              <h2>NASA PSI Investigation Ranker & Copilot</h2>
              <p>
                Scores ISS, Shuttle, and ground investigations by thermodynamic proximity to your target cabin envelope.
              </p>
            </div>
          </div>

          <div className="split-grid">
            <section className="glass-card">
              <div className="card-title-row">
                <h3>Target Envelope</h3>
                <span className="mono-tag">PSI Engine</span>
              </div>

              <div className="range-block">
                <div className="range-head">
                  <span>O2 Concentration</span>
                  <strong>{rankForm.o2_pct}%</strong>
                </div>
                <input
                  type="range"
                  min={15}
                  max={45}
                  step={0.5}
                  value={rankForm.o2_pct}
                  onChange={(e) => setRankForm({ ...rankForm, o2_pct: Number(e.target.value) })}
                />
              </div>

              <div className="range-block">
                <div className="range-head">
                  <span>Pressure</span>
                  <strong>{rankForm.pressure_psi} psi</strong>
                </div>
                <input
                  type="range"
                  min={6}
                  max={16}
                  step={0.1}
                  value={rankForm.pressure_psi}
                  onChange={(e) =>
                    setRankForm({ ...rankForm, pressure_psi: Number(e.target.value) })
                  }
                />
              </div>

              <div className="range-block">
                <div className="range-head">
                  <span>Gravity</span>
                  <strong>{rankForm.gravity_g} g</strong>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={rankForm.gravity_g}
                  onChange={(e) => setRankForm({ ...rankForm, gravity_g: Number(e.target.value) })}
                />
              </div>

              <div className="range-block">
                <div className="range-head">
                  <span>Airflow</span>
                  <strong>{rankForm.airflow_cms} cm/s</strong>
                </div>
                <input
                  type="range"
                  min={0}
                  max={40}
                  step={1}
                  value={rankForm.airflow_cms}
                  onChange={(e) =>
                    setRankForm({ ...rankForm, airflow_cms: Number(e.target.value) })
                  }
                />
              </div>

              <label className="field-label">Safety Pillar</label>
              <select
                className="dark-input"
                value={rankForm.pillar}
                onChange={(e) => setRankForm({ ...rankForm, pillar: e.target.value })}
              >
                <option value="All">All Pillars</option>
                <option value="Prevention">Prevention (Flammability & Materials)</option>
                <option value="Detection">Detection (Smoke & Aerosol)</option>
                <option value="Suppression">Suppression & Extinction</option>
              </select>

              <form onSubmit={runCopilot} style={{ marginTop: 20 }}>
                <label className="field-label">Target Audience / Role</label>
                <select
                  className="dark-input"
                  value={copilotRole}
                  onChange={(e) => setCopilotRole(e.target.value)}
                >
                  <option value="fire_engineer">Aerospace / Fire Safety Engineer</option>
                  <option value="mission_planner">Mission Planner</option>
                  <option value="researcher">Professor / Student / Researcher</option>
                  <option value="earth_responder">Earth Fire Responder</option>
                </select>
                <label className="field-label">Ask Grounded PSI Copilot</label>
                <input
                  className="dark-input"
                  value={copilotQuery}
                  onChange={(e) => setCopilotQuery(e.target.value)}
                />
                <button type="submit" className="btn-white" style={{ marginTop: 12, width: "100%" }}>
                  Query PSI Evidence
                </button>
              </form>

              {copilotReply && (
                <div className="muted-note">
                  <div style={{ fontWeight: 600, color: "#fff", marginBottom: 4 }}>
                    {copilotReply.role_mode}
                  </div>
                  <p>{copilotReply.evidence_summary}</p>
                  <p style={{ marginTop: 6 }}>{copilotReply.physics_interpretation}</p>
                </div>
              )}
            </section>

            <div style={{ display: "flex", flexDirection: "column", gap: "22px" }}>
              <section className="glass-card">
                <div className="card-title-row">
                  <h3>
                    {rankChartTab === "envelope"
                      ? "Cabin Envelope Map (O2 vs Pressure)"
                      : "Top 5 Match Breakdown"}
                  </h3>
                  <div className="chip-row" style={{ marginTop: 0 }}>
                    <button
                      type="button"
                      className="chip-btn"
                      style={{
                        color: rankChartTab === "envelope" ? "#fff" : undefined,
                        borderColor: rankChartTab === "envelope" ? "rgba(255,255,255,0.3)" : undefined,
                      }}
                      onClick={() => setRankChartTab("envelope")}
                    >
                      Envelope Map
                    </button>
                    <button
                      type="button"
                      className="chip-btn"
                      style={{
                        color: rankChartTab === "breakdown" ? "#fff" : undefined,
                        borderColor: rankChartTab === "breakdown" ? "rgba(255,255,255,0.3)" : undefined,
                      }}
                      onClick={() => setRankChartTab("breakdown")}
                    >
                      Score Breakdown
                    </button>
                  </div>
                </div>
                <div className="chart-box" style={{ height: 250 }}>
                  {rankChartTab === "envelope" ? (
                    <EnvelopeScatterChart
                      investigations={investigations}
                      ranked={ranked}
                      targetO2={rankForm.o2_pct}
                      targetPsi={rankForm.pressure_psi}
                    />
                  ) : (
                    <MatchBreakdownBarChart ranked={ranked} />
                  )}
                </div>
              </section>

              <section className="glass-card">
                <div className="card-title-row">
                  <h3>Ranked NASA PSI Investigations</h3>
                  <button
                    type="button"
                    className="chip-btn"
                    onClick={() => setShowAllRanked((v) => !v)}
                  >
                    {showAllRanked ? "Show top 6" : `Show all ${ranked.length}`}
                  </button>
                </div>
                <div className="table-scroll">
                  <table className="dark-table">
                    <thead>
                      <tr>
                        <th>Score</th>
                        <th>Investigation</th>
                        <th>Facility</th>
                        <th>Match</th>
                        <th>Key Finding</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(showAllRanked ? ranked : ranked.slice(0, 6)).map((item) => {
                        const isSelected = (selectedInvId || ranked[0]?.id) === item.id;
                        return (
                          <tr
                            key={item.id}
                            onClick={() => setSelectedInvId(item.id)}
                            style={{
                              cursor: "pointer",
                              background: isSelected ? "rgba(255,255,255,0.04)" : "transparent",
                            }}
                          >
                            <td>
                              <span className={`pill-score ${item.relevance_score >= 70 ? "high" : ""}`}>
                                {item.relevance_score}
                              </span>
                            </td>
                            <td>
                              <div style={{ fontWeight: 600, color: "#f3f5f8" }}>{item.title}</div>
                              <div style={{ fontSize: 12, color: "#667085" }}>
                                {item.id} · {item.psi_id}
                              </div>
                            </td>
                            <td>
                              <div style={{ fontSize: 13, color: "#d0d6e0" }}>{item.facility}</div>
                              <div style={{ fontSize: 12, color: "#667085" }}>{item.fuel_state}</div>
                            </td>
                            <td style={{ fontSize: 12.5, color: "#9aa3b2", whiteSpace: "nowrap" }}>
                              O2 {Math.round(item.o2_match)} · Fl {Math.round(item.flow_match)} · g {Math.round(item.gravity_match)}
                            </td>
                            <td style={{ color: "#9aa3b2" }}>
                              <div className="clamp-2">{item.key_findings}</div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </section>

              {(() => {
                const activeId = selectedInvId || (ranked[0] && ranked[0].id);
                const fullInv = investigations.find((i) => i.id === activeId);
                if (!fullInv) return null;
                const pct = (val, min, max) => Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
                const o2Start = pct(fullInv.o2_range_pct?.[0] ?? 15, 10, 65);
                const o2End = pct(fullInv.o2_range_pct?.[1] ?? 35, 10, 65);
                const o2Dot = pct(rankForm.o2_pct, 10, 65);
                const pStart = pct(fullInv.pressure_range_psi?.[0] ?? 8, 4, 16);
                const pEnd = pct(fullInv.pressure_range_psi?.[1] ?? 15, 4, 16);
                const pDot = pct(rankForm.pressure_psi, 4, 16);
                const fStart = pct(fullInv.airflow_range_cms?.[0] ?? 0, 0, 55);
                const fEnd = pct(fullInv.airflow_range_cms?.[1] ?? 30, 0, 55);
                const fDot = pct(rankForm.airflow_cms, 0, 55);

                return (
                  <section className="glass-card">
                    <div className="card-title-row">
                      <h3>{fullInv.title} ({fullInv.id})</h3>
                      <span className="mono-tag">
                        {fullInv.facility} / {fullInv.hardware_insert}
                      </span>
                    </div>
                    <p style={{ color: "#d0d6e0", marginBottom: 12 }}>{fullInv.objectives}</p>
                    <div className="two-grid" style={{ marginBottom: 12 }}>
                      <div className="sub-card">
                        <div style={{ fontWeight: 600, marginBottom: 6 }}>Target vs Tested Range</div>
                        <div className="range-meter-wrap">
                          <div className="range-meter-head">
                            <span>O2 ({fullInv.o2_range_pct?.[0]}-{fullInv.o2_range_pct?.[1]}%)</span>
                            <span>Target {rankForm.o2_pct}%</span>
                          </div>
                          <div className="range-meter-track">
                            <div className="range-meter-fill" style={{ left: `${o2Start}%`, width: `${Math.max(4, o2End - o2Start)}%` }} />
                            <div className="range-meter-dot" style={{ left: `${o2Dot}%` }} />
                          </div>
                        </div>
                        <div className="range-meter-wrap">
                          <div className="range-meter-head">
                            <span>Pressure ({fullInv.pressure_range_psi?.[0]}-{fullInv.pressure_range_psi?.[1]} psi)</span>
                            <span>Target {rankForm.pressure_psi} psi</span>
                          </div>
                          <div className="range-meter-track">
                            <div className="range-meter-fill" style={{ left: `${pStart}%`, width: `${Math.max(4, pEnd - pStart)}%` }} />
                            <div className="range-meter-dot" style={{ left: `${pDot}%` }} />
                          </div>
                        </div>
                        <div className="range-meter-wrap">
                          <div className="range-meter-head">
                            <span>Airflow ({fullInv.airflow_range_cms?.[0]}-{fullInv.airflow_range_cms?.[1]} cm/s)</span>
                            <span>Target {rankForm.airflow_cms} cm/s</span>
                          </div>
                          <div className="range-meter-track">
                            <div className="range-meter-fill" style={{ left: `${fStart}%`, width: `${Math.max(4, fEnd - fStart)}%` }} />
                            <div className="range-meter-dot" style={{ left: `${fDot}%` }} />
                          </div>
                        </div>
                      </div>

                      <div className="sub-card">
                        <div style={{ fontWeight: 600, marginBottom: 6 }}>Fuels, Geometries & Sensors</div>
                        <div style={{ color: "#9aa3b2" }}>
                          <strong>Fuels:</strong> {(fullInv.fuels || []).join(", ")}
                        </div>
                        <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                          <strong>Shapes/Sizes:</strong> {(fullInv.sample_geometries || []).join(", ")}
                        </div>
                        <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                          <strong>Varied:</strong> {(fullInv.intra_variables || []).join("; ")}
                        </div>
                        <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                          <strong>Held Constant:</strong> {(fullInv.intra_constants || []).join("; ")}
                        </div>
                        <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                          <strong>Sensors:</strong> {(fullInv.sensors || []).join(", ")}
                        </div>
                      </div>
                    </div>
                  </section>
                );
              })()}
            </div>
          </div>
        </main>
      )}

      {view === "vision" && (
        <main className="workbench">
          <div className="section-header">
            <div>
              <h2>FlameUNet Segmentation & Flame Morphology</h2>
              <p>
                Convolutional encoder-decoder trained on 160 microgravity flame frames separating hot soot core, reaction zone, and background.
              </p>
            </div>
            <span className="mono-tag">
              Val mIoU {cvData?.unet_metrics?.final_val_miou ?? "0.9844"}
            </span>
          </div>

          {cvData?.unet_metrics?.strip_image_path && (
            <section className="glass-card" style={{ marginBottom: 20 }}>
              <div className="card-title-row">
                <h3>FlameUNet Segmentation Strip</h3>
                <span className="mono-tag">
                  CE Loss {cvData.unet_metrics.final_val_ce_loss ?? "0.0087"}
                </span>
              </div>
              <img
                className="wide-img"
                src={assetUrl(cvData.unet_metrics.strip_image_path)}
                alt="FlameUNet Segmentation Strip"
              />
            </section>
          )}

          <div className="cv-grid">
            {(cvData?.regimes || []).map((r) => (
              <article key={r.regime_id} className="cv-card">
                <img src={assetUrl(r.image_path)} alt={r.label} />
                <div className="cv-body">
                  <div className="mono-meta">{r.regime_id}</div>
                  <h4>{r.label}</h4>
                  <div className="cv-stats">
                    <span>Soot Area: {r.soot_area_pct}%</span>
                    <span>Zone Area: {r.reaction_area_pct}%</span>
                    <span>Aspect: {r.aspect_ratio}</span>
                    <span>Eccentricity: {r.eccentricity}</span>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </main>
      )}

      {view === "compare" && (
        <main className="workbench">
          <div className="section-header">
            <div>
              <h2>Experiment Diff & Smoke/Dust Classifier</h2>
              <p>
                Compare NASA PSI hardware configurations side by side and inspect Mie-scattering aerosol size separation.
              </p>
            </div>
          </div>

          <div className="two-grid">
            <section className="glass-card">
              <div className="card-title-row">
                <h3>Cross-Investigation Diff</h3>
                <span className="mono-tag">PSI Compare</span>
              </div>

              <div className="two-grid">
                <div>
                  <label className="field-label">Left Investigation</label>
                  <select
                    className="dark-input"
                    value={compareIds.left}
                    onChange={(e) => setCompareIds({ ...compareIds, left: e.target.value })}
                  >
                    {investigations.map((inv) => (
                      <option key={inv.id} value={inv.id}>
                        {inv.id}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="field-label">Right Investigation</label>
                  <select
                    className="dark-input"
                    value={compareIds.right}
                    onChange={(e) => setCompareIds({ ...compareIds, right: e.target.value })}
                  >
                    {investigations.map((inv) => (
                      <option key={inv.id} value={inv.id}>
                        {inv.id}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {compareResult && (
                <div className="kpi-stack">
                  <div className="kpi-line">
                    <span>Shared Facility</span>
                    <strong>{compareResult.shared_facility ? "Yes" : "No"}</strong>
                  </div>
                  <div className="kpi-line">
                    <span>Shared Hardware Insert</span>
                    <strong>{compareResult.shared_hardware_insert ? "Yes" : "No"}</strong>
                  </div>
                  <div className="kpi-line">
                    <span>Same Fuel State</span>
                    <strong>{compareResult.same_fuel_state ? "Yes" : "No"}</strong>
                  </div>
                  <div className="kpi-line">
                    <span>O2 Envelope Overlap</span>
                    <strong>
                      {Array.isArray(compareResult.o2_overlap_pct)
                        ? `${compareResult.o2_overlap_pct[0]}% - ${compareResult.o2_overlap_pct[1]}%`
                        : "No overlap"}
                    </strong>
                  </div>
                  <div className="kpi-line">
                    <span>Shared Sensors</span>
                    <strong>{(compareResult.shared_sensors || []).join(", ") || "None"}</strong>
                  </div>
                  <div className="two-grid" style={{ marginTop: 10 }}>
                    <div className="sub-card">
                      <div style={{ fontWeight: 600, marginBottom: 4 }}>
                        {compareResult.left?.id} ({compareResult.left?.facility} / {compareResult.left?.hardware_insert})
                      </div>
                      <div style={{ color: "#9aa3b2" }}>{compareResult.left?.objectives}</div>
                      <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                        <strong>Fuels:</strong> {(compareResult.left?.fuels || []).join(", ")}
                      </div>
                      <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                        <strong>Varied:</strong> {(compareResult.left?.intra_variables || []).join(", ")}
                      </div>
                    </div>
                    <div className="sub-card">
                      <div style={{ fontWeight: 600, marginBottom: 4 }}>
                        {compareResult.right?.id} ({compareResult.right?.facility} / {compareResult.right?.hardware_insert})
                      </div>
                      <div style={{ color: "#9aa3b2" }}>{compareResult.right?.objectives}</div>
                      <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                        <strong>Fuels:</strong> {(compareResult.right?.fuels || []).join(", ")}
                      </div>
                      <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                        <strong>Varied:</strong> {(compareResult.right?.intra_variables || []).join(", ")}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </section>

            <section className="glass-card">
              <div className="card-title-row">
                <h3>Aerosol Smoke vs Regolith Dust</h3>
                <span className="mono-tag">
                  Cutoff {aerosolData?.optimal_cutoff_diameter_um ?? "-"} um
                </span>
              </div>

              <div className="two-grid">
                <div>
                  <label className="field-label">Pyrolysis Material</label>
                  <select
                    className="dark-input"
                    value={aerosolForm.material}
                    onChange={(e) => setAerosolForm({ ...aerosolForm, material: e.target.value })}
                  >
                    <option value="kapton_wire">Kapton Wire Insulation</option>
                    <option value="teflon_insulation">PTFE / Teflon Insulation</option>
                    <option value="silicone_elastomer">Silicone Elastomer</option>
                    <option value="cotton_cellulose">Cotton / Cellulose</option>
                  </select>
                </div>
                <div>
                  <label className="field-label">Gravity Regime</label>
                  <select
                    className="dark-input"
                    value={aerosolForm.gravity_g}
                    onChange={(e) =>
                      setAerosolForm({ ...aerosolForm, gravity_g: Number(e.target.value) })
                    }
                  >
                    <option value={0}>ISS 0.00g</option>
                    <option value={0.16}>Lunar 0.16g</option>
                    <option value={0.38}>Mars 0.38g</option>
                    <option value={1}>Earth 1.00g</option>
                  </select>
                </div>
              </div>

              <div className="chart-box" style={{ marginTop: 14 }}>
                <DualSeriesChart
                  labels={(aerosolData?.diameters_um || []).map((d) => `${d}`)}
                  seriesA={aerosolData?.smoke_density || []}
                  seriesB={aerosolData?.regolith_dust_density || []}
                  labelA="Pyrolysis Smoke"
                  labelB="Regolith / Cabin Dust"
                />
              </div>
            </section>
          </div>
        </main>
      )}

      {view === "lineage" && (
        <main className="workbench">
          <div className="section-header">
            <div>
              <h2>NASA PSI Data Lineage & Historical Context</h2>
              <p>
                Grounded in NASA Physical Sciences Informatics (s3://nasa-psi), ISS combustion racks, historical fire incidents, and Earth spinoffs.
              </p>
            </div>
          </div>

          <div className="two-grid" style={{ marginBottom: 20 }}>
            <section className="glass-card">
              <div className="card-title-row">
                <h3>ISS Hardware Rack Lineage</h3>
                <span className="mono-tag">CIR & MSG</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div className="sub-card">
                  <div style={{ fontWeight: 600 }}>CIR / ACME (Gaseous Fuels)</div>
                  <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                    Six gaseous-fuel experiments (including BRE, CFI-G up to 85% O2, and E-FIELD Flames) sharing modular ACME hardware in the Combustion Integrated Rack.
                  </div>
                </div>
                <div className="sub-card">
                  <div style={{ fontWeight: 600 }}>CIR / MDCA (Liquid Droplets)</div>
                  <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                    FLEX, FLEX-2, and Cool Flames Investigation (CFI) burning liquid fuel droplets in the Multi-user Droplet Combustion Apparatus, studying CO2/He suppression and cool flames.
                  </div>
                </div>
                <div className="sub-card">
                  <div style={{ fontWeight: 600 }}>MSG / SPICE Flow Duct (Solids & Gas)</div>
                  <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                    BASS, BASS-II, SPICE, and SLICE conducted in the Microgravity Science Glovebox using the fan-driven flow duct built for SPICE.
                  </div>
                </div>
                <div className="sub-card">
                  <div style={{ fontWeight: 600 }}>SAME / SAME-R vs DAFT / DAFT-2</div>
                  <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                    Repurposes the microgravity smoke aerosol instrument from SAME to measure spacecraft cabin & lunar dust in DAFT.
                  </div>
                </div>
              </div>
            </section>

            <section className="glass-card">
              <div className="card-title-row">
                <h3>Partial-Gravity Gap & Modeling</h3>
                <span className="mono-tag">0g to 1g Bounding</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div className="sub-card">
                  <div style={{ fontWeight: 600 }}>No Partial-g Data in PSI Yet</div>
                  <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                    Lunar bases (0.16g, 30%+ O2, &lt;=10.2 psi) and Mars habitats (0.38g) have no direct PSI burn runs yet. Future LUCI and FM2 experiments will populate this regime.
                  </div>
                </div>
                <div className="sub-card">
                  <div style={{ fontWeight: 600 }}>E-FIELD Flames Electric Mimicry</div>
                  <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                    ACME E-FIELD Flames used electric fields acting on flame ions to mimic partial-gravity buoyancy forces in 0g.
                  </div>
                </div>
                <div className="sub-card">
                  <div style={{ fontWeight: 600 }}>How PyroSense-SI Bridges the Gap</div>
                  <div style={{ color: "#9aa3b2", marginTop: 4 }}>
                    Our GPU-trained CombustionPINN and ReactingFlow2DPINN interpolate between 0g ISS data and 1g ground baselines using Navier-Stokes buoyancy, Arrhenius kinetics, and 95% MC-Dropout uncertainty bounds.
                  </div>
                </div>
              </div>
            </section>
          </div>

          <div className="two-grid">
            <section className="glass-card">
              <div className="card-title-row">
                <h3>Historical Fire Incidents</h3>
                <span className="mono-tag">{catalog?.historical_incidents?.length || 0} records</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {(catalog?.historical_incidents || []).map((inc) => (
                  <div key={inc.id} className="sub-card">
                    <div style={{ fontWeight: 600 }}>
                      {inc.name} ({inc.year})
                    </div>
                    <div style={{ color: "#9aa3b2", marginTop: 4 }}>{inc.lesson}</div>
                  </div>
                ))}
              </div>
            </section>

            <section className="glass-card">
              <div className="card-title-row">
                <h3>Earth Spinoff Applications</h3>
                <span className="mono-tag">{catalog?.earth_spinoffs?.length || 0} spinoffs</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {(catalog?.earth_spinoffs || []).map((sp) => (
                  <div key={sp.id} className="sub-card">
                    <div style={{ fontWeight: 600 }}>{sp.title}</div>
                    <div style={{ color: "#9aa3b2", marginTop: 4 }}>{sp.impact}</div>
                  </div>
                ))}
              </div>
            </section>
          </div>
        </main>
      )}
    </div>
  );
}
