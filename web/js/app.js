let appCatalog = null;
let lastRankedRows = [];
let spreadChart = null;
let loiChart = null;
let aerosolChart = null;

document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  await initCatalog();
  bindRankerControls();
  bindPINNControls();
  bindCompareAndAerosolControls();
  bindCopilotControls();

  await runRanker();
  await runPINNPrediction();
  await load2DFields();
  await loadCVMetrics();
  await runComparison();
  await runAerosolCurve();
  await runCopilot();
});

function setupTabs() {
  const buttons = document.querySelectorAll(".nav-item");
  const panels = document.querySelectorAll(".tab-panel");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");
    });
  });
}

async function initCatalog() {
  const res = await fetch("/api/catalog");
  appCatalog = await res.json();

  document.getElementById("hdr-inv-count").textContent = appCatalog.investigations.length;

  const presetSel = document.getElementById("preset-select");
  presetSel.innerHTML = "";
  appCatalog.missions.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.name} (${m.o2_pct}% O2, ${m.pressure_psi} psi, ${m.gravity_g}g)`;
    presetSel.appendChild(opt);
  });

  applyPreset(appCatalog.missions[0]);
  presetSel.addEventListener("change", () => {
    const found = appCatalog.missions.find((x) => x.id === presetSel.value);
    if (found) {
      applyPreset(found);
      runRanker();
    }
  });

  const cmpLeft = document.getElementById("cmp-left");
  const cmpRight = document.getElementById("cmp-right");
  appCatalog.investigations.forEach((inv, idx) => {
    const optA = new Option(`${inv.id} - ${inv.title}`, inv.id, idx === 1, idx === 1);
    const optB = new Option(`${inv.id} - ${inv.title}`, inv.id, idx === 3, idx === 3);
    cmpLeft.add(optA);
    cmpRight.add(optB);
  });

  renderLineageLists();
}

function applyPreset(m) {
  document.getElementById("preset-summary").textContent = m.summary;
  document.getElementById("rng-rank-o2").value = m.o2_pct;
  document.getElementById("val-rank-o2").textContent = `${m.o2_pct.toFixed(1)}%`;
  document.getElementById("rng-rank-p").value = m.pressure_psi;
  document.getElementById("val-rank-p").textContent = `${m.pressure_psi.toFixed(1)} psi`;
  document.getElementById("rng-rank-g").value = m.gravity_g;
  document.getElementById("val-rank-g").textContent = `${m.gravity_g.toFixed(2)}g`;
  document.getElementById("rng-rank-flow").value = m.airflow_cms;
  document.getElementById("val-rank-flow").textContent = `${m.airflow_cms.toFixed(1)} cm/s`;
}

function bindRankerControls() {
  const sliders = [
    ["rng-rank-o2", "val-rank-o2", (v) => `${parseFloat(v).toFixed(1)}%`],
    ["rng-rank-p", "val-rank-p", (v) => `${parseFloat(v).toFixed(1)} psi`],
    ["rng-rank-g", "val-rank-g", (v) => `${parseFloat(v).toFixed(2)}g`],
    ["rng-rank-flow", "val-rank-flow", (v) => `${parseFloat(v).toFixed(1)} cm/s`],
  ];
  sliders.forEach(([rngId, valId, fmt]) => {
    const el = document.getElementById(rngId);
    el.addEventListener("input", () => {
      document.getElementById(valId).textContent = fmt(el.value);
    });
  });

  document.getElementById("btn-run-rank").addEventListener("click", runRanker);
  document.getElementById("pillar-select").addEventListener("change", runRanker);
  document.getElementById("btn-export-csv").addEventListener("click", exportRankedCSV);
}

async function runRanker() {
  const payload = {
    o2_pct: parseFloat(document.getElementById("rng-rank-o2").value),
    pressure_psi: parseFloat(document.getElementById("rng-rank-p").value),
    gravity_g: parseFloat(document.getElementById("rng-rank-g").value),
    airflow_cms: parseFloat(document.getElementById("rng-rank-flow").value),
    pillar: document.getElementById("pillar-select").value,
  };

  const res = await fetch("/api/rank", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  lastRankedRows = data.ranked_investigations;

  const tbody = document.querySelector("#rank-table tbody");
  tbody.innerHTML = "";
  lastRankedRows.forEach((row, i) => {
    const tr = document.createElement("tr");
    const pillClass = row.relevance_score >= 82 ? "score-high" : "score-mid";
    tr.innerHTML = `
      <td style="font-family:var(--font-mono)">${i + 1}</td>
      <td><strong>${row.id}</strong><br/><span style="color:var(--text-muted);font-size:12px">${row.title}</span></td>
      <td style="font-family:var(--font-mono)">${row.psi_id}</td>
      <td>${row.pillar}</td>
      <td>${row.facility} / ${row.hardware_insert}</td>
      <td>${row.fuel_state}</td>
      <td><span class="score-pill ${pillClass}">${row.relevance_score}%</span></td>
      <td>
        <div style="color:var(--nasa-blue);font-size:12px;font-weight:500;margin-bottom:2px">${row.rationale}</div>
        <div style="color:var(--text-secondary);font-size:12px">${row.key_findings}</div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function exportRankedCSV() {
  if (!lastRankedRows.length) return;
  const headers = ["rank", "id", "psi_id", "title", "pillar", "facility", "hardware_insert", "fuel_state", "relevance_score", "rationale"];
  const lines = [headers.join(",")];
  lastRankedRows.forEach((r, idx) => {
    const vals = [
      idx + 1,
      r.id,
      r.psi_id,
      `"${r.title.replace(/"/g, '""')}"`,
      r.pillar,
      r.facility,
      `"${r.hardware_insert}"`,
      r.fuel_state,
      r.relevance_score,
      `"${r.rationale.replace(/"/g, '""')}"`,
    ];
    lines.push(vals.join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "psi_ranked_investigations.csv";
  a.click();
}

function bindPINNControls() {
  const sliders = [
    ["rng-pinn-o2", "val-pinn-o2", (v) => `${parseFloat(v).toFixed(1)}%`],
    ["rng-pinn-p", "val-pinn-p", (v) => `${parseFloat(v).toFixed(1)} psi`],
    ["rng-pinn-flow", "val-pinn-flow", (v) => `${parseFloat(v).toFixed(1)} cm/s`],
    ["rng-pinn-g", "val-pinn-g", (v) => `${parseFloat(v).toFixed(2)}g`],
    ["rng-pinn-efield", "val-pinn-efield", (v) => `${parseInt(v, 10)} V/cm`],
  ];
  sliders.forEach(([rngId, valId, fmt]) => {
    const el = document.getElementById(rngId);
    el.addEventListener("input", () => {
      document.getElementById(valId).textContent = fmt(el.value);
    });
  });

  document.getElementById("btn-run-pinn").addEventListener("click", runPINNPrediction);
  document.getElementById("pinn-material").addEventListener("change", runPINNPrediction);
}

async function runPINNPrediction() {
  const payload = {
    material: document.getElementById("pinn-material").value,
    o2_pct: parseFloat(document.getElementById("rng-pinn-o2").value),
    pressure_psi: parseFloat(document.getElementById("rng-pinn-p").value),
    airflow_cms: parseFloat(document.getElementById("rng-pinn-flow").value),
    gravity_g: parseFloat(document.getElementById("rng-pinn-g").value),
    efield_v_cm: parseFloat(document.getElementById("rng-pinn-efield").value),
  };

  const res = await fetch("/api/pinn/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  const pt = data.point_prediction;

  document.getElementById("hdr-device").textContent = (data.trained_device || "CPU").toUpperCase();
  const spreadCi = pt.flame_spread_ci95 !== undefined ? ` +/- ${pt.flame_spread_ci95}` : "";
  const loiCi = pt.loi_ci95 !== undefined ? ` +/- ${pt.loi_ci95}` : "";
  document.getElementById("kpi-spread").textContent = `${pt.flame_spread_mm_s}${spreadCi} mm/s`;
  document.getElementById("kpi-loi").textContent = `${pt.loi_pct}${loiCi}% O2`;
  document.getElementById("kpi-temp").textContent = `${pt.peak_temp_k} K`;

  const statusEl = document.getElementById("kpi-status");
  statusEl.textContent = `${pt.ignition_status} (${pt.oxygen_margin_over_loi_pct > 0 ? "+" : ""}${pt.oxygen_margin_over_loi_pct}%)`;
  statusEl.style.color = pt.ignition_status === "FLAMMABLE_HAZARD" ? "var(--nasa-red)" : "var(--status-green)";

  const labels = data.gravity_bounding_curve.map((c) => `${c.gravity_g}g`);
  const spreadVals = data.gravity_bounding_curve.map((c) => c.flame_spread_mm_s);
  const spreadUpper = data.gravity_bounding_curve.map((c) => +(c.flame_spread_mm_s + (c.flame_spread_ci95 || 0)).toFixed(3));
  const spreadLower = data.gravity_bounding_curve.map((c) => +Math.max(0, c.flame_spread_mm_s - (c.flame_spread_ci95 || 0)).toFixed(3));
  const loiVals = data.gravity_bounding_curve.map((c) => c.loi_pct);
  const loiUpper = data.gravity_bounding_curve.map((c) => +(c.loi_pct + (c.loi_ci95 || 0)).toFixed(2));
  const loiLower = data.gravity_bounding_curve.map((c) => +Math.max(0, c.loi_pct - (c.loi_ci95 || 0)).toFixed(2));

  renderPINNCharts(labels, spreadVals, spreadUpper, spreadLower, loiVals, loiUpper, loiLower);
}

function renderPINNCharts(labels, spreadVals, spreadUpper, spreadLower, loiVals, loiUpper, loiLower) {
  const ctxSpread = document.getElementById("chart-pinn-spread").getContext("2d");
  if (spreadChart) spreadChart.destroy();
  spreadChart = new Chart(ctxSpread, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "95% CI Upper",
          data: spreadUpper,
          borderColor: "transparent",
          backgroundColor: "rgba(11, 61, 145, 0.12)",
          fill: "+1",
          pointRadius: 0,
          tension: 0.3,
        },
        {
          label: "Flame Spread Rate (mm/s)",
          data: spreadVals,
          borderColor: "#0b3d91",
          backgroundColor: "rgba(11, 61, 145, 0.12)",
          fill: "+1",
          tension: 0.3,
          pointRadius: 3.5,
        },
        {
          label: "95% CI Lower",
          data: spreadLower,
          borderColor: "transparent",
          backgroundColor: "transparent",
          fill: false,
          pointRadius: 0,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#111827", filter: (item) => !item.text.includes("95% CI") } } },
      scales: {
        x: { ticks: { color: "#4b5563" }, grid: { color: "#e5e7eb" } },
        y: { ticks: { color: "#4b5563" }, grid: { color: "#e5e7eb" } },
      },
    },
  });

  const ctxLoi = document.getElementById("chart-pinn-loi").getContext("2d");
  if (loiChart) loiChart.destroy();
  loiChart = new Chart(ctxLoi, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "95% CI Upper",
          data: loiUpper,
          borderColor: "transparent",
          backgroundColor: "rgba(252, 61, 33, 0.12)",
          fill: "+1",
          pointRadius: 0,
          tension: 0.3,
        },
        {
          label: "Limiting Oxygen Index (% O2)",
          data: loiVals,
          borderColor: "#fc3d21",
          backgroundColor: "rgba(252, 61, 33, 0.12)",
          fill: "+1",
          tension: 0.3,
          pointRadius: 3.5,
        },
        {
          label: "95% CI Lower",
          data: loiLower,
          borderColor: "transparent",
          backgroundColor: "transparent",
          fill: false,
          pointRadius: 0,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#111827", filter: (item) => !item.text.includes("95% CI") } } },
      scales: {
        x: { ticks: { color: "#4b5563" }, grid: { color: "#e5e7eb" } },
        y: { ticks: { color: "#4b5563" }, grid: { color: "#e5e7eb" } },
      },
    },
  });
}

async function load2DFields() {
  try {
    const res = await fetch("/api/pinn/fields2d");
    const data = await res.json();
    if (!data.field_heatmap_path || !data.regime_field_summaries.length) return;

    document.getElementById("field-2d-panel").style.display = "block";
    document.getElementById("pinn-2d-img").src = `${data.field_heatmap_path}?t=${Date.now()}`;
    const box = document.getElementById("field-2d-summaries");
    box.innerHTML = "";
    data.regime_field_summaries.forEach((s) => {
      const card = document.createElement("div");
      card.className = "field-summary-card";
      card.innerHTML = `
        <strong>${s.label}</strong>
        <div>Peak Temp: ${s.peak_field_temp_k} K</div>
        <div>Mean Plume: ${s.mean_plume_velocity_cms} cm/s</div>
        <div>Min Core O2: ${(s.min_core_o2_mole_fraction * 100).toFixed(1)}%</div>
      `;
      box.appendChild(card);
    });
  } catch (err) {
    console.warn("2D fields not available:", err);
  }
}

async function loadCVMetrics() {
  const res = await fetch("/api/cv/metrics");
  const data = await res.json();

  if (data.unet_metrics && data.unet_metrics.segmentation_strip_path) {
    const banner = document.getElementById("unet-banner");
    banner.style.display = "block";
    document.getElementById("unet-stats").textContent =
      `FRAMES: ${data.unet_metrics.total_frames_trained} | VAL mIoU: ${data.unet_metrics.final_val_miou} | ${data.unet_metrics.gpu_name}`;
    document.getElementById("unet-strip-img").src = `${data.unet_metrics.segmentation_strip_path}?t=${Date.now()}`;
  }

  const container = document.getElementById("cv-cards-container");
  container.innerHTML = "";

  data.regimes.forEach((r) => {
    const card = document.createElement("div");
    card.className = "cv-card";
    const nasaTag = r.nasa_image_id ? ` | ${r.nasa_image_id}` : "";
    card.innerHTML = `
      <img src="${r.image_path}" alt="${r.title}" />
      <div class="cv-body">
        <div class="cv-meta">${r.investigation} | ${r.gravity_g}g | ${r.o2_pct}% O2${nasaTag}</div>
        <h3>${r.title}</h3>
        <div class="cv-metrics-list">
          <div>Sphericity: <strong>${r.sphericity_index}</strong></div>
          <div>Aspect Ratio: <strong>${r.aspect_ratio}</strong></div>
          <div>Eq. Radius: <strong>${r.equivalent_flame_radius_mm} mm</strong></div>
          <div>Standoff: <strong>${r.standoff_distance_mm} mm</strong></div>
          <div>Blue Core: <strong>${(r.blue_chemiluminescence_fraction * 100).toFixed(1)}%</strong></div>
          <div>Soot Zone: <strong>${(r.soot_radiative_fraction * 100).toFixed(1)}%</strong></div>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

function bindCompareAndAerosolControls() {
  document.getElementById("cmp-left").addEventListener("change", runComparison);
  document.getElementById("cmp-right").addEventListener("change", runComparison);
  document.getElementById("aero-mat").addEventListener("change", runAerosolCurve);
  document.getElementById("aero-g").addEventListener("change", runAerosolCurve);
}

async function runComparison() {
  const leftId = document.getElementById("cmp-left").value;
  const rightId = document.getElementById("cmp-right").value;
  const res = await fetch(`/api/compare?left_id=${encodeURIComponent(leftId)}&right_id=${encodeURIComponent(rightId)}`);
  const d = await res.json();

  const out = document.getElementById("compare-output");
  out.innerHTML = `
    <div class="diff-block">
      <h4>Hardware &amp; Facility Alignment</h4>
      <div><strong>${d.left.id}:</strong> ${d.left.facility} / ${d.left.hardware_insert} (${d.left.fuel_state} fuel)</div>
      <div><strong>${d.right.id}:</strong> ${d.right.facility} / ${d.right.hardware_insert} (${d.right.fuel_state} fuel)</div>
      <div style="margin-top:4px;color:var(--nasa-blue);font-weight:500">
        Shared Facility: ${d.shared_facility ? "YES" : "NO"} | Shared Insert: ${d.shared_hardware_insert ? "YES" : "NO"}
      </div>
    </div>
    <div class="diff-block">
      <h4>Intra-Investigation Variables</h4>
      <div><strong>${d.left.id} Varied:</strong> ${d.left.intra_variables.join(", ")}</div>
      <div><strong>${d.left.id} Constant:</strong> ${d.left.intra_constants.join(", ")}</div>
      <hr style="border:none;border-top:1px solid var(--border);margin:6px 0"/>
      <div><strong>${d.right.id} Varied:</strong> ${d.right.intra_variables.join(", ")}</div>
      <div><strong>${d.right.id} Constant:</strong> ${d.right.intra_constants.join(", ")}</div>
    </div>
    <div class="diff-block">
      <h4>Test Envelope &amp; Sensors</h4>
      <div><strong>O2 Overlap:</strong> ${d.o2_overlap_pct ? `${d.o2_overlap_pct[0]}% to ${d.o2_overlap_pct[1]}% O2` : "Disjoint O2 ranges"}</div>
      <div><strong>Shared Sensors:</strong> ${d.shared_sensors.length ? d.shared_sensors.join(", ") : "Distinct optical/thermal suites"}</div>
    </div>
  `;
}

async function runAerosolCurve() {
  const mat = document.getElementById("aero-mat").value;
  const gVal = document.getElementById("aero-g").value;
  const res = await fetch(`/api/aerosol?material=${encodeURIComponent(mat)}&gravity_g=${encodeURIComponent(gVal)}`);
  const d = await res.json();

  document.getElementById("aerosol-summary").textContent =
    `At ${d.gravity_g}g, ${d.pyrolysis_material} smoke exhibits a count median diameter (CMD) of ${d.count_median_diameter_smoke_um} um, while airborne regolith dust peaks at ${d.count_median_diameter_dust_um} um. A dual-wavelength optical cutoff at ${d.optimal_cutoff_diameter_um} um separates real fire pyrolysis from nuisance dust.`;

  const ctx = document.getElementById("chart-aerosol").getContext("2d");
  if (aerosolChart) aerosolChart.destroy();
  aerosolChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: d.diameters_um.map((x) => `${x} um`),
      datasets: [
        {
          label: "Pyrolysis Smoke Aerosol (SAME / SAME-R)",
          data: d.smoke_density,
          borderColor: "#0b3d91",
          backgroundColor: "rgba(11, 61, 145, 0.14)",
          fill: true,
          tension: 0.3,
        },
        {
          label: "Airborne Regolith / Cabin Dust (DAFT-2)",
          data: d.regolith_dust_density,
          borderColor: "#fc3d21",
          backgroundColor: "rgba(252, 61, 33, 0.12)",
          fill: true,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#111827" } } },
      scales: {
        x: { ticks: { color: "#4b5563", maxTicksLimit: 9 }, grid: { color: "#e5e7eb" } },
        y: { ticks: { color: "#4b5563" }, grid: { color: "#e5e7eb" } },
      },
    },
  });
}

function bindCopilotControls() {
  document.getElementById("btn-ask-si").addEventListener("click", runCopilot);
  document.querySelectorAll(".chip-btn").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.getElementById("si-query").value = chip.dataset.q;
      runCopilot();
    });
  });
}

async function runCopilot() {
  const question = document.getElementById("si-query").value;
  const role = document.getElementById("si-role").value;
  const res = await fetch("/api/copilot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, role }),
  });
  const d = await res.json();

  const box = document.getElementById("si-response-box");
  box.innerHTML = `
    <div style="font-family:var(--font-mono);font-size:11px;color:var(--nasa-blue);margin-bottom:6px;font-weight:600">
      MODE: ${d.role_mode.toUpperCase()} | CITED STUDIES: ${d.cited_investigations.join(", ")}
    </div>
    <ul style="padding-left:18px;margin-bottom:10px">
      ${d.evidence_summary.map((item) => `<li style="margin-bottom:5px">${item}</li>`).join("")}
    </ul>
    <div style="border-top:1px solid var(--border);padding-top:8px;color:var(--text-secondary);font-size:12.5px">
      <strong>Partial-Gravity &amp; Mission Physics Synthesis:</strong> ${d.physics_interpretation}
    </div>
  `;
}

function renderLineageLists() {
  const incBox = document.getElementById("incidents-list");
  incBox.innerHTML = "";
  appCatalog.historical_incidents.forEach((inc) => {
    const div = document.createElement("div");
    div.className = "lineage-item";
    div.innerHTML = `
      <div><strong>${inc.mission} (${inc.year})</strong> | <span style="font-family:var(--font-mono);font-size:11px;color:var(--nasa-red)">${inc.atmosphere}</span></div>
      <div style="color:var(--text-secondary);margin:3px 0">${inc.cause}</div>
      <div style="font-size:12px"><strong>PSI Follow-up:</strong> ${inc.psi_links.join(", ")} - ${inc.engineering_lesson}</div>
    `;
    incBox.appendChild(div);
  });

  const spinBox = document.getElementById("spinoffs-list");
  spinBox.innerHTML = "";
  appCatalog.earth_spinoffs.forEach((sp) => {
    const div = document.createElement("div");
    div.className = "lineage-item";
    div.innerHTML = `
      <div><strong>${sp.title}</strong> | <a href="${sp.source_url}" target="_blank" style="color:var(--nasa-blue);font-size:11px">NASA Spinoff</a></div>
      <div style="color:var(--text-secondary);margin:3px 0"><strong>Origin:</strong> ${sp.nasa_origin} (PSI: ${sp.psi_links.join(", ")})</div>
      <div style="font-size:12px">${sp.terrestrial_impact}</div>
    `;
    spinBox.appendChild(div);
  });
}
