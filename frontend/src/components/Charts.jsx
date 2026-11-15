import React, { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

export function ConfidenceLineChart({ labels, values, upper, lower, label, color, fillRgba }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !labels?.length) return;
    if (chartRef.current) chartRef.current.destroy();

    chartRef.current = new Chart(canvasRef.current, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "95% CI Upper",
            data: upper,
            borderColor: "transparent",
            backgroundColor: fillRgba,
            fill: "+1",
            pointRadius: 0,
            tension: 0.35,
          },
          {
            label,
            data: values,
            borderColor: color,
            backgroundColor: fillRgba,
            fill: "+1",
            pointRadius: 3.5,
            tension: 0.35,
          },
          {
            label: "95% CI Lower",
            data: lower,
            borderColor: "transparent",
            backgroundColor: "transparent",
            fill: false,
            pointRadius: 0,
            tension: 0.35,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: {
              color: "#d0d5dd",
              filter: (item) => !item.text.includes("95% CI"),
            },
          },
        },
        scales: {
          x: { ticks: { color: "#9aa3b2" }, grid: { color: "rgba(255,255,255,0.06)" } },
          y: { ticks: { color: "#9aa3b2" }, grid: { color: "rgba(255,255,255,0.06)" } },
        },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [labels, values, upper, lower, label, color, fillRgba]);

  return <canvas ref={canvasRef} />;
}

export function DualSeriesChart({ labels, seriesA, seriesB, labelA, labelB }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !labels?.length) return;
    if (chartRef.current) chartRef.current.destroy();

    chartRef.current = new Chart(canvasRef.current, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: labelA,
            data: seriesA,
            borderColor: "#ffffff",
            backgroundColor: "rgba(255,255,255,0.10)",
            fill: true,
            tension: 0.35,
          },
          {
            label: labelB,
            data: seriesB,
            borderColor: "#e84b2c",
            backgroundColor: "rgba(232,75,44,0.14)",
            fill: true,
            tension: 0.35,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#d0d5dd" } } },
        scales: {
          x: { ticks: { color: "#9aa3b2", maxTicksLimit: 9 }, grid: { color: "rgba(255,255,255,0.06)" } },
          y: { ticks: { color: "#9aa3b2" }, grid: { color: "rgba(255,255,255,0.06)" } },
        },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [labels, seriesA, seriesB, labelA, labelB]);

  return <canvas ref={canvasRef} />;
}

export function EnvelopeScatterChart({ investigations, ranked, targetO2, targetPsi }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !investigations?.length) return;
    if (chartRef.current) chartRef.current.destroy();

    const scoreMap = new Map((ranked || []).map((r) => [r.id, r.relevance_score]));
    const topPoints = [];
    const otherPoints = [];

    investigations.forEach((inv) => {
      if (!inv.o2_range_pct || !inv.pressure_range_psi) return;
      const o2Mid = (inv.o2_range_pct[0] + inv.o2_range_pct[1]) / 2;
      const pMid = (inv.pressure_range_psi[0] + inv.pressure_range_psi[1]) / 2;
      const score = scoreMap.get(inv.id) || 0;
      const pt = { x: Number(o2Mid.toFixed(1)), y: Number(pMid.toFixed(2)), id: inv.id, score };
      if (score >= 78) topPoints.push(pt);
      else otherPoints.push(pt);
    });

    chartRef.current = new Chart(canvasRef.current, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Your Cabin Target",
            data: [{ x: targetO2, y: targetPsi, id: "TARGET" }],
            backgroundColor: "#e84b2c",
            borderColor: "#ffffff",
            borderWidth: 2,
            pointRadius: 9,
            pointHoverRadius: 11,
          },
          {
            label: "Top PSI Matches",
            data: topPoints,
            backgroundColor: "rgba(255, 255, 255, 0.85)",
            borderColor: "#ffffff",
            pointRadius: 5.5,
          },
          {
            label: "Other PSI Studies",
            data: otherPoints,
            backgroundColor: "rgba(148, 163, 184, 0.32)",
            borderColor: "rgba(148, 163, 184, 0.5)",
            pointRadius: 4,
          },
          {
            label: "NASA Habitats",
            data: [
              { x: 21, y: 14.7, id: "ISS 0g" },
              { x: 34, y: 10.2, id: "Lunar 0.16g" },
              { x: 30, y: 10.2, id: "Mars 0.38g" },
            ],
            backgroundColor: "transparent",
            borderColor: "rgba(255, 255, 255, 0.65)",
            borderWidth: 1.5,
            pointStyle: "rectRot",
            pointRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#d0d5dd", boxWidth: 10, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const raw = ctx.raw || {};
                const scoreTxt = raw.score ? ` (Score ${raw.score})` : "";
                return `${raw.id || ctx.dataset.label}: ${raw.x}% O2, ${raw.y} psi${scoreTxt}`;
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "O2 Concentration (%)", color: "#9aa3b2" },
            min: 10,
            max: 65,
            ticks: { color: "#9aa3b2" },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
          y: {
            title: { display: true, text: "Pressure (psi)", color: "#9aa3b2" },
            min: 4,
            max: 16,
            ticks: { color: "#9aa3b2" },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
        },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [investigations, ranked, targetO2, targetPsi]);

  return <canvas ref={canvasRef} />;
}

export function MatchBreakdownBarChart({ ranked }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    const top = (ranked || []).slice(0, 5);
    if (!canvasRef.current || !top.length) return;
    if (chartRef.current) chartRef.current.destroy();

    chartRef.current = new Chart(canvasRef.current, {
      type: "bar",
      data: {
        labels: top.map((r) => r.id),
        datasets: [
          {
            label: "O2",
            data: top.map((r) => r.o2_match),
            backgroundColor: "rgba(232, 75, 44, 0.88)",
            borderRadius: 3,
          },
          {
            label: "Pressure",
            data: top.map((r) => r.pressure_match),
            backgroundColor: "rgba(255, 255, 255, 0.85)",
            borderRadius: 3,
          },
          {
            label: "Airflow",
            data: top.map((r) => r.flow_match),
            backgroundColor: "rgba(148, 163, 184, 0.6)",
            borderRadius: 3,
          },
          {
            label: "Gravity",
            data: top.map((r) => r.gravity_match),
            backgroundColor: "rgba(71, 85, 105, 0.7)",
            borderRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#d0d5dd", boxWidth: 10, font: { size: 11 } } },
        },
        scales: {
          x: { ticks: { color: "#d0d5dd" }, grid: { display: false } },
          y: {
            min: 0,
            max: 100,
            ticks: { color: "#9aa3b2" },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
        },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [ranked]);

  return <canvas ref={canvasRef} />;
}
