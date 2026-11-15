import React, { useEffect, useRef } from "react";

const HABITAT_PRESETS = [
  { label: "Lunar Base 0.16g (34% O2)", o2: 34, psi: 10.2, g: 0.16, flow: 12 },
  { label: "Mars Habitat 0.38g (30% O2)", o2: 30, psi: 10.2, g: 0.38, flow: 14 },
  { label: "ISS LEO 0.00g (21% O2)", o2: 21, psi: 14.7, g: 0.0, flow: 15 },
  { label: "Spacesuit 0.16g (100% O2)", o2: 45, psi: 6.0, g: 0.16, flow: 35 },
];

export default function Hero({ onLaunch, onOpenRanker }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animId = 0;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const particles = Array.from({ length: 42 }, () => ({
      x: canvas.width * (0.45 + Math.random() * 0.4),
      y: canvas.height * (0.2 + Math.random() * 0.75),
      r: 0.8 + Math.random() * 1.8,
      vy: -0.15 - Math.random() * 0.35,
      vx: (Math.random() - 0.5) * 0.18,
      alpha: 0.15 + Math.random() * 0.45,
    }));

    const tick = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.y < 40) {
          p.y = canvas.height * 0.88;
          p.x = canvas.width * (0.48 + Math.random() * 0.34);
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(232, 95, 58, ${p.alpha})`;
        ctx.fill();
      });
      animId = requestAnimationFrame(tick);
    };
    tick();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <section className="hero-stage">
      <canvas ref={canvasRef} className="hero-canvas" />
      <div className="hero-beam" />
      <div className="hero-horizon" />

      <div className="hero-copy">
        <h1>Spacecraft fire safety from 0g to Mars.</h1>
        <p>
          Explore NASA PSI combustion investigations, predict partial-gravity
          flammability with Physics-Informed Neural Networks, and segment
          microgravity flames in real time.
        </p>
        <div className="hero-cta-row">
          <button type="button" className="btn-white" onClick={() => onLaunch(null)}>
            Predict Flammability
          </button>
          <button type="button" className="btn-ghost" onClick={onOpenRanker}>
            Rank PSI Studies
          </button>
        </div>

        <div className="hero-presets">
          {HABITAT_PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              className="preset-pill"
              onClick={() => onLaunch(p)}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div className="hero-footnote">
        NASA PSI Combustion Intelligence (BASS-II, Saffire, ACME, FLEX, SAME, DAFT-2)
      </div>
      <div className="hero-metrics-bar">
        <span><strong>R2 0.996</strong> 1D PINN</span>
        <span><strong>0.015</strong> 2D Navier-Stokes PDE</span>
        <span><strong>0.985 mIoU</strong> 3,600 NASA Frames</span>
      </div>
    </section>
  );
}
