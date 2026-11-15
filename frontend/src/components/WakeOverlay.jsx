import React, { useEffect, useState } from "react";

const STEPS = [
  { label: "PSI Telemetry", detail: "Connecting to NASA Physical Sciences data layer..." },
  { label: "1D & 2D PINN", detail: "Loading Navier-Stokes & Arrhenius physics fields..." },
  { label: "FlameUNet", detail: "Mounting vision model trained on 3,600 NASA frames..." },
  { label: "Gravity Bounds", detail: "Calibrating 0g to 1g partial-gravity envelopes..." },
  { label: "Console Ready", detail: "Finalizing habitat fire-safety workspace..." },
];

export default function WakeOverlay({ ready }) {
  const [visible, setVisible] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    if (ready) {
      setActiveStep(STEPS.length - 1);
      const t = setTimeout(() => setVisible(false), 450);
      return () => clearTimeout(t);
    }
    const delay = setTimeout(() => setVisible(true), 600);
    return () => clearTimeout(delay);
  }, [ready]);

  useEffect(() => {
    if (!visible || ready) return;
    const iv = setInterval(() => {
      setActiveStep((s) => (s >= STEPS.length - 2 ? STEPS.length - 2 : s + 1));
    }, 3200);
    return () => clearInterval(iv);
  }, [visible, ready]);

  if (!visible || dismissed) return null;

  const current = STEPS[activeStep] || STEPS[0];

  return (
    <div className="wake-overlay" role="status" aria-live="polite">
      <div className="wake-stepper-card">
        <div className="wake-stepper-title">Initializing PyroSense-SI</div>
        <div className="wake-stepper-sub">{current.detail}</div>

        <div className="h-stepper">
          {STEPS.map((step, idx) => {
            const isDone = idx < activeStep;
            const isActive = idx === activeStep;
            return (
              <div key={step.label} className="h-step-item">
                {idx > 0 && (
                  <div className={`h-step-line ${idx <= activeStep ? "lit" : ""}`} />
                )}
                <div className={`h-step-node ${isDone ? "done" : ""} ${isActive ? "active" : ""}`}>
                  {isActive ? (
                    <svg width="14" height="16" viewBox="0 0 16 20" fill="none" aria-hidden="true">
                      <path
                        d="M8 1.5C8 1.5 13.2 6.8 13.2 11.6C13.2 14.9 10.9 17.5 8 17.5C5.1 17.5 2.8 14.9 2.8 11.6C2.8 6.8 8 1.5 8 1.5Z"
                        fill="#e84b2c"
                      />
                      <path
                        d="M8 6.8C8 6.8 10.4 9.9 10.4 12.2C10.4 13.9 9.3 15.2 8 15.2C6.7 15.2 5.6 13.9 5.6 12.2C5.6 9.9 8 6.8 8 6.8Z"
                        fill="#fed7aa"
                      />
                    </svg>
                  ) : isDone ? (
                    <span className="h-step-check" />
                  ) : null}
                </div>
                <span className={`h-step-label ${isActive ? "active" : isDone ? "done" : ""}`}>
                  {step.label}
                </span>
              </div>
            );
          })}
        </div>

        <div className="wake-stepper-foot">
          <button type="button" className="wake-skip" onClick={() => setDismissed(true)}>
            Continue to interface
          </button>
        </div>
      </div>
    </div>
  );
}
