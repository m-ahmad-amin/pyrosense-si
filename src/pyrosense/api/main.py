import json
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pyrosense.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MODELS_CHECKPOINT_DIR,
    WEB_DIR,
    WEB_FRAMES_DIR,
)
from pyrosense.api.schemas import CopilotRequest, PINNPredictRequest, RankRequest
from pyrosense.engine.relevance import (
    compare_investigations,
    compute_aerosol_discrimination_curve,
    query_si_copilot,
    score_investigation_for_mission,
)
from pyrosense.models.pde_field_pinn import (
    load_2d_pinn_checkpoint,
    predict_live_2d_field,
)
from pyrosense.models.pinn import (
    MATERIAL_ENCODING,
    CombustionPINN,
    normalize_features,
    predict_with_uncertainty,
    train_pinn,
)
from pyrosense.vision.flame_analyzer import run_flame_cv_pipeline

app = FastAPI(title="PyroSense SI - NASA PSI Combustion Intelligence Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST_DIR = WEB_DIR.parent / "frontend" / "dist"
WEB_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/assets/frames", StaticFiles(directory=str(WEB_FRAMES_DIR)), name="frames")
if (FRONTEND_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="frontend-assets")
else:
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")
if (WEB_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(WEB_DIR / "css")), name="css")
if (WEB_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(WEB_DIR / "js")), name="js")


def load_catalog() -> dict:
    return json.loads((DATA_RAW_DIR / "psi_investigations.json").read_text(encoding="utf-8"))


def ensure_artifacts():
    grid_path = DATA_PROCESSED_DIR / "pinn_phase_grid.json"
    model_path = MODELS_CHECKPOINT_DIR / "pinn_flammability.pt"
    cv_path = DATA_PROCESSED_DIR / "flame_morphology_metrics.json"
    if not grid_path.exists() or not model_path.exists():
        train_pinn(
            csv_path=DATA_RAW_DIR / "psi_burn_runs.csv",
            model_out=model_path,
            grid_out=grid_path,
            epochs=900,
        )
    if not cv_path.exists():
        run_flame_cv_pipeline(frames_dir=WEB_FRAMES_DIR, metrics_out=cv_path)


@app.on_event("startup")
def on_startup():
    ensure_artifacts()


@app.get("/")
def serve_index():
    dist_index = FRONTEND_DIST_DIR / "index.html"
    if dist_index.exists():
        return FileResponse(str(dist_index))
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/catalog")
def get_catalog():
    catalog = load_catalog()
    runs_df = pd.read_csv(DATA_RAW_DIR / "psi_burn_runs.csv")
    s3_path = DATA_PROCESSED_DIR / "psi_s3_manifest.json"
    s3_manifest = (
        json.loads(s3_path.read_text(encoding="utf-8"))
        if s3_path.exists()
        else {"bucket": "s3://nasa-psi", "top_level_prefixes": [], "sampled_objects": []}
    )
    return {
        "missions": catalog["missions"],
        "investigations": catalog["investigations"],
        "historical_incidents": catalog["historical_incidents"],
        "earth_spinoffs": catalog["earth_spinoffs"],
        "burn_runs": runs_df.to_dict(orient="records"),
        "s3_manifest": s3_manifest,
    }


@app.post("/api/rank")
def rank_investigations(req: RankRequest):
    catalog = load_catalog()
    params = req.dict()
    scored = [score_investigation_for_mission(inv, params) for inv in catalog["investigations"]]
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return {"ranked_investigations": scored, "query_parameters": params}


@app.post("/api/pinn/predict")
def predict_pinn(req: PINNPredictRequest):
    ensure_artifacts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODELS_CHECKPOINT_DIR / "pinn_flammability.pt", map_location=device)
    model = CombustionPINN(in_dim=8, hidden_dim=128).to(device)
    model.load_state_dict(ckpt["state_dict"])

    mat_props = MATERIAL_ENCODING.get(req.material, [0.5, 1.0, 16.5])
    gravity_points = [0.0, 0.05, 0.10, 0.16, 0.25, 0.38, 0.60, 1.00]

    target_in = torch.tensor(
        [
            [
                req.o2_pct,
                req.pressure_psi,
                req.airflow_cms,
                req.gravity_g,
                req.efield_v_cm,
                mat_props[0],
                mat_props[1],
                mat_props[2],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    mean_pt, std_pt = predict_with_uncertainty(model, normalize_features(target_in), n_samples=24)
    pred = mean_pt[0]
    ci_pt = 1.96 * std_pt[0]

    batch_g = []
    for g_val in gravity_points:
        batch_g.append(
            [
                req.o2_pct,
                req.pressure_psi,
                req.airflow_cms,
                g_val,
                req.efield_v_cm,
                mat_props[0],
                mat_props[1],
                mat_props[2],
            ]
        )
    t_batch = torch.tensor(batch_g, dtype=torch.float32, device=device)
    g_preds, g_stds = predict_with_uncertainty(model, normalize_features(t_batch), n_samples=24)

    curve = []
    for idx, g_val in enumerate(gravity_points):
        curve.append(
            {
                "gravity_g": g_val,
                "flame_spread_mm_s": round(float(g_preds[idx, 0]), 3),
                "flame_spread_ci95": round(float(1.96 * g_stds[idx, 0]), 3),
                "loi_pct": round(float(g_preds[idx, 1]), 2),
                "loi_ci95": round(float(1.96 * g_stds[idx, 1]), 2),
                "peak_temp_k": round(float(g_preds[idx, 2]), 1),
                "radiative_fraction": round(float(g_preds[idx, 3]), 3),
            }
        )

    margin_pct = round(float(req.o2_pct - pred[1]), 2)
    status = "FLAMMABLE_HAZARD" if margin_pct > 0 else "SELF_EXTINGUISHING"

    return {
        "material": req.material,
        "point_prediction": {
            "flame_spread_mm_s": round(float(pred[0]), 3),
            "flame_spread_ci95": round(float(ci_pt[0]), 3),
            "loi_pct": round(float(pred[1]), 2),
            "loi_ci95": round(float(ci_pt[1]), 2),
            "peak_temp_k": round(float(pred[2]), 1),
            "radiative_fraction": round(float(pred[3]), 3),
            "oxygen_margin_over_loi_pct": margin_pct,
            "ignition_status": status,
        },
        "gravity_bounding_curve": curve,
        "validation_metrics": ckpt.get("validation_metrics", {}),
        "trained_device": ckpt.get("device", str(device)),
    }


@app.get("/api/pinn/grid")
def get_pinn_grid():
    ensure_artifacts()
    return json.loads((DATA_PROCESSED_DIR / "pinn_phase_grid.json").read_text(encoding="utf-8"))


@app.get("/api/pinn/fields2d")
def get_pinn_fields2d():
    f_path = DATA_PROCESSED_DIR / "pinn_2d_fields.json"
    if f_path.exists():
        return json.loads(f_path.read_text(encoding="utf-8"))
    return {"regime_field_summaries": [], "field_heatmap_path": None}


_LIVE_2D_CACHE = {"model": None, "mtime": None, "device": None}


@app.post("/api/pinn/field2d/live")
def predict_live_fields2d(req: PINNPredictRequest):
    ckpt_path = MODELS_CHECKPOINT_DIR / "pinn_2d_reacting_flow.pt"
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail="2D reacting-flow checkpoint not trained yet")

    mtime = ckpt_path.stat().st_mtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _LIVE_2D_CACHE["model"] is None or _LIVE_2D_CACHE["mtime"] != mtime:
        _LIVE_2D_CACHE["model"] = load_2d_pinn_checkpoint(ckpt_path, device)
        _LIVE_2D_CACHE["mtime"] = mtime
        _LIVE_2D_CACHE["device"] = device

    return predict_live_2d_field(
        model=_LIVE_2D_CACHE["model"],
        device=_LIVE_2D_CACHE["device"],
        gravity_g=req.gravity_g,
        o2_pct=req.o2_pct,
        pressure_psi=req.pressure_psi,
        airflow_cms=req.airflow_cms,
    )


@app.get("/api/cv/metrics")
def get_cv_metrics():
    ensure_artifacts()
    cv_data = json.loads((DATA_PROCESSED_DIR / "flame_morphology_metrics.json").read_text(encoding="utf-8"))
    unet_path = DATA_PROCESSED_DIR / "unet_vision_metrics.json"
    if unet_path.exists():
        cv_data["unet_metrics"] = json.loads(unet_path.read_text(encoding="utf-8"))
    return cv_data


@app.get("/api/compare")
def compare_two(left_id: str = "BASS-II", right_id: str = "SAFFIRE-II"):
    catalog = load_catalog()
    invs = {i["id"]: i for i in catalog["investigations"]}
    if left_id not in invs or right_id not in invs:
        raise HTTPException(status_code=404, detail="Investigation ID not found")
    return compare_investigations(invs[left_id], invs[right_id])


@app.get("/api/aerosol")
def get_aerosol_curve(
    material: str = "kapton_wire",
    dust_load: float = 1.2,
    gravity_g: float = 0.16,
):
    return compute_aerosol_discrimination_curve(material, dust_load, gravity_g)


@app.post("/api/copilot")
def ask_copilot(req: CopilotRequest):
    catalog = load_catalog()
    return query_si_copilot(req.question, req.role, catalog)
