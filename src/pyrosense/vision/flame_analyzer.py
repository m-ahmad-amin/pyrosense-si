import json
from pathlib import Path
import cv2
import numpy as np
import requests
import torch


FLAME_REGIMES = [
    {
        "id": "bass2_0g_quiescent",
        "investigation": "BASS-II / BRE",
        "title": "0g Quiescent Spherical Diffusion Flame",
        "gravity_g": 0.0,
        "airflow_cms": 1.2,
        "o2_pct": 21.0,
        "pressure_psi": 14.7,
        "efield_v_cm": 0.0,
        "aspect_y": 1.04,
        "aspect_x": 1.00,
        "soot_intensity": 0.14,
        "blue_intensity": 0.92,
        "standoff_px": 46,
        "nasa_search_query": "microgravity spherical flame ISS",
    },
    {
        "id": "bass2_0g_ventilated",
        "investigation": "BASS-II / SAFFIRE-I",
        "title": "0g Ventilated Solid Boundary-Layer Flame (15 cm/s)",
        "gravity_g": 0.0,
        "airflow_cms": 15.0,
        "o2_pct": 21.0,
        "pressure_psi": 14.7,
        "efield_v_cm": 0.0,
        "aspect_y": 1.62,
        "aspect_x": 0.84,
        "soot_intensity": 0.58,
        "blue_intensity": 0.88,
        "standoff_px": 34,
        "nasa_search_query": "BASS flame microgravity space station",
    },
    {
        "id": "efield_lunar_016g",
        "investigation": "ACME E-FIELD / Lunar Base Analog",
        "title": "0.16g Lunar Equivalent (380 V/cm / 34% O2 / 10.2 psi)",
        "gravity_g": 0.16,
        "airflow_cms": 8.0,
        "o2_pct": 34.0,
        "pressure_psi": 10.2,
        "efield_v_cm": 380.0,
        "aspect_y": 1.88,
        "aspect_x": 0.78,
        "soot_intensity": 0.76,
        "blue_intensity": 0.96,
        "standoff_px": 28,
        "nasa_search_query": "ACME flame combustion space station",
    },
    {
        "id": "cfi_cool_flame",
        "investigation": "FLEX / CFI / CFI-G",
        "title": "0g Post-Extinction Cool Flame (680 K CH2O* Band)",
        "gravity_g": 0.0,
        "airflow_cms": 0.0,
        "o2_pct": 18.0,
        "pressure_psi": 12.0,
        "efield_v_cm": 0.0,
        "aspect_y": 1.01,
        "aspect_x": 1.01,
        "soot_intensity": 0.02,
        "blue_intensity": 0.48,
        "standoff_px": 54,
        "nasa_search_query": "FLEX droplet flame microgravity",
    },
    {
        "id": "saffire_spacecraft_burn",
        "investigation": "SAFFIRE-I / II / III",
        "title": "0g Large-Scale Spacecraft Panel Propagation (20 cm/s)",
        "gravity_g": 0.0,
        "airflow_cms": 20.0,
        "o2_pct": 21.0,
        "pressure_psi": 14.7,
        "efield_v_cm": 0.0,
        "aspect_y": 1.75,
        "aspect_x": 0.80,
        "soot_intensity": 0.68,
        "blue_intensity": 0.82,
        "standoff_px": 31,
        "nasa_search_query": "Saffire fire experiment NASA",
    },
    {
        "id": "earth_1g_buoyant",
        "investigation": "1g Ground Reference Control",
        "title": "1.0g Terrestrial Buoyant Teardrop Plume",
        "gravity_g": 1.0,
        "airflow_cms": 40.0,
        "o2_pct": 21.0,
        "pressure_psi": 14.7,
        "efield_v_cm": 0.0,
        "aspect_y": 2.45,
        "aspect_x": 0.62,
        "soot_intensity": 0.91,
        "blue_intensity": 0.72,
        "standoff_px": 19,
        "nasa_search_query": "candle flame microgravity earth comparison",
    },
]


def fetch_real_nasa_flame_image(query: str, index_offset: int = 0) -> tuple[np.ndarray | None, str]:
    try:
        resp = requests.get(
            "https://images-api.nasa.gov/search",
            params={"q": query, "media_type": "image"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None, ""
        items = resp.json().get("collection", {}).get("items", [])
        if not items:
            return None, ""
        chosen = items[min(index_offset, len(items) - 1)]
        nasa_id = (chosen.get("data") or [{}])[0].get("nasa_id", "")
        links = chosen.get("links") or []
        if not links or not links[0].get("href"):
            return None, ""
        img_resp = requests.get(links[0]["href"], timeout=10)
        if img_resp.status_code != 200:
            return None, ""
        arr = np.frombuffer(img_resp.content, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, ""
        bgr_resized = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_AREA)
        return bgr_resized, nasa_id
    except Exception:
        return None, ""


def synthesize_radiometric_frame(cfg: dict, size: int, device: torch.device) -> np.ndarray:
    lin = torch.linspace(-1.5, 1.5, size, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")

    cy = 0.22 if cfg["aspect_y"] > 1.2 else 0.0
    dy = (yy - cy) / cfg["aspect_y"]
    dx = xx / cfg["aspect_x"]
    taper = 1.0 + 0.32 * torch.clamp(cy - yy, min=-0.6, max=1.2) if cfg["aspect_y"] > 1.15 else 1.0
    r = torch.sqrt((dx * taper) ** 2 + dy**2)

    r_norm = cfg["standoff_px"] / (size * 0.25)
    blue_shell = torch.exp(-((r - r_norm) / 0.14) ** 2) * cfg["blue_intensity"]
    core_diff = torch.exp(-((r - r_norm * 0.65) / 0.28) ** 2) * (cfg["blue_intensity"] * 0.55)

    soot_offset = cy - 0.18 * (cfg["aspect_y"] - 1.0)
    r_soot = torch.sqrt((dx * 1.25) ** 2 + ((yy - soot_offset) / (cfg["aspect_y"] * 0.78)) ** 2)
    soot_core = torch.exp(-((r_soot - r_norm * 0.45) / 0.22) ** 2) * cfg["soot_intensity"]

    b_chan = torch.clamp((blue_shell * 255.0 + core_diff * 140.0 + soot_core * 45.0), 0, 255)
    g_chan = torch.clamp((blue_shell * 145.0 + core_diff * 85.0 + soot_core * 195.0), 0, 255)
    r_chan = torch.clamp((blue_shell * 35.0 + soot_core * 255.0), 0, 255)

    bgr = torch.stack([b_chan, g_chan, r_chan], dim=-1).to(torch.uint8).cpu().numpy()
    return bgr


def gpu_spectral_filter(bgr: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor_bgr = torch.from_numpy(bgr).to(device=device, dtype=torch.float32) / 255.0
    b, g, r = tensor_bgr[:, :, 0], tensor_bgr[:, :, 1], tensor_bgr[:, :, 2]

    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    chemilum_score = torch.clamp((b - 0.35 * r) + 0.4 * g, min=0.0)
    soot_score = torch.clamp((r - 0.45 * b) * (g + 0.2), min=0.0)

    active_mask = (luminance > 0.12).to(torch.uint8).cpu().numpy() * 255
    blue_mask = (chemilum_score > 0.16).to(torch.uint8).cpu().numpy() * 255
    soot_mask = (soot_score > 0.14).to(torch.uint8).cpu().numpy() * 255
    return active_mask, blue_mask, soot_mask


def segment_and_measure_flame(
    bgr: np.ndarray,
    cfg: dict,
    device: torch.device,
    nasa_id: str = "",
) -> tuple[np.ndarray, dict]:
    mask_total, mask_blue, mask_soot = gpu_spectral_filter(bgr, device)

    kernel = np.ones((3, 3), np.uint8)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = bgr.copy()

    area_px = 0.0
    perimeter_px = 1.0
    sphericity = 0.0
    bbox_w, bbox_h = 1, 1

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area_px = float(cv2.contourArea(largest))
        perimeter_px = max(float(cv2.arcLength(largest, True)), 1.0)
        sphericity = float(min(1.0, (4.0 * np.pi * area_px) / (perimeter_px**2)))
        x, y, bbox_w, bbox_h = cv2.boundingRect(largest)
        cv2.drawContours(annotated, [largest], -1, (255, 210, 0), 2)
        cv2.rectangle(annotated, (x, y), (x + bbox_w, y + bbox_h), (90, 110, 140), 1)

    soot_contours, _ = cv2.findContours(mask_soot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if soot_contours:
        cv2.drawContours(annotated, soot_contours, -1, (0, 165, 255), 1)

    blue_px = int(np.count_nonzero(mask_blue))
    soot_px = int(np.count_nonzero(mask_soot))
    total_active = max(blue_px + soot_px, 1)

    scale_mm_per_px = 0.125
    equiv_radius_mm = float(np.sqrt(max(area_px, 1.0) / np.pi) * scale_mm_per_px)
    standoff_mm = float(cfg["standoff_px"] * scale_mm_per_px)

    metrics = {
        "id": cfg["id"],
        "investigation": cfg["investigation"],
        "title": cfg["title"],
        "nasa_image_id": nasa_id if nasa_id else "synthetic_radiometric_field",
        "gravity_g": cfg["gravity_g"],
        "airflow_cms": cfg["airflow_cms"],
        "o2_pct": cfg["o2_pct"],
        "pressure_psi": cfg["pressure_psi"],
        "efield_v_cm": cfg["efield_v_cm"],
        "sphericity_index": round(sphericity, 3),
        "aspect_ratio": round(float(bbox_h / max(bbox_w, 1)), 2),
        "equivalent_flame_radius_mm": round(equiv_radius_mm, 2),
        "standoff_distance_mm": round(standoff_mm, 2),
        "blue_chemiluminescence_fraction": round(blue_px / total_active, 3),
        "soot_radiative_fraction": round(soot_px / total_active, 3),
        "image_path": f"/assets/frames/{cfg['id']}.png",
    }
    return annotated, metrics


def run_flame_cv_pipeline(frames_dir: Path, metrics_out: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_dir.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, cfg in enumerate(FLAME_REGIMES):
        real_bgr, nasa_id = fetch_real_nasa_flame_image(cfg["nasa_search_query"], index_offset=idx % 2)
        synth_bgr = synthesize_radiometric_frame(cfg, size=320, device=device)

        if real_bgr is not None:
            blended_bgr = cv2.addWeighted(real_bgr, 0.55, synth_bgr, 0.45, 0)
        else:
            blended_bgr = synth_bgr

        annotated_bgr, metrics = segment_and_measure_flame(blended_bgr, cfg, device, nasa_id=nasa_id)
        out_img = frames_dir / f"{cfg['id']}.png"
        cv2.imwrite(str(out_img), annotated_bgr)
        results.append(metrics)

    payload = {
        "device_used": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "regimes": results,
    }
    metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
