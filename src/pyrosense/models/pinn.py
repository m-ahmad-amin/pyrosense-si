import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


MATERIAL_ENCODING = {
    "Sibal_Cotton_Fiberglass": [1.0, 0.30, 15.8],
    "PMMA_Black": [0.42, 1.00, 16.5],
    "PMMA_Clear": [0.35, 3.20, 17.0],
    "Nomex_HT90": [0.18, 0.45, 25.8],
    "Silicone_Sheet": [0.38, 0.60, 18.9],
    "Ethylene_Gas_Simulant": [0.65, 2.00, 15.2],
    "Methane_Gas_Simulant": [0.55, 2.00, 16.0],
    "Propane_Inverse": [0.78, 1.50, 13.5],
    "Heptane_Droplet": [0.60, 3.20, 15.2],
    "Decane_Droplet": [0.56, 3.80, 15.5],
    "Dodecane_Droplet": [0.50, 4.50, 15.0],
}


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_p: float = 0.05):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act1 = nn.SiLU()
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(dim, dim)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.fc1(x))
        out = self.drop(out)
        out = self.fc2(out)
        return self.act2(x + out)


class CombustionPINN(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )
        self.res1 = ResidualBlock(hidden_dim, dropout_p=0.05)
        self.res2 = ResidualBlock(hidden_dim, dropout_p=0.05)
        self.head = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.res1(h)
        h = self.res2(h)
        raw = self.head(h)
        spread_rate = nn.functional.softplus(raw[:, 0:1])
        loi_pct = 8.0 + 24.0 * torch.sigmoid(raw[:, 1:2])
        peak_temp_k = 550.0 + 2200.0 * torch.sigmoid(raw[:, 2:3])
        rad_frac = 0.08 + 0.65 * torch.sigmoid(raw[:, 3:4])
        return torch.cat([spread_rate, loi_pct, peak_temp_k, rad_frac], dim=1)


def compute_effective_velocity(
    flow_cms: torch.Tensor,
    gravity_g: torch.Tensor,
    pressure_psi: torch.Tensor,
    efield_v_cm: torch.Tensor,
) -> torch.Tensor:
    g_equiv = gravity_g + (efield_v_cm / 950.0) ** 1.35
    u_buoyant = 32.0 * torch.sqrt(torch.clamp(g_equiv, min=0.0)) * (pressure_psi / 14.7) ** 0.5
    return torch.sqrt(flow_cms**2 + u_buoyant**2 + 0.25)


def physics_residual_loss(inputs_raw: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    o2_pct = inputs_raw[:, 0:1]
    pressure_psi = inputs_raw[:, 1:2]
    flow_cms = inputs_raw[:, 2:3]
    gravity_g = inputs_raw[:, 3:4]
    efield_v_cm = inputs_raw[:, 4:5]
    reactivity = inputs_raw[:, 5:6]
    base_loi = inputs_raw[:, 7:8]

    u_eff = compute_effective_velocity(flow_cms, gravity_g, pressure_psi, efield_v_cm)
    o2_ratio = torch.clamp((o2_pct - 12.0) / 21.0, min=0.02)
    pressure_factor = (pressure_psi / 14.7) ** 0.65

    quench_factor = 1.0 - torch.exp(-u_eff / 2.2)
    blowoff_factor = torch.exp(-torch.clamp(u_eff - 22.0, min=0.0) / 38.0)
    vel_factor = (u_eff / 10.0) ** 0.48 * quench_factor * blowoff_factor

    theo_spread = 4.8 * reactivity * (o2_ratio**1.55) * pressure_factor * vel_factor
    theo_loi = (
        base_loi
        - 1.9 * torch.exp(-((u_eff - 6.5) / 5.5) ** 2)
        + 2.4 * torch.exp(-u_eff / 1.4)
        - 1.2 * ((pressure_psi - 10.2) / 14.7)
    )
    theo_temp = 1150.0 + 720.0 * (o2_pct / 21.0) ** 0.55 * quench_factor
    theo_rad = 0.18 + 0.38 * torch.exp(-u_eff / 8.5)

    loss_spread = torch.mean((preds[:, 0:1] - theo_spread) ** 2)
    loss_loi = torch.mean((preds[:, 1:2] - theo_loi) ** 2) * 0.15
    loss_temp = torch.mean(((preds[:, 2:3] - theo_temp) / 400.0) ** 2)
    loss_rad = torch.mean((preds[:, 3:4] - theo_rad) ** 2) * 2.0
    return loss_spread + loss_loi + loss_temp + loss_rad


def normalize_features(x_raw: torch.Tensor) -> torch.Tensor:
    scales = torch.tensor(
        [85.0, 14.7, 55.0, 1.0, 2100.0, 1.0, 15.0, 26.0],
        device=x_raw.device,
        dtype=x_raw.dtype,
    )
    return x_raw / scales


def build_training_tensors(csv_path: Path, device: torch.device):
    df = pd.read_csv(csv_path)
    rows_x = []
    rows_y = []
    for _, row in df.iterrows():
        mat_props = MATERIAL_ENCODING.get(row["material"], [0.5, 1.0, 16.5])
        rows_x.append(
            [
                float(row["o2_pct"]),
                float(row["pressure_psi"]),
                float(row["flow_cms"]),
                float(row["gravity_g"]),
                float(row["efield_v_cm"]),
                float(mat_props[0]),
                float(row["thickness_mm"]),
                float(mat_props[2]),
            ]
        )
        rows_y.append(
            [
                float(row["flame_spread_mm_s"]),
                float(row["loi_pct"]),
                float(row["peak_temp_k"]),
                float(row["radiative_fraction"]),
            ]
        )
    x_data = torch.tensor(rows_x, dtype=torch.float32, device=device)
    y_data = torch.tensor(rows_y, dtype=torch.float32, device=device)
    return x_data, y_data


def sample_collocation_points(n_pts: int, device: torch.device) -> torch.Tensor:
    o2 = 15.0 + 70.0 * torch.rand((n_pts, 1), device=device)
    pressure = 4.3 + 10.4 * torch.rand((n_pts, 1), device=device)
    flow = 55.0 * torch.rand((n_pts, 1), device=device)
    gravity = torch.rand((n_pts, 1), device=device)
    efield = torch.zeros((n_pts, 1), device=device)
    reactivity = 0.18 + 0.82 * torch.rand((n_pts, 1), device=device)
    thickness = 0.3 + 6.0 * torch.rand((n_pts, 1), device=device)
    base_loi = 14.5 + 11.5 * torch.rand((n_pts, 1), device=device)
    return torch.cat([o2, pressure, flow, gravity, efield, reactivity, thickness, base_loi], dim=1)


def compute_validation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {}
    names = ["flame_spread_mm_s", "loi_pct", "peak_temp_k", "radiative_fraction"]
    for idx, name in enumerate(names):
        t = y_true[:, idx]
        p = y_pred[:, idx]
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        mae = float(np.mean(np.abs(t - p)))
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2)) + 1e-8
        r2 = float(1.0 - ss_res / ss_tot)
        metrics[name] = {
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
        }
    return metrics


def predict_with_uncertainty(
    model: CombustionPINN,
    x_norm: torch.Tensor,
    n_samples: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x_norm).cpu().numpy())
    model.eval()
    stacked = np.stack(preds, axis=0)
    mean_pred = np.mean(stacked, axis=0)
    std_pred = np.std(stacked, axis=0)
    return mean_pred, std_pred


def train_pinn(
    csv_path: Path,
    model_out: Path,
    grid_out: Path,
    epochs: int = 1500,
    lr: float = 3e-3,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    x_data, y_data = build_training_tensors(csv_path, device)
    model = CombustionPINN(in_dim=8, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred_data = model(normalize_features(x_data))

        loss_spread = torch.mean((pred_data[:, 0] - y_data[:, 0]) ** 2)
        loss_loi = torch.mean((pred_data[:, 1] - y_data[:, 1]) ** 2) * 0.25
        loss_temp = torch.mean(((pred_data[:, 2] - y_data[:, 2]) / 350.0) ** 2)
        loss_rad = torch.mean((pred_data[:, 3] - y_data[:, 3]) ** 2) * 2.5
        data_loss = loss_spread + loss_loi + loss_temp + loss_rad

        x_colloc = sample_collocation_points(768, device)
        pred_colloc = model(normalize_features(x_colloc))
        pde_loss = physics_residual_loss(x_colloc, pred_colloc)

        total_loss = data_loss + 0.30 * pde_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 300 == 0 or epoch == 1 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "total_loss": round(float(total_loss.item()), 5),
                    "data_loss": round(float(data_loss.item()), 5),
                    "physics_loss": round(float(pde_loss.item()), 5),
                }
            )

    model.eval()
    with torch.no_grad():
        final_preds = model(normalize_features(x_data)).cpu().numpy()
    val_metrics = compute_validation_metrics(y_data.cpu().numpy(), final_preds)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "device": str(device),
            "training_history": history,
            "validation_metrics": val_metrics,
            "material_encoding": MATERIAL_ENCODING,
        },
        model_out,
    )

    gravity_grid = [0.0, 0.05, 0.10, 0.16, 0.25, 0.38, 0.55, 0.75, 1.00]
    flow_grid = [0.0, 2.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 45.0]
    materials_to_export = [
        "Sibal_Cotton_Fiberglass",
        "PMMA_Black",
        "PMMA_Clear",
        "Nomex_HT90",
        "Silicone_Sheet",
    ]
    atmospheres = [
        {"id": "iss_21_o2", "o2_pct": 21.0, "pressure_psi": 14.7},
        {"id": "lunar_34_o2", "o2_pct": 34.0, "pressure_psi": 10.2},
        {"id": "mars_30_o2", "o2_pct": 30.0, "pressure_psi": 10.2},
        {"id": "suit_85_o2", "o2_pct": 85.0, "pressure_psi": 4.3},
    ]

    exported_curves = {}
    for mat in materials_to_export:
        mat_props = MATERIAL_ENCODING[mat]
        exported_curves[mat] = {}
        for atm in atmospheres:
            g_inputs = []
            for g_val in gravity_grid:
                g_inputs.append(
                    [
                        atm["o2_pct"],
                        atm["pressure_psi"],
                        6.0,
                        g_val,
                        0.0,
                        mat_props[0],
                        mat_props[1],
                        mat_props[2],
                    ]
                )
            t_g = torch.tensor(g_inputs, dtype=torch.float32, device=device)
            mean_g, std_g = predict_with_uncertainty(model, normalize_features(t_g), n_samples=24)

            gravity_sweep = []
            for idx, g_val in enumerate(gravity_grid):
                gravity_sweep.append(
                    {
                        "gravity_g": g_val,
                        "flame_spread_mm_s": round(float(mean_g[idx, 0]), 3),
                        "flame_spread_ci95": round(float(1.96 * std_g[idx, 0]), 3),
                        "loi_pct": round(float(mean_g[idx, 1]), 2),
                        "loi_ci95": round(float(1.96 * std_g[idx, 1]), 2),
                        "peak_temp_k": round(float(mean_g[idx, 2]), 1),
                        "radiative_fraction": round(float(mean_g[idx, 3]), 3),
                    }
                )

            flow_sweep_0g_vs_lunar = []
            with torch.no_grad():
                for f_val in flow_grid:
                    inp_pair = torch.tensor(
                        [
                            [atm["o2_pct"], atm["pressure_psi"], f_val, 0.0, 0.0, mat_props[0], mat_props[1], mat_props[2]],
                            [atm["o2_pct"], atm["pressure_psi"], f_val, 0.16, 0.0, mat_props[0], mat_props[1], mat_props[2]],
                            [atm["o2_pct"], atm["pressure_psi"], f_val, 1.00, 0.0, mat_props[0], mat_props[1], mat_props[2]],
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    out_pair = model(normalize_features(inp_pair)).cpu().numpy()
                    flow_sweep_0g_vs_lunar.append(
                        {
                            "flow_cms": f_val,
                            "spread_0g": round(float(out_pair[0, 0]), 3),
                            "spread_lunar_016g": round(float(out_pair[1, 0]), 3),
                            "spread_earth_1g": round(float(out_pair[2, 0]), 3),
                            "loi_0g": round(float(out_pair[0, 1]), 2),
                            "loi_lunar_016g": round(float(out_pair[1, 1]), 2),
                            "loi_earth_1g": round(float(out_pair[2, 1]), 2),
                        }
                    )

            exported_curves[mat][atm["id"]] = {
                "gravity_sweep_at_6cms": gravity_sweep,
                "flow_sweep": flow_sweep_0g_vs_lunar,
            }

    payload = {
        "device_used": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "training_history": history,
        "validation_metrics": val_metrics,
        "gravity_grid": gravity_grid,
        "flow_grid": flow_grid,
        "predictions": exported_curves,
    }
    grid_out.parent.mkdir(parents=True, exist_ok=True)
    grid_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
