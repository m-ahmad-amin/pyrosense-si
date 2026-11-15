import base64
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    def __init__(self, in_features: int = 6, mapping_size: int = 64, scale: float = 2.5):
        super().__init__()
        self.register_buffer("B", torch.randn((in_features, mapping_size)) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * torch.matmul(x, self.B)
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(x + h)


class ReactingFlow2DPINN(nn.Module):
    def __init__(
        self,
        in_features: int = 6,
        hidden_dim: int = 160,
        mapping_size: int = 64,
        num_res_blocks: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.mapping_size = mapping_size
        self.num_res_blocks = num_res_blocks
        self.embed = FourierEmbedding(in_features=in_features, mapping_size=mapping_size, scale=2.2)
        in_total = in_features + mapping_size * 2

        if num_res_blocks > 0:
            layers = [nn.Linear(in_total, hidden_dim), nn.SiLU()]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(hidden_dim))
            layers.append(nn.Linear(hidden_dim, 4))
            self.net = nn.Sequential(*layers)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_total, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 4),
            )

    def forward(self, coords_params: torch.Tensor) -> torch.Tensor:
        z_emb = self.embed(coords_params)
        raw = self.net(z_emb)
        temp_norm = torch.sigmoid(raw[:, 0:1])
        y_o2 = torch.sigmoid(raw[:, 1:2])
        u_z = nn.functional.softplus(raw[:, 2:3])
        reaction_rate = torch.sigmoid(raw[:, 3:4])
        return torch.cat([temp_norm, y_o2, u_z, reaction_rate], dim=-1)


def load_2d_pinn_checkpoint(checkpoint_path: Path, device: torch.device) -> ReactingFlow2DPINN:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    model = ReactingFlow2DPINN(
        in_features=arch.get("in_features", 6),
        hidden_dim=arch.get("hidden_dim", 160),
        mapping_size=arch.get("mapping_size", 64),
        num_res_blocks=arch.get("num_res_blocks", 0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def compute_2d_navier_stokes_energy_pde_loss(
    model: ReactingFlow2DPINN,
    batch_pts: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    batch_pts.requires_grad_(True)
    out = model(batch_pts)

    T = out[:, 0:1]
    Y_o2 = out[:, 1:2]
    U_z = out[:, 2:3]
    omega = out[:, 3:4]

    r = batch_pts[:, 0:1]
    z = batch_pts[:, 1:2]
    g_norm = batch_pts[:, 2:3]
    o2_amb = batch_pts[:, 3:4]
    p_norm = batch_pts[:, 4:5]
    u_in = batch_pts[:, 5:6]

    ones = torch.ones_like(T)
    grad_T = torch.autograd.grad(T, batch_pts, grad_outputs=ones, create_graph=True)[0]
    dT_dr = grad_T[:, 0:1]
    dT_dz = grad_T[:, 1:2]
    d2T_dr2 = torch.autograd.grad(dT_dr, batch_pts, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    d2T_dz2 = torch.autograd.grad(dT_dz, batch_pts, grad_outputs=ones, create_graph=True)[0][:, 1:2]

    grad_U = torch.autograd.grad(U_z, batch_pts, grad_outputs=ones, create_graph=True)[0]
    dU_dr = grad_U[:, 0:1]
    dU_dz = grad_U[:, 1:2]
    d2U_dr2 = torch.autograd.grad(dU_dr, batch_pts, grad_outputs=ones, create_graph=True)[0][:, 0:1]

    grad_Y = torch.autograd.grad(Y_o2, batch_pts, grad_outputs=ones, create_graph=True)[0]
    dY_dr = grad_Y[:, 0:1]
    dY_dz = grad_Y[:, 1:2]
    d2Y_dr2 = torch.autograd.grad(dY_dr, batch_pts, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    d2Y_dz2 = torch.autograd.grad(dY_dz, batch_pts, grad_outputs=ones, create_graph=True)[0][:, 1:2]

    u_buoy = 1.45 * torch.sqrt(torch.clamp(g_norm, min=0.0) + 1e-4) * (p_norm**0.5) * T
    u_target = u_in + u_buoy
    res_momentum = U_z * dU_dz - 0.08 * d2U_dr2 - 1.2 * (u_target - U_z)

    stretch = 1.0 + 1.35 * (g_norm**0.45) + 0.55 * u_in
    r_front = 0.36 / (stretch**0.35)
    z_center = -0.15 + 0.28 * (g_norm**0.4) + 0.12 * u_in
    dist_shell = torch.sqrt((r / 0.85) ** 2 + ((z - z_center) / stretch) ** 2 + 1e-4)

    arrhenius_source = (
        (o2_amb / 0.34) ** 0.75
        * (p_norm**0.35)
        * torch.exp(-((dist_shell - r_front) / 0.16) ** 2)
        * (1.0 - 0.35 * torch.exp(-(u_in + u_buoy) / 0.06))
    )
    radiation_sink = 0.22 * (T**4) * torch.exp(-(u_in + u_buoy) / 0.25)

    res_energy = (
        0.12 * U_z * dT_dz
        - 0.04 * (d2T_dr2 + d2T_dz2)
        - 1.65 * (arrhenius_source - T)
        + radiation_sink
    )
    res_species = (
        0.10 * U_z * dY_dz
        - 0.04 * (d2Y_dr2 + d2Y_dz2)
        + 1.4 * (Y_o2 - o2_amb * (1.0 - 0.82 * arrhenius_source))
    )
    res_omega = omega - arrhenius_source

    loss_m = torch.mean(res_momentum**2)
    loss_e = torch.mean(res_energy**2)
    loss_s = torch.mean(res_species**2)
    loss_w = torch.mean(res_omega**2)
    total = loss_m + 1.5 * loss_e + loss_s + 1.2 * loss_w

    breakdown = {
        "total_pde_loss": round(float(total.item()), 5),
        "momentum_pde": round(float(loss_m.item()), 5),
        "energy_radiation_pde": round(float(loss_e.item()), 5),
        "species_transport_pde": round(float(loss_s.item()), 5),
    }
    return total, breakdown


def predict_live_2d_field(
    model: ReactingFlow2DPINN,
    device: torch.device,
    gravity_g: float,
    o2_pct: float,
    pressure_psi: float,
    airflow_cms: float,
    grid_n: int = 112,
) -> dict:
    r_lin = torch.linspace(-1.0, 1.0, grid_n, device=device)
    z_lin = torch.linspace(1.0, -1.0, grid_n, device=device)
    zz, rr = torch.meshgrid(z_lin, r_lin, indexing="ij")
    r_flat = rr.reshape(-1, 1)
    z_flat = zz.reshape(-1, 1)

    n_pts = r_flat.shape[0]
    g_val = float(np.clip(gravity_g, 0.0, 1.0))
    o2_val = float(np.clip(o2_pct / 100.0, 0.15, 0.85))
    p_val = float(np.clip(pressure_psi / 14.7, 0.30, 1.20))
    u_val = float(np.clip(airflow_cms / 55.0, 0.02, 0.75))

    inp = torch.cat(
        [
            r_flat,
            z_flat,
            torch.full((n_pts, 1), g_val, device=device),
            torch.full((n_pts, 1), o2_val, device=device),
            torch.full((n_pts, 1), p_val, device=device),
            torch.full((n_pts, 1), u_val, device=device),
        ],
        dim=-1,
    )

    model.eval()
    with torch.no_grad():
        out = model(inp)

    T_map = (300.0 + 2150.0 * out[:, 0]).reshape(grid_n, grid_n).cpu().numpy()
    O2_map = out[:, 1].reshape(grid_n, grid_n).cpu().numpy()
    U_map = (out[:, 2] * 45.0).reshape(grid_n, grid_n).cpu().numpy()

    norm_t = np.clip((T_map - 300.0) / 2150.0 * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_t, cv2.COLORMAP_INFERNO)
    heatmap = cv2.resize(heatmap, (320, 320), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", heatmap)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    return {
        "gravity_g": round(gravity_g, 3),
        "o2_pct": round(o2_pct, 1),
        "pressure_psi": round(pressure_psi, 2),
        "airflow_cms": round(airflow_cms, 1),
        "peak_field_temp_k": round(float(np.max(T_map)), 1),
        "mean_plume_velocity_cms": round(float(np.mean(U_map)), 2),
        "min_core_o2_mole_fraction": round(float(np.min(O2_map)), 4),
        "heatmap_data_url": f"data:image/png;base64,{b64}",
    }


def render_4panel_2d_fields(
    model: ReactingFlow2DPINN,
    device: torch.device,
    out_img_path: Path,
) -> list[dict]:
    regimes = [
        {"id": "iss_0g", "label": "ISS 0.00g (21% O2, 14.7 psi)", "g": 0.00, "o2": 0.21, "p": 1.00, "u": 0.12},
        {"id": "moon_016g", "label": "Lunar Base 0.16g (34% O2, 10.2 psi)", "g": 0.16, "o2": 0.34, "p": 0.69, "u": 0.18},
        {"id": "mars_038g", "label": "Mars Habitat 0.38g (30% O2, 10.2 psi)", "g": 0.38, "o2": 0.30, "p": 0.69, "u": 0.22},
        {"id": "earth_1g", "label": "Earth 1.00g (21% O2, 14.7 psi)", "g": 1.00, "o2": 0.21, "p": 1.00, "u": 0.45},
    ]

    grid_n = 160
    r_lin = torch.linspace(-1.0, 1.0, grid_n, device=device)
    z_lin = torch.linspace(1.0, -1.0, grid_n, device=device)
    zz, rr = torch.meshgrid(z_lin, r_lin, indexing="ij")
    r_flat = rr.reshape(-1, 1)
    z_flat = zz.reshape(-1, 1)

    panels = []
    summaries = []
    model.eval()
    with torch.no_grad():
        for reg in regimes:
            n_pts = r_flat.shape[0]
            g_col = torch.full((n_pts, 1), reg["g"], device=device)
            o2_col = torch.full((n_pts, 1), reg["o2"], device=device)
            p_col = torch.full((n_pts, 1), reg["p"], device=device)
            u_col = torch.full((n_pts, 1), reg["u"], device=device)

            inp = torch.cat([r_flat, z_flat, g_col, o2_col, p_col, u_col], dim=-1)
            out = model(inp)

            T_map = (300.0 + 2150.0 * out[:, 0]).reshape(grid_n, grid_n).cpu().numpy()
            O2_map = out[:, 1].reshape(grid_n, grid_n).cpu().numpy()
            U_map = (out[:, 2] * 45.0).reshape(grid_n, grid_n).cpu().numpy()

            norm_t = np.clip((T_map - 300.0) / 2150.0 * 255.0, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm_t, cv2.COLORMAP_INFERNO)

            cv2.putText(
                heatmap,
                reg["label"],
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (240, 248, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(heatmap)
            summaries.append(
                {
                    "regime_id": reg["id"],
                    "label": reg["label"],
                    "gravity_g": reg["g"],
                    "peak_field_temp_k": round(float(np.max(T_map)), 1),
                    "mean_plume_velocity_cms": round(float(np.mean(U_map)), 2),
                    "min_core_o2_mole_fraction": round(float(np.min(O2_map)), 4),
                }
            )

    combined = np.hstack(panels)
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), combined)
    return summaries


def train_2d_reacting_flow_pinn(
    checkpoint_out: Path,
    json_out: Path,
    image_out: Path,
    epochs: int = 1800,
    batch_colloc: int = 4096,
    lr: float = 2e-3,
    hidden_dim: int = 160,
    mapping_size: int = 64,
    num_res_blocks: int = 0,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    model = ReactingFlow2DPINN(
        in_features=6,
        hidden_dim=hidden_dim,
        mapping_size=mapping_size,
        num_res_blocks=num_res_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log_every = max(1, epochs // 12)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        r = -1.0 + 2.0 * torch.rand((batch_colloc, 1), device=device)
        z = -1.0 + 2.0 * torch.rand((batch_colloc, 1), device=device)
        g = torch.rand((batch_colloc, 1), device=device)
        o2 = 0.15 + 0.70 * torch.rand((batch_colloc, 1), device=device)
        p = 0.30 + 0.70 * torch.rand((batch_colloc, 1), device=device)
        u = 0.02 + 0.65 * torch.rand((batch_colloc, 1), device=device)

        pts = torch.cat([r, z, g, o2, p, u], dim=-1)
        loss, breakdown = compute_2d_navier_stokes_energy_pde_loss(model, pts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            entry = {"epoch": epoch, **breakdown}
            history.append(entry)
            print(f"[2D-PINN Epoch {epoch}/{epochs}] {entry}")

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "device": str(device),
            "arch": {
                "in_features": 6,
                "hidden_dim": hidden_dim,
                "mapping_size": mapping_size,
                "num_res_blocks": num_res_blocks,
            },
            "history": history,
        },
        checkpoint_out,
    )

    field_summaries = render_4panel_2d_fields(model, device, image_out)
    payload = {
        "device_used": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "epochs": epochs,
        "collocation_points_per_epoch": batch_colloc,
        "hidden_dim": hidden_dim,
        "num_res_blocks": num_res_blocks,
        "pde_training_history": history,
        "regime_field_summaries": field_summaries,
        "field_heatmap_path": "/assets/frames/pinn_2d_fields.png",
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
