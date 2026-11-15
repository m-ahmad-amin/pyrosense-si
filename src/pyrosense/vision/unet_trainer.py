import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FlameUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_c: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_c * 2, base_c * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_c * 4, base_c * 8)

        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_c * 8, base_c * 4)
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_c * 4, base_c * 2)
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_c * 2, base_c)

        self.head = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def create_weak_supervision_mask(bgr: np.ndarray) -> np.ndarray:
    b = bgr[:, :, 0].astype(np.float32) / 255.0
    g = bgr[:, :, 1].astype(np.float32) / 255.0
    r = bgr[:, :, 2].astype(np.float32) / 255.0

    mask = np.zeros(bgr.shape[:2], dtype=np.int64)
    blue_cond = ((b - 0.30 * r) + 0.35 * g) > 0.16
    soot_cond = ((r - 0.40 * b) * (g + 0.20)) > 0.14

    mask[blue_cond] = 1
    mask[soot_cond] = 2
    return mask


def _extract_video_frames(video_url: str, target_size: int, max_per_video: int = 120) -> list[np.ndarray]:
    extracted = []
    try:
        r = requests.get(video_url, stream=True, timeout=20)
        if r.status_code != 200:
            return extracted
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    tmp.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > 45 * 1024 * 1024:
                        break
            tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
            stride = max(1, total // max_per_video)
            idx = 0
            while len(extracted) < max_per_video:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0 and float(frame.mean()) > 4.0:
                    extracted.append(cv2.resize(frame, (target_size, target_size)))
                idx += 1
            cap.release()
    except Exception:
        pass
    return extracted


def _augment_frame(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = bgr
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.3:
        out = cv2.flip(out, 0)
    k = int(rng.integers(0, 4))
    if k > 0:
        out = np.rot90(out, k).copy()
    gain = float(rng.uniform(0.82, 1.18))
    out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out


def collect_nasa_flame_frames(
    target_size: int = 128,
    max_frames: int = 160,
    include_videos: bool = False,
) -> tuple[list[np.ndarray], dict]:
    real_frames = []
    queries = [
        "microgravity flame combustion",
        "space station flame experiment",
        "Saffire fire NASA",
        "droplet combustion space",
        "BASS combustion ISS",
        "ACME flame ISS",
    ]
    image_hits = 0
    video_hits = 0

    for q in queries:
        if len(real_frames) >= int(max_frames * 0.65):
            break
        try:
            resp = requests.get(
                "https://images-api.nasa.gov/search",
                params={"q": q, "media_type": "image"},
                timeout=12,
            )
            if resp.status_code == 200:
                items = resp.json().get("collection", {}).get("items", [])
                for it in items[:20]:
                    links = it.get("links") or []
                    if not links or not links[0].get("href"):
                        continue
                    img_r = requests.get(links[0]["href"], timeout=8)
                    if img_r.status_code == 200:
                        arr = np.frombuffer(img_r.content, dtype=np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if bgr is not None:
                            real_frames.append(cv2.resize(bgr, (target_size, target_size)))
                            image_hits += 1
                            if len(real_frames) >= int(max_frames * 0.65):
                                break
        except Exception:
            continue

    if include_videos and len(real_frames) < int(max_frames * 0.65):
        for q in queries[:4]:
            if len(real_frames) >= int(max_frames * 0.65):
                break
            try:
                vresp = requests.get(
                    "https://images-api.nasa.gov/search",
                    params={"q": q, "media_type": "video"},
                    timeout=12,
                )
                if vresp.status_code != 200:
                    continue
                items = vresp.json().get("collection", {}).get("items", [])
                for it in items[:5]:
                    nasa_id = (it.get("data") or [{}])[0].get("nasa_id")
                    if not nasa_id:
                        continue
                    aresp = requests.get(f"https://images-api.nasa.gov/asset/{nasa_id}", timeout=10)
                    if aresp.status_code != 200:
                        continue
                    hrefs = [
                        item.get("href", "")
                        for item in aresp.json().get("collection", {}).get("items", [])
                    ]
                    mp4s = [h for h in hrefs if h.lower().endswith(".mp4") and "mobile" not in h.lower()]
                    if not mp4s:
                        mp4s = [h for h in hrefs if h.lower().endswith(".mp4")]
                    if mp4s:
                        vframes = _extract_video_frames(mp4s[0], target_size=target_size, max_per_video=120)
                        if vframes:
                            real_frames.extend(vframes)
                            video_hits += len(vframes)
                            if len(real_frames) >= int(max_frames * 0.65):
                                break
            except Exception:
                continue

    rng = np.random.default_rng(42)
    frames = list(real_frames)

    if real_frames and len(frames) < max_frames:
        aug_target = min(max_frames, len(real_frames) * 8)
        while len(frames) < aug_target:
            base = real_frames[int(rng.integers(0, len(real_frames)))]
            frames.append(_augment_frame(base, rng))

    yy, xx = np.meshgrid(
        np.linspace(-1.4, 1.4, target_size),
        np.linspace(-1.4, 1.4, target_size),
        indexing="ij",
    )
    synth_count = 0
    while len(frames) < max_frames:
        aspect_y = float(rng.uniform(1.0, 2.3))
        aspect_x = float(rng.uniform(0.65, 1.05))
        cy = float(rng.uniform(-0.15, 0.25))
        cx = float(rng.uniform(-0.15, 0.15))
        r_shell = float(rng.uniform(0.28, 0.58))
        blue_amp = float(rng.uniform(0.55, 1.0))
        soot_amp = float(rng.uniform(0.05, 0.95))

        dist = np.sqrt(((xx - cx) / aspect_x) ** 2 + ((yy - cy) / aspect_y) ** 2)
        blue_layer = np.exp(-((dist - r_shell) / 0.13) ** 2) * blue_amp
        dist_soot = np.sqrt(((xx - cx) / (aspect_x * 0.8)) ** 2 + ((yy - (cy - 0.12)) / (aspect_y * 0.75)) ** 2)
        soot_layer = np.exp(-((dist_soot - r_shell * 0.45) / 0.20) ** 2) * soot_amp

        b_ch = np.clip((blue_layer * 250.0 + soot_layer * 40.0), 0, 255)
        g_ch = np.clip((blue_layer * 145.0 + soot_layer * 190.0), 0, 255)
        r_ch = np.clip((blue_layer * 35.0 + soot_layer * 255.0), 0, 255)
        synth = np.stack([b_ch, g_ch, r_ch], axis=-1).astype(np.uint8)
        frames.append(synth)
        synth_count += 1

    stats = {
        "nasa_images": image_hits,
        "nasa_video_frames": video_hits,
        "augmented_real_frames": max(0, len(frames) - len(real_frames) - synth_count),
        "synthetic_frames": synth_count,
    }
    return frames[:max_frames], stats


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 3) -> float:
    ious = []
    for cls in range(num_classes):
        p_c = preds == cls
        t_c = targets == cls
        inter = float(torch.logical_and(p_c, t_c).sum().item())
        union = float(torch.logical_or(p_c, t_c).sum().item())
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    inter = torch.sum(probs * one_hot, dim=(0, 2, 3))
    denom = torch.sum(probs + one_hot, dim=(0, 2, 3))
    dice = (2.0 * inter + 1e-4) / (denom + 1e-4)
    return 1.0 - dice.mean()


def train_flame_unet(
    checkpoint_out: Path,
    metrics_out: Path,
    strip_img_out: Path,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 2e-3,
    target_size: int = 128,
    max_frames: int = 160,
    base_c: int = 32,
    include_videos: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    frames, dataset_stats = collect_nasa_flame_frames(
        target_size=target_size,
        max_frames=max_frames,
        include_videos=include_videos,
    )
    print(f"[FlameUNet Dataset] {len(frames)} frames @ {target_size}x{target_size}: {dataset_stats}")
    masks = [create_weak_supervision_mask(f) for f in frames]

    x_np = np.stack([f.transpose(2, 0, 1).astype(np.float32) / 255.0 for f in frames], axis=0)
    y_np = np.stack(masks, axis=0)

    n_total = len(frames)
    n_train = int(n_total * 0.85)
    train_ds = TensorDataset(
        torch.from_numpy(x_np[:n_train]),
        torch.from_numpy(y_np[:n_train]).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_np[n_train:]),
        torch.from_numpy(y_np[n_train:]).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    model = FlameUNet(in_channels=3, num_classes=3, base_c=base_c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_every = max(1, epochs // 10)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = 0.7 * criterion(logits, yb) + 0.3 * soft_dice_loss(logits, yb, num_classes=3)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            steps += 1
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    logits = model(xb)
                    val_loss_sum += float(criterion(logits, yb).item())
                    val_steps += 1
                    all_preds.append(torch.argmax(logits, dim=1).cpu())
                    all_targets.append(yb.cpu())
            preds_cat = torch.cat(all_preds, dim=0)
            targets_cat = torch.cat(all_targets, dim=0)
            val_miou = compute_miou(preds_cat, targets_cat, num_classes=3)
            entry = {
                "epoch": epoch,
                "train_ce_loss": round(epoch_loss / max(steps, 1), 4),
                "val_ce_loss": round(val_loss_sum / max(val_steps, 1), 4),
                "val_miou": round(val_miou, 4),
            }
            history.append(entry)
            print(f"[FlameUNet Epoch {epoch}/{epochs}] {entry}")

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "device": str(device),
            "arch": {"in_channels": 3, "num_classes": 3, "base_c": base_c, "target_size": target_size},
            "dataset_stats": dataset_stats,
            "history": history,
        },
        checkpoint_out,
    )

    model.eval()
    strip_panels = []
    with torch.no_grad():
        sample_x = torch.from_numpy(x_np[n_train : n_train + 4]).to(device)
        sample_preds = torch.argmax(model(sample_x), dim=1).cpu().numpy()
        for idx in range(min(4, len(sample_x))):
            orig_bgr = (sample_x[idx].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            pred_mask = sample_preds[idx]
            overlay = np.zeros_like(orig_bgr)
            overlay[pred_mask == 1] = [255, 190, 0]
            overlay[pred_mask == 2] = [0, 140, 255]
            blended = cv2.addWeighted(orig_bgr, 0.65, overlay, 0.35, 0)
            strip_panels.append(blended)

    if strip_panels:
        strip_img_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(strip_img_out), np.hstack(strip_panels))

    payload = {
        "device_used": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_frames_trained": n_total,
        "resolution": f"{target_size}x{target_size}",
        "dataset_breakdown": dataset_stats,
        "final_val_miou": history[-1]["val_miou"] if history else 0.0,
        "final_val_ce_loss": history[-1]["val_ce_loss"] if history else 0.0,
        "training_history": history,
        "segmentation_strip_path": "/assets/frames/unet_segmentation_strip.png",
        "strip_image_path": "/assets/frames/unet_segmentation_strip.png",
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
