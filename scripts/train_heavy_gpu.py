import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyrosense.config import (
    DATA_PROCESSED_DIR,
    MODELS_CHECKPOINT_DIR,
    WEB_FRAMES_DIR,
)
from pyrosense.models.pde_field_pinn import train_2d_reacting_flow_pinn
from pyrosense.vision.unet_trainer import train_flame_unet


def main():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Starting PyroSense-SI Heavy GPU Training on: {device_name}")

    print("\n=== HEAVY STAGE 1/2: Continuous 2D Navier-Stokes + Arrhenius + Radiation PINN ===")
    pde_res = train_2d_reacting_flow_pinn(
        checkpoint_out=MODELS_CHECKPOINT_DIR / "pinn_2d_reacting_flow.pt",
        json_out=DATA_PROCESSED_DIR / "pinn_2d_fields.json",
        image_out=WEB_FRAMES_DIR / "pinn_2d_fields.png",
        epochs=10000,
        batch_colloc=16384,
        lr=1.5e-3,
        hidden_dim=256,
        mapping_size=96,
        num_res_blocks=4,
    )
    print(
        f"2D-PINN complete on {pde_res['gpu_name']} "
        f"(final PDE loss={pde_res['pde_training_history'][-1]['total_pde_loss']})."
    )

    print("\n=== HEAVY STAGE 2/2: Real NASA Flame Video/Image FlameUNet (256x256, base_c=64) ===")
    unet_res = train_flame_unet(
        checkpoint_out=MODELS_CHECKPOINT_DIR / "flame_unet.pt",
        metrics_out=DATA_PROCESSED_DIR / "unet_vision_metrics.json",
        strip_img_out=WEB_FRAMES_DIR / "unet_segmentation_strip.png",
        epochs=80,
        batch_size=24,
        lr=1.5e-3,
        target_size=256,
        max_frames=3600,
        base_c=64,
        include_videos=True,
    )
    print(
        f"FlameUNet complete on {unet_res['gpu_name']}: "
        f"{unet_res['total_frames_trained']} frames @ {unet_res['resolution']}, "
        f"val mIoU={unet_res['final_val_miou']}, breakdown={unet_res['dataset_breakdown']}."
    )


if __name__ == "__main__":
    main()
