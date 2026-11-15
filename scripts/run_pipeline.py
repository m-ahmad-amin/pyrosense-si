import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyrosense.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MODELS_CHECKPOINT_DIR,
    WEB_FRAMES_DIR,
)
from pyrosense.data.psi_client import fetch_psi_s3_manifest
from pyrosense.models.pde_field_pinn import train_2d_reacting_flow_pinn
from pyrosense.models.pinn import train_pinn
from pyrosense.vision.flame_analyzer import run_flame_cv_pipeline
from pyrosense.vision.unet_trainer import train_flame_unet


def main():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Starting PyroSense SI pipeline on device: {device_name}")

    s3_out = DATA_PROCESSED_DIR / "psi_s3_manifest.json"
    manifest = fetch_psi_s3_manifest(s3_out)
    print(
        f"[1/4] Sync complete: {len(manifest.get('sampled_objects', []))} S3 objects, "
        f"{len(manifest.get('ntrs_combustion_papers', []))} NTRS papers, "
        f"{len(manifest.get('nasa_image_library_flames', []))} NASA flame images."
    )

    pinn_res = train_pinn(
        csv_path=DATA_RAW_DIR / "psi_burn_runs.csv",
        model_out=MODELS_CHECKPOINT_DIR / "pinn_flammability.pt",
        grid_out=DATA_PROCESSED_DIR / "pinn_phase_grid.json",
        epochs=2500,
    )
    print(f"[2/4] Residual PINN complete on {pinn_res['gpu_name']}: {pinn_res['validation_metrics']}")

    pde_res = train_2d_reacting_flow_pinn(
        checkpoint_out=MODELS_CHECKPOINT_DIR / "pinn_2d_reacting_flow.pt",
        json_out=DATA_PROCESSED_DIR / "pinn_2d_fields.json",
        image_out=WEB_FRAMES_DIR / "pinn_2d_fields.png",
        epochs=2400,
        batch_colloc=8192,
    )
    print(f"[3/4] 2D Navier-Stokes PDE PINN complete: {len(pde_res['regime_field_summaries'])} regimes rendered.")

    cv_res = run_flame_cv_pipeline(
        frames_dir=WEB_FRAMES_DIR,
        metrics_out=DATA_PROCESSED_DIR / "flame_morphology_metrics.json",
    )
    unet_res = train_flame_unet(
        checkpoint_out=MODELS_CHECKPOINT_DIR / "flame_unet.pt",
        metrics_out=DATA_PROCESSED_DIR / "unet_vision_metrics.json",
        strip_img_out=WEB_FRAMES_DIR / "unet_segmentation_strip.png",
        epochs=80,
    )
    print(
        f"[4/4] Flame CV ({len(cv_res['regimes'])} regimes) & FlameUNet "
        f"(val mIoU={unet_res['final_val_miou']}) complete."
    )


if __name__ == "__main__":
    main()
