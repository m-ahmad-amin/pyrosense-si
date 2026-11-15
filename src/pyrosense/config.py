import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
WEB_DIR = REPO_ROOT / "web"
WEB_FRAMES_DIR = WEB_DIR / "assets" / "frames"


def load_config() -> dict:
    cfg_file = CONFIG_DIR / "default_config.json"
    if cfg_file.exists():
        return json.loads(cfg_file.read_text(encoding="utf-8"))
    return {}
