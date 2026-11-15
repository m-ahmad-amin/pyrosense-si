import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyrosense.config import DATA_RAW_DIR
from pyrosense.engine.relevance import (
    compute_aerosol_discrimination_curve,
    score_investigation_for_mission,
)


def test_lunar_ranking():
    catalog = json.loads((DATA_RAW_DIR / "psi_investigations.json").read_text(encoding="utf-8"))
    params = {"o2_pct": 34.0, "pressure_psi": 10.2, "gravity_g": 0.16, "airflow_cms": 12.0, "pillar": "All"}
    scores = [score_investigation_for_mission(inv, params) for inv in catalog["investigations"]]
    assert len(scores) == 16
    assert all(0.0 <= s["relevance_score"] <= 100.0 for s in scores)


def test_aerosol_curve():
    curve = compute_aerosol_discrimination_curve("kapton_wire", 1.2, 0.16)
    assert len(curve["diameters_um"]) == 25
    assert curve["count_median_diameter_smoke_um"] < curve["count_median_diameter_dust_um"]
