from pydantic import BaseModel


class RankRequest(BaseModel):
    o2_pct: float = 34.0
    pressure_psi: float = 10.2
    gravity_g: float = 0.16
    airflow_cms: float = 12.0
    pillar: str = "All"


class PINNPredictRequest(BaseModel):
    material: str = "Sibal_Cotton_Fiberglass"
    o2_pct: float = 34.0
    pressure_psi: float = 10.2
    airflow_cms: float = 8.0
    gravity_g: float = 0.16
    efield_v_cm: float = 0.0


class CopilotRequest(BaseModel):
    question: str
    role: str = "engineer"
