"""Central configuration for the AI Traffic Optimization Agent."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "samples"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_DIR, SAMPLE_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---- Traffic data ----
FREQ_MIN = 15                       # minutes per record
STEPS_PER_HOUR = 60 // FREQ_MIN
TRAFFIC_CSV = DATA_DIR / "synthetic_traffic.csv"

# ---- Model artefacts ----
Q_TABLE_PATH = MODELS_DIR / "q_table.npy"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_iforest.joblib"
SB3_MODEL_PATH = MODELS_DIR / "ppo_signal.zip"

def forecast_model_path(horizon: int):
    return MODELS_DIR / f"forecast_h{horizon}.joblib"

# ---- Impact assumptions (planning values, not measurements) ----
IDLE_FUEL_L_PER_HOUR = 0.8          # petrol car idling
CO2_KG_PER_LITRE = 2.31             # petrol combustion
