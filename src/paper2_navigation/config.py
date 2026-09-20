"""Repository-relative paths used by the migrated Paper 2 code."""

from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "processed" / "vlp_calibration"
EXPERIMENT_DATA_DIR = RAW_DATA_DIR

MODELS_DIR = PROJ_ROOT / "models"
VLP_MODELS_DIR = MODELS_DIR / "vlp"
TRAINING_LOGS_DIR = MODELS_DIR / "training_logs"

RESULTS_DIR = PROJ_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FILE = TABLES_DIR / "results.tex"
