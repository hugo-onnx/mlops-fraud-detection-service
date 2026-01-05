import os
import logging
from pathlib import Path

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud-api")


class Config:
    # MLflow (optional)
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)

    MLFLOW_ENABLED = bool(MLFLOW_TRACKING_URI)

    # Paths (absolute, container-safe)
    BASE_DIR = Path(__file__).resolve().parents[2]
    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"

    DB_PATH = Path(os.getenv("DB_PATH", "/app/data/requests.db"))
    DATA_DIR = BASE_DIR / "data"
    REFERENCE_CSV = DATA_DIR / "production" / "creditcard_reference.csv"

    SCALER_PATH = MODELS_DIR / "scaler.pkl"
    FEATURES_PATH = MODELS_DIR / "feature_columns.json"

    REPORT_PATH = REPORTS_DIR / "data_drift_report.html"
    REPORT_JSON = REPORTS_DIR / "data_drift_report.json"
    
    # Model Registry Names (MLflow)
    MODEL_NAMES = {
        "lightgbm": "fraud_detection_lightgbm",
        "randomforest": "fraud_detection_randomforest",
        "logisticregression": "fraud_detection_logisticregression"
    }


settings = Config()