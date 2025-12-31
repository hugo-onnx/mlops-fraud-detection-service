import os
import logging

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud-api")

class Config:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI")

    MLFLOW_ENABLED = MLFLOW_TRACKING_URI is not None

    # Paths
    DB_PATH = "data/production/requests.db"
    REPORT_PATH = "reports/data_drift_report.html"
    REPORT_JSON = "reports/data_drift_report.json"
    REFERENCE_CSV = "data/production/creditcard_reference.csv"
    SCALER_PATH = "models/scaler.pkl"
    FEATURES_PATH = "models/feature_columns.json"
    
    # Model Registry Names
    MODEL_NAMES = {
        "lightgbm": "fraud_detection_lightgbm",
        "randomforest": "fraud_detection_randomforest",
        "logisticregression": "fraud_detection_logisticregression"
    }


settings = Config()