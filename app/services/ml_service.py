import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import onnxruntime as rt

from mlflow.tracking import MlflowClient
from app.config.config import settings, logger
from app.db.db import init_db


class ModelService:
    def __init__(self):
        self.mlflow_enabled = settings.MLFLOW_ENABLED

        if self.mlflow_enabled:
            logger.info("MLflow enabled. Initializing MLflow client.")
            self.client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        else:
            logger.warning("MLflow disabled. Using local model fallback.")
            self.client = None

        # Model state
        self.session = None
        self.scaler = None
        self.feature_columns = []
        self.input_name = None
        self.output_name = None

        # Metadata
        self.model_meta = {
            "name": None,
            "version": None,
            "description": None,
            "type": None,
        }

    # Public API
    def load_model(self):
        """
        Loads model + preprocessors.
        Tries MLflow (if enabled), otherwise uses local ONNX artifacts.
        """
        if self.mlflow_enabled:
            try:
                logger.info("Attempting to load champion model from MLflow registry...")
                self._load_from_registry()
                logger.info(
                    f"Loaded {self.model_meta['name']} "
                    f"(v{self.model_meta['version']})"
                )
            except Exception as e:
                logger.exception("MLflow load failed. Falling back to local model.")
                self._load_fallback()
        else:
            self._load_fallback()

        # Initialize DB schema using feature list
        init_db(self.feature_columns)

    def predict(self, features: dict) -> float:
        if not self.session:
            raise RuntimeError("Model is not loaded.")

        # Validate inputs
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X_df = pd.DataFrame([features], columns=self.feature_columns)
        X_scaled = self.scaler.transform(X_df).astype(np.float32)

        preds = self.session.run(
            [self.output_name], {self.input_name: X_scaled}
        )

        pred_val = preds[0][0]

        if isinstance(pred_val, dict):
            return float(pred_val.get(1, 0.0))
        if isinstance(pred_val, (list, np.ndarray)):
            return float(pred_val[1])

        return float(pred_val)

    # MLflow loading
    def _find_champion_model(self):
        for model_key, model_name in settings.MODEL_NAMES.items():
            try:
                version = self.client.get_model_version_by_alias(
                    model_name, "champion"
                )
                return model_name, version, model_key
            except Exception:
                continue

        raise RuntimeError("No champion model found in MLflow registry.")

    def _load_from_registry(self):
        model_name, model_version, model_key = self._find_champion_model()

        run_id = model_version.run_id
        artifact_uri = f"runs:/{run_id}"

        if "lightgbm" in model_key:
            model_type = "LightGBM"
        elif "logisticregression" in model_key:
            model_type = "LogisticRegression"
        elif "randomforest" in model_key:
            model_type = "RandomForest"
        else:
            model_type = model_key.title()

        onnx_rel_path = f"{model_type}/onnx/{model_type}_best.onnx"
        local_onnx = mlflow.artifacts.download_artifacts(
            f"{artifact_uri}/{onnx_rel_path}"
        )

        self.session = rt.InferenceSession(
            local_onnx, providers=["CPUExecutionProvider"]
        )

        try:
            scaler_path = mlflow.artifacts.download_artifacts(
                f"{artifact_uri}/scaler.pkl"
            )
            features_path = mlflow.artifacts.download_artifacts(
                f"{artifact_uri}/feature_columns.json"
            )
            self.scaler = joblib.load(scaler_path)
            with open(features_path) as f:
                self.feature_columns = json.load(f)
        except Exception:
            logger.warning("Preprocessors missing in MLflow. Using local files.")
            self._load_local_preprocessors()

        self._configure_session()

        self.model_meta = {
            "name": model_name,
            "version": model_version.version,
            "description": model_version.description,
            "type": model_key,
        }

    # Local fallback (Render-safe)
    def _load_fallback(self):
        onnx_path = settings.MODELS_DIR / "LightGBM_best.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Local ONNX model not found at {onnx_path}"
            )

        logger.info(f"Loading local ONNX model from {onnx_path}")

        self.session = rt.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

        self._load_local_preprocessors()
        self._configure_session()

        self.model_meta = {
            "name": "Local LightGBM",
            "version": "local",
            "description": "Bundled ONNX model (Render)",
            "type": "fallback",
        }

    def _load_local_preprocessors(self):
        self.scaler = joblib.load(settings.SCALER_PATH)
        with open(settings.FEATURES_PATH) as f:
            self.feature_columns = json.load(f)

    # ONNX helpers
    def _configure_session(self):
        self.input_name = self.session.get_inputs()[0].name

        prob_output = None
        for out in self.session.get_outputs():
            if "prob" in out.name.lower():
                prob_output = out.name
                break

        self.output_name = prob_output or self.session.get_outputs()[0].name


model_service = ModelService()