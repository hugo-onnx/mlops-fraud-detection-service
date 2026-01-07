import os
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

        # Runtime state
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
        MLflow first (if enabled), otherwise local fallback.
        """
        try:
            if self.mlflow_enabled:
                logger.info("Attempting MLflow model load...")
                self._load_from_registry()
            else:
                self._load_fallback()
        except Exception:
            logger.exception("Model load failed. Falling back to local artifacts.")
            self._load_fallback()

        if not self.feature_columns:
            raise RuntimeError("Feature columns list is empty — cannot initialize DB")

        init_db(self.feature_columns)
        logger.info("Model, preprocessors and DB initialized successfully")

    def predict(self, features: dict) -> float:
        if not self.session:
            raise RuntimeError("Model is not loaded")

        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X_df = pd.DataFrame([features], columns=self.feature_columns)
        X_scaled = self.scaler.transform(X_df).astype(np.float32)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: X_scaled}
        )

        logger.debug(f"RAW ONNX OUTPUT: {outputs}")

        output = outputs[0]

        if isinstance(output, dict):
            if 1 in output:
                return float(output[1])
            if "probabilities" in output:
                return float(output["probabilities"][-1])
            return float(max(output.values()))

        if isinstance(output, np.ndarray):
            arr = output.squeeze()

            if arr.ndim == 1 and arr.size >= 2:
                return float(arr[-1])

            if arr.ndim == 0:
                return float(arr)

            return float(arr.flatten()[-1])

        if isinstance(output, list):
            return self._normalize_list_output(output)

        return float(output)

    def _normalize_list_output(self, output):
        while isinstance(output, list) and len(output) == 1:
            output = output[0]

        if isinstance(output, dict):
            if 1 in output:
                return float(output[1])
            return float(max(output.values()))

        if isinstance(output, np.ndarray):
            return float(output.flatten()[-1])

        return float(output)

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

        raise RuntimeError("No champion model found in MLflow registry")

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

        # ONNX model
        onnx_rel = f"{model_type}/onnx/{model_type}_best.onnx"
        local_onnx = mlflow.artifacts.download_artifacts(
            f"{artifact_uri}/{onnx_rel}"
        )

        self.session = rt.InferenceSession(
            local_onnx, providers=["CPUExecutionProvider"]
        )

        # Preprocessors
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
            logger.warning("MLflow preprocessors missing — using local files")
            self._load_local_preprocessors()

        self._configure_session()

        self.model_meta = {
            "name": model_name,
            "version": model_version.version,
            "description": model_version.description,
            "type": model_key,
        }

    # Local fallback
    def _load_fallback(self):
        logger.warning("Using local model fallback")

        models_dir = settings.MODELS_DIR
        onnx_path = models_dir / "LightGBM_best.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"Local ONNX model not found at {onnx_path}")

        self._load_local_preprocessors()

        self.session = rt.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )

        self._configure_session()

        self.model_meta = {
            "name": "Local LightGBM",
            "version": "0.0.0",
            "description": "Local ONNX fallback (container)",
            "type": "fallback",
        }

    def _load_local_preprocessors(self):
        logger.info("Loading local preprocessors")

        self.scaler = joblib.load(settings.SCALER_PATH)

        with open(settings.FEATURES_PATH) as f:
            self.feature_columns = json.load(f)

        if not self.feature_columns:
            raise RuntimeError("Loaded feature_columns.json is empty")

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