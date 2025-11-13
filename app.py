import os
import json
import time
import joblib
import logging
import sqlite3
import numpy as np
import pandas as pd
import onnxruntime as rt
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud-api")

app = FastAPI(title="Fraud Detection API")

onnx_model_path = "models/LightGBM_best.onnx"
scaler = joblib.load("data/processed/scaler.pkl")
with open("data/processed/feature_columns.json", "r") as f:
    cols = json.load(f)

session = rt.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

prob_output = None
for out in session.get_outputs():
    if "prob" in out.name.lower() or "probability" in out.name.lower():
        prob_output = out.name
        break

output_name = prob_output if prob_output else session.get_outputs()[0].name
input_name = session.get_inputs()[0].name


DB_PATH = "data/production/requests.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    feature_cols = ", ".join([f'"{c}" REAL' for c in cols])
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {feature_cols},
            fraud_probability REAL,
            timestamp TEXT
        );
    """)
    conn.commit()
    conn.close()

init_db()

def log_request_to_db(features: dict, proba: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    columns = ", ".join([f'"{k}"' for k in features.keys()] + ["fraud_probability", "timestamp"])
    placeholders = ", ".join(["?"] * (len(features) + 2))
    values = list(features.values()) + [proba, datetime.now(timezone.utc).isoformat()]

    cursor.execute(f"INSERT INTO requests ({columns}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()


class Transaction(BaseModel):
    features: dict[str, float] = Field(..., description="Feature name to value mapping")

    @model_validator(mode="after")
    def validate_features(self):
        for k, v in self.features.items():
            if not isinstance(v, (int, float)):
                raise ValueError(f"Feature '{k}' must be numeric, got {type(v).__name__}")
        return self


@app.post("/predict")
def predict(transaction: Transaction, request: Request):
    start_time = time.time()
    try:
        logger.info(f"Prediction request from {request.client.host}")

        input_features = transaction.features

        missing = [c for c in cols if c not in input_features]
        extra = [k for k in input_features if k not in cols]

        if missing:
            msg = f"Missing features: {missing}"
            logger.warning(msg)
            raise HTTPException(status_code=400, detail=msg)
        if extra:
            msg = f"Unexpected features: {extra}"
            logger.warning(msg)
            raise HTTPException(status_code=400, detail=msg)

        X_df = pd.DataFrame([input_features], columns=cols)
        X_scaled = scaler.transform(X_df).astype(np.float32)

        preds_list = session.run([output_name], {input_name: X_scaled})

        pred_dict = preds_list[0][0]

        proba = float(pred_dict[1])

        latency = time.time() - start_time
        logger.info(
            json.dumps({
                "event": "prediction",
                "fraud_probability": proba,
                "latency_sec": round(latency, 4)
            })
        )

        log_request_to_db(input_features, proba)

        return {"fraud_probability": proba}
    
    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise e
    
    except Exception as e:
        logger.exception(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok"}