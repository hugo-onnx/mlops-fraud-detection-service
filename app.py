import json
import time
import joblib
import logging
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud-api")

app = FastAPI(title="Fraud Detection API")

# load model and scaler
onnx_model_path = "models/LightGBM_best.onnx"
scaler = joblib.load("data/processed/scaler.pkl")
with open("data/processed/feature_columns.json", "r") as f:
    cols = json.load(f)

# Create ONNX runtime session
session = rt.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

prob_output = None
for out in session.get_outputs():
    if "prob" in out.name.lower() or "probability" in out.name.lower():
        prob_output = out.name
        break

output_name = prob_output if prob_output else session.get_outputs()[0].name
input_name = session.get_inputs()[0].name


class Transaction(BaseModel):
    features: dict[str, float] = Field(..., description="Feature name to value mapping")

    @model_validator(mode="after")
    def validate_features(self):
        for k, v in self.features.items():
            if not isinstance(v, (int, float)):
                raise ValueError(f"Feature '{k}' must be numeric, got {type(v).__name__}")
        return self


# informarte sobre async def
@app.post("/predict")
def predict(transaction: Transaction, request: Request):
    start_time = time.time()
    try:
        logger.info(f"Prediction request from {request.client.host}")

        input_features = transaction.features

        # Validate feature completeness
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

        # Convert and scale
        X = np.array([input_features[c] for c in cols], dtype=np.float32).reshape(1, -1)
        X_scaled = scaler.transform(X).astype(np.float32)

        # Run ONNX inference
        preds = session.run([output_name], {input_name: X_scaled})[0]
        proba = float(preds[0][1]) if preds.ndim == 2 else float(preds[0])

        latency = time.time() - start_time
        logger.info(
            json.dumps({
                "event": "prediction",
                "fraud_probability": proba,
                "latency_sec": round(latency, 4)
            })
        )

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