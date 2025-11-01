from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np
import pandas as pd
import onnxruntime as rt

app = FastAPI(title="Fraud Detection API")

# load model and scaler
onnx_model_path = "models/LightGBM_best.onnx"
scaler = joblib.load("data/processed/scaler.pkl")
cols = pd.read_csv("data/raw/creditcard.csv").columns.tolist()

# Create ONNX runtime session
session = rt.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[1].name if len(session.get_outputs()) > 1 else session.get_outputs()[0].name

class Transaction(BaseModel):
    features: dict

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert input features to numpy array
    values = [transaction.features.get(c, 0) for c in cols]
    X = np.array(values).reshape(1, -1)

    # Apply scaling
    X_scaled = scaler.transform(X).astype(np.float32)

    # Run inference
    preds = session.run([output_name], {input_name: X_scaled})[0]
    proba = float(preds[0][1]) if preds.ndim == 2 else float(preds[0])

    return {"fraud_probability": proba}


@app.get("/health")
def health_check():
    return {"status": "ok"}

