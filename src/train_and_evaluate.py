import os
import json
import joblib
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SKLFloatTensorType
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType



X_train_scaled = pd.read_parquet("data/processed/X_train_scaled.parquet")
X_test_scaled = pd.read_parquet("data/processed/X_test_scaled.parquet")
y_train = pd.read_parquet("data/processed/y_train.parquet").squeeze()
y_test = pd.read_parquet("data/processed/y_test.parquet").squeeze()


os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def train_save_eval(model, name, X_train_scaled, y_train, X_test_scaled, y_test):
    model.fit(X_train_scaled, y_train)
    
    # Paths for saving
    joblib_path = f"models/{name}_best.joblib"
    onnx_path = f"models/{name}_best.onnx"

    joblib.dump(model, joblib_path)
    print(f"Saved {name} model to Joblib: {joblib_path}")

    try:
        model_pkg = type(model).__module__
        if "lightgbm" in model_pkg:
            print("Detected LightGBM model")
            initial_type = [('float_input', FloatTensorType([None, X_train_scaled.shape[1]]))]
            onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type)
        else:
            print("Detected scikit-learn model — exporting via skl2onnx...")
            input_type = [("input", SKLFloatTensorType([None, X_train_scaled.shape[1]]))]
            onnx_model = convert_sklearn(model, initial_types=input_type)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Saved {name} to ONNX: {onnx_path}")
        print()
    except Exception as e:
        print(f"Could not convert {name} to ONNX: {e}")

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1]

    metrics = {
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"results/{name}_confusion_matrix.png", bbox_inches='tight')
    plt.clf()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} AUC={metrics['ROC-AUC']:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name} ROC")
    plt.legend()
    plt.savefig(f"results/{name}_roc.png", bbox_inches='tight')
    plt.clf()

    return metrics


# Load best params
with open("results/logreg_best_params.json") as f:
    log_params = json.load(f)
with open("results/rf_best_params.json") as f:
    rf_params = json.load(f)
with open("results/lgb_best_params.json") as f:
    lgb_params = json.load(f)

models = {
    "LogisticRegression": LogisticRegression(**log_params, class_weight='balanced', max_iter=10000),
    "RandomForest": RandomForestClassifier(**rf_params, class_weight='balanced', n_jobs=-1),
    "LightGBM": lgb.LGBMClassifier(**lgb_params, objective="binary", boosting_type="gbdt", verbose=-1, n_jobs=-1)
}

results = []
for name, model in models.items():
    print("Training:", name)
    m = train_save_eval(model, name, X_train_scaled, y_train, X_test_scaled, y_test)
    m["Model"] = name
    results.append(m)

pd.DataFrame(results).to_csv("results/model_comparison.csv", index=False)
print("Training & evaluation done. Models saved to models/ and metrics to results/")