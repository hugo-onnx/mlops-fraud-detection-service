import os
import json
import joblib
import hashlib
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap
from shap.maskers import Independent

import mlflow
import mlflow.sklearn
import mlflow.onnx
from mlflow.models import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SKLFloatTensorType
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from dotenv import load_dotenv

warnings.filterwarnings('ignore', message='.*LightGBM binary classifier.*')

# Config / Flags
# Ensures metrics & artifacts flush immediately in containerized environments
os.environ["MLFLOW_FLUSH_INTERVAL"] = "1"

load_dotenv(".env.local")

ENABLE_SHAP = True  # set to False to skip SHAP computations for faster runs

# Helpers
def file_hash(path: str) -> str:
    """Return md5 hash of a file (useful to track dataset versions)."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# Paths & Data
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

X_train_path = "data/processed/X_train_scaled.parquet"
X_test_path = "data/processed/X_test_scaled.parquet"
y_train_path = "data/processed/y_train.parquet"
y_test_path = "data/processed/y_test.parquet"

X_train_scaled = pd.read_parquet(X_train_path)
X_test_scaled = pd.read_parquet(X_test_path)
y_train = pd.read_parquet(y_train_path).squeeze()
y_test = pd.read_parquet(y_test_path).squeeze()

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Fraud Detection V2")

def train_save_eval(model, name, X_train_scaled, y_train, X_test_scaled, y_test):
    with mlflow.start_run(run_name=name, nested=True):
        # Dataset reproducibility
        mlflow.log_param("X_train_hash", file_hash(X_train_path))
        mlflow.log_param("y_train_hash", file_hash(y_train_path))
        mlflow.log_param("X_test_hash", file_hash(X_test_path))
        mlflow.log_param("y_test_hash", file_hash(y_test_path))

        # Tags
        mlflow.set_tag("model_type", name)
        mlflow.set_tag("library", type(model).__module__)

        # Hyperparameters
        mlflow.log_params(model.get_params())

        # Fit model
        model.fit(X_train_scaled, y_train)

        # Signature & input_example
        proba_example = model.predict_proba(X_train_scaled)
        signature = infer_signature(X_train_scaled, proba_example)
        input_example = X_train_scaled.iloc[:1]
        
        # Save joblib artifact
        joblib_path = f"models/{name}_best.joblib"
        joblib.dump(model, joblib_path)
        mlflow.log_artifact(joblib_path, artifact_path=f"{name}/joblib")

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                name=f"{name}_pkl",
                signature=signature,
                input_example=input_example
            )
        except Exception as e:
            print(f"Could not mlflow.sklearn.log_model for {name}: {e}")

        try:
            if "lightgbm" in type(model).__module__:
                # Convert LightGBM model using onnxmltools
                initial_type = [('float_input', FloatTensorType([None, X_train_scaled.shape[1]]))]
                onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type, target_opset=15)
            else:
                # Sklearn conversion
                input_type = [("input", SKLFloatTensorType([None, X_train_scaled.shape[1]]))]
                onnx_model = convert_sklearn(model, initial_types=input_type)

            onnx_path = f"models/{name}_best.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            mlflow.log_artifact(onnx_path, artifact_path=f"{name}/onnx")

            try:
                onnx_model_proto = onnx.load(onnx_path)
                mlflow.onnx.log_model(
                    onnx_model_proto,
                    name=f"{name}_onnx",
                    signature=signature,
                    input_example=input_example
                )
            except Exception as e:
                print(f"Could not mlflow.onnx.log_model for {name}: {e}")

            # Model size metrics
            try:
                mlflow.log_metric("joblib_size_MB", os.path.getsize(joblib_path) / 1e6)
                mlflow.log_metric("onnx_size_MB", os.path.getsize(onnx_path) / 1e6)
            except Exception:
                pass

        except Exception as e:
            print(f"Could not convert {name} to ONNX: {e}")

        # Predictions + metrics
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]

        metrics = {
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba)
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        cm_path = f"results/{name}_confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches='tight')
        plt.clf()
        mlflow.log_artifact(cm_path, artifact_path=f"{name}/plots")
        
        # ROC plot
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} AUC={metrics['ROC-AUC']:.3f}")
        plt.plot([0,1],[0,1],"k--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"{name} ROC")
        plt.legend()
        roc_path = f"results/{name}_roc.png"
        plt.savefig(roc_path, bbox_inches='tight')
        plt.clf()
        mlflow.log_artifact(roc_path, artifact_path=f"{name}/plots")

        # SHAP explainability
        if ENABLE_SHAP:
            print(f"Computing SHAP values for {name}...")
            try:
                if len(X_train_scaled) > 2000:
                    X_shap_bg = X_train_scaled.sample(2000, random_state=42)
                else:
                    X_shap_bg = X_train_scaled

                if "LightGBM" in name or "RandomForest" in name:
                    explainer = shap.TreeExplainer(model)
                    
                    shap_values_list = explainer.shap_values(X_shap_bg, check_additivity=False)
                    
                    if isinstance(shap_values_list, list):
                        shap_values = shap_values_list[1]
                    else:
                        if len(shap_values_list.shape) == 3:
                            shap_values = shap_values_list[:, :, 1]
                        else:
                            shap_values = shap_values_list
                else:
                    explainer = shap.LinearExplainer(model, masker=Independent(X_shap_bg))
                    shap_values = explainer.shap_values(X_shap_bg)

                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_shap_bg, show=False)
                shap_path = f"results/{name}_shap_summary.png"
                plt.savefig(shap_path, bbox_inches='tight', dpi=150)
                plt.clf()
                mlflow.log_artifact(shap_path, artifact_path=f"{name}/shap")

                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_shap_bg, plot_type="bar", show=False)
                shap_path = f"results/{name}_shap_bar.png"
                plt.savefig(shap_path, bbox_inches='tight', dpi=150)
                plt.clf()
                mlflow.log_artifact(shap_path, artifact_path=f"{name}/shap")

            except Exception as e:
                print(f"SHAP feature importance failed for {name}: {e}")
            finally:
                plt.close('all')

            # ---- optional: register model to Model Registry (uncomment if using registry) ----
            # try:
            #     # Adjust model_uri to the exact artifact path that was logged above
            #     model_uri = f"runs:/{mlflow.active_run().info.run_id}/{name}_pkl"
            #     res = mlflow.register_model(model_uri, f"{name}_Registry")
            #     mlflow.set_tag("registered_model_version", str(res.version))
            # except Exception as e:
            #     print(f"Model registry step failed for {name}: {e}")

        return metrics

# Load best params and define models
with open("results/logreg_best_params.json") as f:
    log_params = json.load(f)
with open("results/rf_best_params.json") as f:
    rf_params = json.load(f)
with open("results/lgb_best_params.json") as f:
    lgb_params = json.load(f)

models = {
    "LogisticRegression": LogisticRegression(
        **log_params, class_weight='balanced', max_iter=10000
    ),
    "RandomForest": RandomForestClassifier(
        **rf_params, class_weight='balanced', n_jobs=-1
    ),
    "LightGBM": lgb.LGBMClassifier(
        **lgb_params, objective="binary", boosting_type="gbdt", verbose=-1, n_jobs=-1
    )
}

# Parent run for the entire pipeline; child runs for each model
results = []

with mlflow.start_run(run_name="TrainingPipeline"):
    # Log the hyperparameter search results inside the parent run for provenance
    try:
        mlflow.log_artifact("results/logreg_best_params.json", artifact_path="hyperparams")
        mlflow.log_artifact("results/rf_best_params.json", artifact_path="hyperparams")
        mlflow.log_artifact("results/lgb_best_params.json", artifact_path="hyperparams")
    except Exception as e:
        print(f"Could not log hyperparam JSONs to parent run: {e}")

    for name, model in models.items():
        print("Training:", name)
        m = train_save_eval(model, name, X_train_scaled, y_train, X_test_scaled, y_test)
        m["Model"] = name
        results.append(m)

# Save comparison CSV locally and also log as artifact in the parent run
df = pd.DataFrame(results)
df.to_csv("results/model_comparison.csv", index=False)
print("Training & evaluation done. Models saved to models/ and metrics to results/")


# 🔹 MLflow model registry support