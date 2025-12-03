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
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score
)

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SKLFloatTensorType
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from dotenv import load_dotenv

warnings.filterwarnings('ignore', message='.*LightGBM binary classifier.*')

# Config / Flags
os.environ["MLFLOW_FLUSH_INTERVAL"] = "1"
load_dotenv(".env.local")

ENABLE_SHAP = True  # set to False to skip SHAP computations for faster runs

# Helpers
def file_hash(path: str) -> str:
    """Return md5 hash of a file"""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# Paths and Data
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
mlflow.set_experiment("Fraud Detection V3")

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

        # Signature and input_example
        proba_example = model.predict_proba(X_train_scaled)
        signature = infer_signature(X_train_scaled, proba_example)
        input_example = X_train_scaled.iloc[:1]
        
        # Save joblib artifact
        joblib_path = f"models/{name}_best.joblib"
        joblib.dump(model, joblib_path)
        mlflow.log_artifact(joblib_path, artifact_path=f"{name}/joblib")

        # Log sklearn model
        model_info = None
        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name=f"{name}_pkl",
                signature=signature,
                input_example=input_example,
                registered_model_name=f"fraud_detection_{name.lower()}"
            )
        except Exception as e:
            print(f"Could not mlflow.sklearn.log_model for {name}: {e}")

        # ONNX conversion
        try:
            if "lightgbm" in type(model).__module__:
                initial_type = [('float_input', FloatTensorType([None, X_train_scaled.shape[1]]))]
                onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type, target_opset=15)
            else:
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

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]

        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc
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
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1],"k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} ROC Curve")
        plt.legend()
        roc_path = f"results/{name}_roc.png"
        plt.savefig(roc_path, bbox_inches='tight')
        plt.clf()
        mlflow.log_artifact(roc_path, artifact_path=f"{name}/plots")

        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(6,4))
        plt.plot(recall_curve, precision_curve, label=f"{name} PR-AUC={pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{name} Precision-Recall Curve")
        plt.legend()
        pr_path = f"results/{name}_precision_recall.png"
        plt.savefig(pr_path, bbox_inches='tight')
        plt.clf()
        mlflow.log_artifact(pr_path, artifact_path=f"{name}/plots")

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

        # Update model version with description in registry
        if model_info:
            try:
                client = MlflowClient()
                model_name = f"fraud_detection_{name.lower()}"
                 
                # Get the model version from the returned model_info
                version = model_info.registered_model_version
                
                # Add comprehensive description
                description = (
                    f"PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | "
                    f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}"
                )
                
                client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description
                )
                
                # Add tags for tracking
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="pr_auc",
                    value=str(round(pr_auc, 4))
                )
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="trained_date",
                    value=pd.Timestamp.now().strftime("%Y-%m-%d")
                )
                
                print(f"Registered {model_name} version {version}")
                
            except Exception as e:
                print(f"Could not update model registry for {name}: {e}")

        return metrics, model_info

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

# Parent run for the entire pipeline
results = []
model_versions = {}

with mlflow.start_run(run_name="TrainingPipeline"):
    # Log the hyperparameter search results
    try:
        mlflow.log_artifact("results/logreg_best_params.json", artifact_path="hyperparams")
        mlflow.log_artifact("results/rf_best_params.json", artifact_path="hyperparams")
        mlflow.log_artifact("results/lgb_best_params.json", artifact_path="hyperparams")
    except Exception as e:
        print(f"Could not log hyperparam JSONs to parent run: {e}")

    # Train all models
    for name, model in models.items():
        print(f"\nTraining: {name}")
        m, model_info = train_save_eval(model, name, X_train_scaled, y_train, X_test_scaled, y_test)
        m["Model"] = name
        results.append(m)
        if model_info:
            model_versions[name] = {
                "name": f"fraud_detection_{name.lower()}",
                "version": model_info.registered_model_version
            }

    # Save comparison CSV
    df = pd.DataFrame(results)
    df.to_csv("results/model_comparison.csv", index=False)
    mlflow.log_artifact("results/model_comparison.csv", artifact_path="comparison")

    # Select champion model based on PR-AUC
    best_model = max(results, key=lambda x: x['PR-AUC'])
    
    print("\n" + "="*70)
    print("MODEL COMPARISON (sorted by PR-AUC)")
    print("="*70)
    for r in sorted(results, key=lambda x: x['PR-AUC'], reverse=True):
        print(f"{r['Model']:20} | PR-AUC: {r['PR-AUC']:.4f} | ROC-AUC: {r['ROC-AUC']:.4f} | "
              f"F1: {r['F1-score']:.4f} | Precision: {r['Precision']:.4f} | Recall: {r['Recall']:.4f}")
    
    print("\n" + "="*70)
    print(f"CHAMPION MODEL: {best_model['Model']}")
    print(f"PR-AUC: {best_model['PR-AUC']:.4f} (best for imbalanced fraud detection)")
    print("="*70)

    # Promote champion using modern alias-based approach (no more stages!)
    try:
        client = MlflowClient()
        champion_name = best_model['Model']
        model_name = f"fraud_detection_{champion_name.lower()}"
        version = model_versions[champion_name]["version"]
        
        # Set alias "champion" to the best model version
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=version
        )
        
        # Add tag
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key="selection_metric",
            value="PR-AUC"
        )
        
        print(f"\nSet 'champion' and 'production' aliases for {model_name} version {version}")
        
    except Exception as e:
        print(f"\nCould not set model aliases: {e}")

    # Log champion model info to parent run
    mlflow.log_param("champion_model", best_model['Model'])
    mlflow.log_metric("champion_pr_auc", best_model['PR-AUC'])
    mlflow.log_metric("champion_roc_auc", best_model['ROC-AUC'])
    mlflow.log_metric("champion_f1", best_model['F1-score'])

print("\nTraining and evaluation complete")
print("All models registered in MLflow Model Registry")
print(f"Champion model ({best_model['Model']}) marked with 'champion' and 'production' aliases")
print("\nTo load the champion model in production:")
print(f"  model = mlflow.pyfunc.load_model('models:/{model_versions[best_model['Model']]['name']}@champion')")