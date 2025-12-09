import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

import optuna
import optuna.visualization as vis

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import mlflow
from dotenv import load_dotenv

# Config / Flags
os.environ["MLFLOW_FLUSH_INTERVAL"] = "1"
load_dotenv(".env.local")

# Load data
X_train_scaled = pd.read_parquet("data/processed/X_train_scaled.parquet")
y_train = pd.read_parquet("data/processed/y_train.parquet").values.ravel()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = 'average_precision'
N_TRIALS = 2
STORAGE = "sqlite:///optuna.db"

# Store active run ID to use in callbacks
current_run_id = None

# Objective functions
def objective_logreg(trial):
    params = {
        "C": trial.suggest_float('C', 0.01, 10.0, log=True),
        "penalty": trial.suggest_categorical('penalty', ['l2']),
        "solver": trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    }

    model = LogisticRegression(
        **params,
        class_weight='balanced',
        max_iter=5000,
        random_state=42
    )
    scores = cross_val_score(model, X_train_scaled, y_train, scoring=scorer, cv=cv, n_jobs=-1)
    score = float(np.nanmean(scores))

    # Log trial to MLflow
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True, parent_run_id=current_run_id):
        mlflow.log_params(params)
        mlflow.log_metric("cv_score", score)
        mlflow.log_metric("cv_std", float(np.nanstd(scores)))

    return score


def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 200, 800),
        "max_depth": trial.suggest_int('max_depth', 3, 15),
        "min_samples_split": trial.suggest_int('min_samples_split', 2, 10),
        "max_features": trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }

    model = RandomForestClassifier(
        **params,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(model, X_train_scaled, y_train, scoring=scorer, cv=cv, n_jobs=-1)
    score = float(np.nanmean(scores))
    
    # Log trial to MLflow
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True, parent_run_id=current_run_id):
        mlflow.log_params(params)
        mlflow.log_metric("cv_score", score)
        mlflow.log_metric("cv_std", float(np.nanstd(scores)))
    
    return score


def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 200, 800),
        "max_depth": trial.suggest_int('max_depth', 3, 10),
        "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        "subsample": trial.suggest_float('subsample', 0.5, 1.0),
        "colsample_bytree": trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    model = lgb.LGBMClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        boosting_type="gbdt",
        verbose=-1,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(model, X_train_scaled, y_train, scoring=scorer, cv=cv, n_jobs=-1)
    score = float(np.nanmean(scores))
    
    # Log trial to MLflow
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True, parent_run_id=current_run_id):
        mlflow.log_params(params)
        mlflow.log_metric("cv_score", score)
        mlflow.log_metric("cv_std", float(np.nanstd(scores)))
    
    return score

# Wrap Optuna study inside MLflow nested run
def run_study(study_name, objective):
    global current_run_id

    with mlflow.start_run(run_name=f"HPO_{study_name}", nested=True) as run:
        current_run_id = run.info.run_id  # Store the parent run ID

        mlflow.set_tag("task", "hyperparameter_optimization")
        mlflow.set_tag("model", study_name)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5)

        study = optuna.create_study(
            direction="maximize", 
            study_name=study_name, 
            storage=STORAGE, 
            load_if_exists=True,
            pruner=pruner
        )
        
        study.optimize(
            objective, 
            n_trials=N_TRIALS, 
            n_jobs=-1)
        
        # Visualization artifacts
        fig_history = vis.plot_optimization_history(study)
        fig_importance = vis.plot_param_importances(study)
        mlflow.log_figure(fig_history, f"{study_name}_optimization_history.html")
        mlflow.log_figure(fig_importance, f"{study_name}_param_importances.html")
        
        # Log best params and score
        mlflow.log_metric("best_score", study.best_value)
        mlflow.log_params(study.best_params)
        
        return study

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("Fraud Detection V4")

    studies = {}
    # Parent HPO run
    with mlflow.start_run(run_name="HyperparameterOptimization"):
        for name, obj in [
            ("logreg", objective_logreg), 
            ("rf", objective_rf), 
            ("lgb", objective_lgb)
        ]:
            print("\nRunning study:", name)

            study = run_study(name, obj)
            studies[name] = study

            print("Best value:", study.best_value)
            
            # Save best params JSON
            best_params_path = f"results/{name}_best_params.json"
            with open(best_params_path, "w") as f:
                json.dump(study.best_params, f, indent=2)

            mlflow.log_artifact(best_params_path, artifact_path=f"hpo/{name}")

        print("Tuning finished.")