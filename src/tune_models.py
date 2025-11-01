import os
import json
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


X_train_scaled = pd.read_parquet("data/processed/X_train_scaled.parquet")
y_train = pd.read_parquet("data/processed/y_train.parquet").values.ravel()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = 'roc_auc'
N_TRIALS = 5
STORAGE = "sqlite:///optuna.db"


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
    return float(np.nanmean(scores))


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
    return float(np.nanmean(scores))


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
    return float(np.nanmean(scores))


def run_study(name, objective):
    study = optuna.create_study(direction="maximize", study_name=name, storage=STORAGE, load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
    return study

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    studies = {}
    for name, obj in [
        ("logreg", objective_logreg), 
        ("rf", objective_rf), 
        ("lgb", objective_lgb)
    ]:
        print("Running study:", name)
        studies[name] = run_study(name, obj)
        print("Best value:", studies[name].best_value)
        with open(f"results/{name}_best_params.json", "w") as f:
            json.dump(studies[name].best_params, f, indent=2)
    print("Tuning finished.")