import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_raw(path="data/raw/creditcard.csv"):
    return pd.read_csv(path)

def prepare_and_split(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def scale_and_save(X_train, X_test, y_train, y_test, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    pd.DataFrame(X_train).to_csv(os.path.join(out_dir, "X_train_scaled.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(out_dir, "X_test_scaled.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(out_dir, "y_test.csv"), index=False)
    return X_train_scaled, X_test_scaled, scaler