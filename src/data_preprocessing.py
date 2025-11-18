import os
import joblib
import json
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

    cols = list(X_train.columns)
    json.dump(cols, open("data/processed/feature_columns.json", "w"))

    return X_train, X_test, y_train, y_test

def scale_and_save(X_train, X_test, y_train, y_test, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_parquet(
        os.path.join(out_dir, "X_train_scaled.parquet"), index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_parquet(
        os.path.join(out_dir, "X_test_scaled.parquet"), index=False
    )
    pd.DataFrame(y_train, columns=["Class"]).to_parquet(
        os.path.join(out_dir, "y_train.parquet"), index=False
    )
    pd.DataFrame(y_test, columns=["Class"]).to_parquet(
        os.path.join(out_dir, "y_test.parquet"), index=False
    )
    return X_train_scaled, X_test_scaled, scaler

def create_production_copy(df, out_dir="data/production"):
    os.makedirs(out_dir, exist_ok=True)

    df_copy = df.copy()

    output_path = os.path.join(out_dir, "creditcard_reference.csv")
    df_copy.to_csv(output_path, index=False)


if __name__ == "__main__":
    df = load_raw()
    X_train, X_test, y_train, y_test = prepare_and_split(df)
    X_train_scaled, X_test_scaled, scaler = scale_and_save(X_train, X_test, y_train, y_test)
    create_production_copy(df)
    print("Data prep done")