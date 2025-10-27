from data_preprocessing import load_raw, prepare_and_split, scale_and_save
import pandas as pd

if __name__ == "__main__":
    df = load_raw()
    X_train, X_test, y_train, y_test = prepare_and_split(df)
    X_train_scaled, X_test_scaled, scaler = scale_and_save(X_train, X_test, y_train, y_test)
    print("Data prep done. Scaler saved to data/processed/scaler.pkl")