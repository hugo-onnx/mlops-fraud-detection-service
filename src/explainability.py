import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt


model = joblib.load("models/LightGBM_best.joblib")
X_test_scaled = pd.read_parquet("data/processed/X_test_scaled.parquet")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_scaled)

plt.figure()
shap.summary_plot(shap_values, X_test_scaled, show=False)
plt.savefig("results/shap_summary.png", bbox_inches='tight')