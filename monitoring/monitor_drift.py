import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from evidently import Dataset, DataDefinition, Report, BinaryClassification
from evidently.presets import DataDriftPreset, DataSummaryPreset

DB_PATH = "data/production/requests.db"
REFERENCE_CSV = "data/production/creditcard_reference.csv"
REPORT_PATH = "reports/data_drift_report.html"
REPORT_JSON = "reports/data_drift_report.json"

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# Load reference data
reference = pd.read_csv(REFERENCE_CSV)

# Add missing prediction column to reference
reference["fraud_probability"] = np.nan

# Load current data from DB
now_utc = datetime.now(timezone.utc)
since = (now_utc - timedelta(days=30)).isoformat()

query = "SELECT * FROM requests WHERE timestamp >= ?"

try:
    with sqlite3.connect(DB_PATH) as conn:
        current = pd.read_sql_query(query, conn, params=[since])
except Exception as e:
    print("Error reading DB:", e)
    sys.exit(1)

if current.empty:
    print("No recent data found in the database.")
    sys.exit(0)

# Drop unnecessary columns
current = current.drop(columns=[c for c in ["id", "timestamp"] if c in current.columns])

# Ensure same column order (excluding Class)
feature_cols = [col for col in reference.columns if col not in ["Class"]]
current = current[feature_cols]

# ---- FINAL MERGED DATA DEFINITION ----
data_definition = DataDefinition(
    classification=[
        BinaryClassification(
            target="Class",
            prediction_probas="fraud_probability",
        )
    ],
    numerical_columns=[
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12","V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Amount"
    ]
)

# Build Evidently datasets
reference_data = Dataset.from_pandas(
    reference,
    data_definition=data_definition
)

current_data = Dataset.from_pandas(
    current,
    data_definition=data_definition
)

# Report with summary + drift
report = Report(
    metrics=[
        DataDriftPreset(drift_share=0.7),
        DataSummaryPreset()
    ],
    include_tests=True
)

# Run monitoring
report_results = report.run(reference_data=reference_data, current_data=current_data)

# Save visualization
report_results.save_html(REPORT_PATH)

report_results.save_json(REPORT_JSON)