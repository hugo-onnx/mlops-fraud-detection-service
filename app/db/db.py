import sqlite3
import os
from datetime import datetime, timezone
from app.config.config import settings, logger


def init_db(feature_cols: list):
    if not feature_cols:
        raise RuntimeError("Feature columns list is empty — cannot initialize DB")

    os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)

    cols_sql = ", ".join(f'"{c}" REAL' for c in feature_cols)

    sql = f"""
    CREATE TABLE IF NOT EXISTS "requests" (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {cols_sql},
        fraud_probability REAL,
        timestamp TEXT
    );
    """

    logger.info("Initializing SQLite DB")
    logger.info(f"DB path: {settings.DB_PATH}")
    logger.info(f"Create table SQL:\n{sql}")

    with sqlite3.connect(settings.DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()


def log_request(features: dict, proba: float):
    with sqlite3.connect(settings.DB_PATH) as conn:
        cursor = conn.cursor()

        columns = (
            ", ".join(f'"{k}"' for k in features.keys())
            + ', "fraud_probability", "timestamp"'
        )

        placeholders = ", ".join(["?"] * (len(features) + 2))
        values = list(features.values()) + [
            proba,
            datetime.now(timezone.utc).isoformat(),
        ]

        cursor.execute(
            f'INSERT INTO "requests" ({columns}) VALUES ({placeholders})',
            values,
        )
        conn.commit()
