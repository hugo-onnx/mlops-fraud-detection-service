import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List

from app.config.config import settings, logger

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        settings.DB_PATH,
        timeout=30,
        check_same_thread=False,
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db(feature_cols: List[str]) -> None:
    if not feature_cols:
        raise RuntimeError("Feature columns list is empty — cannot initialize DB")

    os.makedirs(settings.DB_PATH.parent, exist_ok=True)

    cols_sql = ", ".join(f'"{c}" REAL' for c in feature_cols)

    sql = f"""
    CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {cols_sql},
        fraud_probability REAL NOT NULL,
        timestamp TEXT NOT NULL
    );
    """

    logger.info("Initializing SQLite DB")
    logger.info(f"DB path: {settings.DB_PATH}")

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
    except Exception as e:
        logger.exception("Failed to initialize database")
        raise RuntimeError("Database initialization failed") from e
    finally:
        conn.close()

def log_request(features: Dict[str, float], proba: float) -> None:
    try:
        conn = get_connection()
        cursor = conn.cursor()

        columns = (
            ", ".join(f'"{k}"' for k in features.keys())
            + ', "fraud_probability", "timestamp"'
        )

        placeholders = ", ".join(["?"] * (len(features) + 2))
        values = list(features.values()) + [
            float(proba),
            datetime.now(timezone.utc).isoformat(),
        ]

        cursor.execute(
            f'INSERT INTO requests ({columns}) VALUES ({placeholders})',
            values,
        )
        conn.commit()

    except Exception:
        logger.exception("Failed to log prediction to DB")
    finally:
        conn.close()

def log_request_bulk(
    features_list: List[Dict[str, float]],
    probabilities: List[float],
) -> None:
    if not features_list:
        return

    try:
        conn = get_connection()
        cursor = conn.cursor()

        feature_keys = features_list[0].keys()
        columns = (
            ", ".join(f'"{k}"' for k in feature_keys)
            + ', "fraud_probability", "timestamp"'
        )

        placeholders = ", ".join(["?"] * (len(feature_keys) + 2))

        rows = []
        timestamp = datetime.now(timezone.utc).isoformat()

        for features, proba in zip(features_list, probabilities):
            rows.append(
                list(features.values()) + [float(proba), timestamp]
            )

        cursor.executemany(
            f'INSERT INTO requests ({columns}) VALUES ({placeholders})',
            rows,
        )
        conn.commit()

    except Exception:
        logger.exception("Failed to bulk log predictions to DB")
    finally:
        conn.close()