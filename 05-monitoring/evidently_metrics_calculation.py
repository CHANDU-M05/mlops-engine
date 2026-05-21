#!/usr/bin/env python
"""
evidently_metrics_calculation.py
Calculates data drift and missing value metrics using Evidently,
stores results in PostgreSQL, visualised via Grafana.

Usage:
    python evidently_metrics_calculation.py

Env vars:
    POSTGRES_HOST     default: localhost
    POSTGRES_PORT     default: 5432
    POSTGRES_DB       default: monitoring
    POSTGRES_USER     default: postgres
    POSTGRES_PASSWORD default: postgres
    REFERENCE_DATA    default: data/reference.parquet
    CURRENT_DATA      default: data/green_tripdata_2022-02.parquet
    MODEL_PATH        default: models/lin_reg.bin
    START_DATE        default: 2022-02-01
    NUM_DAYS          default: 27
"""

import datetime
import logging
import os
import time

import joblib
import pandas as pd
import psycopg
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from prefect import flow, task

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config from env ───────────────────────────────────────────────────────────

PG_HOST     = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT     = os.getenv("POSTGRES_PORT", "5432")
PG_DB       = os.getenv("POSTGRES_DB", "monitoring")
PG_USER     = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_DSN      = f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PASSWORD}"
PG_DSN_ADMIN = f"host={PG_HOST} port={PG_PORT} user={PG_USER} password={PG_PASSWORD}"

REFERENCE_DATA = os.getenv("REFERENCE_DATA", "data/reference.parquet")
CURRENT_DATA   = os.getenv("CURRENT_DATA", "data/green_tripdata_2022-02.parquet")
MODEL_PATH     = os.getenv("MODEL_PATH", "models/lin_reg.bin")
START_DATE     = os.getenv("START_DATE", "2022-02-01")
NUM_DAYS       = int(os.getenv("NUM_DAYS", "27"))
SEND_TIMEOUT   = 10

CREATE_TABLE = """
DROP TABLE IF EXISTS drift_metrics;
CREATE TABLE drift_metrics (
    timestamp           TIMESTAMP PRIMARY KEY,
    prediction_drift    FLOAT,
    num_drifted_columns INTEGER,
    share_missing_values FLOAT
);
"""

# ── Feature config ────────────────────────────────────────────────────────────

NUM_FEATURES = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
CAT_FEATURES = ["PULocationID", "DOLocationID"]

column_mapping = ColumnMapping(
    prediction="prediction",
    numerical_features=NUM_FEATURES,
    categorical_features=CAT_FEATURES,
    target=None,
)

report = Report(metrics=[
    ColumnDriftMetric(column_name="prediction"),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
])

# ── Load data + model once ────────────────────────────────────────────────────

log.info("Loading reference data from %s", REFERENCE_DATA)
reference_data = pd.read_parquet(REFERENCE_DATA)

log.info("Loading current data from %s", CURRENT_DATA)
raw_data = pd.read_parquet(CURRENT_DATA)

log.info("Loading model from %s", MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    model = joblib.load(f)

begin = datetime.datetime.fromisoformat(START_DATE)

# ── Tasks ─────────────────────────────────────────────────────────────────────

@task
def prep_db():
    """Create database and metrics table if they don't exist."""
    with psycopg.connect(PG_DSN_ADMIN, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname=%s", (PG_DB,)
        ).fetchall()
        if not exists:
            conn.execute(f"CREATE DATABASE {PG_DB};")
            log.info("Created database: %s", PG_DB)

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(CREATE_TABLE)
    log.info("Metrics table ready.")


@task
def calculate_metrics(cursor, day_offset: int):
    """Calculate drift metrics for a single day and insert into DB."""
    window_start = begin + datetime.timedelta(days=day_offset)
    window_end   = window_start + datetime.timedelta(days=1)

    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= window_start) &
        (raw_data.lpep_pickup_datetime <  window_end)
    ].copy()

    if current_data.empty:
        log.warning("No data for %s — skipping.", window_start.date())
        return

    current_data["prediction"] = model.predict(
        current_data[NUM_FEATURES + CAT_FEATURES].fillna(0)
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    result = report.as_dict()

    prediction_drift    = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing       = result["metrics"][2]["result"]["current"]["share_of_missing_values"]

    cursor.execute(
        """
        INSERT INTO drift_metrics
            (timestamp, prediction_drift, num_drifted_columns, share_missing_values)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (timestamp) DO NOTHING
        """,
        (window_start, prediction_drift, num_drifted_columns, share_missing),
    )

    log.info(
        "%s | drift=%.4f | drifted_cols=%d | missing=%.4f",
        window_start.date(), prediction_drift, num_drifted_columns, share_missing,
    )


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(name="batch-monitoring")
def batch_monitoring(num_days: int = NUM_DAYS):
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=SEND_TIMEOUT)

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        for i in range(num_days):
            with conn.cursor() as cur:
                calculate_metrics(cur, i)

            now = datetime.datetime.now()
            elapsed = (now - last_send).total_seconds()
            if elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - elapsed)
            last_send = now

    log.info("Monitoring backfill complete for %d days.", num_days)


if __name__ == "__main__":
    batch_monitoring()
