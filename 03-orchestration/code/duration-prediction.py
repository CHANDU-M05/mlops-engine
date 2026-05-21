#!/usr/bin/env python
"""
duration-prediction.py
Trains an XGBoost model to predict NYC taxi trip duration.
Logs params, metrics, and artifacts to MLflow.

Usage:
    python duration-prediction.py --year 2023 --month 1
"""

import argparse
import logging
import pickle
from pathlib import Path

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "taxi-duration-prediction"
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"

BEST_PARAMS = {
    "learning_rate": 0.09585355369315604,
    "max_depth": 30,
    "min_child_weight": 1.060597050922164,
    "objective": "reg:squarederror",   # fixed: reg:linear is deprecated
    "reg_alpha": 0.018060244040060163,
    "reg_lambda": 0.011658731377413597,
    "seed": 42,
}

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────

def read_dataframe(year: int, month: int) -> pd.DataFrame:
    url = DATA_URL.format(year=year, month=month)
    log.info("Loading data from %s", url)
    df = pd.read_parquet(url)

    df["duration"] = (
        df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    ).dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    log.info("Loaded %d rows for %d-%02d", len(df), year, month)
    return df


def build_features(
    df: pd.DataFrame,
    dv: DictVectorizer | None = None,
) -> tuple:
    features = ["PU_DO", "trip_distance"]
    dicts = df[features].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

# ── Training ──────────────────────────────────────────────────────────────────

def train(
    X_train, y_train,
    X_val, y_val,
    dv: DictVectorizer,
) -> str:
    with mlflow.start_run() as run:
        mlflow.log_params(BEST_PARAMS)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        booster = xgb.train(
            params=BEST_PARAMS,
            dtrain=dtrain,
            num_boost_round=30,
            evals=[(dval, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        y_pred = booster.predict(dval)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        log.info("Validation RMSE: %.4f", rmse)

        # Save and log preprocessor
        preprocessor_path = MODELS_DIR / "preprocessor.b"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="model")

        log.info("MLflow run_id: %s", run.info.run_id)
        return run.info.run_id

# ── Main ──────────────────────────────────────────────────────────────────────

def main(year: int, month: int) -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df_train = read_dataframe(year, month)

    # Validation = next month
    next_year  = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(next_year, next_month)

    X_train, dv = build_features(df_train)
    X_val, _    = build_features(df_val, dv)

    y_train = df_train["duration"].values
    y_val   = df_val["duration"].values

    run_id = train(X_train, y_train, X_val, y_val, dv)

    run_id_path = Path("run_id.txt")
    run_id_path.write_text(run_id)
    log.info("run_id saved to %s", run_id_path)

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train taxi duration prediction model.")
    parser.add_argument("--year",  type=int, required=True, help="Training data year")
    parser.add_argument("--month", type=int, required=True, help="Training data month")
    args = parser.parse_args()

    main(year=args.year, month=args.month)
