#!/usr/bin/env python
"""
model.py
ModelService — loads MLflow model from S3, handles Kinesis predictions.

Env vars:
    MODEL_LOCATION       override full S3 path
    MODEL_BUCKET         default: mlflow-models
    MLFLOW_EXPERIMENT_ID default: 1
    PREDICTIONS_STREAM_NAME
    RUN_ID
    TEST_RUN             default: False
    KINESIS_ENDPOINT_URL optional local override
"""

import base64
import json
import logging
import os
from typing import Callable

import boto3
import mlflow

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Model loading ─────────────────────────────────────────────────────────────

def get_model_location(run_id: str) -> str:
    """Resolve model S3 path from env or convention."""
    location = os.getenv("MODEL_LOCATION")
    if location:
        return location

    bucket        = os.getenv("MODEL_BUCKET", "mlflow-models")
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "1")
    return f"s3://{bucket}/{experiment_id}/{run_id}/artifacts/model"


def load_model(run_id: str):
    path = get_model_location(run_id)
    log.info("Loading model from %s", path)
    return mlflow.pyfunc.load_model(path)


def base64_decode(encoded: str) -> dict:
    decoded = base64.b64decode(encoded).decode("utf-8")
    return json.loads(decoded)


# ── Kinesis callback ──────────────────────────────────────────────────────────

class KinesisCallback:
    def __init__(self, client, stream_name: str):
        self.client      = client
        self.stream_name = stream_name

    def put_record(self, event: dict) -> None:
        ride_id = event["prediction"]["ride_id"]
        log.info("Publishing to Kinesis stream=%s ride_id=%s", self.stream_name, ride_id)
        self.client.put_record(
            StreamName=self.stream_name,
            Data=json.dumps(event),
            PartitionKey=str(ride_id),
        )


# ── Model service ─────────────────────────────────────────────────────────────

class ModelService:
    def __init__(
        self,
        model,
        model_version: str | None = None,
        callbacks: list[Callable] | None = None,
    ):
        self.model         = model
        self.model_version = model_version
        self.callbacks     = callbacks or []

    def prepare_features(self, ride: dict) -> dict:
        return {
            "PU_DO": f"{ride['PULocationID']}_{ride['DOLocationID']}",
            "trip_distance": ride["trip_distance"],
        }

    def predict(self, features: dict) -> float:
        pred = self.model.predict(features)
        return round(float(pred[0]), 2)

    def process_record(self, record: dict) -> dict:
        """Decode one Kinesis record and return a prediction event."""
        encoded  = record["kinesis"]["data"]
        payload  = base64_decode(encoded)

        ride    = payload["ride"]
        ride_id = payload["ride_id"]

        features   = self.prepare_features(ride)
        prediction = self.predict(features)

        event = {
            "model":   "ride_duration_prediction_model",
            "version": self.model_version,
            "prediction": {
                "ride_duration": prediction,
                "ride_id":       ride_id,
            },
        }

        log.info("ride_id=%s predicted=%.2f min", ride_id, prediction)

        for cb in self.callbacks:
            cb(event)

        return event

    def lambda_handler(self, event: dict) -> dict:
        predictions = [
            self.process_record(record)
            for record in event.get("Records", [])
        ]
        return {"predictions": predictions}


# ── Factory ───────────────────────────────────────────────────────────────────

def create_kinesis_client():
    endpoint = os.getenv("KINESIS_ENDPOINT_URL")
    if endpoint:
        log.info("Using local Kinesis endpoint: %s", endpoint)
        return boto3.client("kinesis", endpoint_url=endpoint)
    return boto3.client("kinesis")


def init(
    prediction_stream_name: str,
    run_id: str,
    test_run: bool = False,
) -> ModelService:
    model     = load_model(run_id)
    callbacks = []

    if not test_run:
        client   = create_kinesis_client()
        callback = KinesisCallback(client, prediction_stream_name)
        callbacks.append(callback.put_record)
    else:
        log.info("TEST_RUN mode — Kinesis publishing disabled.")

    return ModelService(
        model=model,
        model_version=run_id,
        callbacks=callbacks,
    )
