#!/usr/bin/env python
"""
predict.py
FastAPI service for taxi trip duration prediction.

Usage:
    uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
"""

import logging
import os
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Model loading ─────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "lin_reg.bin")

def load_model(path: str):
    model_file = Path(path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(model_file, "rb") as f:
        dv, model = pickle.load(f)
    log.info("Model loaded from %s", path)
    return dv, model

dv, model = load_model(MODEL_PATH)

# ── Schemas ───────────────────────────────────────────────────────────────────

class RideRequest(BaseModel):
    PULocationID: str = Field(..., example="130")
    DOLocationID: str = Field(..., example="205")
    trip_distance: float = Field(..., gt=0, example=3.5)

class PredictionResponse(BaseModel):
    duration_minutes: float
    pu_do: str

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Duration Prediction Service",
    description="Predicts NYC taxi trip duration in minutes.",
    version="1.0.0",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def prepare_features(ride: RideRequest) -> dict:
    return {
        "PU_DO": f"{ride.PULocationID}_{ride.DOLocationID}",
        "trip_distance": ride.trip_distance,
    }

def predict(features: dict) -> float:
    X = dv.transform([features])
    pred = model.predict(X)
    return round(float(pred[0]), 2)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(ride: RideRequest):
    try:
        features = prepare_features(ride)
        duration = predict(features)
        log.info("Predicted duration: %.2f min for %s", duration, features["PU_DO"])
        return PredictionResponse(
            duration_minutes=duration,
            pu_do=features["PU_DO"],
        )
    except Exception as e:
        log.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(rides: list[RideRequest]):
    if len(rides) > 100:
        raise HTTPException(status_code=400, detail="Max 100 rides per batch.")
    results = []
    for ride in rides:
        features = prepare_features(ride)
        duration = predict(features)
        results.append(PredictionResponse(
            duration_minutes=duration,
            pu_do=features["PU_DO"],
        ))
    return results
