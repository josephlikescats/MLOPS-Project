"""FastAPI inference service for repeat-purchase predictions."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/repeat_purchase_model.joblib")
app = FastAPI(title="Retail Repeat-Purchase API", version="1.0.0")


class CustomerFeatures(BaseModel):
    recency_days: int = Field(ge=0)
    frequency: int = Field(ge=1)
    monetary: float = Field(ge=0)
    average_order_value: float = Field(ge=0)


class Prediction(BaseModel):
    repeat_purchase_probability: float
    repeat_purchase_prediction: int
    model_version: str


def get_bundle():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact is unavailable; run ml/train_model.py first")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if MODEL_PATH.exists() else "model_unavailable"}


@app.post("/predict", response_model=Prediction)
def predict(features: CustomerFeatures) -> Prediction:
    bundle = get_bundle()
    values = features.model_dump()
    feature_frame = pd.DataFrame([values], columns=bundle["features"])
    probability = float(bundle["model"].predict_proba(feature_frame)[:, 1][0])
    return Prediction(
        repeat_purchase_probability=probability,
        repeat_purchase_prediction=int(probability >= 0.5),
        model_version=MODEL_PATH.name,
    )