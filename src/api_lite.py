"""
Turbofan Anomaly Detection — Lite deployment API.

Uses XGBoost for RUL prediction (no TensorFlow / LSTM).
Runs on Render free tier (<= 512 MB RAM).
Full API with SHAP + LSTM anomaly scoring: src/api.py (local only).
"""

import os
import pickle

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XGB_PATH       = os.path.join(BASE, "models", "xgb_rul.json")
FEAT_COLS_PATH = os.path.join(BASE, "data", "processed", "feature_cols.pkl")

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_PATH)

with open(FEAT_COLS_PATH, "rb") as f:
    FEATURE_COLS = pickle.load(f)

N_FEATURES = len(FEATURE_COLS)  # 110
ALERT_RUL  = 30                 # cycles — HIGH RISK threshold

app = FastAPI(
    title="Turbofan Anomaly Detection API",
    description=(
        "Real-time Remaining Useful Life (RUL) prediction for NASA CMAPSS turbofan engines. "
        "XGBoost model trained on 80 engines. MAE 29.4 cycles, R^2=0.614."
    ),
    version="1.0.0",
)


class PredictRequest(BaseModel):
    engine_id: str = Field(..., example="engine_81")
    feature_vector: list[float] = Field(
        ...,
        min_length=110,
        max_length=110,
        description="110-element vector: 11 raw sensors + 99 rolling statistics",
    )


class PredictResponse(BaseModel):
    engine_id:       str
    rul_estimate:    float
    risk_level:      str
    cycles_to_alert: int


@app.get("/")
def root():
    return {
        "service": "Turbofan Anomaly Detection",
        "model":   "XGBoost RUL Regressor",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "n_features": N_FEATURES}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    feat = np.array(req.feature_vector, dtype=np.float32).reshape(1, -1)
    if feat.shape[1] != N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {N_FEATURES} features, got {feat.shape[1]}",
        )

    rul  = float(np.clip(xgb_model.predict(feat), 0, None)[0])
    risk = "HIGH RISK" if rul < ALERT_RUL else "SAFE"

    return PredictResponse(
        engine_id       = req.engine_id,
        rul_estimate    = round(rul, 1),
        risk_level      = risk,
        cycles_to_alert = max(0, int(rul - ALERT_RUL)),
    )
