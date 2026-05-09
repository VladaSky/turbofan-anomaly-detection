"""
Turbofan Anomaly Detection — FastAPI inference server.

Endpoints
---------
POST /predict   Accept a feature vector + sensor window, return RUL,
                risk level, anomaly score, and top SHAP drivers.
GET  /health    Liveness probe used by Docker / Render.
"""

import os
import pickle

import numpy as np
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Paths — relative to the project root (where uvicorn is launched from)
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

XGB_PATH       = os.path.join(BASE, "models", "xgb_rul.json")
AE_PATH        = os.path.join(BASE, "models", "lstm_autoencoder.keras")
THRESHOLD_PATH = os.path.join(BASE, "results", "anomaly_threshold.txt")
FEAT_COLS_PATH = os.path.join(BASE, "data", "processed", "feature_cols.pkl")

# ---------------------------------------------------------------------------
# Load all artifacts once at startup
# ---------------------------------------------------------------------------
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_PATH)

autoencoder = load_model(AE_PATH)

with open(THRESHOLD_PATH) as f:
    AE_THRESHOLD = float(f.read().strip())

with open(FEAT_COLS_PATH, "rb") as f:
    FEATURE_COLS = pickle.load(f)

explainer = shap.TreeExplainer(xgb_model)

N_FEATURES  = len(FEATURE_COLS)   # 110
TIMESTEPS   = 30
N_SENSORS   = 11
ALERT_RUL   = 30                  # cycles — HIGH RISK threshold

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    engine_id: str = Field(..., example="engine_81")
    feature_vector: list[float] = Field(
        ...,
        min_length=110, max_length=110,
        description="110-element vector: 11 raw sensors + 99 rolling stats"
    )
    sensor_window: list[list[float]] = Field(
        ...,
        description="30 × 11 normalized sensor window for autoencoder"
    )

class FeatureContribution(BaseModel):
    feature:    str
    shap_value: float

class PredictResponse(BaseModel):
    engine_id:     str
    rul_estimate:  float
    risk_level:    str           # "SAFE" or "HIGH RISK"
    anomaly_score: float         # autoencoder reconstruction MAE
    anomaly_flag:  bool
    top_features:  list[FeatureContribution]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Turbofan Anomaly Detection API",
    description="Real-time RUL prediction and anomaly detection for NASA CMAPSS turbofan engines.",
    version="1.0.0",
)

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # --- XGBoost RUL regression ---
    feat = np.array(req.feature_vector, dtype=np.float32).reshape(1, -1)
    if feat.shape[1] != N_FEATURES:
        raise HTTPException(400, f"Expected {N_FEATURES} features, got {feat.shape[1]}")

    rul       = float(np.clip(xgb_model.predict(feat), 0, None)[0])
    risk      = "HIGH RISK" if rul < ALERT_RUL else "SAFE"

    # --- SHAP: top 5 feature drivers ---
    shap_vals = explainer.shap_values(feat)[0]
    top_idx   = np.argsort(np.abs(shap_vals))[::-1][:5]
    top_feats = [
        FeatureContribution(feature=FEATURE_COLS[i], shap_value=round(float(shap_vals[i]), 4))
        for i in top_idx
    ]

    # --- LSTM autoencoder anomaly score ---
    window = np.array(req.sensor_window, dtype=np.float32)
    if window.shape != (TIMESTEPS, N_SENSORS):
        raise HTTPException(
            400,
            f"sensor_window must be {TIMESTEPS}×{N_SENSORS}, got {window.shape}"
        )
    window_batch = window.reshape(1, TIMESTEPS, N_SENSORS)
    recon        = autoencoder.predict(window_batch, verbose=0)
    ae_score     = float(np.mean(np.abs(recon - window_batch)))
    ae_flag      = ae_score > AE_THRESHOLD

    return PredictResponse(
        engine_id     = req.engine_id,
        rul_estimate  = round(rul, 1),
        risk_level    = risk,
        anomaly_score = round(ae_score, 4),
        anomaly_flag  = ae_flag,
        top_features  = top_feats,
    )
