# Turbofan Engine Anomaly Detection

Real-time Remaining Useful Life (RUL) prediction and anomaly detection for NASA CMAPSS turbofan engines. Built end-to-end: from raw sensor data through machine learning models to a deployed REST API and interactive operational dashboard.

---

## Live Demos

| | Link |
|---|---|
| **REST API** | https://turbofan-anomaly-api.onrender.com |
| **API Docs** | https://turbofan-anomaly-api.onrender.com/docs |
| **Streamlit Dashboard** | *(deploy via Streamlit Community Cloud — see below)* |

---

## Project Overview

Aircraft engines degrade over time. Detecting anomalies early and predicting when an engine will fail allows maintenance teams to act before a failure occurs — reducing downtime, cost, and risk.

This project builds a complete anomaly detection pipeline on the [NASA CMAPSS FD001 dataset](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6): 100 turbofan engines each run from healthy operation to failure, with 21 sensor readings recorded every cycle.

**Key results:**

| Model | Metric | Score |
|---|---|---|
| XGBoost RUL Regressor | MAE | 29.4 cycles (49% better than baseline) |
| XGBoost RUL Regressor | R² | 0.614 |
| XGBoost HIGH RISK recall | — | 89% |
| LSTM Autoencoder | ROC AUC | 0.732 |
| LSTM Autoencoder | Engines caught | 94 / 100 |
| Median alert lead time | — | 30 cycles before failure |

---

## Pipeline

```
Raw Sensors (21) → Feature Engineering → Two Models → API → Dashboard
                        │                    │
                   30-cycle windows     XGBoost RUL
                   Rolling stats        LSTM Autoencoder
                   (5/10/30-cycle       (anomaly score)
                    mean, std, roc)
```

---

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 01 | `01_eda.ipynb` | Exploratory data analysis — sensor variability, degradation curves, RUL labeling |
| 02 | `02_preprocessing.ipynb` | Feature engineering — normalization, rolling window stats, windowing |
| 03 | `03_modeling.ipynb` | LSTM Autoencoder training — learns normal engine behavior, flags anomalies |
| 04 | `04_evaluation.ipynb` | Model evaluation — ROC/PR curves, alert lead time, per-engine timelines |
| 05 | `05_rul_regression.ipynb` | XGBoost RUL regression — training, SHAP explainability, risk classification |
| 06 | `06_api.ipynb` | API walkthrough — stream simulation, Evidently drift detection |

---

## Models

- **LSTM Autoencoder** (`models/lstm_autoencoder.keras`) — trained on normal engine windows only. Reconstruction error (MAE) is used as the anomaly score. Threshold: 0.0937.
- **XGBoost Regressor** (`models/xgb_rul.json`) — predicts remaining useful life in cycles from 110 engineered features. Engines below 30 cycles RUL are classified HIGH RISK.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & Features | NumPy, Pandas, Scikit-learn |
| Modeling | TensorFlow/Keras (LSTM), XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Drift Detection | Evidently AI |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Deployment | Render (API), Streamlit Community Cloud (dashboard) |
| CI/CD | GitHub Actions |

---

## Repository Structure

```
├── notebooks/          # Analysis and modeling notebooks (01-06)
├── src/
│   ├── api.py          # Full local FastAPI server (XGBoost + SHAP + LSTM)
│   ├── api_lite.py     # Lite API deployed to Render (XGBoost only)
│   └── precompute.py   # Pre-compute predictions and SHAP values
├── app/
│   └── streamlit_app.py  # Operational dashboard
├── models/             # Trained model artifacts
├── data/processed/     # Engineered features and precomputed results
├── results/            # Evaluation plots and metrics
├── Dockerfile          # Container for API deployment
├── render.yaml         # Render infrastructure config
└── Procfile            # Process definition for Render
```

---

## Running Locally

**Requirements:** Python 3.11, packages in `requirements.txt`

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit dashboard
streamlit run app/streamlit_app.py

# Launch the full local API (requires TensorFlow)
uvicorn src.api:app --reload
```

**Streamlit Community Cloud deployment:**
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set main file: `app/streamlit_app.py`
4. Deploy

---

## Dataset

NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) FD001 subset:
- 100 engines, single operating condition, one fault mode
- 21 raw sensor channels, recorded every engine cycle
- Training set: engines 1–80 | Test set: engines 81–100

---

## Author

Vlada Balinsky
