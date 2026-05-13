FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by XGBoost for OpenMP multi-threading
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.136.1 \
    uvicorn[standard]==0.34.3 \
    xgboost==3.2.0 \
    scikit-learn==1.6.1 \
    numpy==1.26.4

COPY src/api_lite.py              src/
COPY models/xgb_rul.json          models/
COPY data/processed/feature_cols.pkl data/processed/

EXPOSE 8000

# Render sets $PORT at runtime; fall back to 8000 locally
CMD ["sh", "-c", "uvicorn src.api_lite:app --host 0.0.0.0 --port ${PORT:-8000}"]
