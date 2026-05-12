# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# libgomp1 is required by XGBoost for OpenMP multi-threading
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# ── Stage 2: runtime (smaller final image) ────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/api_lite.py              src/
COPY models/xgb_rul.json          models/
COPY data/processed/feature_cols.pkl data/processed/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api_lite:app --host 0.0.0.0 --port ${PORT:-8000}"]
