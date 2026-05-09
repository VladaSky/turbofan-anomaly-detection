"""
Pre-compute predictions and SHAP values for the API notebook.
Run once from the project root:  python src/precompute.py
"""

import numpy as np
import pickle, os, time

import xgboost as xgb
import shap
from tensorflow.keras.models import load_model

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = os.path.join(BASE, "data", "processed")
DEMO_ENGINES = [81, 82, 83, 84, 85, 86]   # only run AE on these

def p(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

p("Loading data...")
X_feat  = np.load(os.path.join(BASE, "data/processed/X_features.npy"))
X_win   = np.load(os.path.join(BASE, "data/processed/X_windows.npy"))
y_rul   = np.load(os.path.join(BASE, "data/processed/y_rul.npy"))
eng_ids = np.load(os.path.join(BASE, "data/processed/engine_ids_flat.npy"))

# Full test-set mask (engines 81-100) for XGBoost
test_engine_ids = np.unique(eng_ids)[80:]
test_mask       = np.isin(eng_ids, test_engine_ids)
X_feat_test     = X_feat[test_mask]
y_rul_test      = y_rul[test_mask]
eng_ids_test    = eng_ids[test_mask]
p(f"Test windows: {X_feat_test.shape[0]}")

# XGBoost — fast even on full test set
p("Loading XGBoost and predicting...")
model    = xgb.XGBRegressor()
model.load_model(os.path.join(BASE, "models/xgb_rul.json"))
rul_pred = np.clip(model.predict(X_feat_test), 0, None)
p(f"  done. shape={rul_pred.shape}")

# Autoencoder — only for the 6 demo engines to keep it fast
p(f"Loading autoencoder (demo engines {DEMO_ENGINES} only)...")
ae          = load_model(os.path.join(BASE, "models/lstm_autoencoder.keras"))
demo_mask   = np.isin(eng_ids_test, DEMO_ENGINES)
X_win_demo  = X_win[test_mask][demo_mask]
recon_demo  = ae.predict(X_win_demo, batch_size=64, verbose=0)
ae_score_demo = np.mean(np.abs(recon_demo - X_win_demo), axis=(1, 2))
p(f"  done. {X_win_demo.shape[0]} windows.")

# Full ae_score array — fill with nan for non-demo engines
ae_score = np.full(len(X_feat_test), np.nan)
ae_score[demo_mask] = ae_score_demo

# SHAP — 200 samples
p("Computing SHAP values (200 samples)...")
explainer = shap.TreeExplainer(model)
rng       = np.random.default_rng(42)
shap_idx  = rng.choice(len(X_feat_test), size=200, replace=False)
shap_vals = explainer.shap_values(X_feat_test[shap_idx])
p(f"  done. shape={shap_vals.shape}")

# Single-window SHAP for waterfall (engine 85, last window)
last_i      = np.where(eng_ids_test == 85)[0][-1]
shap_single = explainer.shap_values(X_feat_test[last_i:last_i+1])[0]
p("  single-window SHAP done.")

np.save(os.path.join(OUT, "precomp_rul_pred.npy"),    rul_pred)
np.save(os.path.join(OUT, "precomp_ae_score.npy"),    ae_score)
np.save(os.path.join(OUT, "precomp_shap_vals.npy"),   shap_vals)
np.save(os.path.join(OUT, "precomp_shap_idx.npy"),    shap_idx)
np.save(os.path.join(OUT, "precomp_shap_single.npy"), shap_single)
np.save(os.path.join(OUT, "precomp_last_i.npy"),      np.array([last_i]))
np.save(os.path.join(OUT, "precomp_eng_ids_test.npy"),eng_ids_test)
np.save(os.path.join(OUT, "precomp_y_rul_test.npy"),  y_rul_test)
with open(os.path.join(OUT, "precomp_expected_value.pkl"), "wb") as f:
    pickle.dump(float(explainer.expected_value), f)

p("All results saved. Done!")
