"""
Turbofan Anomaly Detection — Operational Dashboard
Run:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pickle
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Turbofan Anomaly Detection",
    page_icon="✈️",
    layout="wide",
)

ALERT_RUL  = 30
WARN_RUL   = 50
AE_THRESH  = 0.0937
RENDER_URL = "https://turbofan-anomaly-api.onrender.com"


# ── Data loading (cached across reruns) ──────────────────────────────────────
@st.cache_data
def load_all():
    proc = ROOT / "data" / "processed"

    X_all   = np.load(proc / "X_features.npy")
    eng_all = np.load(proc / "engine_ids_flat.npy")
    test_ids = np.unique(eng_all)[80:]          # engines 81-100
    X_feat_test = X_all[np.isin(eng_all, test_ids)]

    return {
        "rul_pred":     np.load(proc / "precomp_rul_pred.npy"),
        "ae_score":     np.load(proc / "precomp_ae_score.npy"),
        "eng_ids":      np.load(proc / "precomp_eng_ids_test.npy"),
        "y_rul":        np.load(proc / "precomp_y_rul_test.npy"),
        "X_feat_test":  X_feat_test,
        "feature_cols": pickle.load(open(proc / "feature_cols.pkl", "rb")),
    }


@st.cache_resource
def load_explainer():
    import shap
    import xgboost as xgb
    m = xgb.XGBRegressor()
    m.load_model(ROOT / "models" / "xgb_rul.json")
    return shap.TreeExplainer(m)


# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_info(rul: float):
    if rul < ALERT_RUL:
        return "HIGH RISK", "#C62828", "🔴"
    if rul < WARN_RUL:
        return "WARNING", "#E65100", "🟡"
    return "SAFE", "#1B5E20", "🟢"


def engine_card_html(eng, rul, risk, color):
    return f"""
    <div style="background:{color};color:white;padding:10px 6px;border-radius:10px;
                text-align:center;margin:3px 0;">
      <div style="font-size:12px;font-weight:600;opacity:.9;">Engine {eng}</div>
      <div style="font-size:26px;font-weight:700;line-height:1.1;">{rul:.0f}</div>
      <div style="font-size:10px;opacity:.85;">cycles RUL</div>
      <div style="font-size:10px;font-weight:700;margin-top:3px;">{risk}</div>
    </div>"""


def fleet_snapshot(pct, rul_pred, eng_ids, engines):
    """Return fleet status at pct% through each engine's life."""
    rows = []
    for eng in engines:
        mask = np.where(eng_ids == eng)[0]
        idx  = min(int(pct / 100 * len(mask)), len(mask) - 1)
        rul  = float(rul_pred[mask[idx]])
        risk, color, emoji = risk_info(rul)
        rows.append(dict(engine=int(eng), rul=round(rul, 1),
                         risk=risk, color=color, emoji=emoji))
    return rows


# ── Load data ─────────────────────────────────────────────────────────────────
D           = load_all()
rul_pred    = D["rul_pred"]
ae_score    = D["ae_score"]
eng_ids     = D["eng_ids"]
y_rul       = D["y_rul"]
X_feat_test = D["X_feat_test"]
feat_cols   = D["feature_cols"]
engines     = sorted(np.unique(eng_ids).tolist())


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")

    selected_engine = st.selectbox(
        "Select Engine",
        engines,
        index=engines.index(85) if 85 in engines else 0,
    )

    mission_pct = st.slider(
        "Mission Progress (%)", 0, 100, 75,
        help="Simulates fleet health at different points in each engine's service life. "
             "100 % = end-of-life (run-to-failure test data).",
    )

    st.divider()
    st.markdown("**Dataset**")
    st.markdown("- NASA CMAPSS FD001")
    st.markdown("- 100 engines, run-to-failure")
    st.markdown("- 20 test engines (81–100)")
    st.markdown("- 3,913 prediction windows")

    st.divider()
    st.markdown("**Models**")
    st.markdown("- XGBoost RUL: MAE 29.4 cyc, R²=0.614")
    st.markdown("- LSTM Autoencoder: AUC 0.732")
    st.markdown("- HIGH RISK threshold: 30 cycles")
    st.markdown("- 94 / 100 engines caught")

    st.divider()
    st.markdown(f"**Live API**")
    st.markdown(f"[turbofan-anomaly-api.onrender.com]({RENDER_URL})")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("✈️ Turbofan Anomaly Detection")
st.markdown(
    "**Real-Time RUL Prediction & Anomaly Monitoring · NASA CMAPSS FD001**  \n"
    "XGBoost RUL regression + LSTM Autoencoder · Deployed on Render"
)

fleet = fleet_snapshot(mission_pct, rul_pred, eng_ids, engines)
n_high = sum(1 for f in fleet if f["risk"] == "HIGH RISK")
n_warn = sum(1 for f in fleet if f["risk"] == "WARNING")
n_safe = sum(1 for f in fleet if f["risk"] == "SAFE")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Engines Monitored", len(fleet))
k2.metric("🔴 HIGH RISK",      n_high)
k3.metric("🟡 WARNING",        n_warn)
k4.metric("🟢 SAFE",           n_safe)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Fleet Health",
    "🔍 Engine Analysis",
    "🌐 Live API",
    "📋 Maintenance Schedule",
])


# ─────────────────────────────── TAB 1: Fleet Health ─────────────────────────
with tab1:
    st.subheader(f"Fleet Health Grid — Mission Progress {mission_pct}%")
    st.caption(
        "Each card shows predicted RUL at the selected mission progress.  "
        "Adjust the slider in the sidebar to time-travel through the fleet's service history."
    )

    GRID_COLS = 5
    for row_data in [fleet[i:i+GRID_COLS] for i in range(0, len(fleet), GRID_COLS)]:
        cols = st.columns(GRID_COLS)
        for col, f in zip(cols, row_data):
            col.markdown(
                engine_card_html(f["engine"], f["rul"], f["risk"], f["color"]),
                unsafe_allow_html=True,
            )

    st.markdown("---")

    df_fleet = pd.DataFrame(fleet)
    fig_bar = px.bar(
        df_fleet, x="engine", y="rul", color="risk",
        color_discrete_map={
            "HIGH RISK": "#C62828",
            "WARNING":   "#E65100",
            "SAFE":      "#1B5E20",
        },
        labels={"engine": "Engine ID", "rul": "Predicted RUL (cycles)", "risk": "Risk Level"},
        title=f"Fleet RUL Comparison — Mission Progress {mission_pct}%",
    )
    fig_bar.add_hline(y=ALERT_RUL, line_dash="dash", line_color="#C62828",
                      annotation_text="HIGH RISK (30 cyc)")
    fig_bar.add_hline(y=WARN_RUL, line_dash="dot", line_color="#E65100",
                      annotation_text="WARNING (50 cyc)")
    fig_bar.update_layout(height=320, showlegend=True)
    st.plotly_chart(fig_bar, use_container_width=True)


# ───────────────────────────── TAB 2: Engine Analysis ────────────────────────
with tab2:
    st.subheader(f"Engine {selected_engine} — Deep Dive")

    eng_mask     = np.where(eng_ids == selected_engine)[0]
    eng_rul_pred = rul_pred[eng_mask]
    eng_rul_true = y_rul[eng_mask]
    eng_ae_arr   = ae_score[eng_mask]

    last_pred = float(eng_rul_pred[-1])
    last_true = float(eng_rul_true[-1])
    risk, _, emoji = risk_info(last_pred)

    valid_ae = eng_ae_arr[~np.isnan(eng_ae_arr)]
    last_ae  = float(valid_ae[-1]) if len(valid_ae) > 0 else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted RUL (end)",  f"{last_pred:.1f} cycles",
              delta=f"{last_pred - last_true:+.1f} vs actual")
    m2.metric("Actual RUL (end)",     f"{last_true:.1f} cycles")
    m3.metric("Status",               f"{emoji} {risk}")
    if last_ae is not None:
        m4.metric("Anomaly Score", f"{last_ae:.4f}",
                  delta="FLAGGED" if last_ae > AE_THRESH else "NORMAL",
                  delta_color="inverse" if last_ae > AE_THRESH else "off")
    else:
        m4.metric("Anomaly Score", "— (demo engines only)")

    st.markdown("---")
    col_rul, col_shap = st.columns(2)

    # RUL trend chart
    with col_rul:
        cycles = np.arange(len(eng_rul_pred))
        fig_rul = go.Figure()
        fig_rul.add_trace(go.Scatter(
            x=cycles, y=eng_rul_true, name="Actual RUL",
            line=dict(color="#1565C0", dash="dash", width=1.5),
        ))
        fig_rul.add_trace(go.Scatter(
            x=cycles, y=eng_rul_pred, name="Predicted RUL",
            line=dict(color="#B71C1C", width=2.5),
        ))
        fig_rul.add_hrect(y0=0, y1=ALERT_RUL, fillcolor="#C62828",
                          opacity=0.07, line_width=0)
        fig_rul.add_hline(y=ALERT_RUL, line_dash="dot", line_color="#C62828",
                          annotation_text="Alert threshold (30 cyc)")
        fig_rul.update_layout(
            title=f"Engine {selected_engine} — RUL Over Time",
            xaxis_title="Cycle", yaxis_title="Remaining Useful Life (cycles)",
            height=360, legend=dict(x=0, y=1),
        )
        st.plotly_chart(fig_rul, use_container_width=True)

    # SHAP waterfall
    with col_shap:
        st.markdown("**Top SHAP Feature Contributions (last window)**")
        with st.spinner("Computing SHAP explanations…"):
            try:
                explainer = load_explainer()
                last_i    = eng_mask[-1]
                sv        = explainer.shap_values(X_feat_test[last_i : last_i + 1])[0]

                TOP_N  = 12
                top_i  = np.argsort(np.abs(sv))[-TOP_N:]
                top_sv = sv[top_i]
                top_nm = [feat_cols[i] for i in top_i]
                colors = ["#C62828" if v < 0 else "#1B5E20" for v in top_sv]

                fig_shap = go.Figure(go.Bar(
                    x=top_sv, y=top_nm, orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.2f}" for v in top_sv],
                    textposition="outside",
                ))
                fig_shap.update_layout(
                    title="SHAP Drivers  (red = fewer cycles, green = more cycles)",
                    xaxis_title="SHAP Value",
                    height=360,
                    margin=dict(l=150),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.warning(f"SHAP requires the venv311 (Metal GPU) kernel. Error: {e}")

    # Anomaly score timeline (only for engines 81-86)
    if last_ae is not None:
        st.markdown("---")
        ae_idx = np.where(~np.isnan(eng_ae_arr))[0]
        fig_ae = go.Figure()
        fig_ae.add_trace(go.Scatter(
            x=ae_idx, y=valid_ae, name="Anomaly Score",
            fill="tozeroy", line=dict(color="#6A1B9A", width=2),
        ))
        fig_ae.add_hline(y=AE_THRESH, line_dash="dash", line_color="#C62828",
                         annotation_text=f"Threshold ({AE_THRESH})")
        fig_ae.update_layout(
            title=f"Engine {selected_engine} — LSTM Autoencoder Anomaly Score",
            xaxis_title="Cycle", yaxis_title="Reconstruction MAE",
            height=240,
        )
        st.plotly_chart(fig_ae, use_container_width=True)


# ───────────────────────────────── TAB 3: Live API ───────────────────────────
with tab3:
    st.subheader("Live API Demonstration")
    st.markdown(
        f"Sends Engine **{selected_engine}**'s last 110-element feature vector to the "
        f"deployed FastAPI service on Render and returns a real-time RUL prediction."
    )

    last_i    = int(np.where(eng_ids == selected_engine)[0][-1])
    feat_list = X_feat_test[last_i].tolist()
    payload   = {"engine_id": f"engine_{selected_engine}", "feature_vector": feat_list}

    c_req, c_resp = st.columns(2)

    with c_req:
        st.markdown("**POST /predict — Request**")
        st.json({
            "engine_id":      payload["engine_id"],
            "feature_vector": feat_list[:6] + ["… (110 values total) …"],
        })
        st.caption(
            "Feature vector: 11 raw sensors + 5/10/30-cycle rolling mean, std, "
            "rate-of-change = 110 features"
        )

    with c_resp:
        st.markdown("**Response**")
        if st.button("🚀 Call Live API", type="primary", use_container_width=True):
            with st.spinner("Calling Render (allow up to 40 s for cold start)…"):
                try:
                    t0   = time.time()
                    resp = requests.post(
                        f"{RENDER_URL}/predict", json=payload, timeout=40
                    )
                    ms = (time.time() - t0) * 1000
                    if resp.status_code == 200:
                        st.success(f"✅ Response in {ms:.0f} ms")
                        st.json(resp.json())
                    else:
                        st.error(f"HTTP {resp.status_code}: {resp.text[:300]}")
                except requests.exceptions.Timeout:
                    st.warning("Timed out. Service is cold-starting. Try again in 30 s.")
                except Exception as e:
                    st.error(str(e))
        else:
            st.info("Click above to call the live endpoint.")
            st.markdown(
                f"Swagger UI (interactive docs): [{RENDER_URL}/docs]({RENDER_URL}/docs)"
            )


# ──────────────────────────── TAB 4: Maintenance Schedule ────────────────────
with tab4:
    st.subheader("Maintenance Priority Schedule")
    st.caption(
        "Ranked by end-of-mission predicted RUL. "
        "In a real deployment this table would refresh every cycle with live sensor data."
    )

    last_fleet = fleet_snapshot(100, rul_pred, eng_ids, engines)
    df = (
        pd.DataFrame(last_fleet)
        .sort_values("rul")
        .reset_index(drop=True)
    )
    df["Priority"] = range(1, len(df) + 1)
    df["Recommended Action"] = df["rul"].apply(
        lambda r: (
            "🔴 GROUND — immediate maintenance" if r < ALERT_RUL else
            "🟡 SCHEDULE within 10 cycles"      if r < WARN_RUL  else
            "🟢 MONITOR — next routine check"
        )
    )

    st.dataframe(
        df[["Priority", "engine", "rul", "risk", "Recommended Action"]].rename(
            columns={"engine": "Engine", "rul": "Predicted RUL (cycles)", "risk": "Risk Level"}
        ),
        use_container_width=True,
        height=520,
        hide_index=True,
    )

    st.markdown("---")
    avg_rul  = np.mean([f["rul"] for f in last_fleet])
    critical = min(last_fleet, key=lambda x: x["rul"])
    best     = max(last_fleet, key=lambda x: x["rul"])

    s1, s2, s3 = st.columns(3)
    s1.metric("Fleet Mean RUL",  f"{avg_rul:.1f} cycles")
    s2.metric("Most Critical",   f"Engine {critical['engine']} ({critical['rul']:.1f} cyc)")
    s3.metric("Least Critical",  f"Engine {best['engine']} ({best['rul']:.1f} cyc)")
