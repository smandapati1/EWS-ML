import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="EWS Dashboard", layout="wide")

REPORTS = Path("reports")
PRED_PATH = REPORTS / "pred_test.csv"
METRICS_PATH = REPORTS / "metrics_test.json"
FEATIMP_PATH = REPORTS / "feature_importance.json"

st.title("🏥 Early Warning System — Demo Dashboard")

# ---- Active Alerts ----
if not PRED_PATH.exists():
    st.warning("No predictions found. Train a model first with:\n\n"
               "`python main.py train --config configs/default.yaml --model-config configs/model_gbm.yaml`")
else:
    df = pd.read_csv(PRED_PATH)
    thr_mode = st.radio("Alerting mode", ["Top-k%", "Threshold"], horizontal=True)

    if thr_mode == "Top-k%":
        k = st.slider("Alert budget (percent of patients)", 1, 30, 10)
        n = max(1, int(len(df) * k / 100))
        thr = df["p"].nlargest(n).min()
    else:
        thr = st.slider("Risk threshold", 0.0, 1.0, 0.5)

    alerts = df[df["p"] >= thr].copy()
    alerts.sort_values(["p", "patient_id", "hour"], ascending=[False, True, True], inplace=True)

    st.subheader("Active Alerts")
    st.dataframe(alerts.head(200))

# ---- Metrics ----
st.subheader("Metrics (test)")
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    st.json(metrics)
else:
    st.info("Run `python main.py evaluate` to compute metrics.")

# ---- Feature Importance ----
st.subheader("Top Contributing Features")
if FEATIMP_PATH.exists():
    with open(FEATIMP_PATH, "r") as f:
        feat_imp = json.load(f)["top_features"]

    feat_df = pd.DataFrame.from_dict(feat_imp, orient="index", columns=["importance"])
    feat_df = feat_df.sort_values("importance", ascending=False)

    st.bar_chart(feat_df)
else:
    st.info("Run `python main.py interpret` to compute feature importances.")

