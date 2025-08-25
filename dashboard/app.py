import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="EWS Dashboard", layout="wide")

REPORTS = Path("reports")
PRED_PATH = REPORTS / "pred_test.csv"
METRICS_PATH = REPORTS / "metrics_test.json"

st.title("🏥 Early Warning System — Demo Dashboard")

if not PRED_PATH.exists():
    st.warning("No predictions found. Train a model first with: "
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

    st.subheader("Metrics (test)")
    if METRICS_PATH.exists():
        st.json(pd.read_json(METRICS_PATH).to_dict())
    else:
        st.info("Run `python main.py evaluate` to compute metrics.")
