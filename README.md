# 🏥 Healthcare Predictive Early Warning System (EWS)

A machine learning project that predicts **patient deterioration within the next 6–12 hours** using multivariate clinical time-series data (vital signs, labs, demographics).  
The goal is to build an **early-warning risk score** that can help clinicians anticipate ICU transfer, sepsis, or critical decline and take preventive action.  

---

## 🚀 Features
- ⏱ **Time-series modeling**: Handles rolling patient windows of vitals (HR, RR, SBP, SpO2, Temp, etc.).  
- 🧮 **Feature engineering**: Trends, slopes, volatility, abnormal counts, missingness masks, and “time since last measurement.”  
- 🤖 **Models**:
  - Baseline: Gradient Boosting (XGBoost/LightGBM)  
  - Sequence: GRU/LSTM/Temporal CNN  
  - Hybrid: Static + temporal features combined  
- 📊 **Evaluation**: AUROC, AUPRC, calibration curves, subgroup metrics by age/sex/comorbidities.  
- 🔎 **Interpretability**: SHAP/feature attribution for “reason codes” behind alerts.  
- 📈 **Deployment Demo**: Streamlit dashboard showing patient risk lists, vitals trends, and top features contributing to risk.  

---

## 📂 Project Structure
