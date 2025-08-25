from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .utils import get_logger, load_pickle, save_json
from .windowing import build_feature_table

logger = get_logger("int")


def _fallback_importance(model, feature_names: list[str], topk: int = 10):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        order = np.argsort(imp)[::-1][:topk]
        return {feature_names[i]: float(imp[i]) for i in order}
    return {}


def run(cfg: dict, model_path: str = "checkpoints/model.pkl"):
    art = load_pickle(model_path)
    model = art["model"]
    scaler = art["scaler"]
    cols = art["cols"]
    medians = art.get("medians", {})

    df = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "timeseries.parquet")
    variables = cfg["features"]["variables"]
    W = cfg["labeling"]["window_hours"]
    H = cfg["labeling"]["horizon_hours"]
    X, _ = build_feature_table(df, variables, W, H)

    # Use a small sample for speed; impute with training medians
    Xs_df = X[cols].iloc[:1000].copy().fillna(medians)
    Xs = scaler.transform(Xs_df)

    try:
        import shap  # optional
        explainer = shap.Explainer(model, Xs)
        sv = explainer(Xs)
        mean_abs = np.abs(sv.values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:15]
        top = {cols[i]: float(mean_abs[i]) for i in order}
    except Exception as e:
        logger.info(f"SHAP unavailable or failed ({e}); using fallback importances.")
        # If model is CalibratedClassifierCV, underlying estimator may be in .base_estimator
        core = getattr(model, "base_estimator", model)
        top = _fallback_importance(core, cols, topk=15)

    Path(cfg["project"]["reports_dir"]).mkdir(parents=True, exist_ok=True)
    save_json({"top_features": top}, Path(cfg["project"]["reports_dir"]) / "feature_importance.json")
