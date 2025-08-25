from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from .utils import get_logger, load_pickle, save_json
from .windowing import build_feature_table

logger = get_logger("evaluate")


def run(cfg: dict, model_path: str = "checkpoints/model.pkl"):
    art = load_pickle(model_path)
    model = art["model"]
    scaler = art["scaler"]
    cols = art["cols"]

    df = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "timeseries.parquet")
    variables = cfg["features"]["variables"]
    W = cfg["labeling"]["window_hours"]
    H = cfg["labeling"]["horizon_hours"]
    X, y = build_feature_table(df, variables, W, H)

    # hold-out test split (same method as in train)
    pids = X["patient_id"].values
    unique = np.unique(pids)
    n_train = int(cfg["split"]["train_frac"] * len(unique))
    n_valid = int(cfg["split"]["valid_frac"] * len(unique))
    test_p = set(unique[n_train + n_valid :])
    idx_te = np.isin(pids, list(test_p))

    X_te = scaler.transform(X.loc[idx_te, cols])
    y_te = y[idx_te]

    p = model.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_te, p)
    auprc = average_precision_score(y_te, p)
    brier = brier_score_loss(y_te, p)

    frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")

    logger.info(f"TEST AUROC={auroc:.3f} AUPRC={auprc:.3f} Brier={brier:.3f}")

    out = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier": float(brier),
        "calibration": {
            "fraction_of_positives": frac_pos.tolist(),
            "mean_predicted_value": mean_pred.tolist(),
        },
    }

    Path(cfg["project"]["reports_dir"]).mkdir(parents=True, exist_ok=True)
    save_json(out, Path(cfg["project"]["reports_dir"]) / "metrics_test.json")
