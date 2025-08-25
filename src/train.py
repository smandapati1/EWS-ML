from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from .utils import get_logger, save_pickle
from .windowing import build_feature_table
from .mos import build_model

logger = get_logger("train")


def _split_by_patient(X: pd.DataFrame, y: np.ndarray, train_frac: float, valid_frac: float):
    pids = X["patient_id"].values
    unique = np.unique(pids)
    n_train = int(train_frac * len(unique))
    n_valid = int(valid_frac * len(unique))
    train_p = set(unique[:n_train])
    valid_p = set(unique[n_train : n_train + n_valid])
    test_p = set(unique[n_train + n_valid :])

    idx_train = np.isin(pids, list(train_p))
    idx_valid = np.isin(pids, list(valid_p))
    idx_test  = np.isin(pids, list(test_p))
    return idx_train, idx_valid, idx_test


def run(cfg: dict):
    processed_path = Path(cfg["data"]["processed_dir"]) / "timeseries.parquet"
    df = pd.read_parquet(processed_path)

    variables = cfg["features"]["variables"]
    W = cfg["labeling"]["window_hours"]
    H = cfg["labeling"]["horizon_hours"]
    X, y = build_feature_table(df, variables, W, H)

    idx_tr, idx_va, idx_te = _split_by_patient(
        X, y, cfg["split"]["train_frac"], cfg["split"]["valid_frac"]
    )

    # Feature columns (exclude IDs/timestamps)
    cols = [c for c in X.columns if c not in ["patient_id", "hour"]]

    # ---- Median imputation learned ONLY on train (prevents leakage) ----
    train_medians = X.loc[idx_tr, cols].median(numeric_only=True)

    X_tr_df = X.loc[idx_tr, cols].fillna(train_medians)
    X_va_df = X.loc[idx_va, cols].fillna(train_medians)
    X_te_df = X.loc[idx_te, cols].fillna(train_medians)
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]

    # ---- Scaling (fit on train only) ----
    scaler = StandardScaler(with_mean=cfg["scaling"]["with_mean"])
    X_tr = scaler.fit_transform(X_tr_df)
    X_va = scaler.transform(X_va_df)
    X_te = scaler.transform(X_te_df)

    # ---- Model ----
    model = build_model(cfg["model"]["name"], cfg["model"].get("params", {}), cfg.get("calibration", {}))
    model.fit(X_tr, y_tr)

    def _metrics(name, Xs, ys):
        p = model.predict_proba(Xs)[:, 1]
        auroc = roc_auc_score(ys, p)
        auprc = average_precision_score(ys, p)
        logger.info(f"{name}: AUROC={auroc:.3f} AUPRC={auprc:.3f}")
        return p

    p_va = _metrics("VALID", X_va, y_va)
    p_te = _metrics("TEST",  X_te, y_te)

    # ---- Save artifacts ----
    ckpt_dir = Path(cfg["project"]["checkpoints_dir"])
    reports_dir = Path(cfg["project"]["reports_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save model, scaler, feature list, and training medians for future imputation
    save_pickle(
        {"model": model, "scaler": scaler, "cols": cols, "medians": train_medians.to_dict()},
        ckpt_dir / "model.pkl",
    )

    # Save predictions for dashboard
    pred_te = X.loc[idx_te, ["patient_id", "hour"]].copy()
    pred_te["y"] = y_te
    pred_te["p"] = p_te
    pred_te.to_csv(reports_dir / "pred_test.csv", index=False)

    pred_va = X.loc[idx_va, ["patient_id", "hour"]].copy()
    pred_va["y"] = y_va
    pred_va["p"] = p_va
    pred_va.to_csv(reports_dir / "pred_valid.csv", index=False)

    logger.info(f"Saved model → {ckpt_dir / 'model.pkl'}")
    logger.info(f"Saved predictions → {reports_dir}")
