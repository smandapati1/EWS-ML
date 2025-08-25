from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import get_logger

logger = get_logger("preprocess")


def _generate_synthetic(cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    N = cfg["data"]["n_patients"]
    H = cfg["data"]["hours"]

    rows = []
    for pid in range(N):
        base_hr = rng.normal(80, 8)
        base_rr = rng.normal(16, 3)
        base_sbp = rng.normal(120, 10)
        age = rng.integers(20, 90)
        charlson = int(np.clip(rng.normal(2, 1.5), 0, 8))
        event_hour = int(rng.integers(18, H)) if rng.random() < 0.18 else -1

        for h in range(H):
            drift = 0.0
            if event_hour >= 0 and h >= event_hour - 12:
                drift = (h - (event_hour - 12)) / 12.0

            hr = base_hr + rng.normal(0, 3) + 20 * drift
            rr = base_rr + rng.normal(0, 1.2) + 6 * drift
            sbp = base_sbp + rng.normal(0, 5) - 12 * drift
            spo2 = 98 - 3 * drift + rng.normal(0, 0.7)
            temp = 36.8 + 0.8 * drift + rng.normal(0, 0.15)

            # add missingness
            if rng.random() < 0.05: hr = np.nan
            if rng.random() < 0.05: rr = np.nan
            if rng.random() < 0.05: sbp = np.nan
            if rng.random() < 0.05: spo2 = np.nan
            if rng.random() < 0.05: temp = np.nan

            rows.append([pid, h, age, charlson, hr, rr, sbp, spo2, temp, event_hour])

    df = pd.DataFrame(rows, columns=[
        "patient_id", "hour", "Age", "Charlson",
        "HR", "RR", "SBP", "SpO2", "Temp", "event_hour"
    ])

    horizon = cfg["labeling"]["horizon_hours"]
    df["y"] = (
        (df["event_hour"] >= 0)
        & (df["hour"] < df["event_hour"])
        & (df["hour"] >= df["event_hour"] - horizon)
    ).astype(int)
    return df


def run(cfg: dict):
    processed_dir = Path(cfg["data"]["processed_dir"]).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    if cfg["data"]["use_synthetic"]:
        logger.info("Generating synthetic dataset…")
        df = _generate_synthetic(cfg)
    else:
        raise NotImplementedError("Real dataset loading not implemented yet.")

    out_path = processed_dir / "timeseries.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved processed dataset → {out_path}")
