from __future__ import annotations
import numpy as np
import pandas as pd


def _agg_stats(s: pd.Series) -> dict:
    if s.notna().any():
        sff = s.ffill()
        return {
            "last": sff.iloc[-1],
            "mean": sff.mean(),
            "min": sff.min(),
            "max": sff.max(),
            "std": sff.std(),
            "slope": (sff.iloc[-1] - sff.iloc[0]) / max(1, len(sff) - 1),
        }
    else:
        return {k: np.nan for k in ["last", "mean", "min", "max", "std", "slope"]}


def build_feature_table(df: pd.DataFrame, variables: list[str], window_hours: int, horizon_hours: int) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    feats = []
    ys = []

    for pid, g in df.groupby("patient_id"):
        g = g.reset_index(drop=True)
        for t in range(window_hours - 1, len(g)):
            win = g.iloc[t - window_hours + 1 : t + 1]
            y = int(g.loc[t, "y"])

            row = {
                "patient_id": pid,
                "hour": int(g.loc[t, "hour"]),
                "Age": float(g.loc[t, "Age"]),
                "Charlson": float(g.loc[t, "Charlson"]),
            }

            for c in variables:
                stats = _agg_stats(win[c])
                row.update({f"{c}_{k}": v for k, v in stats.items()})
                row[f"{c}_missing_frac"] = float(win[c].isna().mean())
                last_idx = win[c].last_valid_index()
                tslm = (win.index[-1] - last_idx) if last_idx is not None else window_hours
                row[f"{c}_tslm"] = int(tslm)

            feats.append(row)
            ys.append(y)

    X = pd.DataFrame(feats)
    y = np.asarray(ys, dtype=int)
    return X, y
