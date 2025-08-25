from __future__ import annotations
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


def build_model(name: str, params: dict | None, calibration: dict | None):
    name = name.lower()
    if name == "gbm":
        base = GradientBoostingClassifier(**(params or {}))
    else:
        raise NotImplementedError(f"Model '{name}' not implemented.")

    if calibration and calibration.get("enabled", False):
        return CalibratedClassifierCV(
            base,
            method=calibration.get("method", "isotonic"),
            cv=calibration.get("cv", 3),
        )
    return base
