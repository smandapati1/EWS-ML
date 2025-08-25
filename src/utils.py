from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import random
import yaml
import numpy as np
import joblib


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def ensure_dir(p: str | os.PathLike):
    Path(p).mkdir(parents=True, exist_ok=True)


def ensure_project_dirs(cfg: dict):
    ensure_dir(cfg["project"]["checkpoints_dir"])
    ensure_dir(cfg["project"]["reports_dir"])
    ensure_dir(cfg["project"]["logs_dir"])
    ensure_dir(cfg["data"]["processed_dir"])


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str = "ews"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def save_pickle(obj, path: str | os.PathLike):
    joblib.dump(obj, path)


def load_pickle(path: str | os.PathLike):
    return joblib.load(path)


def save_json(data: dict, path: str | os.PathLike):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
