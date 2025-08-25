import numpy as np
from src.mos import build_model


def test_gbm_train_step():
    X = np.random.RandomState(0).randn(100, 5)
    y = (X[:,0] + 0.5 * X[:,1] > 0).astype(int)
    model = build_model("gbm", {"n_estimators": 10, "max_depth": 2}, {"enabled": False})
    model.fit(X, y)
    p = model.predict_proba(X)[:,1]
    assert p.shape == (100,)
