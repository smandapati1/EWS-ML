import numpy as np
import pandas as pd
from src.windowing import _agg_stats


def test_agg_stats_handles_all_nan():
    s = pd.Series([np.nan, np.nan, np.nan])
    d = _agg_stats(s)
    for v in d.values():
        assert np.isnan(v)
