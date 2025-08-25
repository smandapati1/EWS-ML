import pandas as pd
import numpy as np
from src.windowing import build_feature_table


def test_build_feature_table_shapes():
    df = pd.DataFrame({
        "patient_id": [0,0,0,1,1,1],
        "hour": [0,1,2,0,1,2],
        "Age": [50,50,50,60,60,60],
        "Charlson": [1,1,1,2,2,2],
        "HR": [80,82,85,75,77,79],
        "RR": [16,16,17,14,15,15],
        "SBP": [120,118,116,130,128,126],
        "SpO2": [98,97,96,99,98,98],
        "Temp": [36.7,36.8,36.9,36.6,36.6,36.7],
        "y": [0,0,1,0,0,0]
    })
    X, y = build_feature_table(df, ["HR","RR","SBP","SpO2","Temp"], 2, 12)
    assert len(X) == len(y) == 4
    assert {"Age","Charlson"}.issubset(set(X.columns))
