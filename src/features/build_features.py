import pandas as pd

NUMERIC_COLS = ["age", "systolic_bp", "heart_rate", "resp_rate", "temperature_c", "spo2"]
CATEGORICAL_COLS = ["sex"]
TARGET_COL = "target_high_risk"

def split_xy(df: pd.DataFrame):
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y