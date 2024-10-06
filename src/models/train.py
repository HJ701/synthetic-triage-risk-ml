from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features.build_features import NUMERIC_COLS, CATEGORICAL_COLS, split_xy
from src.utils.config import PATHS
from src.utils.logging import get_logger

log = get_logger(__name__)

def load_processed() -> pd.DataFrame:
    path = PATHS.data_processed / "patients_processed.csv"
    return pd.read_csv(path)

def build_model() -> Pipeline:
    pre = ColumnTransformer(transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    return Pipeline([("pre", pre), ("model", model)])

def main() -> None:
    df = load_processed()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipe = build_model()
    pipe.fit(X_train, y_train)
    
    prob = pipe.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, prob))
    log.info(f"ROC-AUC (holdout): {auc:.4f}")
    
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    model_dir = PATHS.reports / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "triage_logreg.joblib"
    joblib.dump(pipe, model_path)
    
    meta = {"model_path": str(model_path), "holdout_roc_auc": auc}
    with open(model_dir / "train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    log.info(f"Saved model to: {model_path}")

if __name__ == "__main__":
    main()