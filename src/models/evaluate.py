from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from src.features.build_features import split_xy
from src.utils.config import PATHS
from src.utils.metrics import classification_report
from src.utils.logging import get_logger

log = get_logger(__name__)

def main() -> None:
    df = pd.read_csv(PATHS.data_processed / "patients_processed.csv")
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model_path = PATHS.reports / "models" / "triage_logreg.joblib"
    pipe = joblib.load(model_path)
    
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    rep = classification_report(y_test, prob, pred)
    
    log.info(f"ROC-AUC: {rep.roc_auc:.4f} | PR-AUC: {rep.pr_auc:.4f} | F1: {rep.f1:.4f}")
    
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    
    RocCurveDisplay.from_predictions(y_test, prob)
    plt.title("ROC Curve (Synthetic Triage Risk)")
    roc_path = PATHS.figures / "roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    
    PrecisionRecallDisplay.from_predictions(y_test, prob)
    plt.title("Precision-Recall Curve (Synthetic Triage Risk)")
    pr_path = PATHS.figures / "pr_curve.png"
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    
    log.info(f"Saved figures: {roc_path}, {pr_path}")

if __name__ == "__main__":
    main()