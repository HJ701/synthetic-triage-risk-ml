import argparse
import pandas as pd
import joblib
from src.utils.config import PATHS

def parse_args():
    p = argparse.ArgumentParser(description="Predict synthetic triage risk from a CSV.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()

def main():
    args = parse_args()
    model_path = PATHS.reports / "models" / "triage_logreg.joblib"
    pipe = joblib.load(model_path)
    
    df = pd.read_csv(args.input)
    prob = pipe.predict_proba(df)[:, 1]
    pred = (prob >= args.threshold).astype(int)
    
    out = df.copy()
    out["prob_high_risk"] = prob
    out["pred_high_risk"] = pred
    out.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()