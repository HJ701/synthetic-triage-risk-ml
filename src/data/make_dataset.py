from pathlib import Path
import pandas as pd
from src.data.generate_synthetic import generate_synthetic_patients
from src.utils.config import PATHS
from src.utils.logging import get_logger

log = get_logger(__name__)

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main(n: int = 5000, seed: int = 42) -> None:
    PATHS.data_raw.mkdir(parents=True, exist_ok=True)
    PATHS.data_processed.mkdir(parents=True, exist_ok=True)
    
    raw_path = PATHS.data_raw / "synthetic_patients.csv"
    processed_path = PATHS.data_processed / "patients_processed.csv"
    
    df = generate_synthetic_patients(n=n, seed=seed)
    save_csv(df, raw_path)
    
    df2 = df.drop_duplicates().copy()
    df2["sex"] = df2["sex"].astype("category")
    save_csv(df2, processed_path)
    
    log.info(f"Wrote raw dataset: {raw_path}")
    log.info(f"Wrote processed dataset: {processed_path}")

if __name__ == "__main__":
    main()