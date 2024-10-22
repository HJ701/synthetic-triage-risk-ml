import pandas as pd
from src.data.generate_synthetic import generate_synthetic_patients
from src.models.train import build_model
from src.features.build_features import split_xy

def test_train_smoke():
    df = generate_synthetic_patients(n=300, seed=123)
    X, y = split_xy(df)
    pipe = build_model()
    pipe.fit(X, y)
    prob = pipe.predict_proba(X)[:, 1]
    assert len(prob) == len(df)