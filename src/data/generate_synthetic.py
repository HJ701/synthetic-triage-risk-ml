import numpy as np
import pandas as pd

def generate_synthetic_patients(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic triage-like dataset."""
    rng = np.random.default_rng(seed)
    
    age = rng.integers(18, 90, size=n)
    sex = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])
    systolic_bp = rng.normal(125, 18, size=n).clip(80, 220)
    heart_rate = rng.normal(78, 14, size=n).clip(40, 180)
    resp_rate = rng.normal(16, 4, size=n).clip(8, 40)
    temperature_c = rng.normal(36.8, 0.5, size=n).clip(34.0, 41.0)
    spo2 = rng.normal(97, 2, size=n).clip(75, 100)

    # A simple synthetic risk signal
    logit = (-6.0 
             + 0.03 * (age - 50) 
             + 0.08 * (90 - spo2) 
             + 0.015 * (heart_rate - 80) 
             + 0.02 * (resp_rate - 16) 
             + 0.01 * (systolic_bp - 125) 
             + 0.6 * (temperature_c - 37.0))
    
    logit += np.where(sex == "M", 0.15, -0.05)
    prob = 1 / (1 + np.exp(-logit))
    risk = rng.binomial(1, prob)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "systolic_bp": systolic_bp.round(1),
        "heart_rate": heart_rate.round(1),
        "resp_rate": resp_rate.round(1),
        "temperature_c": temperature_c.round(2),
        "spo2": spo2.round(1),
        "target_high_risk": risk,
    })
    return df