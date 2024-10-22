from src.data.generate_synthetic import generate_synthetic_patients

def test_generate_shape_and_columns():
    df = generate_synthetic_patients(n=100, seed=1)
    assert df.shape[0] == 100
    assert "target_high_risk" in df.columns