# 🏥 Patient Triage Risk Prediction: ML-Powered HealthTech Simulation

![Project Banner](tests/syntetich_triage.png)

## 👋 About the Project

This repository implements an end-to-end machine learning pipeline for predicting patient triage risk levels (high-risk vs. low-risk) based on vital signs (pulse, blood pressure, temperature, SpO2) and demographic data. Built as a HealthTech simulation, the project operates entirely on **synthetic data** generated using statistical distributions—no real patient information is used.

The system demonstrates a complete ML workflow: synthetic data generation, preprocessing, model training (Logistic Regression), performance evaluation (ROC/PR curves), and batch inference for clinical decision support scenarios.

## 🎯 What Does It Do?

- **Synthetic Data Generation**: Creates 5,000 realistic patient records with vital signs and demographics
- **Feature Engineering**: Processes age, gender, systolic BP, heart rate, respiratory rate, temperature, and SpO2
- **Risk Classification**: Predicts binary triage outcomes (high-risk/low-risk) using balanced Logistic Regression
- **Performance Visualization**: Generates ROC and Precision-Recall curves for model evaluation
- **Batch Inference**: Provides CLI tool for scoring new patient lists with configurable thresholds

## 🛠️ Installation

Set up your environment with the following steps:

```bash
# Clone the repository
git clone https://github.com/username/patient-triage-prediction.git
cd patient-triage-prediction

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate

# Install dependencies (including dev tools: pytest, ruff)
pip install -e ".[dev]"
```

**Requirements**: Python 3.10+, Scikit-learn ≥1.3, Pandas ≥2.0, Numpy, Matplotlib

## 🚀 Usage

Follow these steps to run the complete pipeline:

### 1. Generate Synthetic Data

```bash
python -m src.data.make_dataset
```

Creates 5,000 synthetic patient records with:
- **Raw data**: `data/raw/patients_raw.csv` (unprocessed vital signs)
- **Processed data**: `data/processed/patients_processed.csv` (cleaned, ready for training)

**Note**: No external datasets required—all data is generated programmatically.

### 2. Train the Model

```bash
python -m src.models.train
```

Trains a balanced Logistic Regression classifier and saves:
- Model artifact: `reports/models/triage_logreg.joblib`
- Training metrics: Console output with accuracy, precision, recall

### 3. Evaluate Performance

```bash
python -m src.models.evaluate
```

Generates evaluation visualizations:
- ROC curve: `reports/figures/roc_curve.png`
- Precision-Recall curve: `reports/figures/pr_curve.png`
- Classification report with F1 scores

### 4. Make Predictions

```bash
python -m src.models.predict --input new_patients.csv --output predictions.csv --threshold 0.5
```

Applies trained model to new patient data:
- **Input**: CSV with vital sign columns
- **Output**: CSV with risk predictions and probabilities
- **Threshold**: Configurable decision boundary (default: 0.5)

## 🧠 Model Architecture

The system uses a scikit-learn pipeline with preprocessing and classification:

### Feature Set

| Feature | Description | Type | Example Values |
|---------|-------------|------|----------------|
| **age** | Patient age in years | Numeric | 18-90 |
| **gender** | Biological sex | Categorical | Male/Female |
| **systolic_bp** | Systolic blood pressure (mmHg) | Numeric | 90-180 |
| **heart_rate** | Pulse (beats per minute) | Numeric | 50-120 |
| **respiratory_rate** | Breaths per minute | Numeric | 12-30 |
| **temperature** | Body temperature (°C) | Numeric | 36.0-39.5 |
| **spo2** | Oxygen saturation (%) | Numeric | 85-100 |

### Target Variable

- **risk_level**: Binary classification
  - `1` → High-risk (requires immediate attention)
  - `0` → Low-risk (stable condition)

### Preprocessing Pipeline

```python
ColumnTransformer:
  - Numeric features → StandardScaler (z-score normalization)
  - Categorical features → OneHotEncoder (gender encoding)
```

### Model

- **Algorithm**: Logistic Regression
- **Class Weighting**: `class_weight="balanced"` (handles imbalanced triage scenarios)
- **Solver**: lbfgs (default, efficient for small datasets)

## 📊 Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── make_dataset.py       # Synthetic data generator
│   │   └── generate_synthetic.py # Statistical distribution logic
│   ├── features/
│   │   └── build_features.py     # Feature definitions & X/y split
│   └── models/
│       ├── train.py              # Model training pipeline
│       ├── evaluate.py           # ROC/PR curve generation
│       └── predict.py            # Batch inference CLI
├── tests/
│   └── test_smoke.py             # Basic smoke tests
├── reports/
│   ├── models/
│   │   └── triage_logreg.joblib  # Saved model artifact
│   └── figures/
│       ├── roc_curve.png         # ROC-AUC visualization
│       └── pr_curve.png          # Precision-Recall curve
├── data/
│   ├── raw/                      # Generated raw patient data
│   └── processed/                # Cleaned training data
└── README.md
```

## 📈 Sample Output

After training and evaluation, expect results like:

```
Training Metrics:
Accuracy:  0.88
Precision: 0.85
Recall:    0.91
F1-Score:  0.88

Classification Report:
              precision    recall  f1-score   support
           0       0.91      0.85      0.88       512
           1       0.85      0.91      0.88       488

    accuracy                           0.88      1000
   macro avg       0.88      0.88      0.88      1000
weighted avg       0.88      0.88      0.88      1000

ROC-AUC Score: 0.93
```

**Prediction Output** (`predictions.csv`):
```csv
patient_id,risk_probability,risk_prediction,risk_level
P001,0.82,1,high-risk
P002,0.23,0,low-risk
P003,0.67,1,high-risk
```

## 🔬 Technical Details

### Synthetic Data Generation

The data generator uses realistic statistical distributions:

- **Vital Signs**: Normal distributions with clinically plausible means/std
- **Risk Labels**: Generated based on weighted combinations of vital thresholds
- **Imbalance Handling**: Configurable class ratios (default ~50/50 for balanced training)

**Example Logic**:
```python
# High-risk indicators
high_risk = (systolic_bp > 160) | (heart_rate > 100) | 
            (spo2 < 90) | (temperature > 38.5)
```

### Model Training

- **Train/Test Split**: 80/20 stratified split (preserves class distribution)
- **Cross-Validation**: Can be added via `cross_val_score` (currently not implemented)
- **Hyperparameters**: Scikit-learn defaults (tune via GridSearchCV if needed)

### Evaluation Metrics

- **ROC-AUC**: Measures discriminative power across all thresholds
- **Precision-Recall**: Critical for imbalanced medical scenarios (minimizing false negatives)
- **Confusion Matrix**: Identifies specific error patterns (false positives vs. false negatives)

## 🎨 Visualization Examples

The evaluation script generates publication-quality plots:

- **ROC Curve**: Shows true positive rate vs. false positive rate
- **PR Curve**: Shows precision vs. recall trade-off
- Both include AUC scores and baseline reference lines

## 📝 TODO

Future enhancements and improvements:

- [ ] Add temporal features (time since symptom onset, vital trend analysis)
- [ ] Implement multi-class triage (emergency/urgent/semi-urgent/non-urgent)
- [ ] Experiment with ensemble models (Random Forest, XGBoost)
- [ ] Add SHAP/LIME for model explainability
- [ ] Build real-time API (FastAPI) for clinical integration
- [ ] Create interactive dashboard (Streamlit) for clinicians
- [ ] Add cross-validation and hyperparameter tuning
- [ ] Implement alert system for critical threshold violations
- [ ] Generate synthetic time-series data (vital sign monitoring)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or suggest features via Issues
- Submit pull requests for improvements
- Share insights on medical feature engineering
- Propose additional vital signs or clinical markers

## ⚠️ Medical Disclaimer

**IMPORTANT**: This is a **synthetic data simulation** for educational and research purposes only. 

- **NOT for clinical use**: Never use this system for actual patient triage or medical decision-making
- **No real patient data**: All data is artificially generated
- **Consult medical professionals**: Always rely on qualified healthcare providers for medical assessments

This project demonstrates ML techniques in healthcare contexts but does not constitute medical advice, diagnosis, or treatment.

## 📄 License

This project is open source and for educational purposes only. See LICENSE file for details.

---

**Disclaimer**: This is an experimental ML project for HealthTech research and education. The model is trained on synthetic data and has not been validated on real clinical data. Always consult qualified healthcare professionals for medical decisions. 🩺
