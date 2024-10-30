# Synthetic Triage Risk ML (Healthcare Demo)
A small AI-for-healthcare portfolio project that trains a baseline ML model to predict synthetic "higher-risk triage" labels.

## Quickstart
```bash
# 1) Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2) Pipeline
python -m src.data.make_dataset
python -m src.models.train
python -m src.models.evaluate
```