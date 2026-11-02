# Sri Lankan Phishing & Smishing URL Detection

MSc Applied Machine Learning project focused on detecting phishing and SMS smishing URLs targeting Sri Lankan banks (`BOC`, `Sampath Vishwa`, `ComBank`, `HNB`, `FriMi`), government portals (`gov.lk`, `slpost.gov.lk`, `ird.gov.lk`), utilities (`ceb.lk`, `leco.lk`), and telecommunications providers (`Dialog`, `Mobitel`).

## Overview
- **Feature Extraction:** Extracts 22 URL lexical, host structure, and Sri Lankan `.lk` / brand-mismatch features (`src/feature_extractor.py`).
- **Model Training & Comparison:** Trains and compares 6 non-linear classification models (`XGBoost`, `CatBoost`, `LightGBM`, `SVM with RBF Kernel`, `Random Forest`, and `GaussianNB`) using 5-fold stratified cross-validation (`src/train_and_evaluate.py`). Linear Regression, Logistic Regression, and Deep Learning models are not used.
- **Model Explainability:** Computes global and local explanations using `SHAP` (`TreeExplainer`) and `LIME` (`LimeTabularExplainer`) (`src/explainability.py`).
- **MLOps & Drift Monitoring:** Tracks model registry metadata, CI/CD quality gates, and statistical feature drift via Population Stability Index (PSI) and Two-Sample KS tests (`src/drift_monitor.py`).
- **Web Interface:** Streamlit dashboard (`app/streamlit_app.py`) for testing live URLs and viewing model comparisons, SHAP/LIME outputs, and drift monitoring.
  - **Live Demo:** [https://cybersecurity-threat-detection-pro.streamlit.app/](https://cybersecurity-threat-detection-pro.streamlit.app/)

## Setup & Usage

1. Install required packages:
```bash
pip install -r requirements.txt
```
*(Note for macOS users: `XGBoost` and `LightGBM` require OpenMP, which can be installed via `brew install libomp`.)*

2. Launch the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```

3. Re-run the training, explainability, and MLOps scripts (optional):
```bash
python -m src.data_generator
python -m src.train_and_evaluate
python -m src.explainability
python tests/test_mlops_pipeline.py
```
