# Sri Lankan Phishing & Smishing URL Detection

MSc Applied Machine Learning project focused on detecting phishing and SMS smishing URLs targeting Sri Lankan banks (`BOC`, `Sampath Vishwa`, `ComBank`, `HNB`, `FriMi`), government portals (`gov.lk`, `slpost.gov.lk`, `ird.gov.lk`), utilities (`ceb.lk`, `leco.lk`), and telecommunications providers (`Dialog`, `Mobitel`).

## Overview
- **Feature Extraction:** Extracts 22 URL lexical, host structure, and Sri Lankan `.lk` / brand-mismatch features (`src/feature_extractor.py`).
- **Model Training & Comparison:** Trains and compares 6 non-linear classification models (`XGBoost`, `SVM with RBF Kernel`, `LightGBM`, `Random Forest`, `MLPClassifier`, and `GaussianNB`) using 5-fold stratified cross-validation (`src/train_and_evaluate.py`). Linear and Logistic Regression are not used.
- **Model Explainability:** Computes global and local explanations using `SHAP` (`TreeExplainer`) and `LIME` (`LimeTabularExplainer`) (`src/explainability.py`).
- **Web Interface:** Streamlit dashboard (`app/streamlit_app.py`) for testing live URLs and viewing model comparisons and SHAP/LIME outputs.

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

3. Re-run the training and evaluation scripts (optional):
```bash
python -m src.data_generator
python -m src.train_and_evaluate
python -m src.explainability
```
