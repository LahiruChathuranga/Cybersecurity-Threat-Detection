# 🛡️ LankaShield-Phish: Sri Lankan Cybersecurity Threat Detection & XAI

An Explainable Machine Learning (XAI) system for detecting localized Sri Lankan phishing and SMS smishing URLs targeting local banking (`BOC`, `Sampath Vishwa`, `ComBank`, `HNB`, `FriMi`), e-government (`gov.lk`, `slpost.gov.lk`, `ird.gov.lk`), utilities (`ceb.lk`, `leco.lk`), and telecommunications (`Dialog`, `SLT-Mobitel`).

## Key Features
- **22 Real-Time Cybersecurity & Sri Lankan Domain Features:** Lexical, host/structural, and localized `.lk` / brand-mismatch threat intelligence indicators.
- **6 Non-Linear ML Models Compared:** `XGBoost`, `SVM (RBF Kernel)`, `LightGBM`, `Random Forest`, `MLP Neural Network`, and `Gaussian Naive Bayes` (strictly excluding Linear and Logistic Regression).
- **Explainable AI (SHAP & LIME):** Global and local `TreeSHAP` explanations alongside local `LIME` surrogate decision rules.
- **Interactive Streamlit SOC Dashboard:** Live URL scanner, 6-Model Arena comparator, XAI explorer, and dataset EDA.

## Quick Start

### 1. Install Dependencies
*(On macOS Apple Silicon, ensure OpenMP is installed for XGBoost/LightGBM via `brew install libomp`)*
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```

### 3. Re-Run the ML & XAI Pipeline
```bash
python -m src.data_generator
python -m src.train_and_evaluate
python -m src.explainability
```
