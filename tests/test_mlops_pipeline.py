"""
Automated MLOps test suite and model quality gate verification.
Run automatically in CI/CD (GitHub Actions) on every commit or pull request.
"""

import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_extractor import extract_url_features, FEATURE_NAMES
from src.drift_monitor import evaluate_feature_drift, build_model_registry


def test_feature_extractor_schema():
    """Verify that URL feature extraction returns all 22 expected numerical features."""
    url_phish = "http://slpost-customs-clearance-lk.top/track/confirm.php?id=8821"
    feats = extract_url_features(url_phish)
    assert len(feats) == 22, f"Expected 22 features, got {len(feats)}"
    assert list(feats.keys()) == FEATURE_NAMES
    assert feats["high_risk_tld"] == 1.0
    assert feats["sl_brand_domain_mismatch"] == 1.0
    assert feats["has_suspicious_ext"] == 1.0

    url_legit = "https://www.boc.lk/personal-banking"
    feats_legit = extract_url_features(url_legit)
    assert feats_legit["is_official_lk_domain"] == 1.0
    assert feats_legit["sl_brand_domain_mismatch"] == 0.0
    assert feats_legit["uses_https"] == 1.0


def test_model_quality_gates():
    """Enforce MLOps production quality gates (F1 >= 0.95, Recall >= 0.95, ROC-AUC >= 0.97, Latency <= 5ms)."""
    registry = build_model_registry()
    assert registry["all_quality_gates_passed"] is True, f"Quality gates failed: {registry['quality_gates']}"
    assert len(json.load(open("models/metrics_summary.json"))["metrics"]) == 6, "Must compare all 6 models"


def test_baseline_drift_stability():
    """Verify that held-out test set has no severe covariate shift relative to training split."""
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    drift_df = evaluate_feature_drift(train_df, test_df)
    drifted_count = int((drift_df["Status"] == "Drift Detected").sum())
    assert drifted_count == 0, f"Unexpected drift between train and test splits: {drifted_count} features drifted"


if __name__ == "__main__":
    test_feature_extractor_schema()
    print("[PASS] Feature extractor schema and Sri Lankan domain rules verified.")
    test_model_quality_gates()
    print("[PASS] Model quality gates (F1 >= 0.95, Recall >= 0.95, ROC-AUC >= 0.97, Latency <= 5ms) verified.")
    test_baseline_drift_stability()
    print("[PASS] Baseline train/test distribution stability (PSI & KS-test) verified.")
