"""
Data drift and model monitoring utilities.
Computes Population Stability Index (PSI) and Two-Sample Kolmogorov-Smirnov (KS)
statistics to detect covariate shift in incoming URL traffic, and manages model registry metadata.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.feature_extractor import FEATURE_NAMES


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a dataset or model artifact for lineage tracking."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between reference (training) and current distributions.
    PSI < 0.10: No significant drift
    0.10 <= PSI < 0.20: Moderate drift (monitor closely)
    PSI >= 0.20: Significant drift (retraining recommended)
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    unique_vals = np.unique(expected)
    if len(unique_vals) <= 5:
        # Discrete / binary feature
        categories = np.union1d(unique_vals, np.unique(actual))
        exp_props = np.array([np.mean(expected == c) for c in categories])
        act_props = np.array([np.mean(actual == c) for c in categories])
    else:
        quantiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.unique(np.percentile(expected, quantiles))
        if len(bin_edges) < 2:
            return 0.0
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        exp_counts, _ = np.histogram(expected, bins=bin_edges)
        act_counts, _ = np.histogram(actual, bins=bin_edges)
        exp_props = exp_counts / max(len(expected), 1)
        act_props = act_counts / max(len(actual), 1)

    eps = 1e-4
    exp_props = np.clip(exp_props, eps, 1.0)
    act_props = np.clip(act_props, eps, 1.0)
    psi_val = np.sum((act_props - exp_props) * np.log(act_props / exp_props))
    return float(round(psi_val, 4))


def evaluate_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    psi_threshold: float = 0.20,
    ks_alpha: float = 0.01,
) -> pd.DataFrame:
    """
    Run PSI and Two-Sample Kolmogorov-Smirnov (KS) drift tests across all 22 URL features.
    """
    rows = []
    for feat in FEATURE_NAMES:
        ref_vals = reference_df[feat].dropna().values
        cur_vals = current_df[feat].dropna().values

        psi_score = calculate_psi(ref_vals, cur_vals)
        ks_stat, ks_pval = ks_2samp(ref_vals, cur_vals)

        if psi_score >= psi_threshold or (ks_pval < ks_alpha and ks_stat > 0.12):
            status = "Drift Detected"
        elif psi_score >= 0.10 or ks_stat > 0.08:
            status = "Moderate Shift"
        else:
            status = "Stable"

        rows.append({
            "Feature": feat,
            "Ref_Mean": round(float(np.mean(ref_vals)), 4),
            "Current_Mean": round(float(np.mean(cur_vals)), 4),
            "PSI": psi_score,
            "KS_Statistic": round(float(ks_stat), 4),
            "KS_p_value": round(float(ks_pval), 4),
            "Status": status,
        })

    return pd.DataFrame(rows).sort_values(by="PSI", ascending=False).reset_index(drop=True)


def simulate_shifted_traffic(reference_df: pd.DataFrame, scenario: str = "normal", n_samples: int = 500) -> pd.DataFrame:
    """
    Simulate incoming URL production batches under normal conditions vs. adversarial campaign shift.
    """
    rng = np.random.default_rng(42)
    sample_df = reference_df.sample(n=min(n_samples, len(reference_df)), random_state=42).copy()

    if scenario == "normal":
        # Minor natural sampling variation
        sample_df["url_length"] = np.clip(sample_df["url_length"] + rng.normal(0, 1.5, len(sample_df)), 18, 220)
        sample_df["url_entropy"] = np.clip(sample_df["url_entropy"] + rng.normal(0, 0.04, len(sample_df)), 2.5, 5.8)
    elif scenario == "new_attack_wave":
        # Simulate a new smishing wave using longer obfuscated URLs, new high-risk TLDs, and more subdomains
        sample_df["url_length"] = np.clip(sample_df["url_length"] * 1.35 + rng.normal(18, 8, len(sample_df)), 25, 250)
        sample_df["url_entropy"] = np.clip(sample_df["url_entropy"] + 0.48 + rng.normal(0, 0.12, len(sample_df)), 3.0, 6.0)
        sample_df["num_hyphens_in_domain"] = sample_df["num_hyphens_in_domain"] + rng.choice([1.0, 2.0, 3.0], size=len(sample_df), p=[0.4, 0.4, 0.2])
        sample_df["high_risk_tld"] = rng.choice([0.0, 1.0], size=len(sample_df), p=[0.2, 0.8])
        sample_df["urgency_keyword_count"] = sample_df["urgency_keyword_count"] + rng.choice([1.0, 2.0], size=len(sample_df))

    return sample_df


def build_model_registry(
    metrics_summary_path: str = "models/metrics_summary.json",
    dataset_path: str = "data/raw/srilanka_phishing_dataset.csv",
    best_model_path: str = "models/best_model.joblib",
    output_path: str = "models/model_registry.json",
) -> dict:
    """Create or update the MLOps model registry file with version lineage and quality gate checks."""
    with open(metrics_summary_path, "r") as f:
        summary = json.load(f)

    best_metrics = summary["metrics"][0]
    dataset_hash = compute_file_sha256(dataset_path) if os.path.exists(dataset_path) else "unknown"
    model_hash = compute_file_sha256(best_model_path) if os.path.exists(best_model_path) else "unknown"

    gates = {
        "min_f1_score": {"threshold": 0.95, "actual": best_metrics["Test_F1"], "passed": bool(best_metrics["Test_F1"] >= 0.95)},
        "min_recall": {"threshold": 0.95, "actual": best_metrics["Test_Recall"], "passed": bool(best_metrics["Test_Recall"] >= 0.95)},
        "min_roc_auc": {"threshold": 0.97, "actual": best_metrics["Test_ROC_AUC"], "passed": bool(best_metrics["Test_ROC_AUC"] >= 0.97)},
        "max_latency_ms": {"threshold": 5.0, "actual": best_metrics["Inference_Latency_ms"], "passed": bool(best_metrics["Inference_Latency_ms"] <= 5.0)},
    }
    all_passed = all(g["passed"] for g in gates.values())

    registry = {
        "model_name": "sl-phishing-url-detector",
        "production_version": "v1.2.0",
        "selected_algorithm": summary["best_model"],
        "registered_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": "Production" if all_passed else "Staging",
        "data_lineage": {
            "dataset_path": dataset_path,
            "dataset_sha256": dataset_hash[:16],
            "total_records": 6000,
            "num_features": len(FEATURE_NAMES),
            "split_ratio": "70% Train / 15% Val / 15% Test",
        },
        "artifact_lineage": {
            "model_path": best_model_path,
            "model_sha256": model_hash[:16],
        },
        "quality_gates": gates,
        "all_quality_gates_passed": all_passed,
        "retraining_policy": {
            "psi_alert_threshold": 0.20,
            "ks_p_value_alpha": 0.01,
            "max_drifted_features_before_retrain": 3,
        },
    }

    with open(output_path, "w") as f:
        json.dump(registry, f, indent=2)
    return registry


if __name__ == "__main__":
    reg = build_model_registry()
    print(f"[+] Model Registry updated: {reg['model_name']} ({reg['production_version']}) -> Stage: {reg['stage']}")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    drift_df = evaluate_feature_drift(train_df, test_df)
    print("\nBaseline Test vs Train Drift Check (Top 5 features by PSI):")
    print(drift_df.head(5).to_string(index=False))
