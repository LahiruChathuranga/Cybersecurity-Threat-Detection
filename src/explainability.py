"""
SHAP and LIME explainability analysis script.
Computes global feature importance and local instance explanations on the test set.
"""

import json
import os
import time
import warnings
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src.feature_extractor import FEATURE_NAMES, FEATURE_DESCRIPTIONS

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def run_explainability_suite():
    """Run SHAP and LIME analyses on the champion model and save visual & JSON artifacts."""
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["label"]
    X_test = test_df[FEATURE_NAMES]
    y_test = test_df["label"]

    with open("models/metrics_summary.json", "r") as f:
        summary = json.load(f)

    best_model_name = summary["best_model"]
    best_pipeline = joblib.load("models/best_model.joblib")
    scaler = best_pipeline.named_steps["scaler"]
    clf = best_pipeline.named_steps["clf"]

    print(f"[*] Running SHAP & LIME Explainability Suite on Champion Model: {best_model_name}")

    # Transform features via the pipeline's StandardScaler for the inner tree model,
    # while keeping human-readable original feature values for display in SHAP & LIME
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=FEATURE_NAMES, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_NAMES, index=X_test.index)

    fig_dir = "reports/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # --- 1. Compute SHAP Values (Supports both Tree Models and Kernel/Neural Models) ---
    t0_shap = time.perf_counter()
    if hasattr(clf, "estimators_") or "XGB" in best_model_name or "LightGBM" in best_model_name or "Forest" in best_model_name:
        explainer = shap.TreeExplainer(clf)
        shap_explanation = explainer(X_test_scaled)
        shap_time_per_sample_ms = ((time.perf_counter() - t0_shap) / len(X_test_scaled)) * 1000.0
    else:
        bg_summary = shap.kmeans(X_train_scaled, 30)
        explainer = shap.KernelExplainer(lambda x: clf.predict_proba(x)[:, 1], bg_summary)
        shap_vals_raw = explainer.shap_values(X_test_scaled.iloc[:200], nsamples=150)
        shap_time_per_sample_ms = ((time.perf_counter() - t0_shap) / 200.0) * 1000.0
        shap_explanation = shap.Explanation(
            values=shap_vals_raw,
            base_values=np.repeat(explainer.expected_value, len(shap_vals_raw)),
            data=X_test.iloc[:200].values,
            feature_names=FEATURE_NAMES
        )

    # Handle multi-output vs single-output SHAP values array
    shap_vals = shap_explanation.values
    base_vals = shap_explanation.base_values
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
        if np.ndim(base_vals) == 2:
            base_vals = base_vals[:, 1]

    # Build a clean shap.Explanation object that uses the unscaled (raw human-readable) feature values for display!
    clean_shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=base_vals,
        data=X_test.values,
        feature_names=FEATURE_NAMES
    )

    # Save Global Mean |SHAP| Importance Ranking
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_ranking_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Description": [FEATURE_DESCRIPTIONS[f] for f in FEATURE_NAMES],
        "Mean_Abs_SHAP": np.round(mean_abs_shap, 4)
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    shap_ranking_df.to_csv("models/shap_feature_importance.csv", index=False)

    # --- Plot 8: SHAP Summary Beeswarm Plot ---
    plt.figure(figsize=(11, 8))
    shap.plots.beeswarm(clean_shap_exp, max_display=16, show=False)
    plt.title(f"SHAP Global Summary (Beeswarm) — {best_model_name} on Held-Out Test Set", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "08_shap_summary_beeswarm.png"), bbox_inches="tight")
    plt.close()

    # --- Plot 9: SHAP Global Bar Importance Plot ---
    plt.figure(figsize=(10, 7.5))
    shap.plots.bar(clean_shap_exp, max_display=16, show=False)
    plt.title(f"Global Feature Importance (Mean |SHAP Value|) — {best_model_name}", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "09_shap_global_bar.png"), bbox_inches="tight")
    plt.close()

    # --- Plot 10: SHAP Non-Linear Dependence & Interaction Plots (Top 4 Continuous/Key Features) ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    dep_pairs = [
        ("url_length", "sl_brand_domain_mismatch"),
        ("url_entropy", "high_risk_tld"),
        ("min_brand_edit_distance", "is_official_lk_domain"),
        ("urgency_keyword_count", "num_hyphens_in_domain"),
    ]
    for idx, (feat_main, feat_inter) in enumerate(dep_pairs):
        shap.dependence_plot(
            feat_main,
            shap_vals,
            X_test,
            interaction_index=feat_inter,
            ax=axes[idx],
            show=False,
            alpha=0.75
        )
        axes[idx].set_title(f"SHAP Dependence: {feat_main} × {feat_inter}", fontsize=11, fontweight="bold")
        axes[idx].grid(alpha=0.25)
    plt.suptitle("SHAP Non-Linear Feature Dependence & Interaction Effects", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "10_shap_dependence_interactions.png"), bbox_inches="tight")
    plt.close()

    # --- Select 3 Representative Local Case Studies from Held-Out Test Set ---
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = best_pipeline.predict(X_test)
    test_with_preds = test_df.copy()
    test_with_preds["pred_prob"] = y_prob
    test_with_preds["pred_label"] = y_pred
    test_with_preds["pos_idx"] = np.arange(len(test_df))

    # Case 1: High-Confidence True Positive SL Banking/Postal Phishing
    tp_candidates = test_with_preds[
        (test_with_preds["label"] == 1)
        & (test_with_preds["pred_label"] == 1)
        & (test_with_preds["category"].str.contains("SL_Phishing"))
        & (test_with_preds["sl_brand_domain_mismatch"] == 1)
    ].sort_values(by="pred_prob", ascending=False)
    case1_pos = int(tp_candidates.iloc[0]["pos_idx"])

    # Case 2: High-Confidence True Negative Authentic Sri Lankan Official .lk Portal
    tn_candidates = test_with_preds[
        (test_with_preds["label"] == 0)
        & (test_with_preds["pred_label"] == 0)
        & (test_with_preds["is_official_lk_domain"] == 1)
    ].sort_values(by="pred_prob", ascending=True)
    case2_pos = int(tn_candidates.iloc[0]["pos_idx"])

    # Case 3: Borderline / Stealth Phishing or Hard Benign Edge Case
    stealth_candidates = test_with_preds[
        (test_with_preds["label"] == 1)
        & (test_with_preds["category"] == "Stealth_or_Global_Phishing")
        & (test_with_preds["pred_prob"].between(0.55, 0.92))
    ]
    if len(stealth_candidates) > 0:
        case3_pos = int(stealth_candidates.iloc[0]["pos_idx"])
    else:
        case3_pos = int(test_with_preds.iloc[2]["pos_idx"])

    cases = [
        ("case1_sl_phishing_tp", case1_pos, "Case Study 1: Localized SL Phishing Attack (True Positive)"),
        ("case2_sl_legit_tn", case2_pos, "Case Study 2: Authentic Sri Lankan .lk Portal (True Negative)"),
        ("case3_stealth_edge", case3_pos, "Case Study 3: Stealth Phishing URL Analysis"),
    ]

    # --- Initialize LIME Tabular Explainer on raw (unscaled) features ---
    # LIME perturbs raw features in interpretable units and calls best_pipeline.predict_proba
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=FEATURE_NAMES,
        class_names=["Legitimate (0)", "Phishing (1)"],
        mode="classification",
        discretize_continuous=True,
        random_state=SEED
    )

    def predict_fn_raw(arr: np.ndarray) -> np.ndarray:
        df_in = pd.DataFrame(arr, columns=FEATURE_NAMES)
        return best_pipeline.predict_proba(df_in)

    case_details = []
    for slug, pos_idx, title in cases:
        row_info = test_with_preds.iloc[pos_idx]
        # 11. Individual SHAP Waterfall Plot
        plt.figure(figsize=(10, 6.5))
        shap.plots.waterfall(clean_shap_exp[pos_idx], max_display=12, show=False)
        plt.title(
            f"{title}\nURL: {row_info['url'][:78]}...\nPred Prob: {row_info['pred_prob']:.4f} | Actual: {'Phishing (1)' if row_info['label']==1 else 'Legitimate (0)'}",
            fontsize=10.5,
            fontweight="bold",
            pad=12
        )
        plt.tight_layout()
        shap_fig_name = f"11_shap_waterfall_{slug}.png"
        plt.savefig(os.path.join(fig_dir, shap_fig_name), bbox_inches="tight")
        plt.close()

        # 12. LIME Local Explanation for the exact same instance
        t0_lime = time.perf_counter()
        lime_exp = lime_explainer.explain_instance(
            data_row=X_test.iloc[pos_idx].values,
            predict_fn=predict_fn_raw,
            num_features=10,
            num_samples=1500
        )
        lime_latency_ms = (time.perf_counter() - t0_lime) * 1000.0

        lime_list = lime_exp.as_list()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        rules = [r[0] for r in lime_list][::-1]
        weights = [r[1] for r in lime_list][::-1]
        bar_colors = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]
        ax.barh(rules, weights, color=bar_colors, edgecolor="#2c3e50", alpha=0.9)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("LIME Local Linear Surrogate Weight (+ Phishing / - Legitimate)", fontsize=10.5)
        ax.set_title(
            f"LIME Local Explanation — {title}\nURL: {row_info['url'][:78]}... (Prob={row_info['pred_prob']:.4f})",
            fontsize=10.5,
            fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        lime_fig_name = f"12_lime_local_{slug}.png"
        plt.savefig(os.path.join(fig_dir, lime_fig_name), bbox_inches="tight")
        plt.close()

        # Compare Top-5 Feature Agreement between SHAP and LIME on this instance
        instance_shap_abs = pd.Series(np.abs(shap_vals[pos_idx]), index=FEATURE_NAMES).sort_values(ascending=False)
        top5_shap = list(instance_shap_abs.index[:5])

        # Map LIME feature indices to feature names
        lime_map = dict(lime_exp.local_exp[1])
        lime_abs = pd.Series({FEATURE_NAMES[k]: abs(v) for k, v in lime_map.items()}).sort_values(ascending=False)
        top5_lime = list(lime_abs.index[:5])
        overlap_top5 = len(set(top5_shap).intersection(set(top5_lime)))

        case_details.append({
            "slug": slug,
            "title": title,
            "url": row_info["url"],
            "category": row_info["category"],
            "actual_label": int(row_info["label"]),
            "pred_prob": round(float(row_info["pred_prob"]), 4),
            "top5_shap_features": top5_shap,
            "top5_lime_features": top5_lime,
            "top5_overlap_count": overlap_top5,
            "lime_rules": [{"rule": r, "weight": round(float(w), 4)} for r, w in lime_list],
            "shap_figure": shap_fig_name,
            "lime_figure": lime_fig_name,
            "lime_latency_ms": round(lime_latency_ms, 2)
        })

    # --- Quantitative Evaluation of SHAP vs. LIME Across 50 Test Instances ---
    overlaps = []
    spearman_corrs = []
    lime_times = []
    sample_indices = np.random.choice(len(X_test), size=40, replace=False)
    for idx in sample_indices:
        t_l = time.perf_counter()
        l_exp = lime_explainer.explain_instance(
            data_row=X_test.iloc[idx].values,
            predict_fn=predict_fn_raw,
            num_features=len(FEATURE_NAMES),
            num_samples=800
        )
        lime_times.append((time.perf_counter() - t_l) * 1000.0)
        l_map = dict(l_exp.local_exp[1])
        l_vec = np.array([abs(l_map.get(i, 0.0)) for i in range(len(FEATURE_NAMES))])
        s_vec = np.abs(shap_vals[idx])

        top5_s = set(np.argsort(s_vec)[-5:])
        top5_l = set(np.argsort(l_vec)[-5:])
        overlaps.append(len(top5_s.intersection(top5_l)) / 5.0)
        rho, _ = spearmanr(s_vec, l_vec)
        if not np.isnan(rho):
            spearman_corrs.append(rho)

    xai_summary = {
        "champion_model": best_model_name,
        "shap_latency_ms_per_sample": round(shap_time_per_sample_ms, 3),
        "lime_latency_ms_per_sample": round(float(np.mean(lime_times)), 2),
        "mean_top5_feature_agreement": round(float(np.mean(overlaps)), 4),
        "mean_spearman_rank_correlation": round(float(np.mean(spearman_corrs)), 4),
        "global_top10_features": shap_ranking_df.head(10).to_dict(orient="records"),
        "case_studies": case_details,
    }
    with open("models/xai_summary.json", "w") as f:
        json.dump(xai_summary, f, indent=2)

    print("[+] SHAP & LIME Suite complete!")
    print(f"    - Mean SHAP TreeExplainer Latency: {xai_summary['shap_latency_ms_per_sample']:.3f} ms/URL")
    print(f"    - Mean LIME Perturbation Latency:  {xai_summary['lime_latency_ms_per_sample']:.2f} ms/URL")
    print(f"    - Mean Top-5 SHAP vs LIME Overlap: {xai_summary['mean_top5_feature_agreement']*100:.1f}%")
    print(f"    - Mean Spearman Rank Correlation:  {xai_summary['mean_spearman_rank_correlation']:.4f}")


if __name__ == "__main__":
    run_explainability_suite()
