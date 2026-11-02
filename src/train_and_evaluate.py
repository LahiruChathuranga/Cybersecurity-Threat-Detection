"""
Model training, cross-validation, and evaluation script.
Compares 6 non-linear classifiers (Random Forest, XGBoost, LightGBM, SVM-RBF, MLP, and Gaussian Naive Bayes)
without using Linear or Logistic Regression.
"""

import json
import os
import time
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.feature_extractor import FEATURE_NAMES, FEATURE_DESCRIPTIONS

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# Publication-quality matplotlib styling
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

MODEL_COLORS = {
    "XGBoost": "#1f77b4",
    "LightGBM": "#2ca02c",
    "Random Forest": "#ff7f0e",
    "SVM (RBF Kernel)": "#9467bd",
    "MLP Neural Net": "#d62728",
    "Gaussian Naive Bayes": "#8c564b",
}


def build_models() -> dict[str, Pipeline]:
    """
    Construct the 6 non-linear ML models wrapped in leak-free scikit-learn Pipelines.
    Note: Linear Regression and Logistic Regression are strictly excluded.
    """
    models = {
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=220,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.5,
                reg_alpha=0.2,
                random_state=SEED,
                eval_metric="logloss",
                n_jobs=-1
            ))
        ]),
        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=220,
                num_leaves=31,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.2,
                random_state=SEED,
                verbose=-1,
                n_jobs=-1
            ))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=-1
            ))
        ]),
        "SVM (RBF Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=3.0,
                gamma="scale",
                probability=True,
                random_state=SEED
            ))
        ]),
        "MLP Neural Net": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                alpha=0.005,
                learning_rate_init=0.001,
                max_iter=300,
                early_stopping=True,
                random_state=SEED
            ))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB(var_smoothing=1e-7))
        ]),
    }
    return models


def generate_eda_plots(df: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series, fig_dir: str):
    """Generate Exploratory Data Analysis (EDA) visualizations for the report and app."""
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Category & Class Breakdown Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cat_counts = df["category"].value_counts().sort_values(ascending=True)
    colors = ["#e74c3c" if ("Phishing" in c) else "#2ecc71" for c in cat_counts.index]
    axes[0].barh(cat_counts.index, cat_counts.values, color=colors, edgecolor="#2c3e50", alpha=0.9)
    axes[0].set_title("A. Dataset Breakdown by Threat & Benign Category", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of URLs")
    for i, v in enumerate(cat_counts.values):
        axes[0].text(v + 15, i, str(v), va="center", fontsize=8.5, fontweight="bold")
    axes[0].set_xlim(0, max(cat_counts.values) * 1.15)

    # Mutual Information Feature Ranking
    mi_scores = mutual_info_classif(X_train, y_train, random_state=SEED)
    mi_series = pd.Series(mi_scores, index=FEATURE_NAMES).sort_values(ascending=True)
    axes[1].barh(mi_series.index, mi_series.values, color="#3498db", edgecolor="#1f618d", alpha=0.9)
    axes[1].set_title("B. Mutual Information (Information Gain) with Phishing Label", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mutual Information Score (nats)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_eda_categories_and_mutual_info.png"), bbox_inches="tight")
    plt.close()

    # 2. Feature Correlation Matrix Heatmap
    plt.figure(figsize=(14, 11))
    corr = df[FEATURE_NAMES + ["label"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-0.8,
        vmax=0.8,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}
    )
    plt.title("Feature Correlation Matrix Across 22 Cybersecurity & SL Context Features", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "02_feature_correlation_heatmap.png"), bbox_inches="tight")
    plt.close()

    # 3. Key Feature Distributions (Violin / Box comparison)
    top_cont_features = ["url_length", "url_entropy", "hostname_length", "min_brand_edit_distance", "special_char_count", "urgency_keyword_count"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()
    plot_df = df.copy()
    plot_df["Class"] = plot_df["label"].map({0: "Legitimate (0)", 1: "Phishing (1)"})
    for idx, feat in enumerate(top_cont_features):
        sns.violinplot(
            data=plot_df,
            x="Class",
            y=feat,
            palette={"Legitimate (0)": "#2ecc71", "Phishing (1)": "#e74c3c"},
            inner="quartile",
            ax=axes[idx],
            cut=0
        )
        axes[idx].set_title(f"Distribution of {feat}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("")
    plt.suptitle("Distributional Shift Between Legitimate and Phishing URLs in Sri Lanka", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "03_feature_distributions_violin.png"), bbox_inches="tight")
    plt.close()


def evaluate_and_compare():
    """Train all 6 models, run 5-fold Stratified CV, evaluate on test set, and generate comparison charts."""
    data_path = "data/raw/srilanka_phishing_dataset.csv"
    df = pd.read_csv(data_path)

    X = df[FEATURE_NAMES]
    y = df["label"]

    # Stratified 70% Train, 15% Validation, 15% Test Split
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        X, y, df.index, test_size=0.15, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val, y_train_val, idx_train_val, test_size=0.1765, stratify=y_train_val, random_state=SEED
    )

    # Save processed splits for explainability and app
    os.makedirs("data/processed", exist_ok=True)
    df.loc[idx_train].to_csv("data/processed/train.csv", index=False)
    df.loc[idx_val].to_csv("data/processed/val.csv", index=False)
    df.loc[idx_test].to_csv("data/processed/test.csv", index=False)
    print(f"[+] Splits created -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    fig_dir = "reports/figures"
    generate_eda_plots(df, X_train, y_train, fig_dir)
    print("[+] Generated EDA plots in reports/figures/")

    models = build_models()
    os.makedirs("models/all_models", exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    roc_data = {}
    pr_data = {}
    calib_data = {}
    conf_matrices = {}

    for name, pipeline in models.items():
        print(f"[*] Training & Cross-Validating {name}...")
        # 5-Fold Stratified Cross-Validation on Train+Val
        cv_scores = cross_validate(
            pipeline,
            X_train_val,
            y_train_val,
            cv=cv,
            scoring=["accuracy", "f1", "roc_auc", "precision", "recall"],
            n_jobs=-1
        )

        # Fit on training set and time execution
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t0

        # Measure single-instance inference latency (ms) averaged over 200 single calls
        sample_row = X_test.iloc[[0]]
        t_inf_start = time.perf_counter()
        for _ in range(200):
            _ = pipeline.predict_proba(sample_row)
        inf_latency_ms = ((time.perf_counter() - t_inf_start) / 200.0) * 1000.0

        # Predict on Held-Out Test Set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        conf_matrices[name] = cm.tolist()

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        spec = tn / (tn + fp)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)
        fpr_rate = fp / (fp + tn)
        fnr_rate = fn / (fn + tp)

        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr_arr, tpr_arr, roc_auc)

        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        pr_data[name] = (rec_arr, prec_arr, pr_auc)

        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        calib_data[name] = (prob_pred, prob_true, brier)

        safe_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        joblib.dump(pipeline, os.path.join("models/all_models", safe_filename))

        results.append({
            "Model": name,
            "Test_Accuracy": round(acc, 4),
            "Test_Precision": round(prec, 4),
            "Test_Recall": round(rec, 4),
            "Test_Specificity": round(spec, 4),
            "Test_F1": round(f1, 4),
            "Test_ROC_AUC": round(roc_auc, 4),
            "Test_PR_AUC": round(pr_auc, 4),
            "Test_MCC": round(mcc, 4),
            "Test_Kappa": round(kappa, 4),
            "Brier_Score": round(brier, 4),
            "False_Positive_Rate": round(fpr_rate, 4),
            "False_Negative_Rate": round(fnr_rate, 4),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "CV_F1_Mean": round(float(np.mean(cv_scores["test_f1"])), 4),
            "CV_F1_Std": round(float(np.std(cv_scores["test_f1"])), 4),
            "CV_ROC_AUC_Mean": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "CV_ROC_AUC_Std": round(float(np.std(cv_scores["test_roc_auc"])), 4),
            "Train_Time_s": round(train_time_s, 4),
            "Inference_Latency_ms": round(inf_latency_ms, 4),
            "Artifact_File": safe_filename
        })

    results_df = pd.DataFrame(results).sort_values(by="Test_F1", ascending=False).reset_index(drop=True)
    results_df.to_csv("models/model_comparison_metrics.csv", index=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_artifact = results_df.iloc[0]["Artifact_File"]
    best_pipeline = joblib.load(os.path.join("models/all_models", best_artifact))
    joblib.dump(best_pipeline, "models/best_model.joblib")

    summary_payload = {
        "best_model": best_model_name,
        "feature_names": FEATURE_NAMES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "metrics": results_df.to_dict(orient="records"),
        "confusion_matrices": conf_matrices,
    }
    with open("models/metrics_summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n=== 6-MODEL COMPARISON RESULTS (SORTED BY TEST F1) ===")
    print(results_df[["Model", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC", "Test_MCC", "CV_F1_Mean", "Inference_Latency_ms"]].to_string(index=False))
    print(f"\n[+] Best Champion Model: {best_model_name} saved to models/best_model.joblib")

    # --- Generate Model Comparison Visualizations ---
    # 4. Overlaid ROC & Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name in results_df["Model"]:
        fpr_arr, tpr_arr, auc_val = roc_data[name]
        axes[0].plot(fpr_arr, tpr_arr, label=f"{name} (AUC = {auc_val:.4f})", color=MODEL_COLORS[name], lw=2.2)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random Chance (0.5000)")
    axes[0].set_xlim([-0.01, 1.0])
    axes[0].set_ylim([0.0, 1.02])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    axes[0].set_title("A. Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    axes[0].legend(loc="lower right", frameon=True, fontsize=9.5)
    axes[0].grid(alpha=0.25)

    for name in results_df["Model"]:
        rec_arr, prec_arr, pr_val = pr_data[name]
        axes[1].plot(rec_arr, prec_arr, label=f"{name} (AP = {pr_val:.4f})", color=MODEL_COLORS[name], lw=2.2)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.5, 1.02])
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].set_title("B. Precision-Recall (PR) Curves", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower left", frameon=True, fontsize=9.5)
    axes[1].grid(alpha=0.25)

    plt.suptitle("Multi-Model Discriminative Comparison Across 6 Non-Linear Classifiers", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "04_roc_and_pr_curves.png"), bbox_inches="tight")
    plt.close()

    # 5. Confusion Matrices Grid (2x3) for all 6 models
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for idx, row in results_df.iterrows():
        m_name = row["Model"]
        cm = np.array(conf_matrices[m_name])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Pred Legitimate", "Pred Phishing"],
            yticklabels=["Actual Legitimate", "Actual Phishing"]
        )
        axes[idx].set_title(
            f"{m_name}\nF1: {row['Test_F1']:.4f} | FN: {row['FN']} | FP: {row['FP']}",
            fontsize=11,
            fontweight="bold"
        )
    plt.suptitle("Test Set Confusion Matrices (n = 900 Held-Out URLs) Across All 6 Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "05_confusion_matrices_6models.png"), bbox_inches="tight")
    plt.close()

    # 6. Comprehensive Metric Bar Chart & Pareto Trade-off (F1 vs Latency & Calibration)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    metrics_to_plot = ["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC", "Test_MCC"]
    x = np.arange(len(results_df))
    width = 0.13
    palette = ["#2b5c8f", "#388e3c", "#f57c00", "#d32f2f", "#7b1fa2", "#00796b"]
    for i, m_col in enumerate(metrics_to_plot):
        axes[0].bar(x + (i - 2.5) * width, results_df[m_col], width, label=m_col.replace("Test_", ""), color=palette[i], alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results_df["Model"], rotation=20, ha="right", fontsize=9.5, fontweight="bold")
    axes[0].set_ylim(0.65, 1.03)
    axes[0].set_ylabel("Score")
    axes[0].set_title("A. Multi-Metric Performance Comparison (Held-Out Test Set)", fontsize=12, fontweight="bold")
    axes[0].legend(ncol=3, fontsize=8.5, loc="lower right")
    axes[0].grid(axis="y", alpha=0.3)

    # Calibration Curves
    for name in results_df["Model"]:
        prob_pred, prob_true, brier_val = calib_data[name]
        axes[1].plot(prob_pred, prob_true, marker="o", label=f"{name} (Brier = {brier_val:.4f})", color=MODEL_COLORS[name], lw=1.8)
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    axes[1].set_xlabel("Mean Predicted Phishing Probability", fontsize=11)
    axes[1].set_ylabel("Fraction of Positives (Empirical Probability)", fontsize=11)
    axes[1].set_title("B. Probability Calibration Curves (Reliability Diagram)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=8.5)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "06_metrics_and_calibration.png"), bbox_inches="tight")
    plt.close()

    # 7. Speed vs Accuracy Pareto Frontier + 5-Fold Cross-Validation Stability Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Distinct non-overlapping callout offsets (dx, dy in points) for each model on log-x scale
    label_offsets = {
        "XGBoost": (28, 22),
        "SVM (RBF Kernel)": (-132, 16),
        "LightGBM": (38, -22),
        "Random Forest": (-128, -42),
        "MLP Neural Net": (-110, -48),
        "Gaussian Naive Bayes": (25, 6),
    }

    axes[0].set_xscale("log")
    for _, row in results_df.iterrows():
        m_name = row["Model"]
        x_val = row["Inference_Latency_ms"]
        y_val = row["Test_F1"]
        axes[0].scatter(
            x_val,
            y_val,
            s=190,
            color=MODEL_COLORS[m_name],
            edgecolors="black",
            linewidth=1.5,
            zorder=5
        )
        dx, dy = label_offsets.get(m_name, (15, 0))
        axes[0].annotate(
            f"{m_name}\n({x_val:.2f} ms, F1={y_val:.4f})",
            xy=(x_val, y_val),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.3,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=MODEL_COLORS[m_name], alpha=0.92, lw=1.1),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.12", color="#444444", lw=1.0),
            zorder=6,
        )
    axes[0].set_xlabel("Single-URL Inference Latency (ms, Log Scale) [Lower is Better]", fontsize=10.5)
    axes[0].set_ylabel("Test F1-Score [Higher is Better]", fontsize=10.5)
    axes[0].set_title("A. Operational Trade-Off: F1-Score vs. Inference Latency", fontsize=12, fontweight="bold")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].set_xlim(0.16, 28.0)
    axes[0].set_ylim(0.77, 1.005)

    # CV F1 Mean + Error Bars
    y_pos = np.arange(len(results_df))
    axes[1].barh(
        y_pos,
        results_df["CV_F1_Mean"],
        xerr=results_df["CV_F1_Std"],
        capsize=5,
        color=[MODEL_COLORS[m] for m in results_df["Model"]],
        edgecolor="#222",
        alpha=0.88
    )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(results_df["Model"], fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.75, 1.025)
    axes[1].set_xlabel("5-Fold Stratified Cross-Validation F1-Score (Mean ± Std)")
    axes[1].set_title("B. 5-Fold Cross-Validation Stability", fontsize=12, fontweight="bold")
    for i, row in results_df.iterrows():
        axes[1].text(row["CV_F1_Mean"] + row["CV_F1_Std"] + 0.004, i, f"{row['CV_F1_Mean']:.4f} ± {row['CV_F1_Std']:.4f}", va="center", fontsize=8.5, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "07_latency_pareto_and_cv_stability.png"), bbox_inches="tight")
    plt.close()
    print("[+] Saved all model evaluation & comparison charts in reports/figures/")


if __name__ == "__main__":
    evaluate_and_compare()
