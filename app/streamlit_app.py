"""
LankaShield-Phish: Interactive Deployed Cybersecurity Threat Intelligence & XAI Web Application.

Features:
1. Live Sri Lankan URL Threat Scanner with real-time 22-feature extraction, multi-model consensus,
   and instant on-the-fly SHAP Waterfall + LIME Local Surrogate explanations.
2. 6-Model Arena Comparator (XGBoost, SVM-RBF, LightGBM, Random Forest, MLP Neural Net, Gaussian NB)
   strictly excluding Linear & Logistic Regression.
3. Global & Case-Study Explainable AI (XAI) Explorer (SHAP Beeswarm, Dependence, and SHAP vs LIME comparison).
4. Sri Lankan Threat Intelligence & Dataset EDA Explorer.
"""

import json
import os
import sys
import time
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Ensure root workspace is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_extractor import extract_url_features, FEATURE_NAMES, FEATURE_DESCRIPTIONS

st.set_page_config(
    page_title="LankaShield-Phish | SL Cybersecurity ML & XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean cards and metrics
st.markdown(
    """
    <style>
    .risk-high {
        background: linear-gradient(135deg, #4a0e17 0%, #801323 100%);
        color: #ffffff;
        padding: 18px 22px;
        border-radius: 12px;
        border-left: 6px solid #ff4b4b;
        margin-bottom: 16px;
    }
    .risk-low {
        background: linear-gradient(135deg, #0d3b26 0%, #176943 100%);
        color: #ffffff;
        padding: 18px 22px;
        border-radius: 12px;
        border-left: 6px solid #2ecc71;
        margin-bottom: 16px;
    }
    .risk-medium {
        background: linear-gradient(135deg, #4d3309 0%, #87580d 100%);
        color: #ffffff;
        padding: 18px 22px;
        border-radius: 12px;
        border-left: 6px solid #f39c12;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Load all 6 trained models, dataset splits, and SHAP/LIME explainers."""
    with open("models/metrics_summary.json", "r") as f:
        metrics_summary = json.load(f)
    with open("models/xai_summary.json", "r") as f:
        xai_summary = json.load(f)

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    full_df = pd.read_csv("data/raw/srilanka_phishing_dataset.csv")

    all_models = {}
    for m_info in metrics_summary["metrics"]:
        m_name = m_info["Model"]
        art_path = os.path.join("models/all_models", m_info["Artifact_File"])
        all_models[m_name] = joblib.load(art_path)

    best_model = joblib.load("models/best_model.joblib")
    scaler = best_model.named_steps["scaler"]
    clf = best_model.named_steps["clf"]
    shap_explainer = shap.TreeExplainer(clf)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_df[FEATURE_NAMES].values,
        feature_names=FEATURE_NAMES,
        class_names=["Legitimate (0)", "Phishing (1)"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    return metrics_summary, xai_summary, all_models, best_model, scaler, shap_explainer, lime_explainer, train_df, test_df, full_df


metrics_summary, xai_summary, all_models, best_model, scaler, shap_explainer, lime_explainer, train_df, test_df, full_df = load_artifacts()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🛡️ LankaShield-Phish")
    st.caption("**Sri Lankan Banking, E-Gov & Telco Phishing Detection with Explainable AI (SHAP & LIME)**")
    st.divider()
    st.markdown("### 🏆 Champion Model")
    st.success(f"**{metrics_summary['best_model']}**  \nTest F1: **{metrics_summary['metrics'][0]['Test_F1']:.4f}**  \nROC-AUC: **{metrics_summary['metrics'][0]['Test_ROC_AUC']:.4f}**")
    st.divider()
    decision_threshold = st.slider(
        "SOC Alert Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Lower threshold increases Recall (catches more stealth phishing) at the cost of slightly higher False Positives."
    )

# --- Header ---
st.title("🛡️ LankaShield-Phish: Sri Lankan Cybersecurity Threat Detection & XAI")
st.markdown(
    "Real-time detection and explainability (SHAP & LIME) for localized phishing & SMS smishing attacks targeting "
    "**Sri Lankan Banks (`BOC`, `Sampath Vishwa`, `ComBank`, `HNB`, `FriMi`), `SL Post` Customs, `CEB`, `Gov.lk`, and Telcos (`Dialog`, `Mobitel`)**."
)

tab_scanner, tab_arena, tab_xai, tab_eda = st.tabs([
    "🔍 1. Live URL Threat Scanner (Real-Time SHAP & LIME)",
    "⚔️ 2. 6-Model Arena & Comparison",
    "🧠 3. Global & Case-Study XAI (SHAP vs. LIME)",
    "📊 4. Sri Lankan Dataset & Threat Intelligence",
])

# ==============================================================================
# TAB 1: LIVE URL THREAT SCANNER WITH REAL-TIME SHAP & LIME
# ==============================================================================
with tab_scanner:
    st.subheader("Live Sri Lankan Phishing & Smishing URL Analyzer")

    PRESET_URLS = {
        "🚨 [SL Phishing] Fake Sri Lanka Post Customs SMS Link": "http://slpost-customs-clearance-lk.top/track/confirm-address.php?parcel_id=LK88392014&fee=LKR450",
        "🚨 [SL Phishing] Sampath Vishwa Subdomain & OTP Harvester": "https://sampath.lk.secure-login.xyz/SVRClientWeb/login?user_auth=482910",
        "🚨 [SL Phishing] Raw IP Bank of Ceylon Credential Phish": "http://103.145.22.127/boc-lk/login.php?token=992a8f7c1e",
        "🚨 [SL Phishing] CEB Electricity Disconnection Scam": "http://ceb-lk-disconnection-notice.icu/pay/instant-settlement?acct=0719283741",
        "⚠️ [Stealth Phishing] Clean HTTPS Brand Spoof (Hard Case)": "https://vishwa-portal.org/app/q2c07q",
        "✅ [SL Legitimate] Official Bank of Ceylon Portal": "https://www.boc.lk/personal-banking/accounts?id=1042&lang=en",
        "✅ [SL Legitimate] Official Sri Lanka Post Customs Guidelines (Hard Benign)": "http://www.slpost.gov.lk/customs/parcel-clearance-guidelines?doc=9449587&dept=slpost&type=tax",
        "✅ [SL Legitimate] University of Moratuwa Academic Portal": "https://www.mrt.ac.lk/academic/postgraduate-studies/msc-cs",
    }

    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        selected_preset = st.selectbox("Select a Real-World Sri Lankan Preset Scenario:", list(PRESET_URLS.keys()))
    with col_p2:
        user_url = st.text_input("Or Enter / Modify Any URL to Scan Live:", value=PRESET_URLS[selected_preset])

    # Extract 22 features live
    t_start = time.perf_counter()
    extracted_dict = extract_url_features(user_url)
    X_single = pd.DataFrame([extracted_dict])[FEATURE_NAMES]

    # Champion model prediction
    champ_prob = float(best_model.predict_proba(X_single)[0, 1])
    champ_pred = 1 if champ_prob >= decision_threshold else 0
    scan_ms = (time.perf_counter() - t_start) * 1000.0

    if champ_prob >= 0.70:
        st.markdown(
            f"""
            <div class="risk-high">
                <h3 style="margin:0;">🚨 HIGH PHISHING RISK DETECTED — BLOCK RECOMMENDED</h3>
                <p style="margin:6px 0 0 0; font-size:1.05rem;">
                    <b>Champion Model ({metrics_summary['best_model']}) Phishing Probability:</b> {champ_prob*100:.2f}%
                    &nbsp;|&nbsp; <b>Decision Threshold:</b> {decision_threshold*100:.0f}%
                    &nbsp;|&nbsp; <b>Feature Extraction + Inference Latency:</b> {scan_ms:.2f} ms
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif champ_prob >= decision_threshold:
        st.markdown(
            f"""
            <div class="risk-medium">
                <h3 style="margin:0;">⚠️ SUSPICIOUS / STEALTH PHISHING RISK — STEP-UP VERIFICATION</h3>
                <p style="margin:6px 0 0 0; font-size:1.05rem;">
                    <b>Champion Model ({metrics_summary['best_model']}) Phishing Probability:</b> {champ_prob*100:.2f}%
                    &nbsp;|&nbsp; <b>Decision Threshold:</b> {decision_threshold*100:.0f}%
                    &nbsp;|&nbsp; <b>Latency:</b> {scan_ms:.2f} ms
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="risk-low">
                <h3 style="margin:0;">✅ LEGITIMATE / SAFE URL VERIFIED</h3>
                <p style="margin:6px 0 0 0; font-size:1.05rem;">
                    <b>Champion Model ({metrics_summary['best_model']}) Phishing Probability:</b> {champ_prob*100:.2f}%
                    (Benign Confidence: {(1-champ_prob)*100:.2f}%)
                    &nbsp;|&nbsp; <b>Latency:</b> {scan_ms:.2f} ms
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # All 6 Models Live Consensus Table
    st.markdown("#### 🗳️ Live Consensus Across All 6 Non-Linear ML Models")
    consensus_cols = st.columns(6)
    for idx, (m_name, m_pipe) in enumerate(all_models.items()):
        p_val = float(m_pipe.predict_proba(X_single)[0, 1])
        is_phish = p_val >= decision_threshold
        with consensus_cols[idx]:
            st.metric(
                label=m_name,
                value=f"{p_val*100:.1f}%",
                delta="PHISHING" if is_phish else "SAFE",
                delta_color="inverse" if is_phish else "normal",
            )

    st.divider()
    col_shap_live, col_lime_live = st.columns(2)

    with col_shap_live:
        st.markdown(f"#### 🔬 Live SHAP Waterfall Explanation ({metrics_summary['best_model']})")
        st.caption("Exact game-theoretic Shapley additive contributions pushing log-odds from Base Value $E[f(X)]$ to $f(x)$.")
        X_single_scaled = pd.DataFrame(scaler.transform(X_single), columns=FEATURE_NAMES)
        s_exp = shap_explainer(X_single_scaled)
        s_vals = s_exp.values
        b_vals = s_exp.base_values
        if s_vals.ndim == 3:
            s_vals = s_vals[:, :, 1]
            if np.ndim(b_vals) == 2:
                b_vals = b_vals[:, 1]
        live_clean_shap = shap.Explanation(
            values=s_vals[0],
            base_values=b_vals[0],
            data=X_single.iloc[0].values,
            feature_names=FEATURE_NAMES,
        )
        fig_s, _ = plt.subplots(figsize=(8, 5.2))
        shap.plots.waterfall(live_clean_shap, max_display=10, show=False)
        plt.title("Live Instance SHAP Feature Contributions", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)

    with col_lime_live:
        st.markdown(f"#### 🧪 Live LIME Local Surrogate Rules ({metrics_summary['best_model']})")
        st.caption("Local interpretable linear model fit on 1,000 perturbed neighborhood samples around this URL.")
        lime_exp_live = lime_explainer.explain_instance(
            data_row=X_single.iloc[0].values,
            predict_fn=lambda arr: best_model.predict_proba(pd.DataFrame(arr, columns=FEATURE_NAMES)),
            num_features=10,
            num_samples=1000,
        )
        lime_list_live = lime_exp_live.as_list()
        fig_l, ax_l = plt.subplots(figsize=(8, 5.2))
        rules_l = [r[0] for r in lime_list_live][::-1]
        weights_l = [r[1] for r in lime_list_live][::-1]
        colors_l = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights_l]
        ax_l.barh(rules_l, weights_l, color=colors_l, edgecolor="#2c3e50", alpha=0.9)
        ax_l.axvline(0, color="black", lw=1)
        ax_l.set_xlabel("LIME Probability Weight (+ Phishing / - Legitimate)")
        ax_l.set_title("Top 10 LIME Local Decision Rules", fontsize=11, fontweight="bold")
        ax_l.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_l)
        plt.close(fig_l)

    with st.expander("📋 View All 22 Extracted Cybersecurity & Sri Lankan Domain Features for this URL", expanded=False):
        feat_table = pd.DataFrame([
            {
                "Feature Name": f,
                "Extracted Value": extracted_dict[f],
                "SHAP Contribution (Log-Odds)": round(float(s_vals[0][i]), 4),
                "Description": FEATURE_DESCRIPTIONS[f],
            }
            for i, f in enumerate(FEATURE_NAMES)
        ])
        st.dataframe(feat_table, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: 6-MODEL ARENA & COMPARISON
# ==============================================================================
with tab_arena:
    st.subheader("Systematic Comparison of 6 Non-Linear ML Models (Held-Out Test Set, n = 900)")
    st.info(
        "**Methodology Note:** Strictly adheres to the course constraint excluding Linear Regression and Logistic Regression. "
        "All 6 models were trained on identical stratified splits (`70% Train / 15% Val / 15% Test`) and validated via `5-Fold Stratified Cross-Validation`."
    )

    metrics_df = pd.DataFrame(metrics_summary["metrics"])
    display_cols = [
        "Model", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_Specificity",
        "Test_F1", "Test_ROC_AUC", "Test_PR_AUC", "Test_MCC", "Brier_Score",
        "FP", "FN", "CV_F1_Mean", "CV_F1_Std", "Inference_Latency_ms"
    ]
    st.dataframe(
        metrics_df[display_cols].style.highlight_max(
            subset=["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC", "Test_PR_AUC", "Test_MCC"],
            color="#1e5631"
        ).highlight_min(
            subset=["Brier_Score", "FP", "FN", "Inference_Latency_ms"],
            color="#1e5631"
        ),
        use_container_width=True,
        hide_index=True,
    )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.image("reports/figures/04_roc_and_pr_curves.png", caption="ROC and Precision-Recall Curves Across All 6 Models", use_container_width=True)
    with col_c2:
        st.image("reports/figures/06_metrics_and_calibration.png", caption="Multi-Metric Performance & Probability Calibration", use_container_width=True)

    st.image("reports/figures/05_confusion_matrices_6models.png", caption="Test Set Confusion Matrices (n = 900) Across All 6 Models", use_container_width=True)
    st.image("reports/figures/07_latency_pareto_and_cv_stability.png", caption="Inference Latency vs F1 Pareto Frontier & 5-Fold Cross-Validation Stability", use_container_width=True)

# ==============================================================================
# TAB 3: GLOBAL & CASE-STUDY EXPLAINABLE AI (SHAP vs. LIME)
# ==============================================================================
with tab_xai:
    st.subheader("Explainable AI (XAI) Deep Dive: SHAP vs. LIME")

    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Champion Model Explained", xai_summary["champion_model"])
    x2.metric("SHAP TreeExplainer Latency", f"{xai_summary['shap_latency_ms_per_sample']:.3f} ms / URL")
    x3.metric("LIME Perturbation Latency", f"{xai_summary['lime_latency_ms_per_sample']:.2f} ms / URL")
    x4.metric("Top-5 Feature Overlap (SHAP vs LIME)", f"{xai_summary['mean_top5_feature_agreement']*100:.1f}%")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.image("reports/figures/08_shap_summary_beeswarm.png", caption="SHAP Global Beeswarm Plot (Feature Impact & Directionality)", use_container_width=True)
    with col_g2:
        st.image("reports/figures/09_shap_global_bar.png", caption="SHAP Global Mean Absolute Importance Bar Plot", use_container_width=True)

    st.image("reports/figures/10_shap_dependence_interactions.png", caption="Non-Linear Feature Dependence & Interaction Effects Captured by SHAP", use_container_width=True)

    st.divider()
    st.markdown("### 🔎 Side-by-Side Local Case Studies: SHAP Waterfall vs. LIME Surrogate")
    case_titles = [c["title"] for c in xai_summary["case_studies"]]
    chosen_case_title = st.radio("Select Held-Out Test Case Study:", case_titles, horizontal=True)
    chosen_case = next(c for c in xai_summary["case_studies"] if c["title"] == chosen_case_title)

    st.markdown(
        f"**Inspected URL:** `{chosen_case['url']}`  \n"
        f"**Category:** `{chosen_case['category']}` | **Actual Label:** `{chosen_case['actual_label']}` | "
        f"**Predicted Phishing Probability:** `{chosen_case['pred_prob']*100:.2f}%` | "
        f"**Top-5 Feature Agreement:** `{chosen_case['top5_overlap_count']} / 5 features`"
    )
    cs_col1, cs_col2 = st.columns(2)
    with cs_col1:
        st.image(os.path.join("reports/figures", chosen_case["shap_figure"]), caption="SHAP Local Waterfall Plot", use_container_width=True)
    with cs_col2:
        st.image(os.path.join("reports/figures", chosen_case["lime_figure"]), caption="LIME Local Linear Surrogate Explanation", use_container_width=True)

# ==============================================================================
# TAB 4: SRI LANKAN DATASET & THREAT INTELLIGENCE
# ==============================================================================
with tab_eda:
    st.subheader("Sri Lankan Cybersecurity Dataset & Feature Engineering Explorer")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Curated URLs", f"{len(full_df):,}")
    d2.metric("Legitimate URLs (Class 0)", f"{int((full_df['label']==0).sum()):,}")
    d3.metric("Phishing URLs (Class 1)", f"{int((full_df['label']==1).sum()):,}")
    d4.metric("Engineered Features", f"{len(FEATURE_NAMES)}")

    st.image("reports/figures/01_eda_categories_and_mutual_info.png", caption="Threat Categories & Mutual Information Ranking", use_container_width=True)
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.image("reports/figures/02_feature_correlation_heatmap.png", caption="22-Feature Pearson Correlation Matrix", use_container_width=True)
    with col_e2:
        st.image("reports/figures/03_feature_distributions_violin.png", caption="Distributional Shifts Across Key Continuous Features", use_container_width=True)

    st.markdown("#### Sample Records from `data/raw/srilanka_phishing_dataset.csv`")
    st.dataframe(full_df.head(25), use_container_width=True)
