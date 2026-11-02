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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_extractor import extract_url_features, FEATURE_NAMES, FEATURE_DESCRIPTIONS
from src.drift_monitor import evaluate_feature_drift, simulate_shifted_traffic, build_model_registry

st.set_page_config(
    page_title="SL Phishing URL Detector - MSc AML",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .alert-danger {
        background-color: #69141e;
        color: #ffffff;
        padding: 16px 20px;
        border-radius: 8px;
        border-left: 5px solid #ff4d4d;
        margin-bottom: 15px;
    }
    .alert-safe {
        background-color: #134e32;
        color: #ffffff;
        padding: 16px 20px;
        border-radius: 8px;
        border-left: 5px solid #2ecc71;
        margin-bottom: 15px;
    }
    .alert-warn {
        background-color: #5c4010;
        color: #ffffff;
        padding: 16px 20px;
        border-radius: 8px;
        border-left: 5px solid #f39c12;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    with open("models/metrics_summary.json", "r") as f:
        metrics_summary = json.load(f)
    with open("models/xai_summary.json", "r") as f:
        xai_summary = json.load(f)

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    full_df = pd.read_csv("data/raw/srilanka_phishing_dataset.csv")

    all_models = {}
    try:
        for m_info in metrics_summary["metrics"]:
            m_name = m_info["Model"]
            art_path = os.path.join("models/all_models", m_info["Artifact_File"])
            pipe = joblib.load(art_path)
            _ = pipe.predict_proba(train_df[FEATURE_NAMES].iloc[[0]])
            all_models[m_name] = pipe
        best_model = joblib.load("models/best_model.joblib")
        _ = best_model.predict_proba(train_df[FEATURE_NAMES].iloc[[0]])
    except Exception:
        from src.train_and_evaluate import build_models
        raw_models = build_models()
        X_tr = train_df[FEATURE_NAMES]
        y_tr = train_df["label"]
        for m_name, pipe in raw_models.items():
            pipe.fit(X_tr, y_tr)
            all_models[m_name] = pipe
        best_model = all_models[metrics_summary["best_model"]]

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

# Sidebar
with st.sidebar:
    st.markdown("## SL Phishing Detector")
    st.caption("MSc Applied Machine Learning Project — Detecting localized Sri Lankan phishing & smishing URLs")
    st.divider()
    st.markdown("### Selected Best Model")
    st.info(
        f"**{metrics_summary['best_model']}**  \n"
        f"Test F1-Score: **{metrics_summary['metrics'][0]['Test_F1']:.4f}**  \n"
        f"ROC-AUC: **{metrics_summary['metrics'][0]['Test_ROC_AUC']:.4f}**"
    )
    st.divider()
    decision_threshold = st.slider(
        "Classification Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Adjust the probability cutoff for flagging a URL as phishing.",
    )

st.title("Sri Lankan Phishing & Smishing URL Detection")
st.markdown(
    "Detection and model explainability (SHAP & LIME) for malicious URLs targeting Sri Lankan banks "
    "(`BOC`, `Sampath Vishwa`, `ComBank`, `HNB`, `FriMi`), `SL Post`, `CEB`, `Gov.lk`, and local telcos."
)

tab_scanner, tab_arena, tab_xai, tab_eda, tab_mlops = st.tabs([
    "1. Live URL Scanner (SHAP & LIME)",
    "2. Model Comparison (6 Models)",
    "3. Explainability Analysis (SHAP vs LIME)",
    "4. Dataset & EDA",
    "5. MLOps & Drift Monitor",
])

# Tab 1: Live URL Scanner
with tab_scanner:
    st.subheader("Test a URL in Real Time")

    PRESET_URLS = {
        "[Phishing] Fake SL Post Customs SMS Link": "http://slpost-customs-clearance-lk.top/track/confirm-address.php?parcel_id=LK88392014&fee=LKR450",
        "[Phishing] Sampath Vishwa Subdomain Spoof": "https://sampath.lk.secure-login.xyz/SVRClientWeb/login?user_auth=482910",
        "[Phishing] Raw IP Bank of Ceylon Login": "http://103.145.22.127/boc-lk/login.php?token=992a8f7c1e",
        "[Phishing] CEB Bill Disconnection Scam": "http://ceb-lk-disconnection-notice.icu/pay/instant-settlement?acct=0719283741",
        "[Stealth Phishing] Clean HTTPS Brand Spoof": "https://vishwa-portal.org/app/q2c07q",
        "[Legitimate] Official Bank of Ceylon Website": "https://www.boc.lk/personal-banking/accounts?id=1042&lang=en",
        "[Legitimate] Official SL Post Customs Page (HTTP + Keywords)": "http://www.slpost.gov.lk/customs/parcel-clearance-guidelines?doc=9449587&dept=slpost&type=tax",
        "[Legitimate] University of Moratuwa Portal": "https://www.mrt.ac.lk/academic/postgraduate-studies/msc-cs",
    }

    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        selected_preset = st.selectbox("Sample URLs:", list(PRESET_URLS.keys()))
    with col_p2:
        user_url = st.text_input("Enter or edit URL:", value=PRESET_URLS[selected_preset])

    t_start = time.perf_counter()
    extracted_dict = extract_url_features(user_url)
    X_single = pd.DataFrame([extracted_dict])[FEATURE_NAMES]

    champ_prob = float(best_model.predict_proba(X_single)[0, 1])
    scan_ms = (time.perf_counter() - t_start) * 1000.0

    if champ_prob >= 0.70:
        st.markdown(
            f"""
            <div class="alert-danger">
                <h4 style="margin:0;">High Phishing Risk Detected</h4>
                <p style="margin:5px 0 0 0;">
                    <b>{metrics_summary['best_model']} Phishing Probability:</b> {champ_prob*100:.2f}%
                    &nbsp;|&nbsp; <b>Threshold:</b> {decision_threshold*100:.0f}%
                    &nbsp;|&nbsp; <b>Prediction Time:</b> {scan_ms:.2f} ms
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif champ_prob >= decision_threshold:
        st.markdown(
            f"""
            <div class="alert-warn">
                <h4 style="margin:0;">Moderate / Suspicious Phishing Risk</h4>
                <p style="margin:5px 0 0 0;">
                    <b>{metrics_summary['best_model']} Phishing Probability:</b> {champ_prob*100:.2f}%
                    &nbsp;|&nbsp; <b>Threshold:</b> {decision_threshold*100:.0f}%
                    &nbsp;|&nbsp; <b>Prediction Time:</b> {scan_ms:.2f} ms
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-safe">
                <h4 style="margin:0;">Legitimate URL</h4>
                <p style="margin:5px 0 0 0;">
                    <b>{metrics_summary['best_model']} Phishing Probability:</b> {champ_prob*100:.2f}%
                    (Legitimate Score: {(1-champ_prob)*100:.2f}%)
                    &nbsp;|&nbsp; <b>Prediction Time:</b> {scan_ms:.2f} ms
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Predictions Across All 6 Trained Models")
    consensus_cols = st.columns(6)
    for idx, (m_name, m_pipe) in enumerate(all_models.items()):
        p_val = float(m_pipe.predict_proba(X_single)[0, 1])
        is_phish = p_val >= decision_threshold
        with consensus_cols[idx]:
            st.metric(
                label=m_name,
                value=f"{p_val*100:.1f}%",
                delta="Phishing" if is_phish else "Legitimate",
                delta_color="inverse" if is_phish else "normal",
            )

    st.divider()
    col_shap_live, col_lime_live = st.columns(2)

    with col_shap_live:
        st.markdown(f"#### SHAP Waterfall Plot ({metrics_summary['best_model']})")
        st.caption("Shows how each extracted feature pushes the model output from the base value toward Phishing (+) or Legitimate (-).")
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
        plt.title("SHAP Feature Contributions for Input URL", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)

    with col_lime_live:
        st.markdown(f"#### LIME Local Explanation ({metrics_summary['best_model']})")
        st.caption("Local linear surrogate fit on perturbed samples around this URL.")
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
        ax_l.set_xlabel("LIME Weight (+ Phishing / - Legitimate)")
        ax_l.set_title("Top 10 LIME Feature Rules", fontsize=11, fontweight="bold")
        ax_l.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_l)
        plt.close(fig_l)

    with st.expander("View All 22 Extracted Features for this URL", expanded=False):
        feat_table = pd.DataFrame([
            {
                "Feature": f,
                "Value": extracted_dict[f],
                "SHAP Value": round(float(s_vals[0][i]), 4),
                "Description": FEATURE_DESCRIPTIONS[f],
            }
            for i, f in enumerate(FEATURE_NAMES)
        ])
        st.dataframe(feat_table, use_container_width=True, hide_index=True)

# Tab 2: Model Comparison
with tab_arena:
    st.subheader("Evaluation Across 6 Non-Linear Classifiers (Test Set, n = 900)")
    st.write(
        "All 6 models were trained on the same stratified train/validation/test split (`70% / 15% / 15%`) "
        "and evaluated with 5-fold stratified cross-validation. Linear and Logistic Regression were not used."
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
        st.image("reports/figures/04_roc_and_pr_curves.png", caption="ROC and Precision-Recall Curves", use_container_width=True)
    with col_c2:
        st.image("reports/figures/06_metrics_and_calibration.png", caption="Test Set Metrics & Calibration Curves", use_container_width=True)

    st.image("reports/figures/05_confusion_matrices_6models.png", caption="Confusion Matrices on Held-Out Test Set (n = 900)", use_container_width=True)
    st.image("reports/figures/07_latency_pareto_and_cv_stability.png", caption="Inference Latency vs F1-Score & 5-Fold Cross-Validation Results", use_container_width=True)

# Tab 3: Explainability
with tab_xai:
    st.subheader("Global & Local Explainability (SHAP and LIME)")

    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Model Explained", xai_summary["champion_model"])
    x2.metric("Mean SHAP Time", f"{xai_summary['shap_latency_ms_per_sample']:.3f} ms / URL")
    x3.metric("Mean LIME Time", f"{xai_summary['lime_latency_ms_per_sample']:.2f} ms / URL")
    x4.metric("Top-5 Overlap (SHAP vs LIME)", f"{xai_summary['mean_top5_feature_agreement']*100:.1f}%")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.image("reports/figures/08_shap_summary_beeswarm.png", caption="SHAP Summary Plot (Beeswarm)", use_container_width=True)
    with col_g2:
        st.image("reports/figures/09_shap_global_bar.png", caption="Mean Absolute SHAP Values", use_container_width=True)

    st.image("reports/figures/10_shap_dependence_interactions.png", caption="SHAP Dependence Plots for Top Features", use_container_width=True)

    st.divider()
    st.markdown("### Local Case Studies: Comparing SHAP and LIME on Specific Test URLs")
    case_titles = [c["title"] for c in xai_summary["case_studies"]]
    chosen_case_title = st.radio("Select Case Study:", case_titles, horizontal=True)
    chosen_case = next(c for c in xai_summary["case_studies"] if c["title"] == chosen_case_title)

    st.markdown(
        f"**URL:** `{chosen_case['url']}`  \n"
        f"**Category:** `{chosen_case['category']}` | **Actual Label:** `{chosen_case['actual_label']}` | "
        f"**Predicted Probability:** `{chosen_case['pred_prob']*100:.2f}%` | "
        f"**Top-5 Feature Agreement:** `{chosen_case['top5_overlap_count']} / 5`"
    )
    cs_col1, cs_col2 = st.columns(2)
    with cs_col1:
        st.image(os.path.join("reports/figures", chosen_case["shap_figure"]), caption="SHAP Waterfall Explanation", use_container_width=True)
    with cs_col2:
        st.image(os.path.join("reports/figures", chosen_case["lime_figure"]), caption="LIME Local Explanation", use_container_width=True)

# Tab 4: Dataset & EDA
with tab_eda:
    st.subheader("Dataset Overview & Exploratory Data Analysis")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total URLs", f"{len(full_df):,}")
    d2.metric("Legitimate (0)", f"{int((full_df['label']==0).sum()):,}")
    d3.metric("Phishing (1)", f"{int((full_df['label']==1).sum()):,}")
    d4.metric("Features", f"{len(FEATURE_NAMES)}")

    st.image("reports/figures/01_eda_categories_and_mutual_info.png", caption="Dataset Categories and Mutual Information Scores", use_container_width=True)
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.image("reports/figures/02_feature_correlation_heatmap.png", caption="Feature Correlation Heatmap", use_container_width=True)
    with col_e2:
        st.image("reports/figures/03_feature_distributions_violin.png", caption="Feature Distributions by Class", use_container_width=True)

    st.markdown("#### Sample Rows from Dataset")
    st.dataframe(full_df.head(25), use_container_width=True)

# Tab 5: MLOps & Drift Monitor
with tab_mlops:
    st.subheader("MLOps Model Registry, Quality Gates & Data Drift Monitoring")

    reg_path = "models/model_registry.json"
    if os.path.exists(reg_path):
        with open(reg_path, "r") as f:
            registry = json.load(f)
    else:
        registry = build_model_registry()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Version", registry["production_version"])
    m2.metric("Deployment Stage", registry["stage"])
    m3.metric("Dataset SHA-256", registry["data_lineage"]["dataset_sha256"])
    m4.metric("Quality Gates Passed", "4 / 4" if registry["all_quality_gates_passed"] else "Failed")

    st.markdown("#### Automated CI/CD Production Quality Gates")
    gates_rows = []
    for g_name, g_info in registry["quality_gates"].items():
        gates_rows.append({
            "Gate Check": g_name,
            "Required Threshold": g_info["threshold"],
            "Measured Value": g_info["actual"],
            "Status": "PASS" if g_info["passed"] else "FAIL",
        })
    st.dataframe(pd.DataFrame(gates_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Covariate Shift & Data Drift Monitor (PSI & Two-Sample KS Test)")
    st.write(
        "Compares incoming URL feature distributions against the reference training baseline (`train.csv`). "
        "A **Population Stability Index (PSI) >= 0.20** or **KS test p-value < 0.01** triggers a retraining alert."
    )

    sim_choice = st.radio(
        "Select Incoming Traffic Batch to Evaluate:",
        [
            "Held-Out Test Set Baseline (Normal Production Traffic)",
            "Simulated Adversarial Shift (New Smishing Wave with Longer Obfuscated URLs & New TLDs)",
        ],
        horizontal=True,
    )

    if "Normal" in sim_choice:
        incoming_batch = test_df
    else:
        incoming_batch = simulate_shifted_traffic(test_df, scenario="new_attack_wave", n_samples=500)

    drift_table = evaluate_feature_drift(train_df, incoming_batch)
    num_drifted = int((drift_table["Status"] == "Drift Detected").sum())

    if num_drifted >= registry["retraining_policy"]["max_drifted_features_before_retrain"]:
        st.error(
            f"**Retraining Alert Triggered:** {num_drifted} features exceed the drift threshold "
            f"(policy limit: {registry['retraining_policy']['max_drifted_features_before_retrain']} features). "
            "Scheduled pipeline retraining recommended."
        )
    else:
        st.success(
            f"**Distribution Stable:** {num_drifted} features drifted. Model `{registry['production_version']}` "
            "remains within healthy operating bounds."
        )

    st.dataframe(drift_table, use_container_width=True, hide_index=True)

