"""Streamlit dashboard — Telecom Churn Prediction with Apache Spark MLlib.

Tabs:
  1. 🏠 Home            — project overview, Spark environment, quick stats
  2. 📂 Data Explorer   — dataset preview, distributions, churn breakdown
  3. 🤖 Model Scorecard — ALL model metrics, comparison charts, feature importances
  4. 👤 Predict Customer — interactive single-customer scoring with any model
  5. 📁 Batch Scoring   — upload CSV, score all rows, download results
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import (
    CATEGORICAL,
    FEATURES,
    LABEL,
    MODEL_CONFIGS,
    NUMERIC,
    load_model,
    make_spark,
    predict_records,
    train,
)

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "telco_churn_pipeline"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn — Spark MLlib",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS — subtle card styling
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 18px 22px;
        border-left: 4px solid #7c3aed;
    }
    .best-badge {
        background: #7c3aed;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #a78bfa;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  Cached resources
# ─────────────────────────────────────────────
@st.cache_resource
def get_spark():
    return make_spark("TelcoChurnStreamlit")


@st.cache_resource
def get_model(model_path: str, stamp: float):
    return load_model(get_spark(), model_path)


@st.cache_data
def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ─────────────────────────────────────────────
#  Ensure models exist (train on first launch)
# ─────────────────────────────────────────────
def ensure_models() -> None:
    if not DATA_PATH.exists():
        st.error(
            "Dataset not found. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` "
            "from Kaggle and save it to `data/raw/`."
        )
        st.link_button(
            "Open Kaggle dataset",
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        )
        st.stop()

    models_exist = any(
        (MODELS_DIR / key / "metadata").exists() for key in MODEL_CONFIGS
    )
    if not models_exist and not (DEFAULT_MODEL_PATH / "metadata").exists():
        with st.spinner(
            "🔥 Training all Spark MLlib models for the first time — this takes ~2 min…"
        ):
            result = train(DATA_PATH, MODELS_DIR, model_types="all")
            METRICS_PATH.write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        st.cache_resource.clear()
        st.rerun()


ensure_models()

# ─────────────────────────────────────────────
#  Discover trained models
# ─────────────────────────────────────────────
available_models: dict[str, Path] = {}
for _key, _info in MODEL_CONFIGS.items():
    _folder = MODELS_DIR / _key
    if (_folder / "metadata").exists():
        available_models[_info["name"]] = _folder

if not available_models and (DEFAULT_MODEL_PATH / "metadata").exists():
    available_models["Best Model"] = DEFAULT_MODEL_PATH

# ─────────────────────────────────────────────
#  Load saved metrics
# ─────────────────────────────────────────────
metrics: dict = {}
if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}

models_data: dict = metrics.get("models", {})
best_model_name: str = metrics.get("best_model", next(iter(available_models), ""))
default_index = (
    list(available_models.keys()).index(best_model_name)
    if best_model_name in available_models
    else 0
)

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://spark.apache.org/images/spark-logo-trademark.png",
        width=160,
    )
    st.title("📡 Telco Churn")
    st.caption("Apache Spark MLlib · Streamlit")

    st.markdown("---")
    st.markdown("### 🤖 Trained Models")
    for _name, _path in available_models.items():
        _m = models_data.get(_name, {})
        _auc = _m.get("roc_auc")
        _acc = _m.get("accuracy")
        _is_best = _name == best_model_name
        _badge = " 🏆" if _is_best else ""
        _auc_str = f"  AUC: **{_auc:.3f}** | Acc: **{_acc:.1%}**" if _auc else ""
        st.markdown(f"**{_name}**{_badge}\n{_auc_str}")

    st.markdown("---")
    st.markdown("### ⚙️ Spark Environment")
    try:
        _spark = get_spark()
        st.success("Spark session active ✅")
        st.write(f"**Version:** {_spark.version}")
        st.write("**Master:** local[*]")
        st.write("**App:** TelcoChurnStreamlit")
    except Exception:
        st.warning("Spark session not started yet")

    st.markdown("---")
    if st.button("🔄 Retrain All Models", use_container_width=True):
        with st.spinner("Retraining all 4 Spark MLlib models…"):
            _new = train(DATA_PATH, MODELS_DIR, model_types="all")
            METRICS_PATH.write_text(json.dumps(_new, indent=2), encoding="utf-8")
        st.cache_resource.clear()
        st.success("Retrained successfully!")
        st.rerun()

    st.markdown("---")
    st.caption("Dataset: [Kaggle Telco Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")
    st.caption("Engine: PySpark 3.5 · Java 17")

# ─────────────────────────────────────────────
#  Page header
# ─────────────────────────────────────────────
st.title("📡 Telecom Customer Churn Prediction")
st.markdown(
    "**Apache Spark MLlib** end-to-end pipeline · "
    "Logistic Regression · Random Forest · Gradient-Boosted Trees · Decision Tree"
)
st.markdown("---")

# ─────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────
tab_home, tab_eda, tab_scores, tab_predict, tab_batch = st.tabs([
    "🏠 Home",
    "📂 Data Explorer",
    "🤖 Model Scorecard",
    "👤 Predict Customer",
    "📁 Batch Scoring",
])

# ════════════════════════════════════════════════════════
#  TAB 1 — HOME
# ════════════════════════════════════════════════════════
with tab_home:
    st.subheader("Project Overview")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        This project demonstrates a production-style **customer churn prediction system**
        built entirely on **Apache Spark MLlib** with a **Streamlit** frontend.

        #### What is Customer Churn?
        Churn occurs when a telecom customer cancels their subscription. Predicting churn
        early allows the business to offer targeted retention incentives and reduce revenue loss.

        #### Machine Learning Approach
        Four Spark MLlib classification algorithms are trained on the IBM Telco Customer Churn
        dataset and benchmarked on the same held-out 20% test split (seed 42):

        | # | Model | Type |
        |---|-------|------|
        | 1 | **Logistic Regression** | Linear, interpretable baseline |
        | 2 | **Random Forest** | Ensemble bagging (50 trees) |
        | 3 | **Gradient-Boosted Trees** | Sequential boosting, typically strongest |
        | 4 | **Decision Tree** | Single tree for interpretable rules |

        #### Data Pipeline
        ```
        CSV → Spark DataFrame → StringIndexer → OneHotEncoder
            → VectorAssembler → Label Indexer → Classifier → PipelineModel
        ```

        Predictions from any model are available in the **Predict Customer** and **Batch Scoring** tabs.
        """)

    with col_b:
        st.markdown("#### 📊 Dataset at a Glance")
        if metrics:
            st.metric("Total Customers", f"{metrics.get('rows', 0):,}")
            st.metric("Training Rows", f"{metrics.get('train_rows', 0):,}")
            st.metric("Test Rows", f"{metrics.get('test_rows', 0):,}")
            st.metric("Dataset Churn Rate", f"{metrics.get('churn_rate', 0):.1%}")
            st.metric("🏆 Best Model (ROC AUC)", f"{best_model_name}  [{metrics.get('roc_auc', 0):.3f}]")

    st.markdown("---")
    st.subheader("🔑 Key Metrics — All Models at a Glance")
    if models_data:
        cols = st.columns(len(models_data))
        for i, (mname, mdata) in enumerate(models_data.items()):
            is_best = mname == best_model_name
            badge = "🏆 Best" if is_best else ""
            with cols[i]:
                st.markdown(f"##### {mname} {badge}")
                st.metric("Accuracy", f"{mdata.get('accuracy', 0):.1%}")
                st.metric("F1-Score", f"{mdata.get('f1', 0):.3f}")
                st.metric("ROC AUC", f"{mdata.get('roc_auc', 0):.3f}")

    st.markdown("---")
    st.markdown("""
    #### ℹ️ How to Use This App
    - **📂 Data Explorer** — Browse the dataset, see churn distributions and feature statistics
    - **🤖 Model Scorecard** — Compare all 4 models side-by-side with charts and feature importances
    - **👤 Predict Customer** — Fill in customer details and get a churn risk score from any model
    - **📁 Batch Scoring** — Upload a CSV of customers and download scored results
    """)

# ════════════════════════════════════════════════════════
#  TAB 2 — DATA EXPLORER
# ════════════════════════════════════════════════════════
with tab_eda:
    st.subheader("📂 Dataset Exploration")

    if not DATA_PATH.exists():
        st.warning("Dataset CSV not found at `data/raw/`. Please add the file first.")
    else:
        df_raw = load_raw_data(str(DATA_PATH))

        # ── Top stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{len(df_raw):,}")
        churned = (df_raw["Churn"] == "Yes").sum()
        c2.metric("Churned", f"{churned:,}")
        c3.metric("Retained", f"{len(df_raw) - churned:,}")
        c4.metric("Churn Rate", f"{churned / len(df_raw):.1%}")

        st.markdown("---")

        # ── Churn breakdown
        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.markdown("#### Churn Distribution")
            churn_counts = df_raw["Churn"].value_counts().reset_index()
            churn_counts.columns = ["Churn", "Count"]
            st.bar_chart(churn_counts.set_index("Churn"))

        with col_bar:
            st.markdown("#### Churn by Contract Type")
            contract_churn = (
                df_raw.groupby(["Contract", "Churn"])
                .size()
                .reset_index(name="Count")
            )
            pivot = contract_churn.pivot(
                index="Contract", columns="Churn", values="Count"
            ).fillna(0)
            st.bar_chart(pivot)

        st.markdown("---")

        # ── Numeric feature distributions
        st.markdown("#### Numeric Feature Distributions")
        num_col = st.selectbox(
            "Select a numeric feature to explore:",
            options=NUMERIC,
            key="eda_num_col",
        )
        split_by_churn = st.checkbox("Split by Churn (Yes / No)", value=True, key="eda_split")

        if split_by_churn:
            n1, n2 = st.columns(2)
            with n1:
                st.markdown("**Churned (Yes)**")
                vals_yes = df_raw[df_raw["Churn"] == "Yes"][num_col].dropna()
                st.bar_chart(vals_yes.value_counts().sort_index())
            with n2:
                st.markdown("**Retained (No)**")
                vals_no = df_raw[df_raw["Churn"] == "No"][num_col].dropna()
                st.bar_chart(vals_no.value_counts().sort_index())
        else:
            st.bar_chart(df_raw[num_col].dropna().value_counts().sort_index())

        st.markdown("---")

        # ── Categorical feature breakdown
        st.markdown("#### Categorical Feature vs. Churn")
        cat_col = st.selectbox(
            "Select a categorical feature:",
            options=CATEGORICAL,
            key="eda_cat_col",
        )
        cat_pivot = (
            df_raw.groupby([cat_col, "Churn"])
            .size()
            .reset_index(name="Count")
            .pivot(index=cat_col, columns="Churn", values="Count")
            .fillna(0)
        )
        st.bar_chart(cat_pivot)

        st.markdown("---")

        # ── Dataset preview
        with st.expander("📋 Raw Dataset Preview (first 200 rows)", expanded=False):
            st.dataframe(df_raw.head(200), use_container_width=True, hide_index=True)

        with st.expander("📐 Dataset Schema & Summary Statistics", expanded=False):
            st.markdown("**Columns and dtypes:**")
            st.dataframe(
                pd.DataFrame({"Column": df_raw.columns, "dtype": [str(t) for t in df_raw.dtypes]}),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("**Summary statistics (numeric columns):**")
            st.dataframe(df_raw[NUMERIC].describe().T, use_container_width=True)

        with st.expander("📉 Missing Values Report", expanded=False):
            missing = df_raw.isnull().sum().reset_index()
            missing.columns = ["Column", "Missing Count"]
            missing["Missing %"] = (missing["Missing Count"] / len(df_raw) * 100).round(2)
            st.dataframe(missing, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
#  TAB 3 — MODEL SCORECARD  (ALL MODELS)
# ════════════════════════════════════════════════════════
with tab_scores:
    st.subheader("🤖 Model Scorecard — All Algorithms Compared")

    if not models_data:
        st.info("No trained models found. Run `python train.py` or click **Retrain All Models** in the sidebar.")
    else:
        # ── Full comparison table
        st.markdown("#### 📋 Side-by-Side Evaluation (all models, same 80/20 test split)")
        table_rows = []
        for mname, mdata in models_data.items():
            table_rows.append({
                "Model": mname,
                "Description": mdata.get("description", ""),
                "Accuracy": f"{mdata.get('accuracy', 0):.1%}",
                "F1-Score": f"{mdata.get('f1', 0):.3f}",
                "ROC AUC": f"{mdata.get('roc_auc', 0):.3f}",
                "Rank (AUC)": "",
                "Best?": "🏆 Yes" if mname == best_model_name else "—",
            })

        # Sort by AUC descending and add rank
        sorted_rows = sorted(
            table_rows,
            key=lambda r: float(r["ROC AUC"]),
            reverse=True,
        )
        for rank, row in enumerate(sorted_rows, 1):
            row["Rank (AUC)"] = str(rank)

        st.dataframe(pd.DataFrame(sorted_rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Bar charts: one per metric
        st.markdown("#### 📊 Metric Comparison Charts")
        chart_df = pd.DataFrame([
            {
                "Model": name,
                "Accuracy": mdata.get("accuracy", 0),
                "F1-Score": mdata.get("f1", 0),
                "ROC AUC": mdata.get("roc_auc", 0),
            }
            for name, mdata in models_data.items()
        ]).set_index("Model")

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown("**Accuracy**")
            st.bar_chart(chart_df[["Accuracy"]])
        with mc2:
            st.markdown("**F1-Score**")
            st.bar_chart(chart_df[["F1-Score"]])
        with mc3:
            st.markdown("**ROC AUC**")
            st.bar_chart(chart_df[["ROC AUC"]])

        st.markdown("---")

        # ── All-metrics combined chart
        st.markdown("#### 📈 Combined Metric Comparison")
        st.bar_chart(chart_df)

        st.markdown("---")

        # ── Feature importance per EVERY model
        st.markdown("#### 🔍 Feature Importances / Coefficient Weights — Per Model")
        st.caption(
            "Random Forest & GBT show Gini importance. "
            "Logistic Regression shows |coefficient| magnitude. "
            "Decision Tree shows Gini importance."
        )

        fi_tabs = st.tabs(list(models_data.keys()))
        for fi_tab, (mname, mdata) in zip(fi_tabs, models_data.items()):
            with fi_tab:
                top_feats = mdata.get("top_features", [])
                if top_feats:
                    feat_df = pd.DataFrame(top_feats)
                    feat_df.columns = ["Feature", "Importance"]
                    feat_df = feat_df.sort_values("Importance", ascending=False)
                    f_col1, f_col2 = st.columns([1, 2])
                    with f_col1:
                        st.dataframe(feat_df, use_container_width=True, hide_index=True)
                    with f_col2:
                        st.bar_chart(feat_df.set_index("Feature"))
                else:
                    st.info(f"Feature importance data not available for {mname}.")

        st.markdown("---")

        # ── Individual model detail cards
        st.markdown("#### 📝 Model Details")
        for mname, mdata in models_data.items():
            is_best = mname == best_model_name
            with st.expander(
                f"{'🏆 ' if is_best else ''}{mname}" + (" — Best Model" if is_best else ""),
                expanded=is_best,
            ):
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Accuracy", f"{mdata.get('accuracy', 0):.4f}")
                d2.metric("F1-Score", f"{mdata.get('f1', 0):.4f}")
                d3.metric("ROC AUC", f"{mdata.get('roc_auc', 0):.4f}")
                d4.metric("Rank", f"#{sorted([m.get('roc_auc', 0) for m in models_data.values()], reverse=True).index(mdata.get('roc_auc', 0)) + 1} of {len(models_data)}")
                st.markdown(f"**Algorithm:** {mdata.get('description', mname)}")
                st.markdown(f"**Saved model path:** `{mdata.get('path', 'N/A')}`")

# ════════════════════════════════════════════════════════
#  TAB 4 — PREDICT CUSTOMER
# ════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("👤 Interactive Single-Customer Churn Prediction")
    st.markdown(
        "Fill in the customer details below, choose which model to use, "
        "and click **Predict Churn** to get a probability estimate."
    )

    # Model selector + quick metrics display
    sel_col, info_col = st.columns([2, 2])
    with sel_col:
        selected_model_name = st.selectbox(
            "🤖 Select Model",
            options=list(available_models.keys()),
            index=default_index,
            help="Switch between Logistic Regression, Random Forest, GBT, or Decision Tree.",
        )
    with info_col:
        _sel_metrics = models_data.get(selected_model_name, {})
        if _sel_metrics:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Accuracy", f"{_sel_metrics.get('accuracy', 0):.1%}")
            mc2.metric("F1-Score", f"{_sel_metrics.get('f1', 0):.3f}")
            mc3.metric("ROC AUC", f"{_sel_metrics.get('roc_auc', 0):.3f}")

    chosen_model_path = available_models[selected_model_name]
    active_model = get_model(
        str(chosen_model_path), chosen_model_path.stat().st_mtime
    )

    st.markdown("---")

    with st.form("customer_form"):
        st.markdown("##### 👤 Demographics")
        a, b, c = st.columns(3)
        gender = a.selectbox("Gender", ["Female", "Male"])
        senior = a.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = a.selectbox("Has Partner", ["Yes", "No"])
        dependents = b.selectbox("Has Dependents", ["Yes", "No"])

        st.markdown("##### 📱 Services")
        s1, s2, s3 = st.columns(3)
        phone = s1.selectbox("Phone Service", ["Yes", "No"])
        lines = s1.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = s2.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = s2.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = s2.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        protection = s3.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        support = s3.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = s3.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = s3.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown("##### 💳 Billing")
        b1, b2, b3 = st.columns(3)
        contract = b1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = b1.selectbox("Paperless Billing", ["Yes", "No"])
        payment = b2.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        tenure = b3.slider("Tenure (months)", 0, 72, 12)
        monthly = b3.number_input("Monthly Charges ($)", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
        total = b3.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=float(tenure * monthly), step=10.0)

        submitted = st.form_submit_button(
            f"🔍 Predict Churn with {selected_model_name}", type="primary", use_container_width=True
        )

    if submitted:
        record = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": lines, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }
        with st.spinner("Scoring with Spark MLlib…"):
            output = predict_records(get_spark(), active_model, [record])[0]

        probability = output["churn_probability"]
        is_churn = output["prediction"] == "Yes"

        st.markdown("---")
        r1, r2 = st.columns([1, 2])
        with r1:
            if is_churn:
                st.error(f"### ⚠️ High Churn Risk\n**{probability:.1%}** estimated probability")
            else:
                st.success(f"### ✅ Low Churn Risk\n**{probability:.1%}** estimated probability")
            st.progress(probability, text="Churn probability")

        with r2:
            st.markdown(f"**Model used:** `{selected_model_name}`")
            st.markdown(f"**Prediction:** `{'Churn' if is_churn else 'No Churn'}`")
            st.markdown(f"**Churn probability:** `{probability:.4f}`")
            st.markdown(f"**Retain probability:** `{1 - probability:.4f}`")
            prob_df = pd.DataFrame(
                {"Outcome": ["No Churn", "Churn"], "Probability": [1 - probability, probability]}
            )
            st.bar_chart(prob_df.set_index("Outcome"))

        # All models comparison for same customer
        if len(available_models) > 1:
            st.markdown("---")
            with st.expander("📊 See this customer's score across ALL models"):
                all_scores = []
                for mname, mpath in available_models.items():
                    mdl = get_model(str(mpath), mpath.stat().st_mtime)
                    out = predict_records(get_spark(), mdl, [record])[0]
                    all_scores.append({
                        "Model": mname,
                        "Churn Probability": f"{out['churn_probability']:.4f}",
                        "Prediction": out["prediction"],
                        "Best Model?": "🏆 Yes" if mname == best_model_name else "—",
                    })
                st.dataframe(
                    pd.DataFrame(all_scores), use_container_width=True, hide_index=True
                )
                prob_compare = pd.DataFrame([
                    {"Model": r["Model"], "Churn Probability": float(r["Churn Probability"])}
                    for r in all_scores
                ]).set_index("Model")
                st.bar_chart(prob_compare)

# ════════════════════════════════════════════════════════
#  TAB 5 — BATCH SCORING
# ════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("📁 Batch Scoring — Score Multiple Customers at Once")
    st.markdown(
        "Upload a CSV file containing customer feature columns. "
        "`customerID` and `Churn` columns are optional and will be preserved in the output."
    )

    b_sel_col, b_info_col = st.columns([2, 2])
    with b_sel_col:
        batch_model_name = st.selectbox(
            "🤖 Select Model for Batch Scoring",
            options=list(available_models.keys()),
            index=default_index,
            key="batch_model_select",
        )
    with b_info_col:
        _bm = models_data.get(batch_model_name, {})
        if _bm:
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Accuracy", f"{_bm.get('accuracy', 0):.1%}")
            bc2.metric("F1-Score", f"{_bm.get('f1', 0):.3f}")
            bc3.metric("ROC AUC", f"{_bm.get('roc_auc', 0):.3f}")

    batch_model_path = available_models[batch_model_name]
    active_batch_model = get_model(
        str(batch_model_path), batch_model_path.stat().st_mtime
    )

    st.markdown("**Required feature columns:**")
    st.code(", ".join(FEATURES), language=None)

    uploaded = st.file_uploader("Upload Customer CSV", type="csv", key="batch_csv")

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            missing_cols = sorted(set(FEATURES) - set(batch_df.columns))
            if missing_cols:
                st.error(f"❌ Missing required columns: `{', '.join(missing_cols)}`")
            else:
                with st.spinner(f"Scoring {len(batch_df):,} customers with {batch_model_name}…"):
                    scores = predict_records(
                        get_spark(),
                        active_batch_model,
                        batch_df[FEATURES].to_dict(orient="records"),
                    )

                result = batch_df.copy()
                result["churn_probability"] = [round(x["churn_probability"], 4) for x in scores]
                result["predicted_churn"] = [x["prediction"] for x in scores]

                # Summary stats
                total_customers = len(result)
                predicted_churn = (result["predicted_churn"] == "Yes").sum()
                avg_prob = result["churn_probability"].mean()

                st.markdown("---")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Customers", f"{total_customers:,}")
                s2.metric("Predicted to Churn", f"{predicted_churn:,}")
                s3.metric("Churn Rate", f"{predicted_churn / total_customers:.1%}")
                s4.metric("Avg Churn Probability", f"{avg_prob:.3f}")

                # High-risk customers
                high_risk = result[result["churn_probability"] >= 0.7].shape[0]
                med_risk = result[(result["churn_probability"] >= 0.4) & (result["churn_probability"] < 0.7)].shape[0]
                low_risk = result[result["churn_probability"] < 0.4].shape[0]

                r1, r2, r3 = st.columns(3)
                r1.metric("🔴 High Risk (≥70%)", f"{high_risk:,}")
                r2.metric("🟡 Medium Risk (40–70%)", f"{med_risk:,}")
                r3.metric("🟢 Low Risk (<40%)", f"{low_risk:,}")

                # Probability distribution chart
                st.markdown("#### Churn Probability Distribution")
                hist_bins = pd.cut(
                    result["churn_probability"],
                    bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    labels=["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                            "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"],
                ).value_counts().sort_index()
                st.bar_chart(hist_bins)

                st.markdown("---")
                st.markdown("#### Scored Results")
                st.dataframe(result, use_container_width=True, hide_index=True)

                st.download_button(
                    label=f"⬇️ Download Scored CSV ({batch_model_name})",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name=f"churn_predictions_{batch_model_name.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    type="primary",
                )

        except Exception as exc:
            st.error(f"Error scoring file: {exc}")
