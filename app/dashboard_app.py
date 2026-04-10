"""
Streamlit dashboard for Case 01 — User Retention & Churn Prediction.

Recruiter-facing and stakeholder-friendly analytical product layer for the
Retention Intelligence Framework.

Run
---
streamlit run app/dashboard_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Case 01 — Retention Intelligence",
    page_icon="📈",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Lightweight styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.35rem;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            opacity: 0.82;
            margin-bottom: 1.5rem;
            max-width: 980px;
        }

        .section-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.7;
            margin-bottom: 0.25rem;
        }

        .insight-box {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1rem 1rem 0.9rem 1rem;
            background: rgba(255,255,255,0.02);
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }

        .kpi-caption {
            font-size: 0.82rem;
            opacity: 0.7;
            margin-top: -0.2rem;
        }

        .sidebar-copy {
            font-size: 0.9rem;
            opacity: 0.85;
            line-height: 1.5;
        }

        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 0.85rem 1rem 0.85rem 1rem;
            border-radius: 14px;
        }

        div[data-testid="metric-container"] label {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_BENCHMARK_PATH = REPORTS_DIR / "model_benchmark_results.csv"
SHAP_GLOBAL_PATH = REPORTS_DIR / "shap_global_importance.csv"
DECISION_TABLE_PATH = REPORTS_DIR / "decision_table.csv"
DECISION_TIER_SUMMARY_PATH = REPORTS_DIR / "decision_tier_summary.csv"
DECISION_PLAYBOOK_SUMMARY_PATH = REPORTS_DIR / "decision_playbook_summary.csv"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def safe_round(value: float, digits: int = 2) -> float:
    """Safely round numeric value."""
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def format_currency(value: float) -> str:
    """Format numeric value as currency."""
    return f"${value:,.0f}"


def normalize_feature_name(feature_name: str) -> str:
    """Clean transformed feature names for UI display."""
    if not isinstance(feature_name, str):
        return str(feature_name)

    clean = feature_name.replace("numeric__", "").replace("categorical__", "")
    clean = clean.replace("_", " ")
    return clean.title()


def normalize_playbook_name(playbook: str) -> str:
    """Format playbook names for display."""
    if not isinstance(playbook, str):
        return str(playbook)
    return playbook.replace("_", " ").title()


def normalize_model_name(model_name: str) -> str:
    """Format model names for display."""
    if not isinstance(model_name, str):
        return str(model_name)
    return model_name.replace("_", " ").title()


def get_champion_model(benchmark_df: pd.DataFrame) -> str:
    """Infer champion model from validation composite score."""
    validation_df = benchmark_df[benchmark_df["split"] == "validation"].copy()

    if validation_df.empty:
        return "unknown"

    validation_df["selection_score"] = (
        0.40 * validation_df["roc_auc"]
        + 0.25 * validation_df["pr_auc"]
        + 0.20 * validation_df["lift_at_10pct"]
        + 0.15 * validation_df["f1"]
    )

    champion_row = validation_df.sort_values(
        "selection_score", ascending=False
    ).iloc[0]

    return str(champion_row["model_name"])


def get_test_metrics_for_model(
    benchmark_df: pd.DataFrame,
    model_name: str,
) -> pd.Series:
    """Return test metrics for selected model."""
    filtered = benchmark_df[
        (benchmark_df["model_name"] == model_name)
        & (benchmark_df["split"] == "test")
    ]

    if filtered.empty:
        return pd.Series(dtype="object")

    return filtered.iloc[0]


def prepare_benchmark_display(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare benchmark table for display."""
    display_df = benchmark_df.copy()

    percentage_cols = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    for col in percentage_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)

    if "lift_at_10pct" in display_df.columns:
        display_df["lift_at_10pct"] = display_df["lift_at_10pct"].round(3)

    display_df["model_name_display"] = display_df["model_name"].apply(normalize_model_name)
    display_df["split_display"] = display_df["split"].astype(str).str.title()

    return display_df.sort_values(["split", "model_name"]).reset_index(drop=True)


def prepare_shap_display(shap_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Prepare SHAP importance display table."""
    display_df = shap_df.copy()
    display_df["feature_display"] = display_df["feature_name"].apply(normalize_feature_name)
    display_df["mean_abs_shap"] = display_df["mean_abs_shap"].round(4)
    return display_df.head(top_n)


def prepare_decision_table_display(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare decision table for UI."""
    display_df = df.copy()

    if "predicted_probability" in display_df.columns:
        display_df["predicted_probability"] = display_df["predicted_probability"].round(4)

    if "revenue_at_risk" in display_df.columns:
        display_df["revenue_at_risk"] = display_df["revenue_at_risk"].round(2)

    if "priority_score" in display_df.columns:
        display_df["priority_score"] = display_df["priority_score"].round(2)

    if "top_driver" in display_df.columns:
        display_df["top_driver_display"] = display_df["top_driver"].apply(normalize_feature_name)
    elif "top_driver_1" in display_df.columns:
        display_df["top_driver_display"] = display_df["top_driver_1"].apply(normalize_feature_name)
    else:
        display_df["top_driver_display"] = "N/A"

    if "recommended_playbook" in display_df.columns:
        display_df["playbook_display"] = display_df["recommended_playbook"].apply(
            normalize_playbook_name
        )
    else:
        display_df["playbook_display"] = "N/A"

    if "risk_tier" in display_df.columns:
        display_df["risk_tier_display"] = display_df["risk_tier"].astype(str).str.title()
    else:
        display_df["risk_tier_display"] = "N/A"

    return display_df


def build_risk_color_map() -> dict[str, str]:
    """Color map for risk tiers."""
    return {
        "low": "#22C55E",
        "medium": "#EAB308",
        "high": "#F97316",
        "critical": "#DC2626",
    }


def apply_clean_plotly_theme(fig):
    """Apply consistent chart formatting."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        title_font_size=18,
        font=dict(size=13),
    )
    return fig


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
try:
    benchmark_df = load_csv(MODEL_BENCHMARK_PATH)
    shap_global_df = load_csv(SHAP_GLOBAL_PATH)
    decision_table_df = load_csv(DECISION_TABLE_PATH)
    decision_tier_summary_df = load_csv(DECISION_TIER_SUMMARY_PATH)
    decision_playbook_summary_df = load_csv(DECISION_PLAYBOOK_SUMMARY_PATH)
except FileNotFoundError as exc:
    st.error(
        "Required pipeline outputs were not found.\n\n"
        "Run these commands first from the project root:\n"
        "```bash\n"
        "python -m src.pipeline.run_simulation\n"
        "python -m src.pipeline.run_features\n"
        "python -m src.pipeline.run_modeling\n"
        "python -m src.pipeline.run_explainability\n"
        "python -m src.pipeline.run_decisioning\n"
        "```"
    )
    st.exception(exc)
    st.stop()


# -----------------------------------------------------------------------------
# Derived dashboard state
# -----------------------------------------------------------------------------
champion_model = get_champion_model(benchmark_df)
champion_model_display = normalize_model_name(champion_model)
champion_test_metrics = get_test_metrics_for_model(benchmark_df, champion_model)

decision_table_display = prepare_decision_table_display(decision_table_df)
benchmark_display = prepare_benchmark_display(benchmark_df)
shap_display = prepare_shap_display(shap_global_df, top_n=10)

risk_color_map = build_risk_color_map()

users_scored = len(decision_table_df)
critical_users = int(
    (decision_table_df["risk_tier"].astype(str).str.lower() == "critical").sum()
) if "risk_tier" in decision_table_df.columns else 0

total_revenue_at_risk = (
    float(decision_table_df["revenue_at_risk"].sum())
    if "revenue_at_risk" in decision_table_df.columns
    else 0.0
)

avg_predicted_probability = (
    float(decision_table_df["predicted_probability"].mean())
    if "predicted_probability" in decision_table_df.columns
    else 0.0
)

playbooks_activated = (
    int(decision_table_df["recommended_playbook"].nunique())
    if "recommended_playbook" in decision_table_df.columns
    else 0
)

revenue_per_scored_user = total_revenue_at_risk / users_scored if users_scored else 0


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Case 01")
st.sidebar.caption("Retention Intelligence")
st.sidebar.markdown(
    '<div class="sidebar-copy">Interactive presentation layer for subscription '
    'churn prediction, explainability, and retention prioritization.</div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Model Performance",
        "Churn Drivers",
        "Retention Prioritization",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
**Production Model**  
`{champion_model}`

**Users Scored**  
`{users_scored:,}`

**Critical Users**  
`{critical_users:,}`
"""
)

risk_filter_options = ["All"]
if "risk_tier" in decision_table_display.columns:
    risk_filter_options += sorted(
        decision_table_display["risk_tier"].dropna().astype(str).str.title().unique()
    )

selected_risk_tier = st.sidebar.selectbox("Filter Risk Tier", risk_filter_options)

plan_filter_options = ["All"]
if "plan" in decision_table_display.columns:
    plan_filter_options += sorted(
        decision_table_display["plan"].dropna().astype(str).unique()
    )

selected_plan = st.sidebar.selectbox("Filter Plan", plan_filter_options)

filtered_decision_df = decision_table_display.copy()

if selected_risk_tier != "All" and "risk_tier" in filtered_decision_df.columns:
    filtered_decision_df = filtered_decision_df[
        filtered_decision_df["risk_tier"].astype(str).str.lower()
        == selected_risk_tier.lower()
    ]

if selected_plan != "All" and "plan" in filtered_decision_df.columns:
    filtered_decision_df = filtered_decision_df[
        filtered_decision_df["plan"].astype(str) == selected_plan
    ]


# -----------------------------------------------------------------------------
# Hero header
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="section-label">Case 01 · Apple Decision Intelligence Portfolio</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-title">Retention Intelligence Platform</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">An end-to-end retention intelligence system for '
    'predicting subscription churn, explaining behavioral risk drivers, and '
    'prioritizing proactive intervention opportunities.</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Page 1 — Executive Overview
# -----------------------------------------------------------------------------
if page == "Executive Overview":
    st.subheader("Executive Overview")
    st.write(
        "This dashboard summarizes the end-to-end retention intelligence "
        "framework: predictive churn scoring, behavioral explainability, and "
        "operational retention prioritization."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Production Model", champion_model_display)
    c2.metric("ROC-AUC", f"{safe_round(champion_test_metrics.get('roc_auc', 0), 3):.3f}")
    c3.metric("PR-AUC", f"{safe_round(champion_test_metrics.get('pr_auc', 0), 3):.3f}")
    c4.metric("High-Risk Accounts", f"{critical_users:,}")
    c5.metric("Revenue Exposure", format_currency(total_revenue_at_risk))

    st.markdown(
        """
        <div class="insight-box">
        <strong>Executive Interpretation</strong><br>
        A concentrated subset of high-risk users drives disproportionate revenue exposure,
        enabling retention teams to focus intervention resources where economic impact is highest.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Risk and Revenue Overview")
    row1_col1, row1_col2 = st.columns(2)

    if not decision_tier_summary_df.empty:
        tier_plot_df = decision_tier_summary_df.copy()
        if "risk_tier" in tier_plot_df.columns:
            tier_plot_df["risk_tier"] = tier_plot_df["risk_tier"].astype(str).str.lower()

        if "users" in tier_plot_df.columns:
            fig_tier_dist = px.pie(
                tier_plot_df,
                names="risk_tier",
                values="users",
                hole=0.58,
                color="risk_tier",
                color_discrete_map=risk_color_map,
                title="Risk Tier Distribution",
            )
            apply_clean_plotly_theme(fig_tier_dist)
            row1_col1.plotly_chart(fig_tier_dist, use_container_width=True)

        revenue_col = (
            "total_revenue_at_risk"
            if "total_revenue_at_risk" in tier_plot_df.columns
            else "revenue_at_risk"
        )

        if revenue_col in tier_plot_df.columns:
            fig_revenue = px.bar(
                tier_plot_df,
                x="risk_tier",
                y=revenue_col,
                color="risk_tier",
                color_discrete_map=risk_color_map,
                title="Revenue at Risk by Tier",
            )
            fig_revenue.update_layout(
                xaxis_title="Risk Tier",
                yaxis_title="Revenue at Risk",
                showlegend=False,
            )
            apply_clean_plotly_theme(fig_revenue)
            row1_col2.plotly_chart(fig_revenue, use_container_width=True)

    st.markdown("### Executive Highlights")
    h1, h2, h3 = st.columns(3)
    h1.metric("Average Risk Score", f"{safe_round(avg_predicted_probability, 3):.3f}")
    h2.metric("Playbooks Activated", f"{playbooks_activated}")
    h3.metric("Revenue / Scored User", format_currency(revenue_per_scored_user))


# -----------------------------------------------------------------------------
# Page 2 — Model Performance
# -----------------------------------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model Performance")
    st.write(
        "Benchmark comparison across candidate models using a business-aware "
        "evaluation framework."
    )

    st.markdown(
        """
        <div class="insight-box">
        <strong>Modeling Decision</strong><br>
        XGBoost was selected as the production champion due to its strongest balance of
        ranking quality, precision-recall performance, and top-decile prioritization lift.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_options = ["roc_auc", "pr_auc", "precision", "recall", "f1", "lift_at_10pct"]
    selected_metric = st.selectbox("Select comparison metric", metric_options, index=0)

    chart_df = benchmark_display.copy()
    chart_df["highlight_group"] = chart_df["model_name"].apply(
        lambda x: "Champion" if x == champion_model else "Benchmark"
    )

    fig_benchmark = px.bar(
        chart_df,
        x="model_name_display",
        y=selected_metric,
        color="highlight_group",
        pattern_shape="split_display",
        barmode="group",
        title=f"Benchmark Comparison — {selected_metric}",
        color_discrete_map={
            "Champion": "#2563EB",
            "Benchmark": "#94A3B8",
        },
    )
    fig_benchmark.update_layout(
        xaxis_title="Model",
        yaxis_title=selected_metric,
        legend_title_text="Group",
    )
    apply_clean_plotly_theme(fig_benchmark)
    st.plotly_chart(fig_benchmark, use_container_width=True)

    st.markdown("### Benchmark Table")
    st.dataframe(
        benchmark_display[
            [
                "model_name_display",
                "split_display",
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "f1",
                "lift_at_10pct",
            ]
        ].rename(
            columns={
                "model_name_display": "Model",
                "split_display": "Split",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "lift_at_10pct": "Lift@10%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# Page 3 — Churn Drivers
# -----------------------------------------------------------------------------
elif page == "Churn Drivers":
    st.subheader("Churn Drivers")
    st.write(
        "Global model explainability reveals which behavioral patterns most strongly "
        "contribute to 30-day churn risk."
    )

    st.markdown(
        """
        <div class="insight-box">
        <strong>Behavioral Insight</strong><br>
        Churn risk is primarily driven by declining engagement intensity, recency deterioration,
        and lower content consumption quality. Behavioral signals materially outweigh demographic context.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_shap = px.bar(
        shap_display.sort_values("mean_abs_shap", ascending=True),
        x="mean_abs_shap",
        y="feature_display",
        orientation="h",
        title="Top Global Churn Drivers (SHAP)",
    )
    fig_shap.update_layout(
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
    )
    apply_clean_plotly_theme(fig_shap)
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
        <strong>Interpretation Note</strong><br>
        Global SHAP results indicate that churn risk is primarily driven by declining
        recent consumption, inactivity recency, and weaker engagement quality signals.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Driver Table")
    st.dataframe(
        shap_display[["feature_display", "mean_abs_shap"]].rename(
            columns={
                "feature_display": "Driver",
                "mean_abs_shap": "Mean |SHAP|",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# Page 4 — Retention Prioritization
# -----------------------------------------------------------------------------
elif page == "Retention Prioritization":
    st.subheader("Retention Prioritization")
    st.write(
        "Predictions are operationalized into risk tiers, value-aware prioritization, "
        "and recommended retention playbooks."
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Critical Users", f"{critical_users:,}")
    p2.metric("Avg Risk Score", f"{safe_round(avg_predicted_probability, 3):.3f}")
    p3.metric("Revenue Exposure", format_currency(total_revenue_at_risk))
    p4.metric("Intervention Types", f"{playbooks_activated}")

    st.markdown(
        """
        <div class="insight-box">
        <strong>Decision Logic</strong><br>
        The decision engine combines churn probability, SHAP-based driver logic, and
        economic value exposure to prioritize intervention actions across the user base.
        </div>
        """,
        unsafe_allow_html=True,
    )

    row2_col1, row2_col2 = st.columns([1.2, 1])

    if not decision_playbook_summary_df.empty:
        playbook_df = decision_playbook_summary_df.copy()

        playbook_users_col = "users" if "users" in playbook_df.columns else None
        playbook_name_col = (
            "recommended_playbook"
            if "recommended_playbook" in playbook_df.columns
            else ("playbook" if "playbook" in playbook_df.columns else None)
        )

        if playbook_users_col and playbook_name_col:
            playbook_df["playbook_display"] = playbook_df[playbook_name_col].apply(
                normalize_playbook_name
            )
            fig_playbook = px.bar(
                playbook_df.sort_values(playbook_users_col, ascending=False),
                x=playbook_users_col,
                y="playbook_display",
                orientation="h",
                title="Playbook Allocation",
            )
            fig_playbook.update_layout(
                xaxis_title="Users",
                yaxis_title="Playbook",
                yaxis={"categoryorder": "total ascending"},
            )
            apply_clean_plotly_theme(fig_playbook)
            row2_col1.plotly_chart(fig_playbook, use_container_width=True)

    if not decision_tier_summary_df.empty:
        tier_table_df = decision_tier_summary_df.copy()
        if "risk_tier" in tier_table_df.columns:
            tier_table_df["risk_tier"] = tier_table_df["risk_tier"].astype(str).str.title()
        row2_col2.markdown("### Tier Summary")
        row2_col2.dataframe(
            tier_table_df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### User-Level Decision Table")

    display_columns = [
        "user_id",
        "predicted_probability",
        "risk_tier_display",
        "revenue_at_risk",
        "priority_score",
        "top_driver_display",
        "playbook_display",
        "plan",
    ]
    display_columns = [col for col in display_columns if col in filtered_decision_df.columns]

    decision_preview_df = filtered_decision_df[display_columns].copy()
    decision_preview_df = decision_preview_df.rename(
        columns={
            "predicted_probability": "churn_probability",
            "risk_tier_display": "risk_tier",
            "top_driver_display": "top_driver",
            "playbook_display": "recommended_playbook",
        }
    )

    st.dataframe(
        decision_preview_df.sort_values(
            by="priority_score",
            ascending=False,
        ).head(25),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Showing the top 25 highest-priority users after active filters. "
        "This table represents the operational output of the retention decision engine."
    )