from __future__ import annotations

import pandas as pd
import numpy as np

from src.io_utils import save_csv
from src.logging_utils import get_logger


logger = get_logger("case01.decisioning")


class RetentionDecisionEngine:
    """
    Converts churn model outputs into actionable retention decisions.
    """

    DRIVER_CATEGORY_MAP = {
        "watch_time_last_30d": "engagement_decline",
        "sessions_last_30d": "engagement_decline",
        "active_days_last_30d": "engagement_decline",
        "days_since_last_session": "engagement_decline",

        "completion_rate": "consumption_quality",
        "skip_rate": "consumption_quality",
        "avg_session_duration_min": "consumption_quality",

        "content_diversity_score": "discovery_weakness",
        "search_activity_last_30d": "discovery_weakness",
        "watchlist_additions_last_30d": "discovery_weakness",
        "searches_per_session": "discovery_weakness",
        "watchlist_additions_per_session": "discovery_weakness",

        "feature_usage_score": "product_adoption",
        "num_services": "product_adoption",

        "tenure_days": "lifecycle_context",
        "plan": "lifecycle_context",
        "region": "lifecycle_context",
        "age_group": "lifecycle_context",
    }

    PLAYBOOK_MAP = {
        "engagement_decline": "re_engagement_nudge",
        "consumption_quality": "engagement_quality_recovery",
        "discovery_weakness": "content_discovery_boost",
        "product_adoption": "feature_education",
        "lifecycle_context": "commercial_retention_offer",
    }

    PLAN_VALUE_MAP = {
        "student": 7,
        "individual": 12,
        "family": 18,
        "premium": 22,
    }

    def run(
        self,
        predictions: pd.DataFrame,
        shap_local: pd.DataFrame,
        feature_table: pd.DataFrame,
    ) -> dict:

        merged = self._build_decision_table(
            predictions,
            shap_local,
            feature_table,
        )

        tier_summary = self._build_tier_summary(merged)

        playbook_summary = self._build_playbook_summary(merged)

        return {
            "decision_table": merged,
            "tier_summary": tier_summary,
            "playbook_summary": playbook_summary,
        }

    def _build_decision_table(
        self,
        predictions: pd.DataFrame,
        shap_local: pd.DataFrame,
        feature_table: pd.DataFrame,
    ) -> pd.DataFrame:

        champion_preds = predictions[
            predictions["model_name"] == "xgboost"
        ].copy()

        required_shap_cols = ["user_id", "top_driver_1", "top_driver_1_shap"]
        missing_cols = [col for col in required_shap_cols if col not in shap_local.columns]

        if missing_cols:
            raise ValueError(
                f"Missing required SHAP columns in shap_local_top_drivers.csv: {missing_cols}"
            )

        champion_preds = champion_preds[
            champion_preds["split"] == "test"
        ].copy()

        top_driver = shap_local[
        [
            "user_id",
            "top_driver_1",
            "top_driver_1_shap",
            "top_driver_2",
            "top_driver_2_shap",
            "top_driver_3",
            "top_driver_3_shap",
        ]
        ].copy()

        merged = champion_preds.merge(
            top_driver,
            on="user_id",
            how="left",
        )

        merged = merged.merge(
            feature_table[["user_id", "plan", "num_services"]],
            on="user_id",
            how="left",
        )

        merged["risk_tier"] = merged["predicted_probability"].apply(
            self._assign_risk_tier
        )

        merged["estimated_monthly_value"] = (
            merged["plan"].map(self.PLAN_VALUE_MAP)
            + merged["num_services"]
        )

        merged["revenue_at_risk"] = (
            merged["predicted_probability"]
            * merged["estimated_monthly_value"]
        )

        merged["priority_score"] = merged["revenue_at_risk"]

        merged["driver_category"] = merged.apply(
            self._determine_primary_driver_category,
            axis=1,
        )

        merged["recommended_playbook"] = merged.apply(
            self._assign_playbook,
            axis=1,
        )

        return merged.sort_values(
            by="priority_score",
            ascending=False,
        )

    def _assign_risk_tier(self, prob: float) -> str:

        if prob < 0.20:
            return "low"

        if prob < 0.40:
            return "medium"

        if prob < 0.70:
            return "high"

        return "critical"

    def _map_driver_category(self, driver: str) -> str:

        clean_driver = driver.replace("numeric__", "").replace(
            "categorical__", ""
        )

        for key in self.DRIVER_CATEGORY_MAP:
            if key in clean_driver:
                return self.DRIVER_CATEGORY_MAP[key]

        return "unknown"
    
    def _determine_primary_driver_category(
        self,
        row: pd.Series,
    ) -> str:

        categories = []

        for col in ["top_driver_1", "top_driver_2", "top_driver_3"]:

            driver = row.get(col)

            if pd.notna(driver):

                categories.append(
                    self._map_driver_category(driver)
                )

        if not categories:
            return "unknown"

        return pd.Series(categories).mode()[0]


    def _assign_playbook(
        self,
        row: pd.Series,
    ) -> str:

        if (
            row["risk_tier"] == "critical"
            and row["estimated_monthly_value"] >= 20
        ):
            return "commercial_retention_offer"

        return self.PLAYBOOK_MAP.get(
            row["driver_category"],
            "manual_review",
        )

    def _build_tier_summary(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        return (
            df.groupby("risk_tier")
            .agg(
                users=("user_id", "count"),
                avg_probability=("predicted_probability", "mean"),
                total_revenue_at_risk=("revenue_at_risk", "sum"),
            )
            .reset_index()
        )

    def _build_playbook_summary(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        return (
            df.groupby("recommended_playbook")
            .agg(
                users=("user_id", "count"),
                avg_priority=("priority_score", "mean"),
            )
            .reset_index()
        )