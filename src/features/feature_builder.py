"""
Feature engineering module for Case 01.

This module transforms raw simulation outputs into a modeling-ready,
leakage-aware feature table for 30-day churn prediction.

Design principles
-----------------
- Only include information available at scoring time.
- Exclude direct target leakage and simulation-only latent shortcuts.
- Keep features interpretable and business-aligned.
- Keep categorical variables raw; encoding will happen in modeling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pathlib import Path
from config.settings import FEATURES_DATA_DIR, RAW_DATA_DIR
from src.io_utils import ensure_directory, load_csv, save_csv
from src.logging_utils import get_logger
from src.validation import (
    validate_no_nulls_in_columns,
    validate_non_empty_dataframe,
    validate_required_columns,
)


class ChurnFeatureBuilder:
    """
    Build the official feature table for churn modeling.

    Parameters
    ----------
    raw_data_path : str | None, optional
        Optional path override for the simulation master table.
    """

    def __init__(self, raw_data_path: str | None = None) -> None:
        self.raw_data_path = raw_data_path or str(
            RAW_DATA_DIR / "simulation_master_table.csv"
        )
        self.logger = get_logger("case01.features")

        self.required_input_columns = [
            "user_id",
            "signup_date",
            "tenure_days",
            "behavioral_segment",
            "plan",
            "primary_device",
            "region",
            "age_group",
            "bundle_type",
            "num_services",
            "sessions_last_30d",
            "active_days_last_30d",
            "watch_time_last_30d",
            "avg_session_duration_min",
            "completion_rate",
            "skip_rate",
            "content_diversity_score",
            "feature_usage_score",
            "search_activity_last_30d",
            "watchlist_additions_last_30d",
            "days_since_last_session",
            "churn_probability",
            "will_churn_30d",
            "churn_date",
        ]

        self.model_feature_columns = [
            "tenure_days",
            "plan",
            "primary_device",
            "region",
            "age_group",
            "bundle_type",
            "num_services",
            "sessions_last_30d",
            "active_days_last_30d",
            "watch_time_last_30d",
            "avg_session_duration_min",
            "completion_rate",
            "skip_rate",
            "content_diversity_score",
            "feature_usage_score",
            "search_activity_last_30d",
            "watchlist_additions_last_30d",
            "days_since_last_session",
            "sessions_per_active_day",
            "watch_time_per_active_day",
            "searches_per_session",
            "watchlist_additions_per_session",
        ]

        self.analysis_only_columns = [
            "user_id",
            "signup_date",
            "behavioral_segment",
            "churn_probability",
            "churn_date",
            "will_churn_30d",
        ]

    @staticmethod
    def _safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
        fill_value: float = 0.0,
    ) -> pd.Series:
        """
        Safely divide two pandas Series.

        Parameters
        ----------
        numerator : pd.Series
            Numerator values.
        denominator : pd.Series
            Denominator values.
        fill_value : float, optional
            Value to use where denominator is zero.

        Returns
        -------
        pd.Series
            Safely divided result.
        """
        result = np.where(denominator > 0, numerator / denominator, fill_value)
        return pd.Series(result, index=numerator.index)

    def load_simulation_master(self) -> pd.DataFrame:
        """
        Load the simulation master table.

        Returns
        -------
        pd.DataFrame
            Simulation master table.
        """
        df = load_csv(Path(self.raw_data_path))
        validate_non_empty_dataframe(df, dataset_name="simulation_master_table")
        validate_required_columns(
            df,
            required_columns=self.required_input_columns,
            dataset_name="simulation_master_table",
        )

        self.logger.info(
            "Loaded simulation master table with %s rows and %s columns.",
            f"{len(df):,}",
            len(df.columns),
        )
        return df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived behavioral features.

        Parameters
        ----------
        df : pd.DataFrame
            Input simulation master table.

        Returns
        -------
        pd.DataFrame
            DataFrame with derived features added.
        """
        feature_df = df.copy()

        feature_df["sessions_per_active_day"] = self._safe_divide(
            numerator=feature_df["sessions_last_30d"],
            denominator=feature_df["active_days_last_30d"],
            fill_value=0.0,
        )

        feature_df["watch_time_per_active_day"] = self._safe_divide(
            numerator=feature_df["watch_time_last_30d"],
            denominator=feature_df["active_days_last_30d"],
            fill_value=0.0,
        )

        feature_df["searches_per_session"] = self._safe_divide(
            numerator=feature_df["search_activity_last_30d"],
            denominator=feature_df["sessions_last_30d"],
            fill_value=0.0,
        )

        feature_df["watchlist_additions_per_session"] = self._safe_divide(
            numerator=feature_df["watchlist_additions_last_30d"],
            denominator=feature_df["sessions_last_30d"],
            fill_value=0.0,
        )

        derived_cols = [
            "sessions_per_active_day",
            "watch_time_per_active_day",
            "searches_per_session",
            "watchlist_additions_per_session",
        ]

        feature_df[derived_cols] = feature_df[derived_cols].round(4)

        self.logger.info("Created derived feature set.")
        return feature_df

    def select_feature_table_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select final columns for the official feature table.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with raw and derived features.

        Returns
        -------
        pd.DataFrame
            Final feature table.
        """
        final_columns = (
            ["user_id"]
            + self.model_feature_columns
            + ["will_churn_30d"]
            + ["signup_date", "behavioral_segment", "churn_probability", "churn_date"]
        )

        feature_table = df[final_columns].copy()

        self.logger.info(
            "Selected final feature table with %s columns.",
            len(feature_table.columns),
        )
        return feature_table

    def validate_feature_table(self, df: pd.DataFrame) -> None:
        """
        Validate the final feature table.

        Parameters
        ----------
        df : pd.DataFrame
            Final feature table.

        Raises
        ------
        ValueError
            If critical validation checks fail.
        """
        validate_non_empty_dataframe(df, dataset_name="feature_table")

        validate_required_columns(
            df,
            required_columns=["user_id", "will_churn_30d"] + self.model_feature_columns,
            dataset_name="feature_table",
        )

        validate_no_nulls_in_columns(
            df,
            columns=[
                "user_id",
                "tenure_days",
                "plan",
                "sessions_last_30d",
                "will_churn_30d",
            ],
            dataset_name="feature_table",
        )

        if df["user_id"].nunique() != len(df):
            raise ValueError("feature_table contains duplicate user_id values.")

        if not df["will_churn_30d"].isin([0, 1]).all():
            raise ValueError("will_churn_30d must be binary.")

        if (df["sessions_per_active_day"] < 0).any():
            raise ValueError("sessions_per_active_day contains negative values.")

        if (df["watch_time_per_active_day"] < 0).any():
            raise ValueError("watch_time_per_active_day contains negative values.")

        leakage_columns = {"churn_probability", "churn_date"}
        model_feature_set = set(self.model_feature_columns)
        if leakage_columns & model_feature_set:
            raise ValueError("Leakage columns detected inside model feature set.")

        self.logger.info("Feature table passed validation checks.")

    def build_feature_table(self, export_output: bool = True) -> pd.DataFrame:
        """
        Build the official feature table.

        Parameters
        ----------
        export_output : bool, optional
            Whether to export the feature table to disk.

        Returns
        -------
        pd.DataFrame
            Final feature table.
        """
        ensure_directory(FEATURES_DATA_DIR)

        df = self.load_simulation_master()
        df = self.create_derived_features(df)
        feature_table = self.select_feature_table_columns(df)
        self.validate_feature_table(feature_table)

        if export_output:
            save_csv(
                feature_table,
                FEATURES_DATA_DIR / "feature_table.csv",
                index=False,
            )
            self.logger.info(
                "Exported feature table to %s",
                FEATURES_DATA_DIR / "feature_table.csv",
            )

        return feature_table

    def summarize_feature_table(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Summarize the feature table for quick QA.

        Parameters
        ----------
        df : pd.DataFrame
            Final feature table.

        Returns
        -------
        dict[str, Any]
            Summary dictionary.
        """
        summary = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "target_rate": float(df["will_churn_30d"].mean()),
            "n_model_features": len(self.model_feature_columns),
            "categorical_features": [
                "plan",
                "primary_device",
                "region",
                "age_group",
                "bundle_type",
            ],
            "analysis_only_columns": self.analysis_only_columns,
        }

        self.logger.info(
            "Feature summary | rows=%s | columns=%s | target_rate=%.2f%%",
            f"{summary['n_rows']:,}",
            summary["n_columns"],
            summary["target_rate"] * 100,
        )
        return summary