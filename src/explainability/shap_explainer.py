"""
SHAP explainability module for Case 01.

This module rebuilds the champion XGBoost model in a reproducible way,
computes SHAP values, exports global and local explainability outputs,
and generates recruiter-friendly explainability artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config.settings import ASSETS_DIR, FEATURES_DATA_DIR, REPORTS_DIR, SEED
from src.io_utils import ensure_directory, load_csv, save_csv
from src.logging_utils import get_logger
from src.validation import validate_non_empty_dataframe, validate_required_columns


class ChurnShapExplainer:
    """
    Compute SHAP explainability outputs for the champion XGBoost model.
    """

    def __init__(
        self,
        feature_table_path: str | None = None,
        seed: int = SEED,
    ) -> None:
        self.feature_table_path = feature_table_path or str(
            FEATURES_DATA_DIR / "feature_table.csv"
        )
        self.seed = seed
        self.logger = get_logger("case01.explainability")

        self.target_column = "will_churn_30d"
        self.id_column = "user_id"

        self.excluded_from_model = [
            "user_id",
            "signup_date",
            "behavioral_segment",
            "churn_probability",
            "churn_date",
            "will_churn_30d",
        ]

        self.categorical_features = [
            "plan",
            "primary_device",
            "region",
            "age_group",
            "bundle_type",
        ]

        self.numeric_features = [
            "tenure_days",
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

        self.pipeline: Pipeline | None = None
        self.feature_names_transformed: list[str] | None = None
        self.explainer: shap.TreeExplainer | None = None
        self.X_test_raw: pd.DataFrame | None = None
        self.X_test_transformed: np.ndarray | None = None
        self.test_index_data: pd.DataFrame | None = None
        self.shap_values: np.ndarray | None = None
        self.test_predicted_probability: np.ndarray | None = None

    def load_feature_table(self) -> pd.DataFrame:
        """Load the feature table."""
        df = load_csv(Path(self.feature_table_path))
        validate_non_empty_dataframe(df, dataset_name="feature_table")
        validate_required_columns(
            df,
            required_columns=[self.id_column, self.target_column],
            dataset_name="feature_table",
        )
        self.logger.info(
            "Loaded feature table with %s rows and %s columns.",
            f"{len(df):,}",
            len(df.columns),
        )
        return df

    def prepare_inputs(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare X, y, and user IDs."""
        feature_columns = [
            col for col in df.columns if col not in self.excluded_from_model
        ]
        X = df[feature_columns].copy()
        y = df[self.target_column].copy()
        user_ids = df[self.id_column].copy()

        self.logger.info(
            "Prepared explainability inputs with %s predictor columns.",
            len(feature_columns),
        )
        return X, y, user_ids

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        user_ids: pd.Series,
    ) -> dict[str, Any]:
        """Split into train, validation, and test."""
        X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
            X,
            y,
            user_ids,
            test_size=0.40,
            stratify=y,
            random_state=self.seed,
        )

        X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
            X_temp,
            y_temp,
            ids_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=self.seed,
        )

        self.logger.info(
            "Split data for explainability | train=%s | validation=%s | test=%s",
            f"{len(X_train):,}",
            f"{len(X_val):,}",
            f"{len(X_test):,}",
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "ids_train": ids_train,
            "X_val": X_val,
            "y_val": y_val,
            "ids_val": ids_val,
            "X_test": X_test,
            "y_test": y_test,
            "ids_test": ids_test,
        }

    def build_preprocessor(self) -> ColumnTransformer:
        """Build preprocessing transformer."""
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.numeric_features),
                ("categorical", categorical_pipeline, self.categorical_features),
            ]
        )

    def build_champion_pipeline(self) -> Pipeline:
        """Build champion XGBoost pipeline."""
        preprocessor = self.build_preprocessor()

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=self.seed,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def fit_champion_model(self, splits: dict[str, Any]) -> None:
        """Fit champion pipeline and prepare test artifacts."""
        self.pipeline = self.build_champion_pipeline()
        self.pipeline.fit(splits["X_train"], splits["y_train"])

        preprocessor = self.pipeline.named_steps["preprocessor"]
        model = self.pipeline.named_steps["model"]

        self.X_test_raw = splits["X_test"].copy()
        self.X_test_transformed = preprocessor.transform(splits["X_test"])
        self.test_predicted_probability = self.pipeline.predict_proba(
            splits["X_test"]
        )[:, 1]

        self.test_index_data = pd.DataFrame(
            {
                "user_id": splits["ids_test"].to_numpy(),
                "y_true": splits["y_test"].to_numpy(),
                "predicted_probability": self.test_predicted_probability,
            }
        )

        self.feature_names_transformed = list(preprocessor.get_feature_names_out())
        self.explainer = shap.TreeExplainer(model)

        self.logger.info("Fitted champion XGBoost pipeline for explainability.")

    def compute_shap_values(self) -> None:
        """Compute SHAP values for transformed test set."""
        if self.explainer is None or self.X_test_transformed is None:
            raise ValueError("Champion model must be fitted before computing SHAP values.")

        shap_output = self.explainer.shap_values(self.X_test_transformed)
        if isinstance(shap_output, list):
            shap_output = shap_output[0]

        self.shap_values = np.array(shap_output)

        self.logger.info(
            "Computed SHAP values for %s test rows.",
            f"{len(self.shap_values):,}",
        )

    def build_global_importance_table(self) -> pd.DataFrame:
        """Build SHAP global importance table."""
        if self.shap_values is None or self.feature_names_transformed is None:
            raise ValueError("SHAP values must be computed before building importance table.")

        importance_df = pd.DataFrame(
            {
                "feature_name": self.feature_names_transformed,
                "mean_abs_shap": np.abs(self.shap_values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        self.logger.info("Built SHAP global importance table.")
        return importance_df

    def build_local_top_drivers_table(self, top_n: int = 5) -> pd.DataFrame:
        """Build local top drivers table for each test user."""
        if (
            self.shap_values is None
            or self.feature_names_transformed is None
            or self.test_index_data is None
        ):
            raise ValueError("Local explanation artifacts are not ready.")

        rows: list[dict[str, Any]] = []

        for idx in range(len(self.test_index_data)):
            shap_row = self.shap_values[idx]
            abs_order = np.argsort(np.abs(shap_row))[::-1][:top_n]

            driver_names = [self.feature_names_transformed[i] for i in abs_order]
            driver_values = [float(shap_row[i]) for i in abs_order]

            row = {
                "user_id": self.test_index_data.iloc[idx]["user_id"],
                "y_true": int(self.test_index_data.iloc[idx]["y_true"]),
                "predicted_probability": float(
                    self.test_index_data.iloc[idx]["predicted_probability"]
                ),
            }

            for rank, (name, value) in enumerate(zip(driver_names, driver_values), start=1):
                row[f"top_driver_{rank}"] = name
                row[f"top_driver_{rank}_shap"] = value

            rows.append(row)

        local_df = pd.DataFrame(rows)
        self.logger.info("Built SHAP local top drivers table.")
        return local_df

    def plot_shap_summary(self) -> None:
        """Generate and save SHAP summary plot."""
        if self.shap_values is None or self.X_test_transformed is None or self.feature_names_transformed is None:
            raise ValueError("SHAP artifacts must be ready before plotting.")

        ensure_directory(ASSETS_DIR)

        plt.figure()
        shap.summary_plot(
            self.shap_values,
            self.X_test_transformed,
            feature_names=self.feature_names_transformed,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(ASSETS_DIR / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info("Saved SHAP summary plot.")

    def plot_shap_bar_importance(self) -> None:
        """Generate and save SHAP bar importance plot."""
        if self.shap_values is None or self.X_test_transformed is None or self.feature_names_transformed is None:
            raise ValueError("SHAP artifacts must be ready before plotting.")

        ensure_directory(ASSETS_DIR)

        plt.figure()
        shap.summary_plot(
            self.shap_values,
            self.X_test_transformed,
            feature_names=self.feature_names_transformed,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(ASSETS_DIR / "shap_bar_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info("Saved SHAP bar importance plot.")

    def export_outputs(
        self,
        global_importance_df: pd.DataFrame,
        local_drivers_df: pd.DataFrame,
    ) -> None:
        """Export explainability outputs."""
        ensure_directory(REPORTS_DIR)

        save_csv(
            global_importance_df,
            REPORTS_DIR / "shap_global_importance.csv",
            index=False,
        )
        save_csv(
            local_drivers_df,
            REPORTS_DIR / "shap_local_top_drivers.csv",
            index=False,
        )

        self.logger.info("Exported SHAP explainability tables.")

    def run(self, export_outputs: bool = True) -> dict[str, pd.DataFrame]:
        """
        Execute full explainability workflow.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary with global and local explainability tables.
        """
        df = self.load_feature_table()
        X, y, user_ids = self.prepare_inputs(df)
        splits = self.split_data(X, y, user_ids)

        self.fit_champion_model(splits)
        self.compute_shap_values()

        global_importance_df = self.build_global_importance_table()
        local_drivers_df = self.build_local_top_drivers_table(top_n=5)

        self.plot_shap_summary()
        self.plot_shap_bar_importance()

        if export_outputs:
            self.export_outputs(global_importance_df, local_drivers_df)

        return {
            "shap_global_importance": global_importance_df,
            "shap_local_top_drivers": local_drivers_df,
        }