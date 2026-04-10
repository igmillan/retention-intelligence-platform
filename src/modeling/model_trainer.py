"""
Model training and benchmarking module for Case 01.

This module trains and evaluates multiple churn prediction models using a
leakage-safe preprocessing pipeline, stratified data splits, and a business-
aware metric stack.

Design principles
-----------------
- Keep preprocessing inside the modeling pipeline.
- Exclude non-modelable and leakage-prone columns.
- Benchmark interpretable and nonlinear models consistently.
- Export comparable metrics and user-level prediction outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import FEATURES_DATA_DIR, MODELING_CONFIG, REPORTS_DIR, SEED
from src.io_utils import ensure_directory, load_csv, save_csv
from src.logging_utils import get_logger
from src.validation import (
    validate_non_empty_dataframe,
    validate_required_columns,
)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class ModelingArtifacts:
    """
    Container for modeling outputs.

    Attributes
    ----------
    benchmark_results : pd.DataFrame
        Benchmark metrics by model and split.
    prediction_table : pd.DataFrame
        User-level predictions for validation and test splits.
    champion_model_name : str
        Name of the selected champion model.
    """

    benchmark_results: pd.DataFrame
    prediction_table: pd.DataFrame
    champion_model_name: str


class ChurnModelTrainer:
    """
    Train and benchmark churn prediction models.

    Parameters
    ----------
    feature_table_path : str | None, optional
        Optional override path for the feature table CSV.
    seed : int, optional
        Random seed for reproducibility.
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
        self.logger = get_logger("case01.modeling")

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

    def load_feature_table(self) -> pd.DataFrame:
        """
        Load the feature table.

        Returns
        -------
        pd.DataFrame
            Feature table DataFrame.
        """
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

    def prepare_model_inputs(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare model inputs from the feature table.

        Parameters
        ----------
        df : pd.DataFrame
            Full feature table.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, pd.Series]
            Predictors X, target y, and user IDs.
        """
        feature_columns = [
            col for col in df.columns if col not in self.excluded_from_model
        ]
        X = df[feature_columns].copy()
        y = df[self.target_column].copy()
        user_ids = df[self.id_column].copy()

        expected_features = set(self.categorical_features + self.numeric_features)
        actual_features = set(X.columns)

        if expected_features != actual_features:
            missing = expected_features - actual_features
            extra = actual_features - expected_features
            raise ValueError(
                f"Feature mismatch detected. Missing: {missing}. Extra: {extra}."
            )

        self.logger.info(
            "Prepared modeling inputs with %s predictor columns.",
            len(feature_columns),
        )
        return X, y, user_ids

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        user_ids: pd.Series,
    ) -> dict[str, Any]:
        """
        Split data into train, validation, and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor matrix.
        y : pd.Series
            Target series.
        user_ids : pd.Series
            User identifier series.

        Returns
        -------
        dict[str, Any]
            Dictionary containing split datasets.
        """
        test_size = MODELING_CONFIG["test_size"]
        validation_size = MODELING_CONFIG["validation_size"]

        X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
            X,
            y,
            user_ids,
            test_size=test_size + validation_size,
            stratify=y,
            random_state=self.seed,
        )

        relative_validation_size = validation_size / (test_size + validation_size)

        X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
            X_temp,
            y_temp,
            ids_temp,
            test_size=1 - relative_validation_size,
            stratify=y_temp,
            random_state=self.seed,
        )

        split_data = {
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

        self.logger.info(
            (
                "Split data | train=%s | validation=%s | test=%s"
            ),
            f"{len(X_train):,}",
            f"{len(X_val):,}",
            f"{len(X_test):,}",
        )
        return split_data

    def build_preprocessor(self) -> ColumnTransformer:
        """
        Build the preprocessing transformer.

        Returns
        -------
        ColumnTransformer
            Preprocessing transformer for numeric and categorical features.
        """
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.numeric_features),
                ("categorical", categorical_pipeline, self.categorical_features),
            ]
        )

        return preprocessor

    def build_model_registry(self) -> dict[str, Any]:
        """
        Build the benchmark model registry.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping model names to classifier instances.
        """
        model_registry: dict[str, Any] = {
            "logistic_regression": LogisticRegression(
                random_state=self.seed,
                max_iter=2000,
                class_weight="balanced",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                random_state=self.seed,
                class_weight="balanced",
                n_jobs=-1,
            ),
        }

        if XGBOOST_AVAILABLE:
            model_registry["xgboost"] = XGBClassifier(
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

        self.logger.info(
            "Initialized benchmark registry with models: %s",
            ", ".join(model_registry.keys()),
        )
        return model_registry

    @staticmethod
    def compute_lift_at_top_k(
        y_true: pd.Series,
        y_prob: np.ndarray,
        top_k: float = 0.10,
    ) -> float:
        """
        Compute lift at top-k fraction of scored users.

        Parameters
        ----------
        y_true : pd.Series
            Ground truth binary labels.
        y_prob : np.ndarray
            Predicted probabilities.
        top_k : float, optional
            Top fraction of users to consider.

        Returns
        -------
        float
            Lift at top-k.
        """
        evaluation_df = pd.DataFrame(
            {"y_true": y_true.to_numpy(), "y_prob": y_prob}
        ).sort_values("y_prob", ascending=False)

        n_top = max(1, int(len(evaluation_df) * top_k))
        top_rate = evaluation_df.head(n_top)["y_true"].mean()
        base_rate = evaluation_df["y_true"].mean()

        if base_rate == 0:
            return 0.0

        return float(top_rate / base_rate)

    def evaluate_predictions(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        threshold: float = 0.50,
    ) -> dict[str, float]:
        """
        Evaluate model predictions.

        Parameters
        ----------
        y_true : pd.Series
            Ground truth labels.
        y_prob : np.ndarray
            Predicted probabilities.
        threshold : float, optional
            Classification threshold.

        Returns
        -------
        dict[str, float]
            Evaluation metrics dictionary.
        """
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "lift_at_10pct": self.compute_lift_at_top_k(y_true, y_prob, top_k=0.10),
        }
        return metrics

    def benchmark_models(self) -> ModelingArtifacts:
        """
        Train and benchmark all models.

        Returns
        -------
        ModelingArtifacts
            Benchmark results, prediction table, and champion model name.
        """
        df = self.load_feature_table()
        X, y, user_ids = self.prepare_model_inputs(df)
        splits = self.split_data(X, y, user_ids)

        preprocessor = self.build_preprocessor()
        model_registry = self.build_model_registry()

        benchmark_rows: list[dict[str, Any]] = []
        prediction_rows: list[pd.DataFrame] = []

        for model_name, estimator in model_registry.items():
            self.logger.info("Training model: %s", model_name)

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )

            pipeline.fit(splits["X_train"], splits["y_train"])

            for split_name, X_split, y_split, ids_split in [
                ("validation", splits["X_val"], splits["y_val"], splits["ids_val"]),
                ("test", splits["X_test"], splits["y_test"], splits["ids_test"]),
            ]:
                y_prob = pipeline.predict_proba(X_split)[:, 1]
                metrics = self.evaluate_predictions(y_split, y_prob)

                benchmark_row = {
                    "model_name": model_name,
                    "split": split_name,
                    **metrics,
                }
                benchmark_rows.append(benchmark_row)

                prediction_df = pd.DataFrame(
                    {
                        "user_id": ids_split.to_numpy(),
                        "y_true": y_split.to_numpy(),
                        "predicted_probability": y_prob,
                        "predicted_label_050": (y_prob >= 0.50).astype(int),
                        "model_name": model_name,
                        "split": split_name,
                    }
                )
                prediction_rows.append(prediction_df)

                self.logger.info(
                    (
                        "Model=%s | split=%s | roc_auc=%.4f | pr_auc=%.4f "
                        "| precision=%.4f | recall=%.4f | lift@10%%=%.4f"
                    ),
                    model_name,
                    split_name,
                    metrics["roc_auc"],
                    metrics["pr_auc"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["lift_at_10pct"],
                )

        benchmark_results = pd.DataFrame(benchmark_rows)
        prediction_table = pd.concat(prediction_rows, ignore_index=True)

        champion_model_name = self.select_champion_model(benchmark_results)

        self.logger.info("Selected champion model: %s", champion_model_name)

        return ModelingArtifacts(
            benchmark_results=benchmark_results,
            prediction_table=prediction_table,
            champion_model_name=champion_model_name,
        )

    def select_champion_model(self, benchmark_results: pd.DataFrame) -> str:
        """
        Select the champion model using validation performance.

        Parameters
        ----------
        benchmark_results : pd.DataFrame
            Benchmark metrics table.

        Returns
        -------
        str
            Champion model name.
        """
        validation_results = benchmark_results[
            benchmark_results["split"] == "validation"
        ].copy()

        validation_results["selection_score"] = (
            0.40 * validation_results["roc_auc"]
            + 0.25 * validation_results["pr_auc"]
            + 0.20 * validation_results["lift_at_10pct"]
            + 0.15 * validation_results["f1"]
        )

        champion_row = validation_results.sort_values(
            "selection_score",
            ascending=False,
        ).iloc[0]

        return str(champion_row["model_name"])

    def export_modeling_outputs(self, artifacts: ModelingArtifacts) -> None:
        """
        Export modeling benchmark outputs.

        Parameters
        ----------
        artifacts : ModelingArtifacts
            Modeling artifacts container.
        """
        ensure_directory(REPORTS_DIR)

        benchmark_path = REPORTS_DIR / "model_benchmark_results.csv"
        predictions_path = REPORTS_DIR / "model_predictions.csv"

        save_csv(artifacts.benchmark_results, benchmark_path, index=False)
        save_csv(artifacts.prediction_table, predictions_path, index=False)

        self.logger.info("Exported benchmark results to %s", benchmark_path)
        self.logger.info("Exported prediction table to %s", predictions_path)

    def run(self, export_outputs: bool = True) -> ModelingArtifacts:
        """
        Execute the full modeling benchmark workflow.

        Parameters
        ----------
        export_outputs : bool, optional
            Whether to persist benchmark outputs.

        Returns
        -------
        ModelingArtifacts
            Modeling outputs.
        """
        artifacts = self.benchmark_models()

        if export_outputs:
            self.export_modeling_outputs(artifacts)

        return artifacts