"""
Simulation engine for Case 01.

This module generates a synthetic subscription ecosystem with realistic user
heterogeneity, overlapping behavioral distributions, and probabilistic
30-day churn outcomes.

Design principles
-----------------
- Behavior influences churn probability.
- Churn is sampled probabilistically, not hardcoded.
- Users belong to behavioral archetypes with overlapping activity patterns.
- Outputs are exported as reusable raw-layer datasets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    DIRECTORIES_TO_CREATE,
    RAW_DATA_DIR,
    SEED,
    SIMULATION_CONFIG,
)
from src.io_utils import ensure_directory, save_csv
from src.logging_utils import get_logger
from src.validation import (
    validate_no_nulls_in_columns,
    validate_non_empty_dataframe,
    validate_required_columns,
)


class SubscriptionEcosystemSimulator:
    """
    Simulate a subscription ecosystem for churn modeling.

    Parameters
    ----------
    simulation_config : dict[str, Any] | None, optional
        Optional configuration overrides for simulation behavior.
    seed : int, optional
        Random seed for full reproducibility.

    Notes
    -----
    This simulator produces four raw-layer outputs:
    - users.csv
    - engagement_metrics.csv
    - churn_targets.csv
    - simulation_master_table.csv
    """

    def __init__(
        self,
        simulation_config: dict[str, Any] | None = None,
        seed: int = SEED,
    ) -> None:
        self.config = dict(SIMULATION_CONFIG)
        if simulation_config:
            self.config.update(simulation_config)

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.ref_date = pd.Timestamp(self.config["reference_date"])
        self.logger = get_logger("case01.simulation")

        self.segment_profiles = {
            "power": {
                "sessions_lambda": 20.0,
                "duration_mean": 54.0,
                "completion_mean": 0.84,
                "diversity_mean": 0.74,
                "feature_usage_mean": 0.72,
                "search_lambda": 11.0,
                "watchlist_lambda": 7.0,
                "recency_scale": 3.0,
                "segment_risk_offset": -0.85,
                "num_services_mean": 3.9,
            },
            "casual": {
                "sessions_lambda": 12.0,
                "duration_mean": 42.0,
                "completion_mean": 0.72,
                "diversity_mean": 0.56,
                "feature_usage_mean": 0.50,
                "search_lambda": 6.0,
                "watchlist_lambda": 4.0,
                "recency_scale": 6.5,
                "segment_risk_offset": -0.15,
                "num_services_mean": 2.8,
            },
            "passive": {
                "sessions_lambda": 7.0,
                "duration_mean": 31.0,
                "completion_mean": 0.57,
                "diversity_mean": 0.36,
                "feature_usage_mean": 0.31,
                "search_lambda": 3.0,
                "watchlist_lambda": 2.0,
                "recency_scale": 11.5,
                "segment_risk_offset": 0.42,
                "num_services_mean": 2.0,
            },
            "at_risk": {
                "sessions_lambda": 5.0,
                "duration_mean": 26.0,
                "completion_mean": 0.48,
                "diversity_mean": 0.28,
                "feature_usage_mean": 0.22,
                "search_lambda": 2.0,
                "watchlist_lambda": 1.0,
                "recency_scale": 16.0,
                "segment_risk_offset": 0.88,
                "num_services_mean": 2.2,
            },
        }

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate core simulation configuration."""
        target_churn_rate = self.config["target_churn_rate"]
        if not 0.05 <= target_churn_rate <= 0.50:
            raise ValueError("target_churn_rate must be between 0.05 and 0.50.")

        segment_mix_sum = sum(self.config["segment_mix"].values())
        if not np.isclose(segment_mix_sum, 1.0):
            raise ValueError("segment_mix probabilities must sum to 1.0.")

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Apply the logistic sigmoid transformation.

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Sigmoid-transformed values.
        """
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """
        Compute a robust z-score for a pandas Series.

        Parameters
        ----------
        series : pd.Series
            Input series.

        Returns
        -------
        pd.Series
            Standardized series. Returns zeros if standard deviation is zero.
        """
        std = series.std(ddof=0)
        if std == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean()) / std

    def _bootstrap_directories(self) -> None:
        """Create required project directories if they do not already exist."""
        for directory in DIRECTORIES_TO_CREATE:
            ensure_directory(directory)

    def generate_user_population(self) -> pd.DataFrame:
        """
        Generate the base user population.

        Returns
        -------
        pd.DataFrame
            User-level table with semi-static attributes.
        """
        n_users = self.config["n_users"]

        segment_labels = list(self.config["segment_mix"].keys())
        segment_probs = list(self.config["segment_mix"].values())

        plan_labels = list(self.config["plan_distribution"].keys())
        plan_probs = list(self.config["plan_distribution"].values())

        device_labels = list(self.config["device_distribution"].keys())
        device_probs = list(self.config["device_distribution"].values())

        region_labels = list(self.config["region_distribution"].keys())
        region_probs = list(self.config["region_distribution"].values())

        age_labels = list(self.config["age_group_distribution"].keys())
        age_probs = list(self.config["age_group_distribution"].values())

        bundle_labels = list(self.config["bundle_distribution"].keys())
        bundle_probs = list(self.config["bundle_distribution"].values())

        tenure_days = self.rng.integers(
            self.config["min_tenure_days"],
            self.config["max_tenure_days"] + 1,
            size=n_users,
        )
        signup_dates = self.ref_date - pd.to_timedelta(tenure_days, unit="D")

        users = pd.DataFrame(
            {
                "user_id": [f"U{str(i).zfill(6)}" for i in range(n_users)],
                "signup_date": signup_dates,
                "tenure_days": tenure_days,
                "behavioral_segment": self.rng.choice(
                    segment_labels,
                    size=n_users,
                    p=segment_probs,
                ),
                "plan": self.rng.choice(plan_labels, size=n_users, p=plan_probs),
                "primary_device": self.rng.choice(
                    device_labels,
                    size=n_users,
                    p=device_probs,
                ),
                "region": self.rng.choice(region_labels, size=n_users, p=region_probs),
                "age_group": self.rng.choice(age_labels, size=n_users, p=age_probs),
                "bundle_type": self.rng.choice(
                    bundle_labels,
                    size=n_users,
                    p=bundle_probs,
                ),
            }
        )

        num_services = []
        for segment in users["behavioral_segment"]:
            base_mean = self.segment_profiles[segment]["num_services_mean"]
            sampled = int(np.clip(np.round(self.rng.normal(base_mean, 0.9)), 1, 5))
            num_services.append(sampled)

        users["num_services"] = num_services

        validate_non_empty_dataframe(users, dataset_name="users")
        validate_required_columns(
            users,
            required_columns=[
                "user_id",
                "signup_date",
                "tenure_days",
                "behavioral_segment",
            ],
            dataset_name="users",
        )

        self.logger.info(
            "Generated user population with %s users.", f"{len(users):,}"
        )
        return users

    def simulate_engagement_metrics(self, users: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate engagement metrics over the observation window.

        Parameters
        ----------
        users : pd.DataFrame
            User population table.

        Returns
        -------
        pd.DataFrame
            Engagement metrics table at user level.
        """
        records: list[dict[str, Any]] = []
        observation_window = self.config["observation_window_days"]

        device_duration_adjustment = {
            "mobile": -3.0,
            "tv": 8.0,
            "tablet": 2.0,
            "desktop": 1.0,
        }

        plan_session_adjustment = {
            "student": 0.92,
            "individual": 1.00,
            "family": 1.08,
            "premium": 1.15,
        }

        for _, user in users.iterrows():
            profile = self.segment_profiles[user["behavioral_segment"]]

            tenure_multiplier = np.clip(user["tenure_days"] / 365.0, 0.75, 1.20)
            plan_multiplier = plan_session_adjustment[user["plan"]]
            bundle_multiplier = 1.10 if user["bundle_type"] == "bundle" else 1.0

            session_lambda = (
                profile["sessions_lambda"]
                * tenure_multiplier
                * plan_multiplier
                * bundle_multiplier
                * self.rng.normal(1.0, 0.18)
            )
            session_lambda = max(1.0, session_lambda)
            sessions_last_30d = int(self.rng.poisson(session_lambda))

            active_days_last_30d = int(
                np.clip(
                    np.round(
                        sessions_last_30d * self.rng.uniform(0.42, 0.78)
                        + self.rng.normal(0.0, 2.0)
                    ),
                    1,
                    30,
                )
            )

            avg_session_duration = float(
                np.clip(
                    self.rng.normal(
                        profile["duration_mean"]
                        + device_duration_adjustment[user["primary_device"]],
                        10.0,
                    ),
                    5.0,
                    180.0,
                )
            )

            watch_time_last_30d = float(
                np.clip(
                    sessions_last_30d
                    * avg_session_duration
                    * self.rng.normal(1.0, 0.16),
                    10.0,
                    8000.0,
                )
            )

            completion_rate = float(
                np.clip(
                    self.rng.normal(profile["completion_mean"], 0.10),
                    0.05,
                    0.99,
                )
            )

            skip_rate = float(
                np.clip(
                    1.0 - completion_rate + self.rng.normal(0.06, 0.06),
                    0.01,
                    0.95,
                )
            )

            content_diversity_score = float(
                np.clip(
                    self.rng.normal(profile["diversity_mean"], 0.14),
                    0.05,
                    1.0,
                )
            )

            feature_usage_score = float(
                np.clip(
                    self.rng.normal(profile["feature_usage_mean"], 0.15),
                    0.0,
                    1.0,
                )
            )

            search_activity_last_30d = int(
                self.rng.poisson(
                    max(0.2, profile["search_lambda"] * self.rng.normal(1.0, 0.20))
                )
            )

            watchlist_additions_last_30d = int(
                self.rng.poisson(
                    max(0.1, profile["watchlist_lambda"] * self.rng.normal(1.0, 0.22))
                )
            )

            days_since_last_session = int(
                np.clip(
                    np.round(
                        self.rng.exponential(profile["recency_scale"])
                        + self.rng.normal(0.0, 1.0)
                    ),
                    0,
                    observation_window,
                )
            )

            records.append(
                {
                    "user_id": user["user_id"],
                    "sessions_last_30d": sessions_last_30d,
                    "active_days_last_30d": active_days_last_30d,
                    "watch_time_last_30d": round(watch_time_last_30d, 2),
                    "avg_session_duration_min": round(avg_session_duration, 2),
                    "completion_rate": round(completion_rate, 4),
                    "skip_rate": round(skip_rate, 4),
                    "content_diversity_score": round(content_diversity_score, 4),
                    "feature_usage_score": round(feature_usage_score, 4),
                    "search_activity_last_30d": search_activity_last_30d,
                    "watchlist_additions_last_30d": watchlist_additions_last_30d,
                    "days_since_last_session": days_since_last_session,
                }
            )

        engagement = pd.DataFrame(records)

        validate_non_empty_dataframe(engagement, dataset_name="engagement_metrics")
        validate_required_columns(
            engagement,
            required_columns=[
                "user_id",
                "sessions_last_30d",
                "watch_time_last_30d",
                "days_since_last_session",
            ],
            dataset_name="engagement_metrics",
        )
        validate_no_nulls_in_columns(
            engagement,
            columns=["user_id", "sessions_last_30d", "watch_time_last_30d"],
            dataset_name="engagement_metrics",
        )

        self.logger.info(
            "Simulated engagement metrics for %s users.", f"{len(engagement):,}"
        )
        return engagement

    def compute_churn_targets(
        self,
        users: pd.DataFrame,
        engagement: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute probabilistic 30-day churn targets from behavioral risk.

        Parameters
        ----------
        users : pd.DataFrame
            User population table.
        engagement : pd.DataFrame
            Engagement metrics table.

        Returns
        -------
        pd.DataFrame
            Target table with churn probability and binary churn event.
        """
        master = users.merge(engagement, on="user_id", how="inner")

        sessions_z = self._zscore(np.log1p(master["sessions_last_30d"]))
        active_days_z = self._zscore(np.log1p(master["active_days_last_30d"]))
        watch_time_z = self._zscore(np.log1p(master["watch_time_last_30d"]))
        completion_z = self._zscore(master["completion_rate"])
        skip_rate_z = self._zscore(master["skip_rate"])
        diversity_z = self._zscore(master["content_diversity_score"])
        feature_usage_z = self._zscore(master["feature_usage_score"])
        search_z = self._zscore(np.log1p(master["search_activity_last_30d"]))
        watchlist_z = self._zscore(np.log1p(master["watchlist_additions_last_30d"]))
        recency_z = self._zscore(master["days_since_last_session"])
        services_z = self._zscore(master["num_services"])

        early_life_flag = (master["tenure_days"] < 90).astype(int)
        student_plan_flag = (master["plan"] == "student").astype(int)
        premium_plan_flag = (master["plan"] == "premium").astype(int)
        bundle_flag = (master["bundle_type"] == "bundle").astype(int)

        segment_offsets = master["behavioral_segment"].map(
            {
                segment: profile["segment_risk_offset"]
                for segment, profile in self.segment_profiles.items()
            }
        )

        raw_score = (
            -0.58 * sessions_z
            -0.32 * active_days_z
            -0.28 * watch_time_z
            -0.32 * completion_z
            +0.42 * skip_rate_z
            -0.18 * diversity_z
            -0.18 * feature_usage_z
            -0.12 * search_z
            -0.08 * watchlist_z
            +0.62 * recency_z
            -0.10 * services_z
            +0.10 * student_plan_flag
            -0.12 * premium_plan_flag
            -0.10 * bundle_flag
            +0.22 * early_life_flag
            +segment_offsets
            +self.rng.normal(0.0, 0.42, size=len(master))
        )

        churn_probability = self._calibrate_probabilities(
            raw_score=raw_score.to_numpy(),
            target_rate=self.config["target_churn_rate"],
        )

        will_churn_30d = (
            self.rng.uniform(0.0, 1.0, size=len(master)) < churn_probability
        ).astype(int)

        churn_date = np.where(
            will_churn_30d == 1,
            (
                self.ref_date
                + pd.to_timedelta(
                    self.rng.integers(
                        1,
                        self.config["prediction_window_days"] + 1,
                        size=len(master),
                    ),
                    unit="D",
                )
            ).astype("datetime64[ns]"),
            np.datetime64("NaT"),
        )

        targets = pd.DataFrame(
            {
                "user_id": master["user_id"],
                "churn_probability": np.round(churn_probability, 6),
                "will_churn_30d": will_churn_30d,
                "churn_date": churn_date,
            }
        )

        churn_rate = targets["will_churn_30d"].mean()
        self.logger.info("Computed churn targets. Observed churn rate: %.2f%%", churn_rate * 100)

        validate_non_empty_dataframe(targets, dataset_name="churn_targets")
        validate_required_columns(
            targets,
            required_columns=["user_id", "churn_probability", "will_churn_30d"],
            dataset_name="churn_targets",
        )

        return targets

    def _calibrate_probabilities(
        self,
        raw_score: np.ndarray,
        target_rate: float,
        tolerance: float = 1e-4,
        max_iterations: int = 100,
    ) -> np.ndarray:
        """
        Calibrate probabilities to match a target churn rate via intercept shift.

        Parameters
        ----------
        raw_score : np.ndarray
            Uncalibrated latent risk score.
        target_rate : float
            Desired average churn probability.
        tolerance : float, optional
            Convergence tolerance for mean probability.
        max_iterations : int, optional
            Maximum number of bisection iterations.

        Returns
        -------
        np.ndarray
            Calibrated churn probabilities.
        """
        low, high = -10.0, 10.0
        calibrated = self._sigmoid(raw_score)

        for _ in range(max_iterations):
            midpoint = (low + high) / 2.0
            calibrated = self._sigmoid(raw_score + midpoint)
            delta = calibrated.mean() - target_rate

            if abs(delta) < tolerance:
                break

            if delta > 0:
                high = midpoint
            else:
                low = midpoint

        return calibrated

    def build_master_table(
        self,
        users: pd.DataFrame,
        engagement: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build a convenience master table for downstream analysis.

        Parameters
        ----------
        users : pd.DataFrame
            User population table.
        engagement : pd.DataFrame
            Engagement metrics table.
        targets : pd.DataFrame
            Churn target table.

        Returns
        -------
        pd.DataFrame
            Joined master simulation table.
        """
        master = (
            users.merge(engagement, on="user_id", how="inner")
            .merge(targets, on="user_id", how="inner")
        )

        validate_non_empty_dataframe(master, dataset_name="simulation_master_table")
        validate_required_columns(
            master,
            required_columns=["user_id", "behavioral_segment", "will_churn_30d"],
            dataset_name="simulation_master_table",
        )

        return master

    def validate_outputs(
        self,
        users: pd.DataFrame,
        engagement: pd.DataFrame,
        targets: pd.DataFrame,
        master: pd.DataFrame,
    ) -> None:
        """
        Validate final simulation outputs.

        Parameters
        ----------
        users : pd.DataFrame
            User population table.
        engagement : pd.DataFrame
            Engagement metrics table.
        targets : pd.DataFrame
            Churn target table.
        master : pd.DataFrame
            Joined master table.

        Raises
        ------
        ValueError
            If critical data quality checks fail.
        """
        if users["user_id"].nunique() != len(users):
            raise ValueError("users table contains duplicate user_id values.")

        if engagement["user_id"].nunique() != len(engagement):
            raise ValueError("engagement table contains duplicate user_id values.")

        if targets["user_id"].nunique() != len(targets):
            raise ValueError("targets table contains duplicate user_id values.")

        churn_rate = targets["will_churn_30d"].mean()
        if not 0.10 <= churn_rate <= 0.30:
            raise ValueError(
                f"Observed churn rate {churn_rate:.2%} is outside the expected range."
            )

        if master["sessions_last_30d"].min() < 0:
            raise ValueError("sessions_last_30d contains negative values.")

        if master["completion_rate"].max() > 1 or master["completion_rate"].min() < 0:
            raise ValueError("completion_rate must stay within [0, 1].")

        if (
            master["days_since_last_session"].max()
            > self.config["observation_window_days"]
        ):
            raise ValueError(
                "days_since_last_session exceeds the observation window."
            )

        self.logger.info("Simulation outputs passed validation checks.")

    def export_outputs(
        self,
        users: pd.DataFrame,
        engagement: pd.DataFrame,
        targets: pd.DataFrame,
        master: pd.DataFrame,
    ) -> None:
        """
        Export raw simulation outputs to disk.

        Parameters
        ----------
        users : pd.DataFrame
            User population table.
        engagement : pd.DataFrame
            Engagement metrics table.
        targets : pd.DataFrame
            Churn target table.
        master : pd.DataFrame
            Joined master table.
        """
        save_csv(users, RAW_DATA_DIR / "users.csv", index=False)
        save_csv(engagement, RAW_DATA_DIR / "engagement_metrics.csv", index=False)
        save_csv(targets, RAW_DATA_DIR / "churn_targets.csv", index=False)
        save_csv(master, RAW_DATA_DIR / "simulation_master_table.csv", index=False)

        self.logger.info("Exported simulation outputs to %s", RAW_DATA_DIR)

    def run(self, export_outputs: bool = True) -> dict[str, pd.DataFrame]:
        """
        Execute the full simulation workflow.

        Parameters
        ----------
        export_outputs : bool, optional
            Whether to persist outputs to disk.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary containing users, engagement, targets, and master tables.
        """
        self._bootstrap_directories()

        self.logger.info("Starting subscription ecosystem simulation.")
        self.logger.info("Simulation seed: %s", self.seed)
        self.logger.info("Reference date: %s", self.ref_date.date())

        users = self.generate_user_population()
        engagement = self.simulate_engagement_metrics(users)
        targets = self.compute_churn_targets(users, engagement)
        master = self.build_master_table(users, engagement, targets)

        self.validate_outputs(users, engagement, targets, master)

        if export_outputs:
            self.export_outputs(users, engagement, targets, master)

        self.logger.info("Simulation pipeline completed successfully.")

        return {
            "users": users,
            "engagement_metrics": engagement,
            "churn_targets": targets,
            "simulation_master_table": master,
        }