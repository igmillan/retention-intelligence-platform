"""
Global configuration settings for Case 01.

This module centralizes project-level constants, paths, runtime parameters,
simulation defaults, modeling defaults, and business assumptions.

Notes
-----
All project modules should import configuration values from this file rather
than hardcoding parameters locally.
"""

from pathlib import Path


# -----------------------------------------------------------------------------
# Project metadata
# -----------------------------------------------------------------------------
PROJECT_NAME = "case_01_user_retention_churn"
PROJECT_DISPLAY_NAME = "Case 01 — User Retention & Churn Prediction"
VERSION = "0.2.0"


# -----------------------------------------------------------------------------
# Global reproducibility
# -----------------------------------------------------------------------------
SEED = 42


# -----------------------------------------------------------------------------
# Root paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
SCORED_DATA_DIR = DATA_DIR / "scored"

SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"
APP_DIR = PROJECT_ROOT / "app"
DASHBOARDS_DIR = PROJECT_ROOT / "dashboards"
TESTS_DIR = PROJECT_ROOT / "tests"

LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"


# -----------------------------------------------------------------------------
# Directory registry for bootstrapping
# -----------------------------------------------------------------------------
DIRECTORIES_TO_CREATE = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURES_DATA_DIR,
    SCORED_DATA_DIR,
    DOCS_DIR,
    REPORTS_DIR,
    ASSETS_DIR,
    APP_DIR,
    DASHBOARDS_DIR,
    NOTEBOOKS_DIR,
    TESTS_DIR,
    LOGS_DIR,
    MODELS_DIR,
]


# -----------------------------------------------------------------------------
# Simulation defaults
# -----------------------------------------------------------------------------
SIMULATION_CONFIG = {
    "n_users": 10000,
    "reference_date": "2025-07-01",
    "observation_window_days": 90,
    "prediction_window_days": 30,
    "target_churn_rate": 0.18,
    "min_tenure_days": 30,
    "max_tenure_days": 900,
    "segment_mix": {
        "power": 0.20,
        "casual": 0.35,
        "passive": 0.25,
        "at_risk": 0.20,
    },
    "plan_distribution": {
    "student": 0.20,
    "individual": 0.40,
    "family": 0.25,
    "premium": 0.15,
    },
    "device_distribution": {
        "mobile": 0.46,
        "tv": 0.28,
        "tablet": 0.14,
        "desktop": 0.12,
    },
    "region_distribution": {
        "LATAM": 0.50,
        "US": 0.24,
        "EU": 0.16,
        "APAC": 0.10,
    },
    "age_group_distribution": {
    "18_24": 0.20,
    "25_34": 0.30,
    "35_44": 0.25,
    "45_54": 0.15,
    "55_plus": 0.10,
    },
    "bundle_distribution": {
        "standalone": 0.58,
        "bundle": 0.42,
    },
}


# -----------------------------------------------------------------------------
# Modeling defaults
# -----------------------------------------------------------------------------
MODELING_CONFIG = {
    "test_size": 0.20,
    "validation_size": 0.20,
    "cv_folds": 5,
    "random_state": SEED,
}


# -----------------------------------------------------------------------------
# Business assumptions
# -----------------------------------------------------------------------------
BUSINESS_CONFIG = {
    "avg_monthly_revenue_per_user": 12.99,
    "avg_ltv_months": 12,
    "retention_intervention_cost": 2.50,
    "default_expected_save_rate": 0.18,
}


# -----------------------------------------------------------------------------
# Risk thresholds
# -----------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.40,
    "low": 0.00,
}


# -----------------------------------------------------------------------------
# Logging defaults
# -----------------------------------------------------------------------------
LOGGING_CONFIG = {
    "logger_name": "case01",
    "log_filename": "case_01.log",
    "log_level": "INFO",
}