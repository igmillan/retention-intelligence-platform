"""
Validation utilities for Case 01.

This module provides reusable validation helpers for schema, missing values,
column presence, and basic data integrity checks.
"""

from typing import Iterable

import pandas as pd


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str = "dataset",
) -> None:
    """
    Validate that a DataFrame contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : Iterable[str]
        Required column names.
    dataset_name : str, optional
        Human-readable dataset name for error messages.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {dataset_name}: {missing_columns}"
        )


def validate_no_nulls_in_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    dataset_name: str = "dataset",
) -> None:
    """
    Validate that specified columns contain no null values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    columns : Iterable[str]
        Column names that must not contain nulls.
    dataset_name : str, optional
        Human-readable dataset name for error messages.

    Raises
    ------
    ValueError
        If any specified column contains null values.
    """
    null_columns = [col for col in columns if df[col].isna().any()]
    if null_columns:
        raise ValueError(
            f"Null values detected in {dataset_name} columns: {null_columns}"
        )


def validate_non_empty_dataframe(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
) -> None:
    """
    Validate that a DataFrame is not empty.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    dataset_name : str, optional
        Human-readable dataset name for error messages.

    Raises
    ------
    ValueError
        If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError(f"{dataset_name} is empty.")