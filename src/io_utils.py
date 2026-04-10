"""
Input/output utility functions for Case 01.

This module centralizes common filesystem operations such as directory creation,
CSV persistence, and path validation.
"""

from pathlib import Path

import pandas as pd


def ensure_directory(path: Path) -> None:
    """
    Ensure that a directory exists.

    Parameters
    ----------
    path : Path
        Directory path to create if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, output_path: Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV, creating parent directories if needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    output_path : Path
        Target file path.
    index : bool, optional
        Whether to write the DataFrame index.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)


def load_csv(input_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Parameters
    ----------
    input_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"CSV file not found: {input_path}")

    return pd.read_csv(input_path)