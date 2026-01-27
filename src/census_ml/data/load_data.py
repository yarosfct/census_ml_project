"""
Data loading utilities for the Adult Census Income dataset.

This module contains functions to load and perform basic validation
of the dataset, supporting both UCI split format and single CSV format.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from census_ml.config import (
    COLUMN_NAMES,
    ALL_FEATURES,
    CSV_FILE,
    DATA_RAW_DIR,
    TARGET_COL,
    TEST_FILE,
    TRAIN_FILE,
)
from census_ml.utils.logging import get_logger

logger = get_logger(__name__)


def _standardize_target_labels(series: pd.Series) -> pd.Series:
    """
    Standardize target labels by removing trailing periods.

    The UCI test file includes trailing periods (e.g., '>50K.' instead of '>50K').
    This function removes them for consistency.

    Args:
        series: Target column series

    Returns:
        Standardized series
    """
    return series.str.strip().str.rstrip(".")


def _convert_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert '?' missing value indicators to NaN and strip whitespace.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with '?' converted to NaN and whitespace stripped
    """
    df = df.copy()

    # Replace '?' with NaN for all columns
    df = df.replace("?", np.nan)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    return df


def _read_uci_file(filepath: Path, skip_lines: int = 0) -> pd.DataFrame:
    """
    Read a UCI format file (no headers, comma-separated).

    Args:
        filepath: Path to the UCI format file
        skip_lines: Number of lines to skip at the beginning (test file has 1 header line)

    Returns:
        DataFrame with Adult dataset column names
    """
    # Define column names (Adult dataset does not have headers)
    column_names = ALL_FEATURES + [TARGET_COL]

    logger.info(f"Reading UCI file: {filepath}")

    # Read CSV without headers, skip initial whitespace
    df = pd.read_csv(
        filepath,
        names=column_names,
        skipinitialspace=True,
        skiprows=skip_lines,
        na_values=["?"],
    )

    return df


def load_adult_dataset(
    source: Literal["uci_split", "csv", "auto"] = "auto",
    data_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load the Adult Census Income dataset.

    Supports two formats:
    - 'uci_split': Load and combine adult.data and adult.test files
    - 'csv': Load a single adult.csv file
    - 'auto': Auto-detect which format is available (prefers UCI split)

    Args:
        source: Data source format to use
        data_path: Path to the data directory. If None, uses DATA_RAW_DIR from config.

    Returns:
        DataFrame containing the complete dataset with standardized columns

    Raises:
        FileNotFoundError: If the required data files are not found
        ValueError: If source is invalid or no data files are found
    """
    if data_path is None:
        data_path = DATA_RAW_DIR

    # Auto-detect available format
    if source == "auto":
        train_file = data_path / TRAIN_FILE
        test_file = data_path / TEST_FILE
        csv_file = data_path / CSV_FILE

        if train_file.exists() and test_file.exists():
            source = "uci_split"
            logger.info("Auto-detected UCI split format")
        elif csv_file.exists():
            source = "csv"
            logger.info("Auto-detected CSV format")
        else:
            raise FileNotFoundError(
                f"No Adult dataset files found in {data_path}\n"
                f"Please download the dataset using one of these options:\n\n"
                f"Option 1 (UCI split - recommended):\n"
                f"  cd {data_path}\n"
                f"  wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data\n"
                f"  wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test\n\n"
                f"Option 2 (Single CSV):\n"
                f"  Place adult.csv in {data_path}\n"
            )

    # Load based on detected/specified format
    if source == "uci_split":
        train_file = data_path / TRAIN_FILE
        test_file = data_path / TEST_FILE

        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(
                f"UCI split files not found.\n"
                f"Expected: {train_file} and {test_file}\n"
                f"Please download using:\n"
                f"  cd {data_path}\n"
                f"  wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data\n"
                f"  wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
            )

        # Read train and test files
        # Note: test file has 1 header line that needs to be skipped
        df_train = _read_uci_file(train_file, skip_lines=0)
        df_test = _read_uci_file(test_file, skip_lines=1)

        # Combine train and test
        df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

        logger.info(
            f"Loaded UCI split: {len(df_train)} train + {len(df_test)} test = {len(df)} total"
        )

    elif source == "csv":
        csv_file = data_path / CSV_FILE

        if not csv_file.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_file}\n"
                f"Please place adult.csv in {data_path}"
            )

        # Read single CSV file
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded CSV: {len(df)} records")

    else:
        raise ValueError(
            f"Invalid source: {source}. Must be 'uci_split', 'csv', or 'auto'"
        )

    # Standardize target labels (remove trailing periods)
    df[TARGET_COL] = _standardize_target_labels(df[TARGET_COL])

    # Convert missing values
    df = _convert_missing_values(df)

    logger.info(
        f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns "
        f"({len(ALL_FEATURES)} features + target)"
    )

    return df


def get_feature_target_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features (X) and target (y).

    Args:
        df: Input dataframe containing features and target.

    Returns:
        Tuple of (X, y) where X is features and y is target.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return X, y


# Deprecated - kept for backward compatibility
def load_raw_data(
    data_path: Path | None = None,
    filename: str = TRAIN_FILE,
) -> pd.DataFrame:
    """
    Load raw Adult Census Income data from CSV.

    .. deprecated::
        Use :func:`load_adult_dataset` instead for better format support.

    Args:
        data_path: Path to the data directory. If None, uses DATA_RAW_DIR from config.
        filename: Name of the data file to load.

    Returns:
        DataFrame containing the raw data.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    logger.warning(
        "load_raw_data() is deprecated. Use load_adult_dataset() instead."
    )

    if data_path is None:
        data_path = DATA_RAW_DIR

    file_path = data_path / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Please download the Adult dataset and place it in {data_path}"
        )

    logger.info(f"Loading data from {file_path}")

    # Define column names (Adult dataset does not have headers)
    column_names = ALL_FEATURES + [TARGET_COL]

    # Load data
    df = pd.read_csv(file_path, names=column_names, skipinitialspace=True)

    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

    return df

def load_data(
        train_path: Path,
        test_path: Path
) -> pd.DataFrame:
    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Train and test files not found.\n"
        )

    df_train = pd.read_csv(
        train_path,
        sep=",",
        skipinitialspace=True,
        header=None,
        names=COLUMN_NAMES
    )

    df_test = pd.read_csv(
        test_path,
        sep=",",
        skipinitialspace=True,
        header=None,
        names=COLUMN_NAMES
    )
    df_test = df_test[1:] # remove first row which is header in test file
    df_test[TARGET_COL] = df_test[TARGET_COL].str.rstrip('.')  # Remove trailing period from target labels

    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    return df
