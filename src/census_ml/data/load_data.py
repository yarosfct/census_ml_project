"""
Data loading utilities for the Adult Census Income dataset.

This module will contain functions to load and perform basic validation
of the dataset.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from census_ml.config import (
    CATEGORICAL_FEATURES,
    DATA_RAW_DIR,
    NUMERICAL_FEATURES,
    TARGET_COL,
    TRAIN_FILE,
)
from census_ml.utils.logging import get_logger

logger = get_logger(__name__)


def load_raw_data(
    data_path: Optional[Path] = None,
    filename: str = TRAIN_FILE,
) -> pd.DataFrame:
    """
    Load raw Adult Census Income data from CSV.

    Args:
        data_path: Path to the data directory. If None, uses DATA_RAW_DIR from config.
        filename: Name of the data file to load.

    Returns:
        DataFrame containing the raw data.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
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
    column_names = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]

    # Load data
    df = pd.read_csv(file_path, names=column_names, skipinitialspace=True)

    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

    return df


def get_feature_target_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
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
