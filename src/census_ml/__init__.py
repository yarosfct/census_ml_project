"""
Adult Census Income Classification ML Project.

A machine learning project to predict whether an individual's annual income
exceeds $50,000 using demographic and employment-related attributes.
"""

__version__ = "0.1.0"

from census_ml.config import (
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    RANDOM_SEED,
    TARGET_COL,
)

__all__ = [
    "RANDOM_SEED",
    "TARGET_COL",
    "DATA_RAW_DIR",
    "DATA_INTERIM_DIR",
    "DATA_PROCESSED_DIR",
]
