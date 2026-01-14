"""
Configuration module for the census ML project.

Contains all central constants, paths, and configuration settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Random seed for reproducibility
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Reports directories
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Dataset configuration
DATASET_NAME: str = "adult"
TRAIN_FILE: str = "adult.data"
TEST_FILE: str = "adult.test"

# Target column name
TARGET_COL: str = "income"

# Feature columns (Adult dataset)
CATEGORICAL_FEATURES: list[str] = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

NUMERICAL_FEATURES: list[str] = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

# Missing value indicator in the dataset
MISSING_VALUE_INDICATOR: str = "?"

# Cross-validation settings
CV_FOLDS: int = 5
CV_RANDOM_STATE: int = RANDOM_SEED

# Logging configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
