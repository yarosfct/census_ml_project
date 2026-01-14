"""
Preprocessing utilities for the Adult Census Income dataset.

This module will contain preprocessing pipelines and transformations
for the Adult dataset.
"""

from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from census_ml.config import CATEGORICAL_FEATURES, MISSING_VALUE_INDICATOR, NUMERICAL_FEATURES
from census_ml.utils.logging import get_logger

logger = get_logger(__name__)


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values in the Adult dataset.

    Replaces missing indicators (?) with a dedicated 'Missing' category
    for categorical features and uses median imputation for numerical features.
    """

    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        missing_indicator: str = MISSING_VALUE_INDICATOR,
    ):
        """
        Initialize the missing value handler.

        Args:
            categorical_features: List of categorical feature names.
            numerical_features: List of numerical feature names.
            missing_indicator: String indicating missing values in the data.
        """
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES
        self.numerical_features = numerical_features or NUMERICAL_FEATURES
        self.missing_indicator = missing_indicator
        self.numerical_medians_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer by computing medians for numerical features.

        Args:
            X: Input features.
            y: Target (ignored).

        Returns:
            self
        """
        for col in self.numerical_features:
            if col in X.columns:
                self.numerical_medians_[col] = X[col].median()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by handling missing values.

        Args:
            X: Input features.

        Returns:
            Transformed dataframe with missing values handled.
        """
        X_copy = X.copy()

        # Handle categorical missing values
        for col in self.categorical_features:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace(self.missing_indicator, "Missing")

        # Handle numerical missing values (if any)
        for col in self.numerical_features:
            if col in X_copy.columns and col in self.numerical_medians_:
                X_copy[col].fillna(self.numerical_medians_[col], inplace=True)

        return X_copy


# Placeholder for future preprocessing classes
# class FeatureEncoder(BaseEstimator, TransformerMixin):
#     """Encode categorical features."""
#     pass

# class FeatureScaler(BaseEstimator, TransformerMixin):
#     """Scale numerical features."""
#     pass
