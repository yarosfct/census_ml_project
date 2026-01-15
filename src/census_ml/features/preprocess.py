"""
Preprocessing utilities for the Adult Census Income dataset.

This module will contain preprocessing pipelines and transformations
for the Adult dataset.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from census_ml.config import CATEGORICAL_FEATURES, MISSING_VALUE_INDICATOR, NUMERICAL_FEATURES
from census_ml.utils.logging import get_logger

logger = get_logger(__name__)

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Handles missing values and one-hot encoding.
    """

    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.numeric_medians_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        # Numerical medians
        for col in NUMERICAL_FEATURES:
            self.numeric_medians_[col] = X[col].median()

        # Numerical scaling (fit after imputation)
        X_num = X[NUMERICAL_FEATURES].copy()
        for col in NUMERICAL_FEATURES:
            X_num[col] = X_num[col].fillna(self.numeric_medians_[col])
        self.scaler.fit(X_num)

        # Replace missing categorical values
        X_cat = X[CATEGORICAL_FEATURES].replace(
            MISSING_VALUE_INDICATOR, "Missing"
        )

        self.encoder.fit(X_cat)
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        # Numerical
        for col in NUMERICAL_FEATURES:
            X[col] = X[col].fillna(self.numeric_medians_[col])

        # Categorical
        X_cat = X[CATEGORICAL_FEATURES].replace(
            MISSING_VALUE_INDICATOR, "Missing"
        )
        X_cat_enc = self.encoder.transform(X_cat)

        X_num = X[NUMERICAL_FEATURES].to_numpy()
        return pd.DataFrame(
            data=pd.concat(
                [
                    pd.DataFrame(X_num),
                    pd.DataFrame(X_cat_enc),
                ],
                axis=1,
            ).values
        )
