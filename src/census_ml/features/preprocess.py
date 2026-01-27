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

    def __init__(self, impute = True):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.numeric_medians_ = {}
        self.impute = impute

    def fit(self, X: pd.DataFrame, y=None):
        X_num = X[NUMERICAL_FEATURES].copy()
        X_num = X_num.apply(pd.to_numeric, errors='coerce')
        X_cat = X[CATEGORICAL_FEATURES].copy()

        if self.impute:
            # Numerical medians
            for col in NUMERICAL_FEATURES:
                self.numeric_medians_[col] = X[col].median()

            # Numerical scaling (fit after imputation)
            for col in NUMERICAL_FEATURES:
                X_num[col] = X_num[col].fillna(self.numeric_medians_[col])
            
            # Replace missing categorical values
            X_cat = X_cat.replace(MISSING_VALUE_INDICATOR, "Missing")

        self.scaler.fit(X_num)
        self.encoder.fit(X_cat)
        return self

    def transform(self, X: pd.DataFrame):
        X_num = X[NUMERICAL_FEATURES].copy()
        X_num = X_num.apply(pd.to_numeric, errors='coerce')
        X_cat = X[CATEGORICAL_FEATURES].copy()

        if self.impute:
            # Numerical
            for col in NUMERICAL_FEATURES:
                X_num[col] = X_num[col].fillna(self.numeric_medians_[col])

            # Categorical
            X_cat = X_cat.replace(MISSING_VALUE_INDICATOR, "Missing")
        
        X_num = self.scaler.transform(X_num)
        X_cat_enc = self.encoder.transform(X_cat)
        
        return pd.DataFrame(
            data=pd.concat(
                [
                    pd.DataFrame(X_num),
                    pd.DataFrame(X_cat_enc),
                ],
                axis=1,
            ).values
        )
