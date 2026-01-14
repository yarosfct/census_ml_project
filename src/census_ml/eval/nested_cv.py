"""
Nested cross-validation and evaluation utilities.

This module will contain functions for performing nested cross-validation,
computing evaluation metrics, and statistical testing.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from census_ml.utils.logging import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional, for ROC-AUC).

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }

    # Add ROC-AUC if probabilities are provided
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            logger.warning("Could not compute ROC-AUC score")
            metrics["roc_auc"] = np.nan

    return metrics


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> Dict[str, List[float]]:
    """
    Perform cross-validation and return metrics for each fold.

    Args:
        model: Sklearn-compatible model.
        X: Feature matrix.
        y: Target vector.
        cv: Number of cross-validation folds.

    Returns:
        Dictionary of metric names to lists of fold-wise scores.
    """
    logger.info(f"Running {cv}-fold cross-validation")

    # Get cross-validated predictions
    y_pred = cross_val_predict(model, X, y, cv=cv)

    # Compute metrics
    metrics = compute_classification_metrics(y, y_pred)

    logger.info(f"Cross-validation metrics: {metrics}")

    return metrics


# Placeholder for future evaluation functions
# def nested_cross_validation(...):
#     """Perform nested cross-validation with hyperparameter tuning."""
#     pass

# def statistical_comparison(...):
#     """Perform statistical tests to compare models."""
#     pass
