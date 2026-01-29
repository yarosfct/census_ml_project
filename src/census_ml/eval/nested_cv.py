"""
Nested cross-validation and evaluation utilities.

This module will contain functions for performing nested cross-validation,
computing evaluation metrics, and statistical testing.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from census_ml.utils.logging import get_logger
from census_ml.fairness.fairness_analysis import FairnessAnalyzer

from tqdm import tqdm

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logger = get_logger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> dict[str, float]:
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
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }

    # Add ROC-AUC if probabilities are provided
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except (ValueError, Warning) as e:
            logger.warning(f"Could not compute ROC-AUC score: {e}")
            metrics["roc_auc"] = np.nan

    return metrics


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict[str, list[float]]:
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


def nested_cross_validation(model, X, y, param_grid, outer_cv_splits=5, outer_cv_repeats=2, inner_cv=3, sensitive_features_df=None):
    """
    Perform nested cross-validation with hyperparameter tuning and fairness metrics.
    
    Args:
        model: Estimator pipeline or model
        X: Feature matrix (DataFrame or array-like)
        y: Target vector (Series or array-like)
        param_grid: Hyperparameter grid for inner CV
        outer_cv_splits: Number of outer CV splits
        outer_cv_repeats: Number of outer CV repeats
        inner_cv: Number of inner CV folds
        sensitive_features_df: DataFrame with sensitive attributes for fairness analysis
        
    Returns:
        Dictionary containing metrics for each fold, including fairness metrics if sensitive_features provided
    """
    # Use 5 splits with 2 repeats for more robust statistical testing
    outer_splitter = RepeatedStratifiedKFold(
        n_splits=outer_cv_splits, n_repeats=outer_cv_repeats, random_state=42
    )

    metrics = {
        "outer_fold": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "best_inner_auc": [],
        "best_params": []
    }
    
    # Add fairness metrics structure if sensitive features provided
    has_fairness = sensitive_features_df is not None
    if has_fairness:
        metrics["demographic_parity_difference"] = []
        metrics["equalized_odds_difference"] = []
        metrics["fold_results"] = []  # Store full fairness results per fold
        fairness_analyzer = FairnessAnalyzer()

    for i, (train_idx, test_idx) in enumerate(tqdm(outer_splitter.split(X, y), total=outer_cv_splits * outer_cv_repeats)):
        metrics["outer_fold"].append(i)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        try:
            y_proba = best_model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        fold_metrics = compute_classification_metrics(
            y_test, y_pred, y_proba
        )

        for k in fold_metrics:
            metrics[k].append(fold_metrics[k])
        
        metrics["best_inner_auc"].append(grid.best_score_)
        metrics["best_params"].append(grid.best_params_)
        
        # Compute fairness metrics for this fold
        if has_fairness:
            sensitive_test = sensitive_features_df.iloc[test_idx]
            fairness_result = fairness_analyzer.compute_fairness_metrics(
                y_test.values if hasattr(y_test, 'values') else y_test,
                y_pred,
                y_proba,
                sensitive_features=sensitive_test
            )
            metrics["demographic_parity_difference"].append(
                fairness_result["fairness_metrics"].get("demographic_parity_difference", np.nan)
            )
            metrics["equalized_odds_difference"].append(
                fairness_result["fairness_metrics"].get("equalized_odds_difference", np.nan)
            )
            # Store the full result (includes metric_frame with per-group data)
            metrics["fold_results"].append(fairness_result)

    return metrics

# def statistical_comparison(...):
#     """Perform statistical tests to compare models."""
#     pass
