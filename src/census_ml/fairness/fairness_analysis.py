import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_positive_rate_difference,
    false_negative_rate_difference,
    MetricFrame,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from census_ml.utils.logging import get_logger

logger = get_logger(__name__)


class FairnessAnalyzer:
    """
    Computes fairness metrics for model predictions across protected groups.

    This analyzer evaluates demographic parity, equalized odds, and other
    fairness metrics to detect potential bias in model predictions across
    different demographic groups (e.g., sex, race).
    """

    def __init__(self, sensitive_features: list[str] = None):
        self.sensitive_features = sensitive_features or ["sex", "race"]

    def compute_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        sensitive_features: pd.DataFrame = None,
    ) -> dict:

        if sensitive_features is None:
            logger.warning("No sensitive features provided. Skipping fairness analysis.")
            return {
                "metric_frame": None,
                "fairness_metrics": {},
                "group_metrics": {},
                "num_groups": 0,
            }

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics_dict = {
            "accuracy": accuracy_score,
            "precision": lambda yt, yp: precision_score(
                yt, yp, average="binary", zero_division=0
            ),
            "recall": lambda yt, yp: recall_score(
                yt, yp, average="binary", zero_division=0
            ),
        }

        # Wrapped to handle cases where a group has only one class
        if y_proba is not None:
            y_proba = np.array(y_proba)
            
            def safe_roc_auc(yt, yp):
                try:
                    unique_classes = np.unique(yt)
                    if len(unique_classes) < 2:
                        return np.nan
                    return roc_auc_score(yt, yp)
                except Exception:
                    return np.nan
            
            metrics_dict["roc_auc"] = safe_roc_auc
        else:
            y_proba = None

        # Create MetricFrame for detailed group-level metrics
        # If multiple sensitive features, create separate MetricFrames for each
        mf = None
        try:
            if isinstance(sensitive_features, pd.DataFrame) and sensitive_features.shape[1] > 1:
                # Create separate MetricFrames for each sensitive attribute
                # and combine them for reporting (will be handled in report generation)
                mf_dict = {}
                for col in sensitive_features.columns:
                    mf_dict[col] = MetricFrame(
                        metrics=metrics_dict,
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_features=sensitive_features[col],
                    )
                # Store as dict of MetricFrames - one per attribute
                mf = mf_dict
            else:
                # Single sensitive feature
                mf = MetricFrame(
                    metrics=metrics_dict,
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive_features,
                )
        except Exception as e:
            logger.error(f"Error creating MetricFrame: {e}")
            return {
                "metric_frame": None,
                "fairness_metrics": {},
                "group_metrics": {},
                "num_groups": 0,
            }

        fairness_metrics = {}

        try:
            # If sensitive_features has multiple columns, compute metrics per column
            if isinstance(sensitive_features, pd.DataFrame) and sensitive_features.shape[1] > 1:
                dpd_values = []
                eod_values = []
                
                for col in sensitive_features.columns:
                    col_features = sensitive_features[col]
                    dpd_val = demographic_parity_difference(
                        y_true, y_pred, sensitive_features=col_features
                    )
                    eod_val = equalized_odds_difference(
                        y_true, y_pred, sensitive_features=col_features
                    )
                    dpd_values.append(dpd_val)
                    eod_values.append(eod_val)
                
                fairness_metrics["demographic_parity_difference"] = max(dpd_values, key=abs)
                fairness_metrics["equalized_odds_difference"] = max(eod_values, key=abs)
            else:
                fairness_metrics["demographic_parity_difference"] = (
                    demographic_parity_difference(
                        y_true, y_pred, sensitive_features=sensitive_features
                    )
                )
                fairness_metrics["equalized_odds_difference"] = (
                    equalized_odds_difference(
                        y_true, y_pred, sensitive_features=sensitive_features
                    )
                )
            
            fairness_metrics["demographic_parity_ratio"] = (
                demographic_parity_ratio(
                    y_true, y_pred, sensitive_features=sensitive_features
                )
            )
            fairness_metrics["equalized_odds_ratio"] = equalized_odds_ratio(
                y_true, y_pred, sensitive_features=sensitive_features
            )
            fairness_metrics["false_positive_rate_difference"] = (
                false_positive_rate_difference(
                    y_true, y_pred, sensitive_features=sensitive_features
                )
            )
            fairness_metrics["false_negative_rate_difference"] = (
                false_negative_rate_difference(
                    y_true, y_pred, sensitive_features=sensitive_features
                )
            )
        except Exception as e:
            logger.warning(f"Could not compute some fairness metrics: {e}")

        # Build return dict, handling both single MetricFrame and dict of MetricFrames
        if isinstance(mf, dict):
            # Multiple attributes - combine group metrics from all
            group_metrics_dict = {}
            num_groups = 0
            for attr_name, attr_mf in mf.items():
                if attr_mf is not None and hasattr(attr_mf, 'by_group'):
                    group_metrics_dict[attr_name] = attr_mf.by_group.to_dict()
                    num_groups += len(attr_mf.by_group)
        else:
            # Single MetricFrame
            group_metrics_dict = mf.by_group.to_dict() if mf is not None else {}
            num_groups = len(mf.by_group) if mf is not None else 0

        return {
            "metric_frame": mf,
            "fairness_metrics": fairness_metrics,
            "group_metrics": group_metrics_dict,
            "num_groups": num_groups,
        }

