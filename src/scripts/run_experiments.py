"""
Main script for running experiments.

This script will orchestrate the full ML pipeline:
- Load data
- Preprocess
- Train models
- Evaluate and compare
- Perform fairness analysis

"""

import sys
from pathlib import Path

# Add census_ml_project to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from census_ml.features.preprocess import Preprocessor
from census_ml.eval.nested_cv import nested_cross_validation
from census_ml.fairness.fairness_analysis import FairnessAnalyzer

from census_ml.utils.logging import get_logger
from census_ml.config import DATA_RAW_DIR, TRAIN_FILE, TEST_FILE, RESULTS_DIR
from census_ml.data.load_data import load_data, get_feature_target_split, get_protected_attributes

logger = get_logger(__name__)

def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing values indicated by MISSING_VALUE_INDICATOR.
    """
    from census_ml.config import MISSING_VALUE_INDICATOR

    return df.replace(MISSING_VALUE_INDICATOR, pd.NA).dropna()

def perform_statistical_comparison(all_results):
    """
    Perform pairwise Wilcoxon signed-rank tests between models.
    Saves results to file and prints to console.
    """
    # Combine all results into one DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Get unique models and sort for consistent ordering
    models = sorted(combined_df['model'].unique())
    
    # Build report text
    report_lines = []
    report_lines.append("\n" + "="*70)
    report_lines.append("STATISTICAL COMPARISON OF MODELS")
    report_lines.append("="*70)
    report_lines.append("Using Wilcoxon signed-rank test on ROC-AUC scores")
    report_lines.append("Paired by CV fold (10 total: 5 splits × 2 repeats)")
    report_lines.append("Significance level: Alpha = 0.05 (not adjusted for multiple comparisons)")
    report_lines.append("")
    
    # For each pair of models
    comparison_count = 0
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:
                comparison_count += 1
                
                df1 = combined_df[combined_df['model'] == model1].sort_values('outer_fold')
                df2 = combined_df[combined_df['model'] == model2].sort_values('outer_fold')
                
                scores1 = df1['roc_auc'].values
                scores2 = df2['roc_auc'].values
                
                if len(scores1) != len(scores2):
                    msg = f"Could not compare {model1} vs {model2}: Different number of folds ({len(scores1)} vs {len(scores2)})"
                    report_lines.append(msg)
                    continue
                
                try:
                    stat, p_value = stats.wilcoxon(scores1, scores2)
                    significant = "YES" if p_value < 0.05 else "NO"
                    
                    report_lines.append(f"{model1} vs {model2}:")
                    report_lines.append(f"  Test Statistic: {stat:.4f}")
                    report_lines.append(f"  P-value: {p_value:.6f}")
                    report_lines.append(f"  Significant (α=0.05): {significant}")
                    report_lines.append(f"  Mean ROC-AUC: {scores1.mean():.4f} vs {scores2.mean():.4f}")
                    report_lines.append("")
                    
                except ValueError as e:
                    report_lines.append(f"Could not compare {model1} vs {model2}: {e}")
                    report_lines.append("")
    
    report_lines.append("="*70)
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    report_path = RESULTS_DIR / "statistical_comparison.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Statistical comparison saved to {report_path}")


def _generate_model_fairness_report(model_name: str, results: dict, analyzer: FairnessAnalyzer) -> str:

    report = f"\n{'='*70}\n"
    report += f"MODEL: {model_name}\n"
    report += f"{'='*70}\n\n"
    
    if 'demographic_parity_difference' not in results:
        report += "No fairness metrics available for this model.\n"
        return report
    
    dpd_values = results['demographic_parity_difference']
    eod_values = results['equalized_odds_difference']
    
    report += "FAIRNESS METRICS ACROSS CV FOLDS:\n"
    report += "-" * 70 + "\n"
    
    if dpd_values and not all(pd.isna(dpd_values)):
        dpd_mean = pd.Series(dpd_values).mean()
        dpd_std = pd.Series(dpd_values).std()
        report += f"Demographic Parity Difference:\n"
        report += f"  Mean: {dpd_mean:8.4f} ± {dpd_std:.4f}\n"
        report += f"  Range: [{min(dpd_values):.4f}, {max(dpd_values):.4f}]\n\n"
    
    if eod_values and not all(pd.isna(eod_values)):
        eod_mean = pd.Series(eod_values).mean()
        eod_std = pd.Series(eod_values).std()
        report += f"Equalized Odds Difference:\n"
        report += f"  Mean: {eod_mean:8.4f} ± {eod_std:.4f}\n"
        report += f"  Range: [{min(eod_values):.4f}, {max(eod_values):.4f}]\n\n"
    

    report += "DEMOGRAPHIC GROUP BREAKDOWN:\n"
    report += "-" * 70 + "\n"
    
    fold_results = results.get('fold_results', [])
    if fold_results:
        # Collect per-group metrics across folds
        # Handle both single MetricFrame and dict of MetricFrames (per-attribute)
        group_metrics = {}
        
        for fold_idx, fold_result in enumerate(fold_results):
            if fold_result is None:
                continue
            if isinstance(fold_result, dict) and 'metric_frame' in fold_result:
                mf = fold_result['metric_frame']
                
                # Handle dict of MetricFrames (one per attribute) or single MetricFrame
                if isinstance(mf, dict):
                    # Multiple attributes - process each separately
                    for attr_name, attr_mf in mf.items():
                        if attr_mf is not None and hasattr(attr_mf, 'by_group') and attr_mf.by_group is not None:
                            for metric_col in attr_mf.by_group.columns:
                                key = f"{attr_name}_{metric_col}"
                                if key not in group_metrics:
                                    group_metrics[key] = {"attr": attr_name, "metric": metric_col, "data": {}}
                                for group_name in attr_mf.by_group.index:
                                    if group_name not in group_metrics[key]["data"]:
                                        group_metrics[key]["data"][group_name] = []
                                    group_metrics[key]["data"][group_name].append(
                                        attr_mf.by_group.loc[group_name, metric_col]
                                    )
                elif mf is not None and hasattr(mf, 'by_group') and mf.by_group is not None and not mf.by_group.empty:
                    # Single MetricFrame
                    for metric_col in mf.by_group.columns:
                        if metric_col not in group_metrics:
                            group_metrics[metric_col] = {"attr": None, "metric": metric_col, "data": {}}
                        for group_name in mf.by_group.index:
                            if group_name not in group_metrics[metric_col]["data"]:
                                group_metrics[metric_col]["data"][group_name] = []
                            group_metrics[metric_col]["data"][group_name].append(
                                mf.by_group.loc[group_name, metric_col]
                            )
        
        if group_metrics:
            # Group by metric, then by attribute for cleaner output
            metrics_by_name = {}
            for key, info in group_metrics.items():
                metric_name = info["metric"]
                if metric_name not in metrics_by_name:
                    metrics_by_name[metric_name] = {}
                attr_name = info["attr"] if info["attr"] else "all"
                metrics_by_name[metric_name][attr_name] = info["data"]
            
            for metric_name in sorted(metrics_by_name.keys()):
                report += f"\n{metric_name}:\n"
                
                for attr_name in sorted(metrics_by_name[metric_name].keys()):
                    metric_data = metrics_by_name[metric_name][attr_name]
                    
                    if attr_name != "all":
                        report += f"  {attr_name}:\n"
                        indent = "    "
                    else:
                        indent = "  "
                    
                    # Calculate mean and std per group
                    group_stats = {}
                    for group_name in sorted(metric_data.keys()):
                        values = [v for v in metric_data[group_name] if not pd.isna(v)]
                        if values:
                            mean_val = np.mean(values)
                            std_val = np.std(values) if len(values) > 1 else 0
                            group_stats[group_name] = (mean_val, std_val)
                    
                    if group_stats:
                        max_group = max(group_stats.items(), key=lambda x: x[1][0])
                        min_group = min(group_stats.items(), key=lambda x: x[1][0])
                        
                        for group_name in sorted(group_stats.keys()):
                            mean_val, std_val = group_stats[group_name]
                            
                            # Handle both single values and tuples (for MultiIndex)
                            if isinstance(group_name, tuple):
                                group_label = " | ".join(str(g) for g in group_name)
                            else:
                                group_label = str(group_name)
                            
                            indicator = "★" if group_name == max_group[0] else "▼" if group_name == min_group[0] else " "
                            report += f"{indent}{indicator} {group_label:35s}: {mean_val:7.4f} ± {std_val:.4f}\n"
                        
                        # Show disparity
                        disparity = max_group[1][0] - min_group[1][0]
                        pct_diff = (disparity / max_group[1][0] * 100) if max_group[1][0] != 0 else 0
                        
                        # Format min group name
                        if isinstance(min_group[0], tuple):
                            min_group_label = " | ".join(str(g) for g in min_group[0])
                        else:
                            min_group_label = str(min_group[0])
                        
                        report += f"{indent}→ {min_group_label} disadvantaged by {abs(pct_diff):.2f}%\n"

        else:
            report += "No per-group metrics available.\n"
    else:
        report += "No fold results available.\n"
    
    report += "\nFAIRNESS ASSESSMENT:\n"
    report += "-" * 70 + "\n"
    
    if dpd_values and not all(pd.isna(dpd_values)):
        avg_dpd = abs(pd.Series(dpd_values).mean())
        if avg_dpd < 0.1:
            report += f"✓ Demographic Parity: FAIR (|diff| = {avg_dpd:.4f} < 0.1)\n"
        elif avg_dpd < 0.2:
            report += f"⚠ Demographic Parity: MODERATE BIAS (|diff| = {avg_dpd:.4f})\n"
        else:
            report += f"✗ Demographic Parity: SIGNIFICANT BIAS (|diff| = {avg_dpd:.4f})\n"
    
    if eod_values and not all(pd.isna(eod_values)):
        avg_eod = abs(pd.Series(eod_values).mean())
        if avg_eod < 0.1:
            report += f"✓ Equalized Odds: FAIR (|diff| = {avg_eod:.4f} < 0.1)\n"
        elif avg_eod < 0.2:
            report += f"⚠ Equalized Odds: MODERATE BIAS (|diff| = {avg_eod:.4f})\n"
        else:
            report += f"✗ Equalized Odds: SIGNIFICANT BIAS (|diff| = {avg_eod:.4f})\n"
    
    report += "=" * 70 + "\n"
    
    return report


def run_experiments(X, y, sensitive_features_df=None):
    models = {
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", n_jobs=1),
            {
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05, 0.1],
                "clf__n_estimators": [100, 200],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
            },
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=3000),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "NaiveBayes": (
            GaussianNB(),
            {},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7]},
        ),
        # SVM huge performance sink and awful results on this dataset
        #"SVM": (
        #    SVC(probability=True),
        #    {"clf__C": [0.1, 1, 10]},
        #),
        "RandomForest": (
            RandomForestClassifier(),
            {"clf__n_estimators": [100, 200]},
        ),

    }

    all_results = []
    fairness_analyzer = FairnessAnalyzer()
    all_fairness_reports = []

    for name, (clf, grid) in models.items():
        if name in ["XGBoost", "RandomForest"]:
            # Tree-based models: skip feature selection
            pipeline = Pipeline(
                steps=[
                    ("preprocess", Preprocessor()),
                    ("clf", clf),
                ]
            )
        else:
            # Other models: include feature selection
            pipeline = Pipeline(
                steps=[
                    ("preprocess", Preprocessor()),
                    ("selector", SelectKBest(mutual_info_classif)),
                    ("clf", clf),
                ]
            )
            # Use experimentally determined k values for each classifier
            if name == "LogisticRegression":
                grid["selector__k"] = [50]
            elif name == "NaiveBayes":
                grid["selector__k"] = [20]
            elif name == "KNN":
                grid["selector__k"] = [10]

        results = nested_cross_validation(
            pipeline,
            X,
            y,
            grid,
            outer_cv_splits=5,
            outer_cv_repeats=2,
            inner_cv=3,
            sensitive_features_df=sensitive_features_df
        )

        # Extract fold_results before creating DataFrame (it contains complex nested objects)
        fold_results_data = results.pop('fold_results', None)
        
        results_df = pd.DataFrame(results)
        results_df.insert(0, "model", name)
        all_results.append(results_df.copy())
        print(f'{results_df}\n')
        results_df.to_csv(RESULTS_DIR / f"{name}_results.csv", index=False, header=True)
        
        # Generate and save fairness report if sensitive features available
        if sensitive_features_df is not None:
            # Restore fold_results to results dict for fairness report generation
            if fold_results_data is not None:
                results['fold_results'] = fold_results_data
            fairness_report = _generate_model_fairness_report(
                name, results, fairness_analyzer
            )
            all_fairness_reports.append(fairness_report)
            print(fairness_report)
            
            # Save fairness metrics to CSV
            fairness_metrics_df = pd.DataFrame({
                'demographic_parity_difference': results.get('demographic_parity_difference', []),
                'equalized_odds_difference': results.get('equalized_odds_difference', []),
            })
            fairness_metrics_df.insert(0, 'fold', range(len(fairness_metrics_df)))
            fairness_metrics_df.to_csv(
                RESULTS_DIR / f"{name}_fairness.csv", 
                index=False, 
                header=True
            )
    
    # Statistical comparison of models
    perform_statistical_comparison(all_results)
    
    # Save combined fairness report
    if all_fairness_reports:
        combined_fairness_report = "\n".join(all_fairness_reports)
        report_path = RESULTS_DIR / "fairness_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(combined_fairness_report)
        logger.info(f"Fairness report saved to {report_path}")
        

def main():
    """Run experiments."""
    logger.info("=" * 60)
    logger.info("Adult Census Income Classification - Experiments")
    logger.info("=" * 60)

    adult_train_path = DATA_RAW_DIR / TRAIN_FILE
    adult_test_path = DATA_RAW_DIR / TEST_FILE

    df = load_data(adult_train_path, adult_test_path)
    df = df[:200]
    df = drop_missing_values(df)
    
    # Extract protected attributes before removing them from features
    sensitive_features_df = get_protected_attributes(df, sensitive_features=['sex', 'race'])
    logger.info(f"Extracted protected attributes: {sensitive_features_df.columns.tolist()}")
    
    X, y = get_feature_target_split(df)

    y = y.map({">50K": 1, "<=50K": 0})
    
    run_experiments(X, y, sensitive_features_df=sensitive_features_df)
    

if __name__ == "__main__":
    main()


