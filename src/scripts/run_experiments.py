"""
Main script for running experiments.

This script will orchestrate the full ML pipeline:
- Load data
- Preprocess
- Train models
- Evaluate and compare

(To be implemented in later milestones)
"""

import sys
from pathlib import Path

# Add census_ml_project to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from census_ml.features.preprocess import Preprocessor
from census_ml.eval.nested_cv import nested_cross_validation

from census_ml.utils.logging import get_logger
from census_ml.config import DATA_RAW_DIR, TRAIN_FILE, TEST_FILE, RESULTS_DIR
from census_ml.data.load_data import load_data, get_feature_target_split

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
    """
    # Combine all results into one DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Get unique models
    models = combined_df['model'].unique()
    
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON OF MODELS")
    print("="*60)
    print("Using Wilcoxon signed-rank test on ROC-AUC scores (paired by CV fold)")
    print("Alpha = 0.05 (not adjusted for multiple comparisons)")
    print()
    
    # For each pair of models
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:  # Avoid duplicate comparisons
                # Get ROC-AUC scores for each model
                scores1 = combined_df[combined_df['model'] == model1]['roc_auc'].values
                scores2 = combined_df[combined_df['model'] == model2]['roc_auc'].values
                
                # Perform Wilcoxon test
                try:
                    stat, p_value = stats.wilcoxon(scores1, scores2)
                    
                    # Determine if significant
                    significant = "YES" if p_value < 0.05 else "NO"
                    
                    print(f"{model1} vs {model2}:")
                    print(".4f")
                    print(".4f")
                    print(f"  Significant difference: {significant}")
                    print()
                    
                except ValueError as e:
                    print(f"Could not compare {model1} vs {model2}: {e}")
                    print()

def run_experiments(X, y):
    models = {
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
    }

    all_results = []

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
            grid["selector__k"] = [10, 20, 35, 50]

        results = nested_cross_validation(
            pipeline,
            X,
            y,
            grid,
            outer_cv_splits=5,
            outer_cv_repeats=2,
            inner_cv=3
        )

        results_df = pd.DataFrame(results)
        results_df.insert(0, "model", name)
        all_results.append(results_df.copy())
        print(f'{results_df}\n')
        results_df.to_csv(RESULTS_DIR / f"{name}_results.csv", index=False, header=True)
    
    # Statistical comparison of models
    perform_statistical_comparison(all_results)
        

def main():
    """Run experiments."""
    logger.info("=" * 60)
    logger.info("Adult Census Income Classification - Experiments")
    logger.info("=" * 60)


    adult_train_path = DATA_RAW_DIR / TRAIN_FILE
    adult_test_path = DATA_RAW_DIR / TEST_FILE

    df = load_data(adult_train_path, adult_test_path)
    #df = drop_missing_values(df)
    X, y = get_feature_target_split(df)

    y = y.map({">50K": 1, "<=50K": 0})
    
    run_experiments(X, y)
    
    # Future implementation:
    # 5. Compare models statistically
    # 6. Generate reports and visualizations


if __name__ == "__main__":
    main()


