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

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from census_ml.features.preprocess import Preprocessor
from census_ml.eval.nested_cv import nested_cross_validation

from census_ml.utils.logging import get_logger
from census_ml.config import COLUMN_NAMES, DATA_RAW_DIR, TRAIN_FILE, TARGET_COL, RESULTS_DIR

logger = get_logger(__name__)

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
        "SVM": (
            SVC(probability=True),
            {"clf__C": [0.1, 1, 10]},
        ),
        "RandomForest": (
            RandomForestClassifier(),
            {"clf__n_estimators": [100, 200]},
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss"),
            {"clf__max_depth": [3, 5]},
        ),
    }

    results = {}

    for name, (clf, grid) in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", Preprocessor()),
                ("clf", clf),
            ]
        )

        metrics = nested_cross_validation(
            pipeline,
            X,
            y,
            grid,
            outer_cv=2,
            inner_cv=2,
        )

        results[name] = metrics
        print(f"{name}: {metrics}")

    return results

def main():
    """Run experiments."""
    logger.info("=" * 60)
    logger.info("Adult Census Income Classification - Experiments")
    logger.info("=" * 60)


    adult_data_path = DATA_RAW_DIR / TRAIN_FILE
    df = pd.read_csv(
        adult_data_path,
        sep=",",
        skipinitialspace=True,
        header=None,
        names=COLUMN_NAMES
    )
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    y = y.map({">50K": 1, "<=50K": 0})

    results = run_experiments(X, y)

    results_path = RESULTS_DIR / "experiment_results.txt"
    with open(results_path, "w") as f:
        f.write(pd.DataFrame(results).to_string())
    
    # Future implementation:
    # 5. Compare models statistically
    # 6. Generate reports and visualizations


if __name__ == "__main__":
    main()


