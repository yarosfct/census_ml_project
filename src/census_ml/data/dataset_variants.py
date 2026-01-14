"""
Dataset inspection and variant creation utilities.

This module provides functions to inspect the Adult dataset and create
two clean dataset variants: DROP (missing rows removed) and IMPUTE (NaNs preserved).
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from census_ml.config import (
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    NUMERICAL_FEATURES,
    TABLES_DIR,
    TARGET_COL,
)
from census_ml.utils.logging import get_logger

logger = get_logger(__name__)


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Perform comprehensive dataset inspection and generate summary statistics.

    Creates CSV tables and visualizations saved to reports/ directory.

    Args:
        df: Input dataframe to inspect

    Returns:
        Dictionary containing summary statistics
    """
    logger.info("Starting dataset inspection...")

    # Ensure output directories exist
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Basic shape info
    n_rows, n_cols = df.shape
    n_features = n_cols - 1  # Exclude target

    # 2. Column types
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]

    # 3. Missingness analysis
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    missingness_df = pd.DataFrame(
        {
            "column": missing_counts.index,
            "missing_count": missing_counts.values,
            "missing_percentage": missing_percentages.values,
        }
    )
    missingness_df = missingness_df[missingness_df["missing_count"] > 0].sort_values(
        "missing_percentage", ascending=False
    )
    missingness_df.to_csv(TABLES_DIR / "missingness.csv", index=False)
    logger.info(f"Saved missingness stats to {TABLES_DIR / 'missingness.csv'}")

    # 4. Target distribution
    target_dist = df[TARGET_COL].value_counts()
    target_pct = (target_dist / len(df)) * 100
    target_df = pd.DataFrame(
        {
            "class": target_dist.index,
            "count": target_dist.values,
            "percentage": target_pct.values,
        }
    )
    target_df.to_csv(TABLES_DIR / "target_distribution.csv", index=False)
    logger.info(f"Saved target distribution to {TABLES_DIR / 'target_distribution.csv'}")

    # 5. Categorical cardinality
    cardinality_data = []
    for col in categorical_cols:
        n_unique = df[col].nunique()
        cardinality_data.append({"column": col, "unique_values": n_unique})
    cardinality_df = pd.DataFrame(cardinality_data).sort_values("unique_values", ascending=False)
    cardinality_df.to_csv(TABLES_DIR / "categorical_cardinality.csv", index=False)
    logger.info(f"Saved categorical cardinality to {TABLES_DIR / 'categorical_cardinality.csv'}")

    # 6. Numerical statistics
    numeric_stats = df[numerical_cols].describe().T
    numeric_stats["column"] = numeric_stats.index
    numeric_stats = numeric_stats[
        ["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    numeric_stats.to_csv(TABLES_DIR / "numeric_stats.csv", index=False)
    logger.info(f"Saved numeric stats to {TABLES_DIR / 'numeric_stats.csv'}")

    # === Create visualizations ===

    # Plot 1: Missing percentage by column (if any)
    if len(missingness_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.barh(missingness_df["column"], missingness_df["missing_percentage"], color="coral")
        plt.xlabel("Missing Percentage (%)")
        plt.ylabel("Column")
        plt.title("Missing Values by Column")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "missingness_percentage.png", dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {FIGURES_DIR / 'missingness_percentage.png'}")

    # Plot 2: Target distribution
    plt.figure(figsize=(8, 6))
    plt.bar(target_df["class"], target_df["count"], color=["skyblue", "lightcoral"])
    plt.xlabel("Income Class")
    plt.ylabel("Count")
    plt.title("Target Distribution")
    for i, (_, cnt, pct) in enumerate(
        zip(
            target_df["class"],
            target_df["count"],
            target_df["percentage"],
            strict=True,
        )
    ):
        plt.text(
            i,
            cnt,
            f"{cnt}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {FIGURES_DIR / 'target_distribution.png'}")

    # Plot 3-6: Histograms for key numeric features
    key_features = ["age", "hours_per_week", "capital_gain", "capital_loss"]
    for feature in key_features:
        if feature in df.columns:
            plt.figure(figsize=(8, 6))
            plt.hist(df[feature].dropna(), bins=50, edgecolor="black", color="steelblue", alpha=0.7)
            plt.xlabel(feature.replace("_", " ").title())
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {feature.replace('_', ' ').title()}")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{feature}_histogram.png", dpi=100, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved plot to {FIGURES_DIR / f'{feature}_histogram.png'}")

    # Compile summary dictionary
    summary = {
        "shape": {"rows": n_rows, "columns": n_cols, "features": n_features},
        "column_types": {
            "categorical": categorical_cols,
            "numerical": numerical_cols,
        },
        "missingness": {
            "total_missing_values": int(missing_counts.sum()),
            "columns_with_missing": list(missingness_df["column"].values),
            "missing_percentages": dict(
                zip(
                    missingness_df["column"],
                    missingness_df["missing_percentage"],
                    strict=True,
                )
            ),
        },
        "target_distribution": dict(zip(target_df["class"], target_df["count"], strict=True)),
        "categorical_cardinality": dict(
            zip(
                cardinality_df["column"],
                cardinality_df["unique_values"],
                strict=True,
            )
        ),
    }

    logger.info("Dataset inspection complete.")
    return summary


def make_dataset_variant(
    df: pd.DataFrame,
    variant: Literal["impute", "drop"],
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Create a dataset variant with different missing value handling strategies.

    Args:
        df: Input dataframe with features and target
        variant: Strategy to use:
            - "drop": Remove rows with any missing values
            - "impute": Keep NaNs for future pipeline-based imputation (no leakage)

    Returns:
        Tuple of (X, y, metadata) where:
        - X: Feature matrix
        - y: Binary target (1 = '>50K', 0 = '<=50K')
        - metadata: Dict with variant statistics
    """
    logger.info(f"Creating '{variant}' dataset variant...")

    # Separate features and target
    X = df.drop(columns=[TARGET_COL]).copy()
    y_raw = df[TARGET_COL].copy()

    # Convert target to binary ('>50K' -> 1, '<=50K' -> 0)
    y = (y_raw == ">50K").astype(int)

    initial_rows = len(X)

    # Apply variant-specific handling
    if variant == "drop":
        # Remove rows with any missing values
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]

        rows_kept = len(X)
        rows_dropped = initial_rows - rows_kept

        logger.info(
            f"DROP variant: kept {rows_kept}/{initial_rows} rows ({rows_dropped} rows dropped)"
        )

        metadata = {
            "variant": "drop",
            "initial_rows": initial_rows,
            "rows_kept": rows_kept,
            "rows_dropped": rows_dropped,
            "missing_values_remaining": 0,
            "class_balance": {
                "class_0": int((y == 0).sum()),
                "class_1": int((y == 1).sum()),
                "class_0_pct": float((y == 0).mean() * 100),
                "class_1_pct": float((y == 1).mean() * 100),
            },
            "note": "All rows with missing values removed. Ready for modeling.",
        }

    elif variant == "impute":
        # Keep NaNs for future imputation in Pipeline
        rows_kept = len(X)
        missing_count = X.isnull().sum().sum()

        logger.info(
            f"IMPUTE variant: kept all {rows_kept} rows ({missing_count} missing values preserved)"
        )

        metadata = {
            "variant": "impute",
            "initial_rows": initial_rows,
            "rows_kept": rows_kept,
            "rows_dropped": 0,
            "missing_values_remaining": int(missing_count),
            "columns_with_missing": list(X.columns[X.isnull().any()]),
            "missing_per_column": dict(X.isnull().sum()[X.isnull().sum() > 0]),
            "class_balance": {
                "class_0": int((y == 0).sum()),
                "class_1": int((y == 1).sum()),
                "class_0_pct": float((y == 0).mean() * 100),
                "class_1_pct": float((y == 1).mean() * 100),
            },
            "note": (
                "Missing values preserved as NaN. Must use SimpleImputer "
                "in sklearn Pipeline to avoid leakage."
            ),
        }

    else:
        raise ValueError(f"Invalid variant: {variant}. Must be 'drop' or 'impute'")

    return X, y, metadata


def dataset_report(
    df: pd.DataFrame,
    variant_outputs: dict,
    out_dir: Path | None = None,
) -> None:
    """
    Generate a comprehensive dataset summary report in Markdown format.

    Args:
        df: Original dataframe
        variant_outputs: Dictionary with variant results (keys: 'drop', 'impute')
        out_dir: Output directory for the report (default: REPORTS_DIR)
    """
    if out_dir is None:
        from census_ml.config import REPORTS_DIR

        out_dir = REPORTS_DIR

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_file = out_dir / "dataset_summary.md"

    logger.info(f"Generating dataset report at {report_file}...")

    with open(report_file, "w") as f:
        f.write("# Adult Census Income Dataset - Summary Report\n\n")

        # Overall stats
        f.write("## Overall Dataset Statistics\n\n")
        f.write(f"- **Total Rows**: {len(df):,}\n")
        f.write(f"- **Total Columns**: {df.shape[1]}\n")
        f.write(f"- **Features**: {df.shape[1] - 1}\n")
        f.write(f"  - Numerical: {len(NUMERICAL_FEATURES)}\n")
        f.write(f"  - Categorical: {len(CATEGORICAL_FEATURES)}\n\n")

        # Missing values
        missing = df.isnull().sum()
        total_missing = missing.sum()
        f.write("### Missing Values\n\n")
        f.write(f"- **Total Missing Values**: {total_missing:,}\n")
        if total_missing > 0:
            f.write("- **Columns with Missing Values**:\n\n")
            for col, count in missing[missing > 0].items():
                pct = (count / len(df)) * 100
                f.write(f"  - `{col}`: {count:,} ({pct:.2f}%)\n")
        f.write("\n")

        # Target distribution
        f.write("### Target Distribution\n\n")
        target_counts = df[TARGET_COL].value_counts()
        for cls, count in target_counts.items():
            pct = (count / len(df)) * 100
            f.write(f"- `{cls}`: {count:,} ({pct:.2f}%)\n")
        f.write("\n")

        # Variant summaries
        f.write("## Dataset Variants\n\n")
        f.write("Two dataset variants have been prepared for different modeling approaches:\n\n")

        for variant_name, outputs in variant_outputs.items():
            metadata = outputs["metadata"]
            f.write(f"### Variant: {variant_name.upper()}\n\n")

            f.write(f"- **Strategy**: {metadata['variant'].title()}\n")
            f.write(f"- **Rows Kept**: {metadata['rows_kept']:,}\n")
            f.write(f"- **Rows Dropped**: {metadata['rows_dropped']:,}\n")
            f.write(f"- **Missing Values Remaining**: {metadata['missing_values_remaining']:,}\n")

            f.write("\n**Class Balance**:\n")
            cb = metadata["class_balance"]
            f.write(f"- Class 0 (<=50K): {cb['class_0']:,} ({cb['class_0_pct']:.2f}%)\n")
            f.write(f"- Class 1 (>50K): {cb['class_1']:,} ({cb['class_1_pct']:.2f}%)\n")

            if "columns_with_missing" in metadata and metadata["columns_with_missing"]:
                cols_str = ", ".join([f"`{c}`" for c in metadata["columns_with_missing"]])
                f.write(f"\n**Columns with Missing Values**: {cols_str}\n")

            f.write(f"\n**Note**: {metadata['note']}\n\n")

        # Leakage-safe approach
        f.write("## Leakage Prevention\n\n")
        f.write("To prevent data leakage, preprocessing follows these principles:\n\n")
        f.write(
            "1. **IMPUTE variant**: Missing values are preserved as NaN. "
            "Imputation must be performed inside an sklearn Pipeline during "
            "cross-validation, ensuring imputation parameters are fitted only "
            "on training folds.\n\n"
        )
        f.write(
            "2. **DROP variant**: Complete-case analysis. All rows with missing "
            "values removed. No imputation needed, but sample size is reduced.\n\n"
        )
        f.write(
            "3. **No preprocessing fitted on full data**: This inspection phase "
            "only examines the data; no preprocessing transformations have been "
            "fitted that could leak information from test sets.\n\n"
        )

        # Generated artifacts
        f.write("## Generated Artifacts\n\n")
        f.write("### Tables (CSV)\n")
        f.write(f"- `{TABLES_DIR.name}/missingness.csv` - Missing value statistics\n")
        f.write(f"- `{TABLES_DIR.name}/target_distribution.csv` - Target class distribution\n")
        f.write(
            f"- `{TABLES_DIR.name}/categorical_cardinality.csv` - "
            f"Unique values per categorical feature\n"
        )
        f.write(
            f"- `{TABLES_DIR.name}/numeric_stats.csv` - "
            f"Descriptive statistics for numeric features\n\n"
        )

        f.write("### Figures (PNG)\n")
        f.write(f"- `{FIGURES_DIR.name}/missingness_percentage.png`\n")
        f.write(f"- `{FIGURES_DIR.name}/target_distribution.png`\n")
        f.write(f"- `{FIGURES_DIR.name}/age_histogram.png`\n")
        f.write(f"- `{FIGURES_DIR.name}/hours_per_week_histogram.png`\n")
        f.write(f"- `{FIGURES_DIR.name}/capital_gain_histogram.png`\n")
        f.write(f"- `{FIGURES_DIR.name}/capital_loss_histogram.png`\n\n")

    logger.info(f"Report saved to {report_file}")
