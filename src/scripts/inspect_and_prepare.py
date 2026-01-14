"""
Inspect and prepare dataset variants for the Adult Census Income dataset.

This script performs the following:
1. Loads the Adult dataset (auto-detecting format)
2. Runs comprehensive dataset inspection (EDA)
3. Creates two dataset variants (DROP and IMPUTE)
4. Generates reports and visualizations

Usage:
    python -m src.scripts.inspect_and_prepare
    python -m src.scripts.inspect_and_prepare --source uci_split
    python -m src.scripts.inspect_and_prepare --source csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from census_ml.data.dataset_variants import (
    dataset_report,
    inspect_dataset,
    make_dataset_variant,
)
from census_ml.data.load_data import load_adult_dataset
from census_ml.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Run dataset inspection and preparation."""
    parser = argparse.ArgumentParser(
        description="Inspect and prepare Adult Census Income dataset variants"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["uci_split", "csv", "auto"],
        default="auto",
        help="Data source format (default: auto-detect)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Adult Census Income Dataset - Inspection and Preparation")
    logger.info("=" * 70)

    # Step 1: Load dataset
    logger.info(f"\nStep 1: Loading dataset (source={args.source})...")
    try:
        df = load_adult_dataset(source=args.source)
    except FileNotFoundError as e:
        logger.error(f"\n{e}")
        sys.exit(1)

    # Step 2: Inspect dataset
    logger.info("\nStep 2: Running dataset inspection and EDA...")
    summary = inspect_dataset(df)

    # Print basic summary to console
    logger.info("\n" + "=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Shape: {summary['shape']['rows']:,} rows × {summary['shape']['columns']} columns")
    n_num = len(summary["column_types"]["numerical"])
    n_cat = len(summary["column_types"]["categorical"])
    logger.info(
        f"Features: {summary['shape']['features']} ({n_num} numerical + {n_cat} categorical)"
    )

    logger.info(f"\nMissing Values: {summary['missingness']['total_missing_values']:,} total")
    if summary["missingness"]["columns_with_missing"]:
        logger.info("  Affected columns:")
        for col in summary["missingness"]["columns_with_missing"]:
            pct = summary["missingness"]["missing_percentages"][col]
            logger.info(f"    - {col}: {pct:.2f}%")

    logger.info("\nTarget Distribution:")
    for cls, count in summary["target_distribution"].items():
        pct = (count / summary["shape"]["rows"]) * 100
        logger.info(f"  - {cls}: {count:,} ({pct:.2f}%)")

    # Step 3: Create dataset variants
    logger.info("\nStep 3: Creating dataset variants...")

    variant_outputs = {}

    # DROP variant
    X_drop, y_drop, meta_drop = make_dataset_variant(df, variant="drop")
    variant_outputs["drop"] = {
        "X": X_drop,
        "y": y_drop,
        "metadata": meta_drop,
    }

    # IMPUTE variant
    X_impute, y_impute, meta_impute = make_dataset_variant(df, variant="impute")
    variant_outputs["impute"] = {
        "X": X_impute,
        "y": y_impute,
        "metadata": meta_impute,
    }

    # Print variant summaries
    logger.info("\n" + "=" * 70)
    logger.info("DATASET VARIANTS")
    logger.info("=" * 70)

    for variant_name, outputs in variant_outputs.items():
        meta = outputs["metadata"]
        logger.info(f"\n{variant_name.upper()} Variant:")
        logger.info(f"  Rows: {meta['rows_kept']:,} (dropped: {meta['rows_dropped']:,})")
        logger.info(f"  Missing values: {meta['missing_values_remaining']:,}")
        cb0 = meta["class_balance"]["class_0"]
        cb1 = meta["class_balance"]["class_1"]
        logger.info(f"  Class balance: {cb0:,} (<=50K) vs {cb1:,} (>50K)")
        logger.info(f"  Note: {meta['note']}")

    # Step 4: Generate comprehensive report
    logger.info("\nStep 4: Generating comprehensive report...")
    dataset_report(df, variant_outputs)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("ARTIFACTS GENERATED")
    logger.info("=" * 70)

    from census_ml.config import FIGURES_DIR, REPORTS_DIR, TABLES_DIR

    logger.info(f"\nReport: {REPORTS_DIR / 'dataset_summary.md'}")

    logger.info(f"\nTables ({TABLES_DIR}):")
    logger.info("  - missingness.csv")
    logger.info("  - target_distribution.csv")
    logger.info("  - categorical_cardinality.csv")
    logger.info("  - numeric_stats.csv")

    logger.info(f"\nFigures ({FIGURES_DIR}):")
    logger.info("  - missingness_percentage.png")
    logger.info("  - target_distribution.png")
    logger.info("  - age_histogram.png")
    logger.info("  - hours_per_week_histogram.png")
    logger.info("  - capital_gain_histogram.png")
    logger.info("  - capital_loss_histogram.png")

    logger.info("\n" + "=" * 70)
    logger.info("Dataset inspection and preparation complete!")
    logger.info("=" * 70)

    logger.info("\nNext steps:")
    logger.info("  1. Review the generated report: reports/dataset_summary.md")
    logger.info("  2. Examine visualizations in reports/figures/")
    logger.info("  3. Proceed with model training using the prepared variants")


if __name__ == "__main__":
    main()
