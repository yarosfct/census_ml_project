"""
Quick check script to verify package installation and imports.

This script checks that all core modules can be imported and basic
functionality works as expected.
"""

import sys
from pathlib import Path

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from census_ml import RANDOM_SEED, TARGET_COL
from census_ml.config import DATA_RAW_DIR, PROJECT_ROOT
from census_ml.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Run quick checks."""
    logger.info("=" * 60)
    logger.info("Running quick check for census_ml package")
    logger.info("=" * 60)

    # Check imports
    logger.info("\n1. Checking imports...")
    try:
        from census_ml.data import load_data  # noqa: F401
        from census_ml.eval import nested_cv  # noqa: F401
        from census_ml.features import preprocess  # noqa: F401
        from census_ml.models import model_zoo

        logger.info("✓ All modules imported successfully")
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        sys.exit(1)

    # Check configuration
    logger.info("\n2. Checking configuration...")
    logger.info(f"  - Random seed: {RANDOM_SEED}")
    logger.info(f"  - Target column: {TARGET_COL}")
    logger.info(f"  - Project root: {PROJECT_ROOT}")
    logger.info(f"  - Data directory: {DATA_RAW_DIR}")
    logger.info("✓ Configuration loaded successfully")

    # Check models
    logger.info("\n3. Checking model zoo...")
    try:
        models = model_zoo.get_baseline_models()
        logger.info(f"  - Available models: {list(models.keys())}")
        logger.info("✓ Model zoo loaded successfully")
    except Exception as e:
        logger.error(f"✗ Model zoo failed: {e}")
        sys.exit(1)

    # Check directory structure
    logger.info("\n4. Checking directory structure...")
    required_dirs = [
        DATA_RAW_DIR,
        DATA_RAW_DIR.parent / "interim",
        DATA_RAW_DIR.parent / "processed",
        PROJECT_ROOT / "reports" / "figures",
        PROJECT_ROOT / "reports" / "tables",
        PROJECT_ROOT / "notebooks",
    ]

    for directory in required_dirs:
        if directory.exists():
            logger.info(f"  ✓ {directory.relative_to(PROJECT_ROOT)}")
        else:
            logger.warning(f"  ✗ {directory.relative_to(PROJECT_ROOT)} (missing)")

    logger.info("\n" + "=" * 60)
    logger.info("Quick check completed successfully!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Download the Adult Census Income dataset")
    logger.info(f"  2. Place it in {DATA_RAW_DIR}")
    logger.info("  3. Run experiments using src/scripts/run_experiments.py")


if __name__ == "__main__":
    main()
