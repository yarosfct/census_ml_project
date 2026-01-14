"""
Test that all key modules can be imported.

This ensures package integrity and that all dependencies are correctly installed.
"""

import pytest


def test_import_main_package():
    """Test importing the main package."""
    import census_ml

    assert hasattr(census_ml, "__version__")
    assert hasattr(census_ml, "RANDOM_SEED")
    assert hasattr(census_ml, "TARGET_COL")


def test_import_config():
    """Test importing config module."""
    from census_ml import config

    assert hasattr(config, "RANDOM_SEED")
    assert hasattr(config, "TARGET_COL")
    assert hasattr(config, "DATA_RAW_DIR")
    assert config.RANDOM_SEED == 42


def test_import_utils():
    """Test importing utils modules."""
    from census_ml.utils import get_logger

    logger = get_logger("test")
    assert logger is not None


def test_import_data():
    """Test importing data modules."""
    from census_ml.data import load_data

    assert hasattr(load_data, "load_raw_data")
    assert hasattr(load_data, "get_feature_target_split")


def test_import_features():
    """Test importing features modules."""
    from census_ml.features import preprocess

    assert hasattr(preprocess, "MissingValueHandler")


def test_import_models():
    """Test importing models modules."""
    from census_ml.models import model_zoo

    assert hasattr(model_zoo, "get_baseline_models")
    assert hasattr(model_zoo, "get_hyperparameter_grids")


def test_import_eval():
    """Test importing evaluation modules."""
    from census_ml.eval import nested_cv

    assert hasattr(nested_cv, "compute_classification_metrics")
    assert hasattr(nested_cv, "cross_validate_model")


def test_baseline_models_instantiation():
    """Test that baseline models can be instantiated."""
    from census_ml.models.model_zoo import get_baseline_models

    models = get_baseline_models()
    assert len(models) > 0
    assert "logistic_regression" in models
    assert "random_forest" in models


def test_logger_creation():
    """Test that logger can be created and configured."""
    from census_ml.utils.logging import get_logger

    logger1 = get_logger("test_logger_1")
    logger2 = get_logger("test_logger_2", level="DEBUG")

    assert logger1.name == "test_logger_1"
    assert logger2.name == "test_logger_2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
