"""
Tests for data loading and dataset variant creation.

These tests use in-memory DataFrames and do not require actual dataset files.
"""

import numpy as np
import pandas as pd
import pytest

from census_ml.data.dataset_variants import make_dataset_variant
from census_ml.data.load_data import (
    _convert_missing_values,
    _standardize_target_labels,
    load_adult_dataset,
)


def test_target_label_cleaning():
    """Test that target labels are correctly standardized."""
    # Test with trailing periods (from UCI test file)
    series = pd.Series([">50K.", "<=50K.", ">50K", "<=50K"])
    result = _standardize_target_labels(series)

    assert result[0] == ">50K"
    assert result[1] == "<=50K"
    assert result[2] == ">50K"
    assert result[3] == "<=50K"


def test_target_label_cleaning_with_whitespace():
    """Test that target labels handle whitespace correctly."""
    series = pd.Series([" >50K. ", "  <=50K.  ", " >50K ", " <=50K"])
    result = _standardize_target_labels(series)

    assert result[0] == ">50K"
    assert result[1] == "<=50K"
    assert result[2] == ">50K"
    assert result[3] == "<=50K"


def test_missing_value_conversion():
    """Test that '?' is converted to NaN."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "workclass": ["Private", "?", "Government"],
            "occupation": ["?", "Sales", "Tech"],
        }
    )

    result = _convert_missing_values(df)

    # Check that '?' was converted to NaN
    assert result["age"].isnull().sum() == 0
    assert result["workclass"].isnull().sum() == 1
    assert result["occupation"].isnull().sum() == 1

    # Check that valid values are preserved
    assert result["age"].tolist() == [25, 30, 35]
    assert result["workclass"].iloc[0] == "Private"
    assert result["workclass"].iloc[2] == "Government"


def test_missing_value_conversion_strips_whitespace():
    """Test that whitespace is stripped from string columns."""
    df = pd.DataFrame(
        {
            "age": [25, 30],
            "occupation": [" Sales ", "  Tech  "],
        }
    )

    result = _convert_missing_values(df)

    assert result["occupation"].iloc[0] == "Sales"
    assert result["occupation"].iloc[1] == "Tech"


def test_load_missing_file_error():
    """Test that a clear error message is shown when files are missing."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        with pytest.raises(FileNotFoundError) as exc_info:
            load_adult_dataset(source="uci_split", data_path=tmppath)

        error_msg = str(exc_info.value)
        assert "UCI split files not found" in error_msg
        assert "adult.data" in error_msg
        assert "adult.test" in error_msg
        assert "wget" in error_msg


def test_load_missing_csv_error():
    """Test clear error message for missing CSV file."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        with pytest.raises(FileNotFoundError) as exc_info:
            load_adult_dataset(source="csv", data_path=tmppath)

        error_msg = str(exc_info.value)
        assert "CSV file not found" in error_msg
        assert "adult.csv" in error_msg


def test_make_dataset_variant_drop():
    """Test DROP variant removes rows with NaNs."""
    # Create test dataframe with some missing values
    df = pd.DataFrame(
        {
            "age": [25, 30, np.nan, 40, 35],
            "hours_per_week": [40, 45, 50, np.nan, 38],
            "workclass": ["Private", "Government", "Private", "Private", np.nan],
            "income": [">50K", "<=50K", ">50K", "<=50K", ">50K"],
        }
    )

    X, y, metadata = make_dataset_variant(df, variant="drop")

    # Should only keep rows with no missing values (rows 0 and none others are complete)
    # Actually row 0 is complete, row 1 is complete, row 4 has NaN in workclass
    # row 2 has NaN in age, row 3 has NaN in hours_per_week
    # So only rows 0 and 1 should remain
    assert len(X) == 2
    assert len(y) == 2
    assert metadata["rows_kept"] == 2
    assert metadata["rows_dropped"] == 3
    assert metadata["missing_values_remaining"] == 0


def test_make_dataset_variant_impute():
    """Test IMPUTE variant keeps NaNs."""
    df = pd.DataFrame(
        {
            "age": [25, 30, np.nan, 40],
            "hours_per_week": [40, 45, 50, np.nan],
            "workclass": ["Private", "Government", "Private", "Private"],
            "income": [">50K", "<=50K", ">50K", "<=50K"],
        }
    )

    X, y, metadata = make_dataset_variant(df, variant="impute")

    # Should keep all rows
    assert len(X) == 4
    assert len(y) == 4
    assert metadata["rows_kept"] == 4
    assert metadata["rows_dropped"] == 0

    # Should still have missing values
    assert metadata["missing_values_remaining"] == 2
    assert "age" in metadata["columns_with_missing"]
    assert "hours_per_week" in metadata["columns_with_missing"]


def test_target_binary_conversion():
    """Test that target is correctly converted to binary."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "hours_per_week": [40, 45, 50, 55],
            "income": [">50K", "<=50K", ">50K", "<=50K"],
        }
    )

    X, y, metadata = make_dataset_variant(df, variant="drop")

    # Check binary conversion
    assert y.dtype == int or y.dtype == np.int64
    assert list(y) == [1, 0, 1, 0]


def test_make_dataset_variant_class_balance():
    """Test that class balance is correctly computed."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50],
            "hours_per_week": [40, 45, 50, 55, 60, 65],
            "income": [">50K", ">50K", "<=50K", "<=50K", "<=50K", "<=50K"],
        }
    )

    X, y, metadata = make_dataset_variant(df, variant="drop")

    # 2 of class 1 (>50K), 4 of class 0 (<=50K)
    assert metadata["class_balance"]["class_0"] == 4
    assert metadata["class_balance"]["class_1"] == 2
    assert abs(metadata["class_balance"]["class_0_pct"] - 66.67) < 0.1
    assert abs(metadata["class_balance"]["class_1_pct"] - 33.33) < 0.1


def test_make_dataset_variant_invalid():
    """Test that invalid variant raises error."""
    df = pd.DataFrame(
        {
            "age": [25, 30],
            "income": [">50K", "<=50K"],
        }
    )

    with pytest.raises(ValueError) as exc_info:
        make_dataset_variant(df, variant="invalid")

    assert "Invalid variant" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
