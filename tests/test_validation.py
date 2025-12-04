"""Test for DataFrame column validation."""

import pandas as pd
import pytest

from geopdp import make_copy_of_dataset_with_midpoints


def test_make_copy_validates_column_existence():
    """Test that missing columns raise KeyError with helpful message."""
    df = pd.DataFrame({"foo": [1, 2, 3]})

    with pytest.raises(KeyError) as exc_info:
        make_copy_of_dataset_with_midpoints(df, "Test", (10.0, 20.0))

    error_msg = str(exc_info.value)
    assert "longitude" in error_msg or "latitude" in error_msg
    assert "Available columns" in error_msg


def test_make_copy_validates_custom_columns():
    """Test validation works with custom column names."""
    df = pd.DataFrame({"wrong_col": [1, 2]})

    with pytest.raises(KeyError) as exc_info:
        make_copy_of_dataset_with_midpoints(
            df, "Test", (10.0, 20.0), lon_col="x", lat_col="y"
        )

    error_msg = str(exc_info.value)
    assert "x" in error_msg or "y" in error_msg
