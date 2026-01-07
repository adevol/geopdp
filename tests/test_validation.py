import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geopdp import compute_geopdp, define_midpoint_of_regions


@pytest.fixture
def sample_geojson_path(tmp_path):
    """Creates a temporary sample GeoJSON file."""
    gdf = gpd.GeoDataFrame(
        {"NAME_1": ["Region A", "Region B"], "geometry": [Point(0, 0), Point(1, 1)]}
    )
    path = tmp_path / "sample.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {"lon": [0, 1], "lat": [0, 1], "region": ["Region A", "Region B"]}
    )


class InvalidModel:
    """Model without predict or predict_proba."""

    pass


class ValidModel:
    """Valid model with predict."""

    def predict(self, X):
        return np.array([0, 1] * len(X))


def test_define_midpoint_missing_property_error(sample_geojson_path):
    """Test that missing property in GeoJSON raises informative KeyError."""
    with pytest.raises(KeyError, match="Region property 'MISSING_PROP' not found"):
        define_midpoint_of_regions(sample_geojson_path, "MISSING_PROP")


def test_compute_geopdp_invalid_model_error(sample_df, sample_geojson_path):
    """Test that model without required methods raises TypeError."""
    with pytest.raises(
        TypeError, match="must have either 'predict_proba' .* or 'predict'"
    ):
        compute_geopdp(
            sample_df,
            InvalidModel(),
            geojson=sample_geojson_path,
            region_col="region",
            geojson_region_property="NAME_1",
            lon_col="lon",
            lat_col="lat",
        )


def test_compute_geopdp_missing_columns_error(sample_df, sample_geojson_path):
    """Test that missing columns in DF raise informative KeyError."""
    # Drop required columns
    bad_df = sample_df.drop(columns=["lon"])

    with pytest.raises(KeyError, match="Required column.*lon.*not found"):
        compute_geopdp(
            bad_df,
            ValidModel(),
            geojson=sample_geojson_path,
            region_col="region",
            geojson_region_property="NAME_1",
            lon_col="lon",
            lat_col="lat",
        )


def test_compute_geopdp_missing_multiple_columns_error(sample_df, sample_geojson_path):
    """Test that multiple missing columns are listed in error."""
    bad_df = pd.DataFrame({"other": [1, 2]})

    with pytest.raises(KeyError, match="region.*lon.*lat"):
        compute_geopdp(
            bad_df,
            ValidModel(),
            geojson=sample_geojson_path,
            region_col="region",
            geojson_region_property="NAME_1",
            lon_col="lon",
            lat_col="lat",
        )
