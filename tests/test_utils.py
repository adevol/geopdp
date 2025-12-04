"""Tests for utility functions."""


import geopandas as gpd
import pytest

from geopdp.data import SAMPLE_REGIONS_GEOJSON
from geopdp.utils import load_geodataframe


def test_load_geodataframe_with_path():
    """Test loading GeoDataFrame from Path object."""
    gdf = load_geodataframe(SAMPLE_REGIONS_GEOJSON)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0


def test_load_geodataframe_with_string():
    """Test loading GeoDataFrame from string path."""
    gdf = load_geodataframe(str(SAMPLE_REGIONS_GEOJSON))
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0


def test_load_geodataframe_with_geodataframe():
    """Test passing GeoDataFrame through unchanged."""
    original_gdf = gpd.read_file(SAMPLE_REGIONS_GEOJSON)
    result_gdf = load_geodataframe(original_gdf)
    assert result_gdf is original_gdf  # Should be the same object
    assert isinstance(result_gdf, gpd.GeoDataFrame)


def test_load_geodataframe_nonexistent_file():
    """Test error handling for non-existent file."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_geodataframe("non_existent_file.geojson")

    assert "not found at" in str(exc_info.value)
    assert "Please verify" in str(exc_info.value)


def test_load_geodataframe_custom_error_label():
    """Test custom error label in error messages."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_geodataframe("missing.geojson", error_label="Custom Data")

    assert "Custom Data" in str(exc_info.value)
