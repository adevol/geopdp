"""Tests for visualization functions."""

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

from geopdp.data import SAMPLE_REGIONS_GEOJSON
from geopdp.visualization import choropleth_basic, compare_geojson_geometry


def test_choropleth_basic_with_path():
    """Test choropleth creation with file path."""
    df = pd.DataFrame({"region": ["A", "B", "C"], "value": [10, 20, 30]})

    fig = choropleth_basic(
        df,
        locations_col="region",
        value_col="value",
        geojson=SAMPLE_REGIONS_GEOJSON,
        geojson_region_property="NAME",
        title="Test Choropleth",
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Choropleth"


def test_choropleth_basic_with_geodataframe():
    """Test choropleth creation with GeoDataFrame."""
    df = pd.DataFrame({"region": ["A", "B", "C"], "value": [10, 20, 30]})

    gdf = gpd.read_file(SAMPLE_REGIONS_GEOJSON)

    fig = choropleth_basic(
        df,
        locations_col="region",
        value_col="value",
        geojson=gdf,
        geojson_region_property="NAME",
        title=None,
    )

    assert isinstance(fig, go.Figure)


def test_choropleth_basic_custom_color_scale():
    """Test choropleth with custom color scale."""
    df = pd.DataFrame({"region": ["A", "B"], "value": [10, 20]})

    fig = choropleth_basic(
        df,
        locations_col="region",
        value_col="value",
        geojson=SAMPLE_REGIONS_GEOJSON,
        geojson_region_property="NAME",
        title="Custom Colors",
        color_scale="Blues",
        range_color=[0, 100],
    )

    assert isinstance(fig, go.Figure)


def test_compare_geojson_geometry_with_paths():
    """Test geometry comparison with file paths."""
    fig = compare_geojson_geometry(
        SAMPLE_REGIONS_GEOJSON,
        SAMPLE_REGIONS_GEOJSON,  # Compare to itself for test
        title="Comparison Test",
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Comparison Test"


def test_compare_geojson_geometry_with_geodataframes():
    """Test geometry comparison with GeoDataFrames."""
    gdf_orig = gpd.read_file(SAMPLE_REGIONS_GEOJSON)
    gdf_alt = gpd.read_file(SAMPLE_REGIONS_GEOJSON)

    fig = compare_geojson_geometry(gdf_orig, gdf_alt, title="GDF Comparison")

    assert isinstance(fig, go.Figure)


def test_compare_geojson_geometry_mixed_inputs():
    """Test geometry comparison with mixed input types."""
    gdf = gpd.read_file(SAMPLE_REGIONS_GEOJSON)

    fig = compare_geojson_geometry(
        SAMPLE_REGIONS_GEOJSON, gdf, title="Mixed Input Test"  # Path  # GeoDataFrame
    )

    assert isinstance(fig, go.Figure)


def test_compare_geojson_geometry_no_side_effects():
    """Test that input GeoDataFrames are not mutated."""
    gdf_orig = gpd.read_file(SAMPLE_REGIONS_GEOJSON)
    gdf_alt = gpd.read_file(SAMPLE_REGIONS_GEOJSON)

    # Store original columns
    orig_cols = list(gdf_orig.columns)
    alt_cols = list(gdf_alt.columns)

    compare_geojson_geometry(gdf_orig, gdf_alt)

    # Verify columns haven't changed (no _dummy, Type, id added)
    assert list(gdf_orig.columns) == orig_cols
    assert list(gdf_alt.columns) == alt_cols
    assert "Type" not in gdf_orig.columns
