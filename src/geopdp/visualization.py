"""Visualization helpers for PDP results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from geopdp.utils import load_geodataframe

__all__ = ["choropleth_basic", "compare_geojson_geometry"]


def choropleth_basic(
    df: pd.DataFrame,
    *,
    locations_col: str,
    value_col: str,
    geojson: Path | str | gpd.GeoDataFrame,
    geojson_region_property: str,
    title: str | None,
    color_scale: str,
    range_color: Sequence[float] | None = None,
) -> go.Figure:
    """Create a Plotly choropleth for regional PDP values.

    Args:
        df: DataFrame containing the values to plot.
        locations_col: Column with region identifiers matching the GeoJSON.
        value_col: Column with the numeric values to render.
        geojson: Location of the GeoJSON file with geometries or a GeoDataFrame.
        geojson_region_property: GeoJSON feature property that stores region names.
        title: Optional chart title. Pass None to let Plotly handle it.
        color_scale: Plotly color scale name (e.g., Viridis, Cividis).
        range_color: Optional min/max range for the color scale.

    Returns:
        Rendered choropleth figure.
    """
    gdf = load_geodataframe(geojson, "GeoJSON")

    figure = px.choropleth(
        df,
        geojson=gdf,
        locations=locations_col,
        color=value_col,
        featureidkey=f"properties.{geojson_region_property}",
        color_continuous_scale=color_scale,
        range_color=range_color,
    )
    if title:
        figure.update_layout(title_text=title)
    return figure


def compare_geojson_geometry(
    original_geojson: Path | str | gpd.GeoDataFrame,
    alternate_geojson: Path | str | gpd.GeoDataFrame,
    title: str = "GeoJSON Simplification Comparison",
) -> go.Figure:
    """Plots original and simplified GeoJSONs side-by-side.

    Args:
        original_geojson: Path to a GeoJSON file or a GeoDataFrame for the
            original GeoJSON file.
        alternate_geojson: Path to a GeoJSON file or a GeoDataFrame for the
            alternate GeoJSON file.
        title: Title for the comparison plot.

    Returns:
        Side-by-side choropleth comparison.

    Raises:
        FileNotFoundError: If any of the input files do not exist.
    """
    gdf_orig = load_geodataframe(original_geojson, "Original GeoJSON").copy()
    gdf_simp = load_geodataframe(alternate_geojson, "Alternate GeoJSON").copy()

    gdf_orig["_dummy"] = 1
    gdf_simp["_dummy"] = 1

    fig = go.Figure()

    gdf_orig["Type"] = "Original"
    gdf_simp["Type"] = "Alternate"

    common_gdf = pd.concat(
        [gdf_orig[["geometry", "Type"]], gdf_simp[["geometry", "Type"]]]
    )

    common_gdf = common_gdf.reset_index(drop=True)
    common_gdf["id"] = common_gdf.index.astype(str)

    geojson_combined = json.loads(common_gdf.to_json())

    fig = px.choropleth(
        common_gdf,
        geojson=geojson_combined,
        locations="id",
        color="Type",
        featureidkey="properties.id",
        facet_col="Type",
        title=title,
    )

    fig.update_geos(fitbounds="locations", visible=False)

    return fig
