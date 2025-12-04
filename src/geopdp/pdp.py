"""Partial dependency plotting utilities for GeoPandas datasets."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import TANZANIA_GEOJSON
from .visualization import choropleth_basic

logger = logging.getLogger(__name__)


class ProbabilisticClassifier(Protocol):
    """Protocol for models with predict_proba method."""

    def predict_proba(self, X: Any) -> np.ndarray: ...


def define_midpoint_of_regions(
    geojson_path: Path | str | None = None,
    geojson_region_property: str = "NAME_1",
) -> dict[str, tuple[float, float]]:
    """Compute representative point coordinates for each region in a GeoJSON file.

    Args:
        geojson_path: Path to a GeoJSON containing region geometries. Defaults to
            the bundled Tanzania example when None.
        geojson_region_property: Property name in GeoJSON features that identifies
            the region.

    Returns:
        Dictionary mapping region names to (longitude, latitude) tuples.

    Raises:
        ImportError: If GeoPandas is not installed.
        FileNotFoundError: If the GeoJSON file cannot be located.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("GeoPandas is required for midpoint calculations.") from exc

    if geojson_path is None:
        geojson_path = TANZANIA_GEOJSON
    else:
        geojson_path = Path(geojson_path)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found at {geojson_path}")

    gdf = gpd.read_file(geojson_path)
    midpoints: dict[str, tuple[float, float]] = {}

    for idx in gdf.index:
        region_name = gdf.loc[idx, geojson_region_property]
        geometry = gdf.loc[idx, "geometry"]
        longitude, latitude = geometry.representative_point().coords[:][0]
        midpoints[region_name] = (longitude, latitude)

    return midpoints


def make_copy_of_dataset_with_midpoints(
    X: pd.DataFrame,
    new_region: str,
    midpoint_coords: Sequence[float],
    region_col: str = "region",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """Return a dataset copy where coordinates match the provided midpoint.

    Args:
        X: Original dataset.
        new_region: Name of the region to stamp on the copy.
        midpoint_coords: Tuple or list with (longitude, latitude).
        region_col: Name of the column in X that holds region labels.
        lon_col: Name of the longitude column.
        lat_col: Name of the latitude column.

    Returns:
        Dataset copy with updated longitude, latitude, and region values
        (if region column exists).

    Raises:
        ValueError: If midpoint_coords does not contain exactly two values.
        KeyError: If required columns are missing from the DataFrame.
    """
    if len(midpoint_coords) != 2:
        raise ValueError("midpoint_coords must contain exactly two values.")

    missing_cols = []
    if lon_col not in X.columns:
        missing_cols.append(lon_col)
    if lat_col not in X.columns:
        missing_cols.append(lat_col)

    if missing_cols:
        raise KeyError(
            f"Required column(s) {missing_cols} not found in DataFrame. "
            f"Available columns: {list(X.columns)}"
        )

    longitude, latitude = midpoint_coords
    X_copy = X.copy()
    X_copy[lon_col] = longitude
    X_copy[lat_col] = latitude
    if region_col in X_copy.columns:
        X_copy[region_col] = new_region
    return X_copy


def _create_permutation_dataset(
    X: pd.DataFrame,
    midpoints_coords: dict[str, tuple[float, float]],
    lon_col: str,
    lat_col: str,
    region_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a dataset with all region permutations for vectorized inference.

    Args:
        X: Original dataset.
        midpoints_coords: Dictionary mapping region names to (lon, lat) tuples.
        lon_col: Name of the longitude column.
        lat_col: Name of the latitude column.
        region_col: Name of the region column.

    Returns:
        Tuple of (expanded dataset, list of region names).
    """
    n_regions = len(midpoints_coords)
    n_samples = len(X)

    big_X = pd.concat([X] * n_regions, ignore_index=True)

    all_lons = []
    all_lats = []
    all_regions = []

    sorted_regions = sorted(midpoints_coords.items())

    for region, (lon, lat) in sorted_regions:
        all_lons.extend([lon] * n_samples)
        all_lats.extend([lat] * n_samples)
        all_regions.extend([region] * n_samples)

    big_X[lon_col] = all_lons
    big_X[lat_col] = all_lats

    if region_col in big_X.columns:
        big_X[region_col] = all_regions

    return big_X, all_regions


def _aggregate_pdp_results(
    preds: np.ndarray,
    col_index_to_predict: int,
    all_regions: list[str],
    region_col: str,
) -> pd.DataFrame:
    """Aggregate predictions by region.

    Args:
        preds: Prediction array from model.
        col_index_to_predict: Column index to extract from predictions.
        all_regions: List of region names corresponding to predictions.
        region_col: Name of the region column.

    Returns:
        DataFrame with region names and averaged predictions.

    Raises:
        ValueError: If col_index_to_predict is out of bounds.
    """
    if preds.ndim != 2 or col_index_to_predict >= preds.shape[1]:
        raise ValueError("col_index_to_predict exceeds available prediction columns.")

    target_preds = preds[:, col_index_to_predict]
    results_df = pd.DataFrame({region_col: all_regions, "prediction": target_preds})
    return results_df.groupby(region_col, as_index=False)["prediction"].mean()


def compute_geopdp(
    X: pd.DataFrame,
    pipe: ProbabilisticClassifier,
    *,
    col_index_to_predict: int = 0,
    geojson_path: Path | str | None = None,
    region_col: str = "region",
    geojson_region_property: str = "NAME_1",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """Generate PDP predictions by sweeping over each region's midpoint.

    Args:
        X: Input dataset used for making predictions.
        pipe: Trained model or pipeline exposing a predict_proba method.
        col_index_to_predict: Probability column to average. Defaults to 0
            (first class). Only needed for multi-class classification.
        geojson_path: GeoJSON file used to derive region midpoints. Defaults to
            the bundled Tanzania example when None.
        region_col: Name of the column in X that holds region labels.
        geojson_region_property: Property name in GeoJSON features that identifies
            the region.
        lon_col: Name of the longitude column.
        lat_col: Name of the latitude column.

    Returns:
        DataFrame mapping region names to averaged prediction probabilities.

    Raises:
        ValueError: When col_index_to_predict is outside the pipeline output.
    """
    logger.info(f"Computing GeoPDP for {len(X)} samples across regions")
    midpoints_coords = define_midpoint_of_regions(
        geojson_path, geojson_region_property=geojson_region_property
    )
    n_regions = len(midpoints_coords)
    logger.info(f"Processing {n_regions} regions (Vectorized)")

    big_X, all_regions = _create_permutation_dataset(
        X, midpoints_coords, lon_col, lat_col, region_col
    )

    preds = pipe.predict_proba(big_X)

    final_df = _aggregate_pdp_results(
        preds, col_index_to_predict, all_regions, region_col
    )

    logger.info("GeoPDP computation complete")
    return final_df


def plot_geopdp(
    df: pd.DataFrame,
    geojson: Path | str | gpd.GeoDataFrame | None = None,
    region_col: str = "region",
    geojson_region_property: str = "NAME_1",
    title: str | None = None,
    color_scale: str = "Viridis",
    range_color: Sequence[float] | None = None,
) -> go.Figure:
    """Plot a choropleth visualizing PDP values by region.

    Args:
        df: DataFrame containing PDP results with a region column.
        geojson: Path to the GeoJSON or GeoDataFrame containing region
            boundaries. Defaults to the bundled Tanzania example when None.
        region_col: Name of the column with region labels in df.
        geojson_region_property: GeoJSON feature property containing region names.
        title: Optional figure title.
        color_scale: Plotly color scale name.
        range_color: Optional min/max range for the color scale.

    Returns:
        Plotly Figure object with the choropleth visualization.
    """
    if geojson is None:
        geojson = TANZANIA_GEOJSON

    return choropleth_basic(
        df,
        locations_col=region_col,
        value_col="prediction",
        geojson=geojson,
        geojson_region_property=geojson_region_property,
        title=title or "PDP Values by Region",
        color_scale=color_scale,
        range_color=range_color,
    )
