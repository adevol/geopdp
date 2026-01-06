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

from .utils import load_geodataframe
from .visualization import choropleth_basic

logger = logging.getLogger(__name__)


class ProbabilisticClassifier(Protocol):
    """Protocol for models with predict_proba method."""

    def predict_proba(self, X: Any) -> np.ndarray: ...


class Regressor(Protocol):
    """Protocol for models with predict method (regressors)."""

    def predict(self, X: Any) -> np.ndarray: ...


def _is_regressor(model: ProbabilisticClassifier | Regressor) -> bool:
    """Detect if model is a regressor (no predict_proba, only predict).

    Args:
        model: A trained model object.

    Returns:
        True if model is a regressor (has predict but not predict_proba).
    """
    return not hasattr(model, "predict_proba") and hasattr(model, "predict")


def define_midpoint_of_regions(
    geojson: Path | str | gpd.GeoDataFrame,
    geojson_region_property: str,
) -> dict[str, tuple[float, float]]:
    """Compute representative point coordinates for each region in a GeoJSON file.

    Args:
        geojson: Path to a GeoJSON file or a GeoDataFrame containing region geometries.
        geojson_region_property: Property name in the GeoJSON that identifies regions.

    Returns:
        Dictionary mapping region names to (longitude, latitude) tuples.

    Raises:
        FileNotFoundError: If the GeoJSON path does not exist.
    """
    gdf = load_geodataframe(geojson, "GeoJSON")
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
    region_col: str,
    lon_col: str,
    lat_col: str,
) -> pd.DataFrame:
    """Return a dataset copy where coordinates match the provided midpoint.

    Args:
        X: Original dataset.
        new_region: Name of the region to stamp on the copy.
        midpoint_coords: Tuple or list with (longitude, latitude).
        region_col: Name of the column containing region identifiers.
        lon_col: Name of the longitude column.
        lat_col: Name of the latitude column.

    Returns:
        DataFrame copy with updated region and coordinates.

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


def _extract_target_predictions(
    preds: np.ndarray,
    col_index: int | None,
) -> np.ndarray:
    """Extract target predictions from model output.

    Handles three cases:
    - 1D array (single-output regression): returns as-is
    - 2D array with col_index: returns specified column
    - 2D array without col_index: returns first column (multi-output regression)

    Args:
        preds: Prediction array from model.
        col_index: Column index to extract. None for regression models.

    Returns:
        1D array of target predictions.

    Raises:
        ValueError: If col_index exceeds available columns.
    """
    if preds.ndim == 1:
        return preds

    if col_index is None:
        # Multi-output regression: default to first output
        return preds[:, 0]

    if col_index >= preds.shape[1]:
        raise ValueError(
            f"col_index_to_predict ({col_index}) exceeds available columns "
            f"({preds.shape[1]})."
        )
    return preds[:, col_index]


def _aggregate_pdp_results(
    preds: np.ndarray,
    col_index_to_predict: int | None,
    all_regions: list[str],
    region_col: str,
) -> pd.DataFrame:
    """Aggregate predictions by region.

    Args:
        preds: Prediction array from model (1D or 2D).
        col_index_to_predict: Column index to extract from predictions.
        all_regions: List of region names corresponding to predictions.
        region_col: Name of the region column.

    Returns:
        DataFrame with region names and averaged predictions.
    """
    target_preds = _extract_target_predictions(preds, col_index_to_predict)
    results_df = pd.DataFrame({region_col: all_regions, "prediction": target_preds})
    return results_df.groupby(region_col, as_index=False)["prediction"].mean()


def compute_geopdp(
    X: pd.DataFrame,
    pipe: ProbabilisticClassifier | Regressor,
    *,
    geojson: Path | str | gpd.GeoDataFrame,
    region_col: str,
    geojson_region_property: str,
    lon_col: str,
    lat_col: str,
    col_index_to_predict: int | None = None,
) -> pd.DataFrame:
    """Generate PDP predictions by sweeping over each region's midpoint.

    Automatically detects whether the model is a classifier or regressor:
    - Classifiers (with `predict_proba`): uses probability predictions
    - Regressors (with only `predict`): uses direct predictions

    Args:
        X: Input dataset used for making predictions.
        pipe: Trained model or pipeline. Can be a classifier with `predict_proba`
            or a regressor with `predict`.
        geojson: Path to a GeoJSON file or a GeoDataFrame defining region boundaries.
        region_col: Column name in X that contains region labels.
        geojson_region_property: Property in the GeoJSON that matches region_col.
        lon_col: Column name in X for longitude.
        lat_col: Column name in X for latitude.
        col_index_to_predict: Index of the class probability to return (e.g. 1 for
            positive class). Defaults to 0 for binary classification. Required for
            multi-class problems. Ignored for regression models.

    Returns:
        DataFrame with two columns: region and prediction (mean predicted value).

    Raises:
        ValueError: When col_index_to_predict is outside the pipeline output, or
            when it is missing for multi-class problems.
    """
    logger.info(f"Computing GeoPDP for {len(X)} samples across regions")
    midpoints_coords = define_midpoint_of_regions(
        geojson, geojson_region_property=geojson_region_property
    )
    n_regions = len(midpoints_coords)
    logger.info(f"Processing {n_regions} regions (Vectorized)")

    big_X, all_regions = _create_permutation_dataset(
        X, midpoints_coords, lon_col, lat_col, region_col
    )

    is_regressor = _is_regressor(pipe)

    if is_regressor:
        logger.info("Detected regression model, using predict()")
        preds = pipe.predict(big_X)
        col_index_to_predict = None
    else:
        logger.info("Detected classification model, using predict_proba()")
        preds = pipe.predict_proba(big_X)
        if col_index_to_predict is None:
            if preds.shape[1] == 2:
                col_index_to_predict = 0
            else:
                raise ValueError(
                    "col_index_to_predict must be specified for multi-class "
                    f"problems (found {preds.shape[1]} classes)."
                )

    final_df = _aggregate_pdp_results(
        preds, col_index_to_predict, all_regions, region_col
    )

    logger.info("GeoPDP computation complete")
    return final_df


def plot_geopdp(
    df: pd.DataFrame,
    *,
    geojson: Path | str | gpd.GeoDataFrame,
    region_col: str,
    geojson_region_property: str,
    color_scale: str,
    title: str | None = None,
    range_color: Sequence[float] | None = None,
) -> go.Figure:
    """Plot a choropleth visualizing PDP values by region.

    Args:
        df: DataFrame containing PDP results with a region column.
        geojson: Path to the GeoJSON or GeoDataFrame containing region
            boundaries.
        region_col: Column in df containing region names.
        geojson_region_property: Property in the GeoJSON that matches region_col.
        color_scale: Name of the Plotly color scale to use (e.g. "Viridis").
        title: Title of the chart.
        range_color: Optional min/max range for the color scale.

    Returns:
        Plotly Figure object.
    """

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
