"""Geometry utilities for GeoPandas datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import shapely

from geopdp.utils import load_geodataframe

__all__ = ["simplify_geojson"]

logger = logging.getLogger(__name__)


def simplify_geojson(
    geojson: Path | str | gpd.GeoDataFrame,
    tolerance: float = 0.01,
    precision: float | None = None,
) -> gpd.GeoDataFrame:
    """Simplifies GeoJSON geometries and reduces coordinate precision.

    Args:
        geojson: Path to a GeoJSON file or a GeoDataFrame to simplify.
        tolerance: Tolerance for simplification (higher = more simple).
        precision: Grid size for coordinate snapping (e.g. 1e-3 for 3 decimal
            places). If None, no precision reduction is applied.

    Returns:
        GeoDataFrame with simplified geometries.

    Raises:
        FileNotFoundError: If the GeoJSON path does not exist.
    """
    logger.info(
        f"Simplifying GeoJSON with tolerance={tolerance}, precision={precision}"
    )
    gdf = load_geodataframe(geojson, "GeoJSON").copy()

    gdf["geometry"] = gdf["geometry"].simplify(
        tolerance=tolerance, preserve_topology=True
    )

    if precision is not None:
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: shapely.set_precision(geom, grid_size=precision)
        )

    logger.info(f"Simplification complete. Output GeoDataFrame has {len(gdf)} features")
    return gdf
