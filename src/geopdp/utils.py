"""Common utility functions for the geopdp package."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

__all__ = ["load_geodataframe"]


def load_geodataframe(
    path_or_gdf: Path | str | gpd.GeoDataFrame,
    error_label: str = "GeoJSON",
) -> gpd.GeoDataFrame:
    """Load a GeoDataFrame from a path or return it if already a GeoDataFrame.

    Args:
        path_or_gdf: Either a GeoDataFrame or a path to a GeoJSON file.
        error_label: Label to use in error messages.

    Returns:
        The loaded or passed GeoDataFrame.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if isinstance(path_or_gdf, gpd.GeoDataFrame):
        return path_or_gdf
    else:
        path = Path(path_or_gdf)
        if not path.exists():
            raise FileNotFoundError(
                f"{error_label} not found at {path}. "
                f"Please verify the path exists and you have read permissions."
            )
        return gpd.read_file(path)
