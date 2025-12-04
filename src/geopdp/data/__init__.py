import importlib.resources
from pathlib import Path


def _get_path(filename: str) -> Path:
    return Path(importlib.resources.files("geopdp.data") / filename)


TANZANIA_GEOJSON = _get_path("tanzania.geojson")
SAMPLE_REGIONS_GEOJSON = _get_path("sample_regions.geojson")
