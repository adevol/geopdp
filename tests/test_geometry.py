import geopandas as gpd
import pytest

from geopdp.data import SAMPLE_REGIONS_GEOJSON, TANZANIA_GEOJSON
from geopdp.geometry import simplify_geojson


def test_simplify_geojson_reduces_vertices():
    original = gpd.read_file(TANZANIA_GEOJSON)
    simplified = simplify_geojson(TANZANIA_GEOJSON, tolerance=0.2)

    def get_total_vertices(gdf):
        return sum(
            len(geom.exterior.coords)
            for geom in gdf.explode(index_parts=False).geometry
        )

    assert get_total_vertices(simplified) < get_total_vertices(original)


def test_simplify_geojson_reduces_precision():
    simplified = simplify_geojson(SAMPLE_REGIONS_GEOJSON, tolerance=0.0, precision=1.0)

    coords = simplified.geometry[0].exterior.coords
    for x, y in coords:
        assert x == int(x)
        assert y == int(y)

    with pytest.raises(FileNotFoundError):
        simplify_geojson("non_existent.geojson")


def test_simplify_json_tanzanian_data():
    from geopdp.data import TANZANIA_GEOJSON

    original = gpd.read_file(TANZANIA_GEOJSON)
    simplified = simplify_geojson(TANZANIA_GEOJSON, tolerance=0.05)

    def get_total_vertices(gdf):
        return sum(
            len(geom.exterior.coords)
            for geom in gdf.explode(index_parts=False).geometry
        )

    assert get_total_vertices(simplified) < get_total_vertices(original)
