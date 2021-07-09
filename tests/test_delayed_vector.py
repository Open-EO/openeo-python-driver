from openeo_driver.delayed_vector import DelayedVector
from .data import get_path
from pyproj import CRS


def test_feature_collection_bounds():
    dv = DelayedVector(str(get_path("FeatureCollection.geojson")))
    assert dv.bounds == (4.45, 51.1, 4.52, 51.2)


def test_geometry_collection_bounds():
    dv = DelayedVector(str(get_path("GeometryCollection.geojson")))
    assert dv.bounds == (5.05, 51.21, 5.15, 51.3)

def test_geojson_crs_unspecified():
    dv = DelayedVector(str(get_path("geojson/test_geojson_crs_unspecified.geojson")))
    assert dv.crs == CRS.from_user_input("+init=epsg:4326")

def test_geojson_crs_from_epsg():
    dv = DelayedVector(str(get_path("geojson/test_geojson_crs_from_epsg.geojson")))
    assert dv.crs == CRS.from_user_input("+init=epsg:4326")

def test_geojson_crs_from_ogc_urn():
    dv = DelayedVector(str(get_path("geojson/test_geojson_crs_from_ogc_urn.geojson")))
    assert dv.crs == CRS.from_user_input("+init=epsg:4326")
