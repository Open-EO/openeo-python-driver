import re

import pytest

from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.errors import OpenEOApiException
from .data import get_path
from pyproj import CRS


def test_feature_collection_bounds():
    dv = DelayedVector(str(get_path("geojson/FeatureCollection01.json")))
    assert dv.bounds == (4.45, 51.1, 4.52, 51.2)


def test_geometry_collection_bounds():
    dv = DelayedVector(str(get_path("geojson/GeometryCollection01.json")))
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


def test_geojson_url_invalid(requests_mock):
    requests_mock.get("https://dl.test/features.json", text="\n\n<p>not json<p>", headers={"Content-Type": "text/html"})
    dv = DelayedVector("https://dl.test/features.json")

    with pytest.raises(OpenEOApiException, match="Failed to parse GeoJSON from URL"):
        _ = dv.bounds


def test_geojson_invalid_coordinates():
    dv = DelayedVector(str(get_path("geojson/test_geojson_invalid_coordinates.geojson")))
    expected_error = "Failed to parse Geojson. Invalid coordinate: [-361.0, 50.861345984658136]"
    with pytest.raises(OpenEOApiException, match=re.escape(expected_error)):
        _ = dv.bounds
