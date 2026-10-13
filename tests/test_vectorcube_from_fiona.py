"""
Tests for loading vector files through `DriverVectorCube.from_fiona`
(replacing the former `DelayedVector` tests: bounds, CRS handling, error handling).
"""

import pytest
import shapely.geometry
from pyproj import CRS

from openeo_driver.datacube import DriverVectorCube
from openeo_driver.errors import OpenEOApiException

from .data import get_path


def test_feature_collection_bounds():
    cube = DriverVectorCube.from_fiona([str(get_path("geojson/FeatureCollection01.json"))])
    assert cube.get_bounding_box() == (4.45, 51.1, 4.52, 51.2)


def test_geometry_collection_bounds():
    cube = DriverVectorCube.from_fiona([str(get_path("geojson/GeometryCollection01.json"))])
    assert cube.get_bounding_box() == (5.05, 51.21, 5.15, 51.3)


def test_geojson_crs_unspecified():
    cube = DriverVectorCube.from_fiona([str(get_path("geojson/test_geojson_crs_unspecified.geojson"))])
    assert cube.get_crs() == CRS.from_user_input("epsg:4326")


def test_geojson_crs_from_epsg():
    cube = DriverVectorCube.from_fiona([str(get_path("geojson/test_geojson_crs_from_epsg.geojson"))])
    assert cube.get_crs() == CRS.from_user_input("epsg:4326")


def test_geojson_crs_from_ogc_urn():
    cube = DriverVectorCube.from_fiona([str(get_path("geojson/test_geojson_crs_from_ogc_urn.geojson"))])
    assert cube.get_crs() == CRS.from_user_input("epsg:4326")


def test_geojson_crs_from_utm_epsg():
    cube = DriverVectorCube.from_fiona([str(get_path("geojson/test_geojson_crs_from_epsg_utm.geojson"))])
    assert cube.get_crs() == CRS.from_user_input("epsg:32631")
    # Geometries are kept in the original CRS (UTM, so no lat/lon coordinates)
    (geometry, *_) = cube.get_geometries()
    assert isinstance(geometry, shapely.geometry.base.BaseGeometry)
    west, south, east, north = cube.get_bounding_box()
    assert abs(west) > 180 or abs(east) > 180


def test_geojson_url_invalid(requests_mock):
    requests_mock.get("https://dl.test/features.json", text="\n\n<p>not json<p>", headers={"Content-Type": "text/html"})
    with pytest.raises(OpenEOApiException, match="load_url failed for"):
        DriverVectorCube.from_fiona(["https://dl.test/features.json"])


def test_geojson_invalid_coordinates():
    # TODO: `DelayedVector` rejected these with "Failed to parse Geojson. Invalid coordinate: [-361.0, ...]",
    #       `from_fiona` does not validate coordinates (yet)
    cube = DriverVectorCube.from_fiona([str(get_path("geojson/test_geojson_invalid_coordinates.geojson"))])
    assert cube.geometry_count() > 0

