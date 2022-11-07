import math

import pyproj
import pytest
import shapely.geometry
from numpy.testing import assert_allclose
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geos import WKTWriter

from openeo_driver.util.geometry import (
    geojson_to_multipolygon,
    reproject_bounding_box,
    spatial_extent_union,
    GeometryBufferer,
    as_geojson_feature,
    as_geojson_feature_collection,
)


from ..data import get_path

EARTH_CIRCUMFERENCE_KM = 40075.017


def test_geojson_to_multipolygon():
    poly1_coords = [(0, 0), (1, 0), (1, 1), (0, 0)]
    poly2_coords = [(0, 2), (1, 2), (1, 3), (0, 2)]
    poly3_coords = [(2, 0), (3, 0), (3, 1), (2, 0)]
    gj_poly_1 = {"type": "Polygon", "coordinates": (poly1_coords,)}
    gj_poly_2 = {"type": "Polygon", "coordinates": (poly2_coords,)}
    gj_poly_3 = {"type": "Polygon", "coordinates": (poly3_coords,)}
    gj_multipoly_p1_p2 = {
        "type": "MultiPolygon",
        "coordinates": [(poly1_coords,), (poly2_coords,)],
    }
    gj_geometry_collection_p1_p2 = {
        "type": "GeometryCollection",
        "geometries": [gj_poly_1, gj_poly_2],
    }
    gj_geometry_collection_mp12_p3 = {
        "type": "GeometryCollection",
        "geometries": [gj_multipoly_p1_p2, gj_poly_3],
    }
    gj_geometry_collection_p3 = {
        "type": "GeometryCollection",
        "geometries": [gj_poly_3],
    }
    gj_feature_p1 = {"type": "Feature", "geometry": gj_poly_1}
    gj_feature_mp12 = {"type": "Feature", "geometry": gj_multipoly_p1_p2}
    gj_feature_gc12 = {"type": "Feature", "geometry": gj_geometry_collection_p1_p2}
    gj_feature_gc123 = {"type": "Feature", "geometry": gj_geometry_collection_mp12_p3}
    gj_featcol_p1_p2 = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": gj_poly_1},
            {"type": "Feature", "geometry": gj_poly_2},
        ],
    }
    gj_featcol_mp12_p3 = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": gj_multipoly_p1_p2},
            {"type": "Feature", "geometry": gj_poly_3},
        ],
    }
    gj_featcol_gc12_p3 = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": gj_geometry_collection_p1_p2},
            {"type": "Feature", "geometry": gj_poly_3},
        ],
    }
    gj_featcol_mp12_gc3 = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": gj_multipoly_p1_p2},
            {"type": "Feature", "geometry": gj_geometry_collection_p3},
        ],
    }

    def assert_equal_multipolygon(a, b):
        assert isinstance(a, shapely.geometry.MultiPolygon)
        assert isinstance(b, shapely.geometry.MultiPolygon)
        assert len(a) == len(b)
        assert a.equals(b), f"{a!s} ?= {b!s}"

    poly_1 = shapely.geometry.Polygon(poly1_coords)
    poly_2 = shapely.geometry.Polygon(poly2_coords)
    poly_3 = shapely.geometry.Polygon(poly3_coords)
    multipoly_p1_p2 = shapely.geometry.MultiPolygon([poly_1, poly_2])
    multipoly_p1_p2_p3 = shapely.geometry.MultiPolygon([poly_1, poly_2, poly_3])

    assert geojson_to_multipolygon(gj_poly_1) == poly_1
    assert geojson_to_multipolygon(gj_feature_p1) == poly_1

    assert_equal_multipolygon(
        geojson_to_multipolygon(gj_multipoly_p1_p2), multipoly_p1_p2
    )
    assert_equal_multipolygon(
        geojson_to_multipolygon(gj_geometry_collection_p1_p2), multipoly_p1_p2
    )
    assert_equal_multipolygon(
        geojson_to_multipolygon(gj_geometry_collection_mp12_p3), multipoly_p1_p2_p3
    )
    assert_equal_multipolygon(geojson_to_multipolygon(gj_feature_mp12), multipoly_p1_p2)
    assert_equal_multipolygon(geojson_to_multipolygon(gj_feature_gc12), multipoly_p1_p2)
    assert_equal_multipolygon(
        geojson_to_multipolygon(gj_feature_gc123), multipoly_p1_p2_p3
    )
    assert_equal_multipolygon(
        geojson_to_multipolygon(gj_featcol_p1_p2), multipoly_p1_p2
    )
    assert_equal_multipolygon(
        geojson_to_multipolygon(gj_featcol_mp12_p3), multipoly_p1_p2_p3
    )
    assert_equal_multipolygon(
        geojson_to_multipolygon(gj_featcol_gc12_p3), multipoly_p1_p2_p3
    )
    assert_equal_multipolygon(
        geojson_to_multipolygon(gj_featcol_mp12_gc3), multipoly_p1_p2_p3
    )


@pytest.mark.parametrize(
    ["crs", "bbox"],
    [
        (
            "EPSG:32631",
            {"west": 640800, "south": 5676000, "east": 642200, "north": 5677000},
        ),
        ("EPSG:4326", {"west": 5.01, "south": 51.2, "east": 5.1, "north": 51.5}),
    ],
)
def test_reproject_bounding_box_same(crs, bbox):
    reprojected = reproject_bounding_box(bbox, from_crs=crs, to_crs=crs)
    assert reprojected == dict(crs=crs, **bbox)


def test_reproject_bounding_box():
    bbox = {"west": 640800, "south": 5676000, "east": 642200.0, "north": 5677000.0}
    reprojected = reproject_bounding_box(
        bbox, from_crs="EPSG:32631", to_crs="EPSG:4326"
    )
    assert reprojected == {
        "west": pytest.approx(5.016118467277098),
        "south": pytest.approx(51.217660146353246),
        "east": pytest.approx(5.036548264535997),
        "north": pytest.approx(51.22699369149726),
        "crs": "EPSG:4326",
    }


def test_spatial_extent_union():
    assert spatial_extent_union({"west": 1, "south": 51, "east": 2, "north": 52}) == {
        "west": 1,
        "south": 51,
        "east": 2,
        "north": 52,
        "crs": "EPSG:4326",
    }
    assert spatial_extent_union(
        {"west": 1, "south": 51, "east": 2, "north": 52},
        {"west": 3, "south": 53, "east": 4, "north": 54},
    ) == {"west": 1, "south": 51, "east": 4, "north": 54, "crs": "EPSG:4326"}
    assert spatial_extent_union(
        {"west": 1, "south": 51, "east": 2, "north": 52},
        {"west": -5, "south": 50, "east": 3, "north": 53},
        {"west": 3, "south": 53, "east": 4, "north": 54},
    ) == {"west": -5, "south": 50, "east": 4, "north": 54, "crs": "EPSG:4326"}


def test_spatial_extent_union_mixed_crs():
    bbox1 = {
        "west": 640860.0,
        "south": 5676170.0,
        "east": 642140.0,
        "north": 5677450.0,
        "crs": "EPSG:32631",
    }
    bbox2 = {
        "west": 5.017,
        "south": 51.21,
        "east": 5.035,
        "north": 51.23,
        "crs": "EPSG:4326",
    }
    assert spatial_extent_union(bbox1, bbox2) == {
        "west": 640824.9450876965,
        "south": 5675111.354480841,
        "north": 5677450.0,
        "east": 642143.1739784481,
        "crs": "EPSG:32631",
    }
    assert spatial_extent_union(bbox2, bbox1) == {
        "west": 5.017,
        "south": 51.21,
        "east": 5.035868037661473,
        "north": 51.2310228422003,
        "crs": "EPSG:4326",
    }


def to_wkt(geometry: BaseGeometry, rounding_precision=2):
    wkt_writer = WKTWriter(shapely.geos.lgeos, rounding_precision=rounding_precision)
    return wkt_writer.write(geometry)


class TestGeometryBufferer:
    def test_basic_point(self):
        bufferer = GeometryBufferer(distance=1)
        point = Point(2, 3)
        polygon = bufferer.buffer(point)
        assert isinstance(polygon, Polygon)
        assert_allclose(
            polygon.exterior.coords,
            [
                (3, 3),
                (2.7, 2.3),
                (2, 2),
                (1.3, 2.3),
                (1, 3),
                (1.3, 3.7),
                (2, 4),
                (2.7, 3.7),
                (3, 3),
            ],
            atol=0.01,
        )

    @pytest.mark.parametrize(
        ["resolution", "expected"],
        [
            (1, [(8, 8), (5, 5), (2, 8), (5, 11), (8, 8)]),
            (
                2,
                [
                    (8.0, 8.0),
                    (7.12, 5.88),
                    (5.0, 5.0),
                    (2.88, 5.88),
                    (2.0, 8.0),
                    (2.88, 10.12),
                    (5.0, 11.0),
                    (7.12, 10.12),
                    (8.0, 8.0),
                ],
            ),
        ],
    )
    def test_resolution(self, resolution, expected):
        bufferer = GeometryBufferer(distance=3, resolution=resolution)
        point = Point(5, 8)
        polygon = bufferer.buffer(point)
        assert isinstance(polygon, Polygon)
        assert_allclose(polygon.exterior.coords, expected, atol=0.01)

    @pytest.mark.parametrize(
        ["distance", "expected"],
        [
            (0, 0),
            (1, 360 / (EARTH_CIRCUMFERENCE_KM * 1000)),
            (1000, 360 / EARTH_CIRCUMFERENCE_KM),
        ],
    )
    def test_transform_meters_to_lonlat_default(self, distance, expected):
        distance = GeometryBufferer.transform_meter_to_crs(
            distance=distance, crs="EPSG:4326"
        )
        assert_allclose(distance, expected, rtol=1e-3)

    def test_transform_meters_to_pyproj_crs(self):
        distance = GeometryBufferer.transform_meter_to_crs(
            distance=1000,
            crs=pyproj.CRS.from_epsg(32631),
            loi=(3, 0),
            loi_crs=pyproj.CRS.from_epsg(4326),
        )
        assert_allclose(distance, 1000, rtol=1e-3)

    @pytest.mark.parametrize(
        ["latitude", "expected"],
        [
            (0, 360 / EARTH_CIRCUMFERENCE_KM),
            # Rough back-of-envelope calculation for circumference at latitude
            (52, 360 / (EARTH_CIRCUMFERENCE_KM * math.cos(52 * math.pi / 180))),
            (75, 360 / (EARTH_CIRCUMFERENCE_KM * math.cos(75 * math.pi / 180))),
        ],
    )
    def test_transform_meters_to_lonlat_latitude_correction(self, latitude, expected):
        distance = GeometryBufferer.transform_meter_to_crs(
            distance=1000, crs="EPSG:4326", loi=(3, latitude)
        )
        assert_allclose(distance, expected, rtol=1e-2)

    def test_transform_meters_to_utm_roundtrip(self):
        distance = GeometryBufferer.transform_meter_to_crs(
            distance=1000,
            crs="epsg:32631",
            loi=(4, 52),
            loi_crs="EPSG:4326",
        )
        assert_allclose(distance, 1000, rtol=1e-6)

    def test_from_meter_for_crs(self):
        bufferer = GeometryBufferer.from_meter_for_crs(distance=1000, crs="EPSG:4326")
        point = Point(4, 51)
        polygon = bufferer.buffer(point)
        assert isinstance(polygon, Polygon)

        assert_allclose(
            polygon.exterior.coords,
            [
                (4.009, 51),
                (4.006, 50.99),
                (4, 50.99),
                (3.994, 50.99),
                (3.991, 51),
                (3.994, 51.01),
                (4, 51.01),
                (4.006, 51.01),
                (4.009, 51),
            ],
            atol=0.01,
        )


class TestAsGeoJsonFeatureCollection:
    def test_as_geojson_feature_point(self):
        feature = as_geojson_feature(Point(1, 2))
        assert feature == {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
            "properties": None,
        }

    def test_as_geojson_feature_point_with_properties(self):
        feature = as_geojson_feature(Point(1, 2), properties={"color": "red"})
        assert feature == {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
            "properties": {"color": "red"},
        }

    def test_as_geojson_feature_dict(self):
        feature = as_geojson_feature({"type": "Point", "coordinates": (1.0, 2.0)})
        assert feature == {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
            "properties": None,
        }

    def test_as_geojson_feature_dict_with_properties(self):
        feature = as_geojson_feature(
            {"type": "Point", "coordinates": (1.0, 2.0)}, properties={"color": "red"}
        )
        assert feature == {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
            "properties": {"color": "red"},
        }

    def test_as_geojson_feature_path(self):
        feature = as_geojson_feature(get_path("geojson/Polygon01.json"))
        assert feature == {
            "type": "Feature",
            "geometry": {
                "coordinates": [
                    [
                        [5.1, 51.22],
                        [5.11, 51.23],
                        [5.14, 51.21],
                        [5.12, 51.2],
                        [5.1, 51.22],
                    ]
                ],
                "type": "Polygon",
            },
            "properties": None,
        }

    def test_as_geojson_feature_from_feature(self):
        feature = as_geojson_feature(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
                "properties": None,
            }
        )
        assert feature == {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
            "properties": None,
        }

    def test_as_geojson_feature_from_feature_with_properties(self):
        feature = as_geojson_feature(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
                "properties": {"color": "red", "size": 4},
            },
        )
        assert feature == {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
            "properties": {"color": "red", "size": 4},
        }

    def test_as_geojson_feature_from_feature_with_properties_override(self):
        feature = as_geojson_feature(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
                "properties": {"color": "red", "size": 4},
            },
            properties={"color": "GREEN!", "shape": "circle"},
        )
        assert feature == {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
            "properties": {"color": "GREEN!", "shape": "circle"},
        }

    def test_as_geojson_feature_collection_simple_point(self):
        feature_collection = as_geojson_feature_collection(Point(1, 2))
        assert feature_collection == {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
                    "properties": None,
                }
            ],
        }

    def test_as_geojson_feature_collection_multiple(self):
        feature_collection = as_geojson_feature_collection(
            Point(1, 2), Polygon.from_bounds(1, 2, 3, 4), Point(5, 6)
        )
        assert feature_collection == {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
                    "properties": None,
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": (
                            (
                                (1.0, 2.0),
                                (1.0, 4.0),
                                (3.0, 4.0),
                                (3.0, 2.0),
                                (1.0, 2.0),
                            ),
                        ),
                    },
                    "properties": None,
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": (5.0, 6.0)},
                    "properties": None,
                },
            ],
        }

    def test_as_geojson_feature_collection_multiple_as_list(self):
        feature_collection = as_geojson_feature_collection(
            [Point(1, 2), Polygon.from_bounds(1, 2, 3, 4), Point(5, 6)]
        )
        assert feature_collection == {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
                    "properties": None,
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": (
                            (
                                (1.0, 2.0),
                                (1.0, 4.0),
                                (3.0, 4.0),
                                (3.0, 2.0),
                                (1.0, 2.0),
                            ),
                        ),
                    },
                    "properties": None,
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": (5.0, 6.0)},
                    "properties": None,
                },
            ],
        }

    def test_as_geojson_feature_collection_mix(self):
        feature_collection = as_geojson_feature_collection(
            {"type": "Point", "coordinates": (1.0, 2.0)},
            Polygon.from_bounds(1, 2, 3, 4),
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": (5.0, 6.0)},
                "properties": None,
            },
            get_path("geojson/Polygon01.json"),
        )
        assert feature_collection == {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
                    "properties": None,
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": (
                            (
                                (1.0, 2.0),
                                (1.0, 4.0),
                                (3.0, 4.0),
                                (3.0, 2.0),
                                (1.0, 2.0),
                            ),
                        ),
                    },
                    "properties": None,
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": (5.0, 6.0)},
                    "properties": None,
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "coordinates": [
                            [
                                [5.1, 51.22],
                                [5.11, 51.23],
                                [5.14, 51.21],
                                [5.12, 51.2],
                                [5.1, 51.22],
                            ]
                        ],
                        "type": "Polygon",
                    },
                    "properties": None,
                },
            ],
        }
