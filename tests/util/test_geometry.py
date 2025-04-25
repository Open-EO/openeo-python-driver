import contextlib
import math
from typing import List, Union

import pyproj
import pytest
import shapely.geometry
from numpy.testing import assert_allclose
from pyproj.exceptions import CRSError
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry


from openeo_driver.util.geometry import (
    BoundingBox,
    BoundingBoxException,
    CrsRequired,
    GeometryBufferer,
    as_geojson_feature,
    as_geojson_feature_collection,
    geojson_to_multipolygon,
    reproject_bounding_box,
    reproject_geometry,
    spatial_extent_union,
    validate_geojson_basic,
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
        assert len(a.geoms) == len(b.geoms)
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


def test_reproject_geometry():
    geometry = shapely.geometry.GeometryCollection(
        [
            shapely.geometry.Point(5.0, 51.0),
            shapely.geometry.LineString([(5.0, 51.0), (5.1, 51.1)]),
        ]
    )
    expected_reprojected = shapely.geometry.GeometryCollection(
        [
            shapely.geometry.Point(640333.2963383198, 5651728.68267166),
            shapely.geometry.LineString(
                [
                    (640333.2963383198, 5651728.68267166),
                    (647032.2494640718, 5663042.764772809),
                ]
            ),
        ]
    )
    reprojected = reproject_geometry(
        geometry, from_crs="EPSG:4326", to_crs="EPSG:32631"
    )
    assert reprojected.equals(expected_reprojected)

    # Individual reprojected geometries should be equal to the reprojected collection.
    reprojected_geoms = []
    for geom in geometry.geoms:
        geom = reproject_geometry(geom, from_crs="EPSG:4326", to_crs="EPSG:32631")
        reprojected_geoms.append(geom)
    reprojected_expected = shapely.geometry.GeometryCollection(reprojected_geoms)
    assert reprojected.equals(reprojected_expected)

    # Inverse
    reprojected = reproject_geometry(
        expected_reprojected, from_crs="EPSG:32631", to_crs="EPSG:4326"
    )
    # Reprojection introduces very small errors.
    assert not reprojected.within(geometry)
    assert reprojected.within(geometry.buffer(1e-6))


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
    try:
        from shapely import to_wkt
        to_wkt(geometry,rounding_precision=rounding_precision)
    except ImportError:
        from shapely.geos import WKTWriter
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


class TestBoundingBox:
    def test_basic(self):
        bbox = BoundingBox(1, 2, 3, 4)
        assert bbox.west == 1
        assert bbox.south == 2
        assert bbox.east == 3
        assert bbox.north == 4
        assert bbox.crs is None

    @pytest.mark.parametrize("crs", [4326, "EPSG:4326", "epsg:4326"])
    def test_basic_with_crs(self, crs):
        bbox = BoundingBox(1, 2, 3, 4, crs=crs)
        assert bbox.west == 1
        assert bbox.south == 2
        assert bbox.east == 3
        assert bbox.north == 4
        assert bbox.crs == "EPSG:4326"

    def test_immutable(self):
        bbox = BoundingBox(1, 2, 3, 4, crs=4326)
        assert bbox.west == 1
        with pytest.raises(AttributeError):
            bbox.west = 100
        with pytest.raises(AttributeError):
            bbox.crs = "EPSG:32631"
        assert bbox.west == 1
        assert bbox.crs == "EPSG:4326"

    def test_repr(self):
        bbox = BoundingBox(1, 2, 3, 4, crs=4326)
        expected = "BoundingBox(west=1, south=2, east=3, north=4, crs='EPSG:4326')"
        assert repr(bbox) == expected

    def test_str(self):
        bbox = BoundingBox(1, 2, 3, 4, crs=4326)
        expected = "BoundingBox(west=1, south=2, east=3, north=4, crs='EPSG:4326')"
        assert str(bbox) == expected

    def test_missing_bounds(self):
        with pytest.raises(BoundingBoxException, match=r"Missing bounds: \['south'\]"):
            _ = BoundingBox(1, None, 3, 4)

    def test_missing_invalid_crs(self):
        with pytest.raises(CRSError):
            _ = BoundingBox(1, 2, 3, 4, crs="foobar:42")

    @pytest.mark.parametrize(
        ["data", "default_crs", "expected"],
        [
            ({"west": 1, "south": 2, "east": 3, "north": 4}, None, (1, 2, 3, 4, None)),
            (
                {"west": 1, "south": 2, "east": 3, "north": 4},
                4326,
                (1, 2, 3, 4, "EPSG:4326"),
            ),
            (
                {"west": 1, "south": 2, "east": 3, "north": 4, "crs": 4326},
                None,
                (1, 2, 3, 4, "EPSG:4326"),
            ),
            (
                {"west": 1, "south": 2, "east": 3, "north": 4, "crs": "EPSG:4326"},
                None,
                (1, 2, 3, 4, "EPSG:4326"),
            ),
            (
                {"west": 1, "south": 2, "east": 3, "north": 4, "crs": "EPSG:4326"},
                32631,
                (1, 2, 3, 4, "EPSG:4326"),
            ),
        ],
    )
    def test_from_dict(self, data, default_crs, expected):
        bbox = BoundingBox.from_dict(data, default_crs=default_crs)
        assert (bbox.west, bbox.south, bbox.east, bbox.north, bbox.crs) == expected

    def test_from_dict_invalid(self):
        data = {"west": 1, "south": 2, "east": None, "northhhhhh": 4}
        with pytest.raises(
            BoundingBoxException, match=r"Missing bounds: \['east', 'north'\]"
        ):
            _ = BoundingBox.from_dict(data)

    def test_from_dict_or_none(self):
        data = {"west": 1, "south": 2, "east": None, "northhhhhh": 4}
        bbox = BoundingBox.from_dict_or_none(data)
        assert bbox is None

    def test_from_wsen_tuple(self):
        bbox = BoundingBox.from_wsen_tuple((4, 3, 2, 1))
        expected = (4, 3, 2, 1, None)
        assert (bbox.west, bbox.south, bbox.east, bbox.north, bbox.crs) == expected

    def test_from_wsen_tuple_with_crs(self):
        bbox = BoundingBox.from_wsen_tuple((4, 3, 2, 1), crs=32631)
        expected = (4, 3, 2, 1, "EPSG:32631")
        assert (bbox.west, bbox.south, bbox.east, bbox.north, bbox.crs) == expected

    def test_is_georeferenced(self):
        assert BoundingBox(1, 2, 3, 4).is_georeferenced() is False
        assert BoundingBox(1, 2, 3, 4, crs=4326).is_georeferenced() is True

    def test_as_tuple(self):
        bbox = BoundingBox(1, 2, 3, 4)
        assert bbox.as_tuple() == (1, 2, 3, 4, None)
        bbox = BoundingBox(1, 2, 3, 4, crs="epsg:4326")
        assert bbox.as_tuple() == (1, 2, 3, 4, "EPSG:4326")

    def test_as_wsen_tuple(self):
        assert BoundingBox(1, 2, 3, 4).as_wsen_tuple() == (1, 2, 3, 4)
        assert BoundingBox(1, 2, 3, 4, crs="epsg:4326").as_wsen_tuple() == (1, 2, 3, 4)

    def test_reproject(self):
        bbox = BoundingBox(3, 51, 3.1, 51.1, crs="epsg:4326")
        reprojected = bbox.reproject(32631)
        assert isinstance(reprojected, BoundingBox)
        assert reprojected.as_tuple() == (
            pytest.approx(500000, abs=10),
            pytest.approx(5649824, abs=10),
            pytest.approx(507016, abs=10),
            pytest.approx(5660950, abs=10),
            "EPSG:32631",
        )

    def test_best_utm_no_crs(self):
        bbox = BoundingBox(1, 2, 3, 4)
        with pytest.raises(CrsRequired):
            _ = bbox.best_utm()

    def test_best_utm(self):
        bbox = BoundingBox(4, 51, 4.1, 51.1, crs="EPSG:4326")
        assert bbox.best_utm() == 32631

        bbox = BoundingBox(-72, -13, -71, -12, crs="EPSG:4326")
        assert bbox.best_utm() == 32719


class TestValidateGeoJSON:
    @staticmethod
    @contextlib.contextmanager
    def _checker(expected_issue: Union[str, None], raise_exception: bool):
        """
        Helper context manager to easily check a validate_geojson_basic result
        for both raise_exception modes:

        - "exception mode": context manger __exit__ phase checks result
        - "return issue mode": returned `check` function should be used inside context manageer body
        """
        checked = False

        def check(result: List[str]):
            """Check validation result in case no actual exception was thrown"""
            nonlocal checked
            checked = True
            if expected_issue:
                if raise_exception:
                    pytest.fail("Exception should have been raised")
                if not result:
                    pytest.fail("No issue was reported")
                assert expected_issue in "\n".join(result)
            else:
                if result:
                    pytest.fail(f"Unexpected issue reported: {result}")

        try:
            yield check
        except Exception as e:
            # Check validation result in case of actual exception
            if not raise_exception:
                pytest.fail(f"Unexpected {e!r}: issue should be returned")
            if not expected_issue:
                pytest.fail(f"Unexpected {e!r}: no issue expected")
            assert expected_issue in str(e)
        else:
            # No exception was thrown: check that the `check` function has been called.
            if not checked:
                raise RuntimeError("`check` function was not used")

    @pytest.mark.parametrize(
        ["value", "expected_issue"],
        [
            ("nope nope", "JSON object (mapping/dictionary) expected, but got str"),
            (123, "JSON object (mapping/dictionary) expected, but got int"),
            ({}, "No 'type' field"),
            ({"type": 123}, "Invalid 'type' type: int"),
            ({"type": {"Poly": "gon"}}, "Invalid 'type' type: dict"),
            ({"type": "meh"}, "Invalid type 'meh'"),
            ({"type": "Point"}, "No 'coordinates' field (type 'Point')"),
            ({"type": "Point", "coordinates": [1, 2]}, None),
            ({"type": "Polygon"}, "No 'coordinates' field (type 'Polygon')"),
            ({"type": "Polygon", "coordinates": [[1, 2]]}, None),
            ({"type": "MultiPolygon"}, "No 'coordinates' field (type 'MultiPolygon')"),
            ({"type": "MultiPolygon", "coordinates": [[[1, 2]]]}, None),
            ({"type": "GeometryCollection", "coordinates": []}, "No 'geometries' field (type 'GeometryCollection')"),
            ({"type": "GeometryCollection", "geometries": []}, None),
            ({"type": "Feature", "coordinates": []}, "No 'geometry' field (type 'Feature')"),
            ({"type": "Feature", "geometry": {}}, "No 'properties' field (type 'Feature')"),
            ({"type": "Feature", "geometry": {}, "properties": {}}, "No 'type' field"),
            (
                {"type": "Feature", "geometry": {"type": "Polygon"}, "properties": {}},
                "No 'coordinates' field (type 'Polygon')",
            ),
            (
                {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[1, 2]]}, "properties": {}},
                None,
            ),
            (
                {"type": "Feature", "geometry": {"type": "Polygonnnnn", "coordinates": [[1, 2]]}, "properties": {}},
                "Found type 'Polygonnnnn', but expects one of ",
            ),
            ({"type": "FeatureCollection"}, "No 'features' field (type 'FeatureCollection')"),
            ({"type": "FeatureCollection", "features": []}, None),
            ({"type": "FeatureCollection", "features": [{"type": "Feature"}]}, "No 'geometry' field (type 'Feature')"),
            (
                {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {}}]},
                "No 'properties' field (type 'Feature')",
            ),
            (
                {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {}, "properties": {}}]},
                "No 'type' field",
            ),
            (
                {
                    "type": "FeatureCollection",
                    "features": [{"type": "Feature", "geometry": {"type": "Polygon"}, "properties": {}}],
                },
                "No 'coordinates' field (type 'Polygon')",
            ),
            (
                {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[1, 2]]}, "properties": {}},
                        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[3, 4]]}, "properties": {}},
                    ],
                },
                None,
            ),
        ],
    )
    @pytest.mark.parametrize("raise_exception", [False, True])
    def test_validate_geojson_basic(self, value, expected_issue, raise_exception):
        with self._checker(expected_issue=expected_issue, raise_exception=raise_exception) as check:
            result = validate_geojson_basic(value, raise_exception=raise_exception)
            check(result)

    @pytest.mark.parametrize(
        ["value", "allowed_types", "expected_issue"],
        [
            (
                {"type": "Point", "coordinates": [1, 2]},
                {"Polygon", "MultiPolygon"},
                "Found type 'Point', but expects one of ['MultiPolygon', 'Polygon']",
            ),
            ({"type": "Polygon", "coordinates": [[1, 2]]}, {"Polygon", "MultiPolygon"}, None),
            ({"type": "MultiPolygon", "coordinates": [[[1, 2]]]}, {"Polygon", "MultiPolygon"}, None),
            (
                {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[1, 2]]}, "properties": {}},
                {"Polygon", "MultiPolygon"},
                "Found type 'Feature', but expects one of ['MultiPolygon', 'Polygon']",
            ),
            (
                {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[1, 2]]}, "properties": {}},
                {"Feature"},
                None,
            ),
            (
                {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[1, 2]]}, "properties": {}},
                        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[3, 4]]}, "properties": {}},
                    ],
                },
                {"Polygon", "MultiPolygon"},
                "Found type 'FeatureCollection', but expects one of ['MultiPolygon', 'Polygon']",
            ),
            (
                {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[1, 2]]}, "properties": {}},
                        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[3, 4]]}, "properties": {}},
                    ],
                },
                {"FeatureCollection"},
                None,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "raise_exception",
        [
            False,
            True,
        ],
    )
    def test_validate_geojson_basic_allowed_types(self, value, allowed_types, expected_issue, raise_exception):
        with self._checker(expected_issue=expected_issue, raise_exception=raise_exception) as check:
            result = validate_geojson_basic(value, allowed_types=allowed_types, raise_exception=raise_exception)
            check(result)
