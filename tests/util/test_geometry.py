import re
import typing

import pytest
import shapely.geometry

from openeo_driver.testing import RegexMatcher
from openeo_driver.util.geometry import (
    geojson_to_geometry,
    geojson_to_multipolygon,
    reproject_bounding_box,
    spatial_extent_union,
)


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
        "west": 5.016118467277098,
        "south": 51.217660146353246,
        "east": 5.036548264535997,
        "north": 51.22699369149726,
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
