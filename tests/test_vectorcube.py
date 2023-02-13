import geopandas as gpd
import numpy.testing
import pyproj
import pytest
import xarray
from shapely.geometry import Polygon, MultiPolygon, Point

from openeo_driver.datacube import DriverVectorCube
from openeo_driver.testing import DictSubSet, ApproxGeometry
from openeo_driver.util.geometry import as_geojson_feature_collection

from .data import get_path


class TestDriverVectorCube:

    @pytest.fixture
    def gdf(self) -> gpd.GeoDataFrame:
        """Fixture for a simple GeoPandas DataFrame from file"""
        path = str(get_path("geojson/FeatureCollection02.json"))
        df = gpd.read_file(path)
        return df

    def test_basic(self, gdf):
        vc = DriverVectorCube(gdf)
        assert vc.get_bounding_box() == (1, 1, 5, 4)

    def test_to_multipolygon(self, gdf):
        vc = DriverVectorCube(gdf)
        mp = vc.to_multipolygon()
        assert isinstance(mp, MultiPolygon)
        assert len(mp) == 2
        assert mp.equals(MultiPolygon([
            Polygon([(1, 1), (2, 3), (3, 1), (1, 1)]),
            Polygon([(4, 2), (5, 4), (3, 4), (4, 2)]),
        ]))

    def test_get_geometries(self, gdf):
        vc = DriverVectorCube(gdf)
        geometries = vc.get_geometries()
        assert len(geometries) == 2
        expected_geometries = [
            Polygon([(1, 1), (2, 3), (3, 1), (1, 1)]),
            Polygon([(4, 2), (5, 4), (3, 4), (4, 2)]),
        ]
        for geometry, expected in zip(geometries, expected_geometries):
            assert geometry.equals(expected)

    def test_to_geojson(self, gdf):
        vc = DriverVectorCube(gdf)
        assert vc.to_geojson() == DictSubSet({
            "type": "FeatureCollection",
            "features": [
                DictSubSet({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": (((1, 1), (3, 1), (2, 3), (1, 1)),)},
                    "properties": {"id": "first", "pop": 1234},
                }),
                DictSubSet({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": (((4, 2), (5, 4), (3, 4), (4, 2)),)},
                    "properties": {"id": "second", "pop": 5678},
                }),
            ]
        })

    def test_to_wkt(self, gdf):
        vc = DriverVectorCube(gdf)
        assert vc.to_wkt() == (
            ['POLYGON ((1 1, 3 1, 2 3, 1 1))', 'POLYGON ((4 2, 5 4, 3 4, 4 2))']
        )

    def test_get_crs(self, gdf):
        vc = DriverVectorCube(gdf)
        assert vc.get_crs() == pyproj.CRS.from_epsg(4326)

    def test_with_cube_to_geojson(self, gdf):
        vc1 = DriverVectorCube(gdf)
        dims, coords = vc1.get_xarray_cube_basics()
        dims += ("bands",)
        coords["bands"] = ["red", "green"]
        cube = xarray.DataArray(data=[[1, 2], [3, 4]], dims=dims, coords=coords)
        vc2 = vc1.with_cube(cube, flatten_prefix="bandz")
        assert vc1.to_geojson() == DictSubSet({
            "type": "FeatureCollection",
            "features": [
                DictSubSet({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": (((1, 1), (3, 1), (2, 3), (1, 1)),)},
                    "properties": {"id": "first", "pop": 1234},
                }),
                DictSubSet({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": (((4, 2), (5, 4), (3, 4), (4, 2)),)},
                    "properties": {"id": "second", "pop": 5678},
                }),
            ]
        })
        assert vc2.to_geojson() == DictSubSet({
            "type": "FeatureCollection",
            "features": [
                DictSubSet({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": (((1, 1), (3, 1), (2, 3), (1, 1)),)},
                    "properties": {"id": "first", "pop": 1234, "bandz~red": 1, "bandz~green": 2},
                }),
                DictSubSet({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": (((4, 2), (5, 4), (3, 4), (4, 2)),)},
                    "properties": {"id": "second", "pop": 5678, "bandz~red": 3, "bandz~green": 4},
                }),
            ]
        })

    @pytest.mark.parametrize(["geojson", "expected"], [
        (
                {"type": "Polygon", "coordinates": [[(1, 1), (3, 1), (2, 3), (1, 1)]]},
                [
                    DictSubSet({
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": (((1, 1), (3, 1), (2, 3), (1, 1),),)},
                        "properties": {},
                    }),
                ],
        ),
        (
                {"type": "MultiPolygon", "coordinates": [[[(1, 1), (3, 1), (2, 3), (1, 1)]]]},
                [
                    DictSubSet({
                        "type": "Feature",
                        "geometry": {"type": "MultiPolygon", "coordinates": [(((1, 1), (3, 1), (2, 3), (1, 1),),)]},
                        "properties": {},
                    }),
                ],
        ),
        (
                {"type": "Point", "coordinates": [1, 1]},
                [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": (1, 1)},
                            "properties": {},
                        }
                    ),
                ],
            ),
            (
                {"type": "MultiPoint", "coordinates": [[1, 1], [2, 3]]},
                [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "MultiPoint",
                                "coordinates": ((1, 1), (2, 3)),
                            },
                            "properties": {},
                        }
                    ),
                ],
            ),
            (
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiPolygon", "coordinates": [[[(1, 1), (3, 1), (2, 3), (1, 1)]]]},
                    "properties": {"id": "12_3"},
                },
                [
                    DictSubSet({
                        "type": "Feature",
                        "geometry": {"type": "MultiPolygon", "coordinates": [(((1, 1), (3, 1), (2, 3), (1, 1),),)]},
                        "properties": {"id": "12_3"},
                    }),
                ],
        ),
        (
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": [[(1, 1), (3, 1), (2, 3), (1, 1)]]},
                            "properties": {"id": 1},
                        },
                        {
                            "type": "Feature",
                            "geometry": {"type": "MultiPolygon", "coordinates": [[[(1, 1), (3, 1), (2, 3), (1, 1)]]]},
                            "properties": {"id": 2},
                        },
                    ],
                },
                [
                    DictSubSet({
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": (((1, 1), (3, 1), (2, 3), (1, 1),),)},
                        "properties": {"id": 1},
                    }),
                    DictSubSet({
                        "type": "Feature",
                        "geometry": {"type": "MultiPolygon", "coordinates": [(((1, 1), (3, 1), (2, 3), (1, 1),),)]},
                        "properties": {"id": 2},
                    }),
                ],
            ),
            (
                {
                    "type": "GeometryCollection",
                    "geometries": [
                        {
                            "type": "Polygon",
                            "coordinates": [[(1, 1), (3, 1), (2, 3), (1, 1)]],
                        },
                        {
                            "type": "MultiPolygon",
                            "coordinates": [[[(1, 1), (3, 1), (2, 3), (1, 1)]]],
                        },
                    ],
                },
                [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": (((1, 1), (3, 1), (2, 3), (1, 1)),),
                            },
                        }
                    ),
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "MultiPolygon",
                                "coordinates": [(((1, 1), (3, 1), (2, 3), (1, 1)),)],
                            },
                        }
                    ),
                ],
            ),
        ],
    )
    def test_from_geojson(self, geojson, expected):
        vc = DriverVectorCube.from_geojson(geojson)
        assert vc.to_geojson() == DictSubSet({
            "type": "FeatureCollection",
            "features": expected,
        })

    @pytest.mark.parametrize(
        ["geometry", "expected"],
        [
            (
                Point(1.2, 3.4),
                [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": (1.2, 3.4)},
                        }
                    ),
                ],
            ),
            (
                Polygon([(1, 1), (2, 3), (4, 1), (1, 1)]),
                [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": (
                                    ((1.0, 1.0), (2.0, 3.0), (4.0, 1.0), (1.0, 1.0)),
                                ),
                            },
                        }
                    ),
                ],
            ),
            (
                [Point(1.2, 3.4), Polygon([(1, 1), (2, 3), (4, 1), (1, 1)])],
                [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": (1.2, 3.4)},
                        }
                    ),
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": (
                                    ((1.0, 1.0), (2.0, 3.0), (4.0, 1.0), (1.0, 1.0)),
                                ),
                            },
                        }
                    ),
                ],
            ),
        ],
    )
    def test_from_geometry(self, geometry, expected):
        vc = DriverVectorCube.from_geometry(geometry)
        assert vc.to_geojson() == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": expected,
            }
        )

    def test_get_bounding_box(self, gdf):
        vc = DriverVectorCube(gdf)
        assert vc.get_bounding_box() == (1, 1, 5, 4)
        assert vc.get_bounding_box_geometry() == Polygon.from_bounds(1, 1, 5, 4)
        assert vc.get_bounding_box_geojson() == {
            "type": "Polygon",
            "coordinates": (
                ((1.0, 1.0), (1.0, 4.0), (5.0, 4.0), (5.0, 1.0), (1.0, 1.0)),
            ),
        }

    def test_get_bounding_box_area(self):
        path = str(get_path("geojson/FeatureCollection06.json"))
        vc = DriverVectorCube(gpd.read_file(path))
        area = vc.get_bounding_box_area()
        # TODO: area of FeatureCollection06 (square with side of approx 2.3km, based on length of Watersportbaan Gent)
        #       is roughly 2.3 km * 2.3 km = 5.29 km2,
        #       but current implementation gives result that is quite a bit larger than that
        numpy.testing.assert_allclose(area, 8e6, rtol=0.1)

    def test_get_area(self):
        path = str(get_path("geojson/FeatureCollection07.json"))
        vc = DriverVectorCube(gpd.read_file(path))
        area = vc.get_area()
        # TODO: see remark in test_get_bounding_box_area
        numpy.testing.assert_allclose(area, 16e6, rtol=0.2)

    def test_buffer_points(self):
        geometry = as_geojson_feature_collection(
            Point(2, 3), Polygon.from_bounds(5, 8, 13, 21)
        )
        vc = DriverVectorCube.from_geojson(geometry)
        buffered = vc.buffer_points(distance=1000)
        assert buffered.to_geojson() == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": [
                    DictSubSet(
                        ApproxGeometry.from_wkt(
                            "POLYGON ((2.009 3, 2.006 2.994, 2 2.991, 1.994 2.994, 1.991 3, 1.994 3.006, 2 3.009, 2.006 3.006, 2.009 3))",
                            abs=0.001,
                        ).to_geojson_feature(properties={})
                    ),
                    DictSubSet(
                        ApproxGeometry(
                            Polygon.from_bounds(5, 8, 13, 21), abs=0.000001
                        ).to_geojson_feature(properties={})
                    ),
                ],
            }
        )
