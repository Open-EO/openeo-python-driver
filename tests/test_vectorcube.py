import textwrap

import geopandas
import geopandas as gpd
import numpy.testing
import pyproj
import pytest
import xarray
from shapely.geometry import MultiPolygon, Point, Polygon

from openeo_driver.datacube import DriverVectorCube
from openeo_driver.errors import OpenEOApiException
from openeo_driver.testing import ApproxGeometry, DictSubSet, IsNan, ApproxGeoJSONByBounds
from openeo_driver.util.geometry import as_geojson_feature_collection
from openeo_driver.utils import EvalEnv

from .data import get_path


class TestDriverVectorCube:

    @pytest.fixture
    def gdf(self) -> gpd.GeoDataFrame:
        """Fixture for a simple GeoPandas DataFrame from file"""
        path = str(get_path("geojson/FeatureCollection02.json"))
        df = gpd.read_file(path)
        return df

    @pytest.fixture
    def vc(self, gdf) -> DriverVectorCube:
        return DriverVectorCube(geometries=gdf)

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

    def test_geometry_count(self, gdf):
        vc = DriverVectorCube(gdf)
        assert vc.geometry_count() == 2

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

    def test_to_internal_json_defaults(self, gdf):
        vc = DriverVectorCube(gdf)
        assert vc.to_internal_json() == {
            "geometries": DictSubSet(
                {
                    "type": "FeatureCollection",
                    "features": [
                        DictSubSet(
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": (((1.0, 1.0), (3.0, 1.0), (2.0, 3.0), (1.0, 1.0)),),
                                },
                                "properties": {"id": "first", "pop": 1234},
                            }
                        ),
                        DictSubSet(
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": (((4.0, 2.0), (5.0, 4.0), (3.0, 4.0), (4.0, 2.0)),),
                                },
                                "properties": {"id": "second", "pop": 5678},
                            }
                        ),
                    ],
                }
            ),
            "cube": None,
        }

    @pytest.mark.parametrize(
        ["columns_for_cube", "expected_cube"],
        [
            (
                "numerical",
                {
                    "name": None,
                    "dims": ("geometry", "properties"),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                        "properties": {"attrs": {}, "data": ["pop"], "dims": ("properties",)},
                    },
                    "data": [[1234], [5678]],
                    "attrs": {},
                },
            ),
            (
                "all",
                {
                    "name": None,
                    "dims": ("geometry", "properties"),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                        "properties": {"attrs": {}, "data": ["id", "pop"], "dims": ("properties",)},
                    },
                    "data": [["first", 1234], ["second", 5678]],
                    "attrs": {},
                },
            ),
            (
                [],
                {
                    "name": None,
                    "dims": ("geometry",),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                    },
                    "data": [IsNan(), IsNan()],
                    "attrs": {"vector_cube_dummy": True},
                },
            ),
            (
                ["pop", "id"],
                {
                    "name": None,
                    "dims": ("geometry", "properties"),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                        "properties": {"attrs": {}, "data": ["pop", "id"], "dims": ("properties",)},
                    },
                    "data": [[1234, "first"], [5678, "second"]],
                    "attrs": {},
                },
            ),
            (
                ["pop", "color"],
                {
                    "name": None,
                    "dims": ("geometry", "properties"),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                        "properties": {"attrs": {}, "data": ["pop", "color"], "dims": ("properties",)},
                    },
                    "data": [[1234.0, IsNan()], [5678.0, IsNan()]],
                    "attrs": {},
                },
            ),
            (
                ["color"],
                {
                    "name": None,
                    "dims": ("geometry", "properties"),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                        "properties": {"attrs": {}, "data": ["color"], "dims": ("properties",)},
                    },
                    "data": [[IsNan()], [IsNan()]],
                    "attrs": {},
                },
            ),
        ],
    )
    def test_to_internal_json_columns_for_cube(self, gdf, columns_for_cube, expected_cube):
        vc = DriverVectorCube.from_geodataframe(gdf, columns_for_cube=columns_for_cube)
        internal = vc.to_internal_json()
        assert internal == {
            "geometries": DictSubSet(
                {
                    "type": "FeatureCollection",
                    "features": [
                        DictSubSet(
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": (((1.0, 1.0), (3.0, 1.0), (2.0, 3.0), (1.0, 1.0)),),
                                },
                                "properties": {"id": "first", "pop": 1234},
                            }
                        ),
                        DictSubSet(
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": (((4.0, 2.0), (5.0, 4.0), (3.0, 4.0), (4.0, 2.0)),),
                                },
                                "properties": {"id": "second", "pop": 5678},
                            }
                        ),
                    ],
                }
            ),
            "cube": expected_cube,
        }

    def test_get_crs(self, gdf):
        vc = DriverVectorCube(gdf)
        assert vc.get_crs() == pyproj.CRS.from_epsg(4326)

    def test_with_cube_to_geojson(self, gdf):
        vc1 = DriverVectorCube(gdf)
        dims, coords = vc1.get_xarray_cube_basics()
        dims += ("bands",)
        coords["bands"] = ["red", "green"]
        cube = xarray.DataArray(data=[[1, 2], [3, 4]], dims=dims, coords=coords)
        vc2 = vc1.with_cube(cube)
        assert vc1.to_geojson() == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": (((1, 1), (3, 1), (2, 3), (1, 1)),)},
                            "properties": {"id": "first", "pop": 1234},
                        }
                    ),
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": (((4, 2), (5, 4), (3, 4), (4, 2)),)},
                            "properties": {"id": "second", "pop": 5678},
                        }
                    ),
                ],
            }
        )
        assert vc2.to_geojson() == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": (((1, 1), (3, 1), (2, 3), (1, 1)),)},
                            "properties": {"id": "first", "pop": 1234, "red": 1, "green": 2},
                        }
                    ),
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": (((4, 2), (5, 4), (3, 4), (4, 2)),)},
                            "properties": {"id": "second", "pop": 5678, "red": 3, "green": 4},
                        }
                    ),
                ],
            }
        )
        assert vc2.to_geojson(flatten_prefix="bandz") == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": (((1, 1), (3, 1), (2, 3), (1, 1)),)},
                            "properties": {"id": "first", "pop": 1234, "bandz~red": 1, "bandz~green": 2},
                        }
                    ),
                    DictSubSet(
                        {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": (((4, 2), (5, 4), (3, 4), (4, 2)),)},
                            "properties": {"id": "second", "pop": 5678, "bandz~red": 3, "bandz~green": 4},
                        }
                    ),
                ],
            }
        )

    def test_from_geodataframe_default(self, gdf):
        vc = DriverVectorCube.from_geodataframe(gdf)
        assert vc.to_geojson() == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "properties": {"id": "first", "pop": 1234},
                            "geometry": {
                                "coordinates": (((1.0, 1.0), (3.0, 1.0), (2.0, 3.0), (1.0, 1.0)),),
                                "type": "Polygon",
                            },
                        }
                    ),
                    DictSubSet(
                        {
                            "type": "Feature",
                            "properties": {"id": "second", "pop": 5678},
                            "geometry": {
                                "coordinates": (((4.0, 2.0), (5.0, 4.0), (3.0, 4.0), (4.0, 2.0)),),
                                "type": "Polygon",
                            },
                        }
                    ),
                ],
            }
        )
        cube = vc.get_cube()
        assert cube.dims == ("geometry", "properties")
        assert cube.shape == (2, 1)
        assert {k: list(v.values) for k, v in cube.coords.items()} == {"geometry": [0, 1], "properties": ["pop"]}

    @pytest.mark.parametrize(
        ["columns_for_cube", "expected_cube"],
        [
            (
                "numerical",
                {
                    "name": None,
                    "dims": ("geometry", "properties"),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                        "properties": {"attrs": {}, "data": ["pop"], "dims": ("properties",)},
                    },
                    "data": [[1234], [5678]],
                    "attrs": {},
                },
            ),
            (
                "all",
                {
                    "name": None,
                    "dims": ("geometry", "properties"),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                        "properties": {"attrs": {}, "data": ["id", "pop"], "dims": ("properties",)},
                    },
                    "data": [["first", 1234], ["second", 5678]],
                    "attrs": {},
                },
            ),
            (
                [],
                {
                    "name": None,
                    "dims": ("geometry",),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                    },
                    "data": [IsNan(), IsNan()],
                    "attrs": {"vector_cube_dummy": True},
                },
            ),
            (
                ["id"],
                {
                    "name": None,
                    "dims": ("geometry", "properties"),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                        "properties": {"attrs": {}, "data": ["id"], "dims": ("properties",)},
                    },
                    "data": [["first"], ["second"]],
                    "attrs": {},
                },
            ),
            (
                ["pop", "id"],
                {
                    "name": None,
                    "dims": ("geometry", "properties"),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                        "properties": {"attrs": {}, "data": ["pop", "id"], "dims": ("properties",)},
                    },
                    "data": [[1234, "first"], [5678, "second"]],
                    "attrs": {},
                },
            ),
            (
                ["color"],
                {
                    "name": None,
                    "dims": ("geometry", "properties"),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                        "properties": {"attrs": {}, "data": ["color"], "dims": ("properties",)},
                    },
                    "data": [[IsNan()], [IsNan()]],
                    "attrs": {},
                },
            ),
            (
                ["pop", "color"],
                {
                    "name": None,
                    "dims": ("geometry", "properties"),
                    "coords": {
                        "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                        "properties": {"attrs": {}, "data": ["pop", "color"], "dims": ("properties",)},
                    },
                    "data": [[1234, IsNan()], [5678, IsNan()]],
                    "attrs": {},
                },
            ),
        ],
    )
    def test_from_geodataframe_columns_for_cube(self, gdf, columns_for_cube, expected_cube):
        vc = DriverVectorCube.from_geodataframe(gdf, columns_for_cube=columns_for_cube)

        assert vc.get_dimension_names() == list(expected_cube["dims"])
        assert vc.to_internal_json() == {
            "geometries": DictSubSet(
                {
                    "type": "FeatureCollection",
                    "features": [
                        DictSubSet(
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": (((1.0, 1.0), (3.0, 1.0), (2.0, 3.0), (1.0, 1.0)),),
                                },
                                "properties": {"id": "first", "pop": 1234},
                            }
                        ),
                        DictSubSet(
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": (((4.0, 2.0), (5.0, 4.0), (3.0, 4.0), (4.0, 2.0)),),
                                },
                                "properties": {"id": "second", "pop": 5678},
                            }
                        ),
                    ],
                }
            ),
            "cube": expected_cube,
        }


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

    def test_from_geojson_invalid_coordinates(self):
        geojson = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-361, 2]},
        }
        with pytest.raises(OpenEOApiException) as e:
            DriverVectorCube.from_geojson(geojson)
        assert e.value.message.startswith(
            "Failed to parse Geojson. Invalid coordinate: [-361, 2]"
        )

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

    @pytest.mark.parametrize(
        ["path", "driver"],
        [
            (get_path("shapefile/mol.shp"), None),
            (get_path("gpkg/mol.gpkg"), None),
            (get_path("parquet/mol.pq"), "parquet"),
        ],
    )
    def test_from_fiona(self, path, driver):
        vc = DriverVectorCube.from_fiona([path], driver=driver, options={"columns_for_cube": []})
        assert vc.to_geojson() == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "id": "0",
                            "geometry": DictSubSet({"type": "Polygon"}),
                            "properties": {"class": 4, "id": 23, "name": "Mol"},
                        }
                    ),
                    DictSubSet(
                        {
                            "type": "Feature",
                            "id": "1",
                            "geometry": DictSubSet({"type": "Polygon"}),
                            "properties": {"class": 5, "id": 58, "name": "TAP"},
                        }
                    ),
                ],
            }
        )

    def test_from_parquet(self):
        path = get_path("parquet/mol.pq")
        vc = DriverVectorCube.from_parquet([path])
        assert vc.to_geojson() == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": [
                    DictSubSet(
                        {
                            "type": "Feature",
                            "id": "0",
                            "geometry": DictSubSet({"type": "Polygon"}),
                            "properties": {"class": 4, "id": 23, "name": "Mol"},
                        }
                    ),
                    DictSubSet(
                        {
                            "type": "Feature",
                            "id": "1",
                            "geometry": DictSubSet({"type": "Polygon"}),
                            "properties": {"class": 5, "id": 58, "name": "TAP"},
                        }
                    ),
                ],
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
        numpy.testing.assert_allclose(area, 5134695.615, rtol=0.1)

    def test_get_bounding_box_area_not_wgs84(self):
        path = str(get_path("geojson/FeatureCollection08.json"))
        vc = DriverVectorCube(gpd.read_file(path))
        area = vc.get_bounding_box_area()
        numpy.testing.assert_allclose(area, 5134695.615, rtol=0.1)

    def test_get_bounding_box_area_northpole_not_wgs84(self):
        path = str(get_path("geojson/FeatureCollection09.json"))
        vc = DriverVectorCube(gpd.read_file(path))
        area = vc.get_bounding_box_area()
        numpy.testing.assert_allclose(area, 1526291.296426, rtol=0.1)

    def test_get_area(self):
        path = str(get_path("geojson/FeatureCollection07.json"))
        vc = DriverVectorCube(gpd.read_file(path))
        area = vc.get_area()
        numpy.testing.assert_allclose(area, 10269391.016361, rtol=0.1)

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

    @pytest.mark.parametrize("dimension", ["bands", "properties"])
    def test_apply_dimension_run_udf_change_geometry(self, gdf, backend_implementation, dimension):
        vc = DriverVectorCube.from_geodataframe(gdf, dimension_name=dimension)
        udf = textwrap.dedent(
            """
            from openeo.udf import UdfData, FeatureCollection
            def process_vector_cube(udf_data: UdfData) -> UdfData:
                [feature_collection] = udf_data.get_feature_collection_list()
                gdf = feature_collection.data
                gdf["geometry"] = gdf["geometry"].buffer(distance=1, resolution=2)
                udf_data.set_feature_collection_list([
                    FeatureCollection(id="_", data=gdf),
                ])
            """
        )
        callback = {
            "runudf1": {
                "process_id": "run_udf",
                "arguments": {"data": {"from_parameter": "data"}, "udf": udf, "runtime": "Python"},
                "result": True,
            }
        }
        env = EvalEnv({"backend_implementation": backend_implementation})
        result = vc.apply_dimension(process=callback, dimension=dimension, env=env)
        assert isinstance(result, DriverVectorCube)
        feature_collection = result.to_geojson()
        assert feature_collection == DictSubSet(
            {
                "type": "FeatureCollection",
                "bbox": pytest.approx((0, 0, 6, 5), abs=0.1),
                "features": [
                    {
                        "type": "Feature",
                        "bbox": pytest.approx((0, 0, 4, 4), abs=0.1),
                        "geometry": DictSubSet({"type": "Polygon"}),
                        "id": "0",
                        "properties": {"id": "first", "pop": 1234},
                    },
                    {
                        "type": "Feature",
                        "bbox": pytest.approx((2, 1, 6, 5), abs=0.1),
                        "geometry": DictSubSet({"type": "Polygon"}),
                        "id": "1",
                        "properties": {"id": "second", "pop": 5678},
                    },
                ],
            }
        )

    @pytest.mark.parametrize("dimension", ["bands", "properties"])
    def test_apply_dimension_run_udf_add_properties(self, gdf, backend_implementation, dimension):
        vc = DriverVectorCube.from_geodataframe(gdf, dimension_name=dimension)
        udf = textwrap.dedent(
            """
            from openeo.udf import UdfData, FeatureCollection
            def process_vector_cube(udf_data: UdfData) -> UdfData:
                [feature_collection] = udf_data.get_feature_collection_list()
                gdf = feature_collection.data
                gdf["popone"] = gdf["pop"] + 1
                gdf["poppop"] = gdf["pop"] ** 2
                udf_data.set_feature_collection_list([
                    FeatureCollection(id="_", data=gdf),
                ])
            """
        )
        callback = {
            "runudf1": {
                "process_id": "run_udf",
                "arguments": {"data": {"from_parameter": "data"}, "udf": udf, "runtime": "Python"},
                "result": True,
            }
        }
        env = EvalEnv({"backend_implementation": backend_implementation})
        result = vc.apply_dimension(process=callback, dimension=dimension, env=env)
        assert isinstance(result, DriverVectorCube)
        assert result.to_internal_json() == {
            "geometries": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": ApproxGeoJSONByBounds(1, 1, 3, 3, types=["Polygon"], abs=0.01),
                        "id": "0",
                        "properties": {"id": "first", "pop": 1234, "popone": 1235, "poppop": 1522756},
                        "bbox": pytest.approx((1, 1, 3, 3), abs=0.01),
                    },
                    {
                        "type": "Feature",
                        "geometry": ApproxGeoJSONByBounds(3, 2, 5, 4, types=["Polygon"], abs=0.01),
                        "id": "1",
                        "properties": {"id": "second", "pop": 5678, "popone": 5679, "poppop": 32239684},
                        "bbox": pytest.approx((3, 2, 5, 4), abs=0.01),
                    },
                ],
                "bbox": pytest.approx((1, 1, 5, 4), abs=0.01),
            },
            "cube": {
                "name": None,
                "dims": ("geometry", "properties"),
                "coords": {
                    "geometry": {"attrs": {}, "data": [0, 1], "dims": ("geometry",)},
                    "properties": {"attrs": {}, "data": ["pop", "popone", "poppop"], "dims": ("properties",)},
                },
                "data": [[1234, 1235, 1522756], [5678, 5679, 32239684]],
                "attrs": {},
            },
        }
