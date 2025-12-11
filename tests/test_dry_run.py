import logging
import json
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import openeo.processes
import pytest
import shapely.geometry
import dirty_equals
from openeo.internal.graph_building import PGNode
from openeo.metadata import SpatialDimension
from openeo.rest.datacube import DataCube
from openeo.testing.stac import StacDummyBuilder

from openeo_driver.datacube import DriverVectorCube
from openeo_driver.datastructs import SarBackscatterArgs
from openeo_driver.dry_run import (
    DataSource,
    DataTrace,
    DryRunDataCube,
    DryRunDataTracer,
    ProcessType,
)
from openeo_driver.dummy.dummy_backend import DummyVectorCube
from openeo_driver.errors import OpenEOApiException, ProcessParameterInvalidException
from openeo_driver.ProcessGraphDeserializer import (
    ENV_DRY_RUN_TRACER,
    ENV_MAX_BUFFER,
    ENV_SAVE_RESULT,
    ENV_SOURCE_CONSTRAINTS,
    _extract_load_parameters,
    custom_process_from_process_graph,
    evaluate,
    process_registry_100,
)
from openeo_driver.save_result import SaveResult
from openeo_driver.testing import DictSubSet, approxify, ephemeral_fileserver
from openeo_driver.util.geometry import as_geojson_feature_collection
from openeo_driver.utils import EvalEnv
from openeo_driver.views import OPENEO_API_VERSION_DEFAULT
from openeo_driver.workspacerepository import WorkspaceRepository
from tests.data import TEST_DATA_ROOT, get_path, load_json

CRS_UTM = {
    "$schema": "https://proj.org/schemas/v0.2/projjson.schema.json",
    "area": "World",
    "bbox": {"east_longitude": 180, "north_latitude": 90, "south_latitude": -90, "west_longitude": -180},
    "coordinate_system": {
        "axis": [
            {"abbreviation": "Lat", "direction": "north", "name": "Geodetic " "latitude", "unit": "degree"},
            {"abbreviation": "Lon", "direction": "east", "name": "Geodetic " "longitude", "unit": "degree"},
        ],
        "subtype": "ellipsoidal",
    },
    "datum": {
        "ellipsoid": {"inverse_flattening": 298.257223563, "name": "WGS 84", "semi_major_axis": 6378137},
        "name": "World Geodetic System 1984",
        "type": "GeodeticReferenceFrame",
    },
    "id": {"authority": "OGC", "code": "Auto42001", "version": "1.3"},
    "name": "AUTO 42001 (Universal Transverse " "Mercator)",
    "type": "GeodeticCRS",
}


@pytest.fixture
def dry_run_tracer() -> DryRunDataTracer:
    return DryRunDataTracer()


@pytest.fixture
def dry_run_env(dry_run_tracer, backend_implementation) -> EvalEnv:
    return EvalEnv(
        {
            ENV_DRY_RUN_TRACER: dry_run_tracer,
            "backend_implementation": backend_implementation,
            "version": OPENEO_API_VERSION_DEFAULT,
            "openeo_api_version": OPENEO_API_VERSION_DEFAULT,
            ENV_MAX_BUFFER: {},
        }
    )


def test_source_load_collection():
    s1 = DataSource.load_collection(collection_id="FOOBAR")
    s2 = DataSource.load_collection(collection_id="FOOBAR")
    s3 = DataSource.load_collection(collection_id="FOOBARV2")
    assert s1.get_source_id() == s2.get_source_id()
    assert s1.get_source_id() != s3.get_source_id()


def test_source_load_uploaded_files():
    s1 = DataSource.load_uploaded_files(paths=["foo.tiff"], format="GTiff", options={})
    s2 = DataSource.load_uploaded_files(paths=["foo.tiff"], format="GTiff", options={})
    s3 = DataSource.load_uploaded_files(paths=["foo.tiff"], format="GTiff", options={"meh": "xev"})
    assert s1.get_source_id() == s2.get_source_id()
    assert s1.get_source_id() != s3.get_source_id()


def test_source_load_result():
    s1 = DataSource.load_result(job_id="a1a6dabc-20ae-43b8-b7c1-f9c9bd3f6dfd")
    s2 = DataSource.load_result(job_id="a1a6dabc-20ae-43b8-b7c1-f9c9bd3f6dfd")
    s3 = DataSource.load_result(job_id="a16b4310-45de-49ae-824b-f501038057c6")
    assert s1.get_source_id() == s2.get_source_id()
    assert s1.get_source_id() != s3.get_source_id()


def test_data_source_hashable():
    s1 = DataSource("load_foo", arguments={"a": "b", "c": "d", "e": "f", "g": "h"})
    s2 = DataSource("load_foo", arguments={"g": "h", "e": "f", "c": "d", "a": "b"})
    assert s1.get_source_id() == s2.get_source_id()


def test_data_trace():
    source = DataSource.load_collection("S2")
    t1 = trace = DataTrace(parent=source, operation="filter_bbox", arguments={"bbox": "Belgium"})
    t2 = trace = DataTrace(parent=trace, operation="filter_bbox", arguments={"bbox": "Mol"})
    t3 = trace = DataTrace(parent=trace, operation="ndvi", arguments={"red": "B04"})
    assert trace.get_source() is source
    assert trace.get_arguments_by_operation("filter_bbox") == [{"bbox": "Belgium"}, {"bbox": "Mol"}]
    assert list(trace.get_arguments_by_operation("ndvi")) == [{"red": "B04"}]

    assert trace.get_operation_closest_to_source("load_collection") is source
    assert trace.get_operation_closest_to_source("filter_bbox") is t1
    assert trace.get_operation_closest_to_source("ndvi") is t3
    assert trace.get_operation_closest_to_source(["load_collection"]) is source
    assert trace.get_operation_closest_to_source(["filter_bbox", "ndvi"]) is t1
    assert trace.get_operation_closest_to_source(["load_collection", "filter_bbox", "ndvi"]) is source
    assert trace.get_operation_closest_to_source("foobar") is None
    assert trace.get_operation_closest_to_source(["foobar"]) is None
    assert trace.get_operation_closest_to_source(["foobar", "filter_bbox"]) is t1


def test_dry_run_data_tracer():
    tracer = DryRunDataTracer()
    source = DataSource.load_collection("S2")
    trace = DataTrace(parent=source, operation="ndvi", arguments={})
    res = tracer.add_trace(trace)
    assert res is trace
    assert tracer.get_trace_leaves() == [trace]


def test_dry_run_data_tracer_process_traces():
    tracer = DryRunDataTracer()
    source = DataSource.load_collection("S2")
    trace1 = DataTrace(parent=source, operation="ndvi", arguments={})
    tracer.add_trace(trace1)
    trace2 = DataTrace(parent=source, operation="evi", arguments={})
    tracer.add_trace(trace2)
    assert tracer.get_trace_leaves() == [trace1, trace2]
    traces = tracer.process_traces([trace1, trace2], operation="filter_bbox", arguments={"bbox": "mol"})
    assert tracer.get_trace_leaves() == traces
    assert set(t.describe() for t in traces) == {
        "load_collection<-ndvi<-filter_bbox",
        "load_collection<-evi<-filter_bbox",
    }


def test_tracer_load_collection():
    tracer = DryRunDataTracer()
    arguments = {
        "temporal_extent": ("2020-01-01", "2020-02-02"),
        "spatial_extent": {"west": 1, "south": 51, "east": 2, "north": 52},
        "bands": ["red", "blue"],
    }
    cube = tracer.load_collection("S2", arguments)
    traces = tracer.get_trace_leaves()
    assert [t.describe() for t in traces] == ["load_collection<-temporal_extent<-spatial_extent<-bands"]


def test_evaluate_basic_no_load_collection(dry_run_env, dry_run_tracer):
    pg = {
        "add": {"process_id": "add", "arguments": {"x": 1, "y": 2}, "result": True},
    }
    res = evaluate(pg, env=dry_run_env)
    assert res == 3
    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert source_constraints == []


def test_evaluate_basic_load_collection(dry_run_env, dry_run_tracer):
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}, "result": True},
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert source_constraints == [(("load_collection", ("S2_FOOBAR", ())), {})]


def test_evaluate_basic_filter_temporal(dry_run_env, dry_run_tracer):
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "ft": {
            "process_id": "filter_temporal",
            "arguments": {"data": {"from_node": "lc"}, "extent": ["2020-02-02", "2020-03-03"]},
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    """
    source_constraints = dry_run_tracer.get_source_constraints(merge=False)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == [{"temporal_extent": [("2020-02-02", "2020-03-03")]}]
    """

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {"temporal_extent": ("2020-02-02", "2020-03-03")}


def test_evaluate_temporal_extent_dynamic(dry_run_env, dry_run_tracer):
    pg = {
        "load": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "extent": {"process_id": "constant", "arguments": {"x": ["2020-01-01", "2020-02-02"]}},
        "filtertemporal": {
            "process_id": "filter_temporal",
            "arguments": {"data": {"from_node": "load"}, "extent": {"from_node": "extent"}},
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)
    source_constraints = dry_run_tracer.get_source_constraints()
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {"temporal_extent": ("2020-01-01", "2020-02-02")}


def test_evaluate_temporal_extent_dynamic_item(dry_run_env, dry_run_tracer):
    pg = {
        "load": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "start": {"process_id": "constant", "arguments": {"x": "2020-01-01"}},
        "filtertemporal": {
            "process_id": "filter_temporal",
            "arguments": {"data": {"from_node": "load"}, "extent": [{"from_node": "start"}, "2020-02-02"]},
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)
    source_constraints = dry_run_tracer.get_source_constraints()
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {"temporal_extent": ("2020-01-01", "2020-02-02")}


def test_evaluate_graph_diamond(dry_run_env, dry_run_tracer):
    """
    Diamond graph:
    load -> band red -> mask -> bbox
        `-> band grass -^
    """
    pg = {
        "load": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "band_red": {
            "process_id": "filter_bands",
            "arguments": {"data": {"from_node": "load"}, "bands": ["red"]},
        },
        "band_grass": {
            "process_id": "filter_bands",
            "arguments": {"data": {"from_node": "load"}, "bands": ["grass"]},
        },
        "mask": {
            "process_id": "mask",
            "arguments": {"data": {"from_node": "band_red"}, "mask": {"from_node": "band_grass"}},
        },
        "bbox": {
            "process_id": "filter_bbox",
            "arguments": {"data": {"from_node": "mask"}, "extent": {"west": 1, "east": 2, "south": 51, "north": 52}},
            "result": True,
        },
    }

    cube = evaluate(pg, env=dry_run_env, do_dry_run=False)

    """
    source_constraints = dry_run_tracer.get_source_constraints(merge=False)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert sorted(constraints, key=str) == [
    """

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 2
    assert source_constraints == [
        (
            ("load_collection", ("S2_FOOBAR", ())),
            {
                "bands": ["grass"],
                "resample": {"method": "near", "resolution": (10, 10), "target_crs": CRS_UTM},
                "spatial_extent": {"west": 1, "east": 2, "south": 51, "north": 52, "crs": "EPSG:4326"},
            },
        ),
        (
            ("load_collection", ("S2_FOOBAR", ())),
            {"bands": ["red"], "spatial_extent": {"west": 1, "east": 2, "south": 51, "north": 52, "crs": "EPSG:4326"}},
        ),
    ]


def test_evaluate_load_collection_and_filter_extents(dry_run_env, dry_run_tracer):
    """temporal/bbox/band extents in load_collection *and* filter_ processes"""
    pg = {
        "load": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FOOBAR",
                "spatial_extent": {"west": 0, "south": 50, "east": 5, "north": 55},
                "temporal_extent": ["2020-01-01", "2020-10-10"],
                "bands": ["red", "green", "blue"],
            },
        },
        "filter_temporal": {
            "process_id": "filter_temporal",
            "arguments": {"data": {"from_node": "load"}, "extent": ["2020-02-02", "2020-03-03"]},
        },
        "filter_bbox": {
            "process_id": "filter_bbox",
            "arguments": {
                "data": {"from_node": "filter_temporal"},
                "extent": {"west": 1, "south": 51, "east": 3, "north": 53},
            },
        },
        "filter_bands": {
            "process_id": "filter_bands",
            "arguments": {"data": {"from_node": "filter_bbox"}, "bands": ["red"]},
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    """
    source_constraints = dry_run_tracer.get_source_constraints(merge=False)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == [{
        "temporal_extent": [("2020-01-01", "2020-10-10"), ("2020-02-02", "2020-03-03"), ],
        "spatial_extent": [
            {"west": 0, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
            {"west": 1, "south": 51, "east": 3, "north": 53, "crs": "EPSG:4326"}
        ],
        "bands": [["red", "green", "blue"], ["red"]],
    }]
    """

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", (), ("red", "green", "blue")))
    assert constraints == {
        "temporal_extent": ("2020-01-01", "2020-10-10"),
        "spatial_extent": {"west": 0, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
        "bands": ["red", "green", "blue"],
    }


@pytest.mark.parametrize(
    "additional_arguments",
    [
        {"level": "error"},
        {},
    ],
)
def test_inspect(dry_run_env, dry_run_tracer, additional_arguments):
    pg = {
        "inspect": {
            "process_id": "inspect",
            "arguments": {**{"data": 123, "message": "logging a message"}, **additional_arguments},
            "result": True,
        },
    }

    assert evaluate(pg, env=dry_run_env) == 123


def test_evaluate_merge_collections(dry_run_env, dry_run_tracer):
    pg = {
        "load": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FOOBAR",
                "spatial_extent": {"west": 0, "south": 50, "east": 5, "north": 55},
                "temporal_extent": ["2020-01-01", "2020-10-10"],
                "bands": ["red", "green", "blue"],
            },
        },
        "load_s1": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FAPAR_CLOUDCOVER",
                "spatial_extent": {"west": -1, "south": 50, "east": 5, "north": 55},
                "temporal_extent": ["2020-01-01", "2020-10-10"],
                "bands": ["VV"],
            },
        },
        "merge": {
            "process_id": "merge_cubes",
            "arguments": {"cube1": {"from_node": "load"}, "cube2": {"from_node": "load_s1"}},
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 2

    source, constraints = source_constraints[0]

    assert source == ("load_collection", ("S2_FOOBAR", (), ("red", "green", "blue")))
    assert constraints == {
        "temporal_extent": ("2020-01-01", "2020-10-10"),
        "spatial_extent": {"west": 0, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
        "bands": ["red", "green", "blue"],
    }

    source, constraints = source_constraints[1]

    assert source == ("load_collection", ("S2_FAPAR_CLOUDCOVER", (), ("VV",)))
    assert constraints == {
        "temporal_extent": ("2020-01-01", "2020-10-10"),
        "spatial_extent": {"west": -1, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
        "bands": ["VV"],
    }

    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    loadparams = _extract_load_parameters(dry_run_env, ("load_collection", ("S2_FOOBAR", (), ("red", "green", "blue"))))

    assert {
        "west": 213370,
        "south": 5540540,
        "east": 643330,
        "north": 6102120,
        "crs": "EPSG:32631",
    } == loadparams.global_extent


def test_evaluate_load_collection_and_filter_extents_dynamic(dry_run_env, dry_run_tracer):
    """ "Dynamic temporal/bbox/band extents in load_collection *and* filter_ processes"""
    pg = {
        "west1": {"process_id": "add", "arguments": {"x": 2, "y": -1}},
        "west2": {"process_id": "divide", "arguments": {"x": 4, "y": 2}},
        "start01": {"process_id": "constant", "arguments": {"x": "2020-01-01"}},
        "start02": {"process_id": "constant", "arguments": {"x": "2020-02-02"}},
        "bandsbgr": {"process_id": "constant", "arguments": {"x": ["blue", "green", "red"]}},
        "bandblue": {"process_id": "constant", "arguments": {"x": "blue"}},
        "load": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FOOBAR",
                "spatial_extent": {"west": {"from_node": "west1"}, "south": 50, "east": 5, "north": 55},
                "temporal_extent": [{"from_node": "start01"}, "2020-10-10"],
                "bands": {"from_node": "bandsbgr"},
            },
        },
        "filter_temporal": {
            "process_id": "filter_temporal",
            "arguments": {"data": {"from_node": "load"}, "extent": [{"from_node": "start02"}, "2020-03-03"]},
        },
        "filter_bbox": {
            "process_id": "filter_bbox",
            "arguments": {
                "data": {"from_node": "filter_temporal"},
                "extent": {"west": {"from_node": "west2"}, "south": 51, "east": 3, "north": 53},
            },
        },
        "filter_bands": {
            "process_id": "filter_bands",
            "arguments": {"data": {"from_node": "filter_bbox"}, "bands": [{"from_node": "bandblue"}]},
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    """
    source_constraints = dry_run_tracer.get_source_constraints(merge=False)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == [{
        "temporal_extent": [("2020-01-01", "2020-10-10"), ("2020-02-02", "2020-03-03")],
        "spatial_extent": [
            {"west": 1, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
            {"west": 2.0, "south": 51, "east": 3, "north": 53, "crs": "EPSG:4326"}
        ],
        "bands": [["blue", "green", "red"], ["blue"]],
    }]
    """

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", (), ("blue", "green", "red")))
    assert constraints == {
        "temporal_extent": ("2020-01-01", "2020-10-10"),
        "spatial_extent": {"west": 1, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
        "bands": ["blue", "green", "red"],
    }


@pytest.mark.parametrize(
    ["inside", "replacement", "expect_spatial_extent"],
    [
        (None, None, True),
        (False, None, True),
        (True, None, False),
        (None, 123, False),
    ],
)
def test_mask_polygon_only(dry_run_env, dry_run_tracer, inside, replacement, expect_spatial_extent):
    polygon = {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]}
    cube = DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None)
    cube = cube.mask_polygon(mask=polygon, inside=inside, replacement=replacement)
    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    if expect_spatial_extent:
        expected = {
            "spatial_extent": {"west": 0.0, "south": 0.0, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"},
            "weak_spatial_extent": {"west": 0.0, "south": 0.0, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"},
        }
    else:
        expected = {}
    assert constraints == expected


def test_mask_polygon_and_load_collection_spatial_extent(dry_run_env, dry_run_tracer):
    polygon = {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]}
    cube = DataCube(
        PGNode("load_collection", id="S2_FOOBAR", spatial_extent={"west": -1, "south": -1, "east": 10, "north": 10}),
        connection=None,
    )
    cube = cube.mask_polygon(mask=polygon)
    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "spatial_extent": {"west": -1, "south": -1, "east": 10, "north": 10, "crs": "EPSG:4326"},
        "weak_spatial_extent": {"west": 0.0, "south": 0.0, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"},
    }


@pytest.mark.parametrize("bbox_first", [True, False])
def test_mask_polygon_and_filter_bbox(dry_run_env, dry_run_tracer, bbox_first):
    polygon = {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]}
    bbox = {"west": -1, "south": -1, "east": 9, "north": 9, "crs": "EPSG:4326"}
    # Use client lib to build process graph in flexible way
    cube = DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None)
    if bbox_first:
        cube = cube.filter_bbox(bbox=bbox).mask_polygon(mask=polygon)
    else:
        cube = cube.mask_polygon(mask=polygon).filter_bbox(bbox=bbox)

    pg = cube.flat_graph()

    res = evaluate(pg, env=dry_run_env, do_dry_run=False)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "spatial_extent": {"west": -1, "south": -1, "east": 9, "north": 9, "crs": "EPSG:4326"},
        "weak_spatial_extent": {"west": 0.0, "south": 0.0, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"},
    }


def test_aggregate_spatial_only(dry_run_env, dry_run_tracer):
    polygon = {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]}
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "agg": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "lc"},
                "geometries": polygon,
                "reducer": {
                    "process_graph": {
                        "mean": {
                            "process_id": "mean",
                            "arguments": {"data": {"from_parameter": "data"}},
                            "result": True,
                        }
                    }
                },
            },
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))

    assert constraints == {
        "spatial_extent": {
            "crs": "EPSG:32631",
            "east": 1056748.2872412915,
            "north": 552664.2968779367,
            "south": 0.0,
            "west": 166021.44308054057,
        },
        "aggregate_spatial": {"geometries": DriverVectorCube.from_geojson(polygon)},
        "weak_spatial_extent": {
            "crs": "EPSG:32631",
            "east": 1056748.2872412915,
            "north": 552664.2968779367,
            "south": 0.0,
            "west": 166021.44308054057,
        },
    }
    (geometries,) = dry_run_tracer.get_geometries()
    assert isinstance(geometries, DriverVectorCube)
    assert geometries.to_geojson() == DictSubSet(
        type="FeatureCollection",
        features=[
            DictSubSet(
                geometry={
                    "type": "Polygon",
                    "coordinates": (((0.0, 0.0), (3.0, 5.0), (8.0, 2.0), (0.0, 0.0)),),
                }
            ),
        ],
    )


@pytest.mark.parametrize(
    ["geometries", "expected"],
    [
        (
            {"type": "Point", "coordinates": (2, 3)},
            approxify(
                {"crs": "EPSG:32631", "east": 388880.8418, "north": 331653.859, "south": 331634.017, "west": 388860.89}
            ),
        ),
        (
            as_geojson_feature_collection(
                shapely.geometry.Point(2, 3),
                shapely.geometry.Point(4, 5),
            ),
            approxify(
                {
                    "crs": "EPSG:32631",
                    "east": 611139.132350062,
                    "north": 552758.6209151235,
                    "south": 331633.9380733739,
                    "west": 388860.86764994,
                }
            ),
        ),
        (
            {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]},
            {
                "crs": "EPSG:32631",
                "east": 1056748.2872412915,
                "north": 552664.2968779367,
                "south": 0.0,
                "west": 166021.44308054057,
            },
        ),
        (
            as_geojson_feature_collection(
                shapely.geometry.Polygon.from_bounds(2, 3, 5, 8),
                shapely.geometry.Polygon.from_bounds(3, 5, 8, 13),
            ),
            {
                "crs": "EPSG:32631",
                "east": 1054970.5285543476,
                "north": 1442473.7600420578,
                "south": 331643.9380733739,
                "west": 388870.8676425279,
            },
        ),
        (
            as_geojson_feature_collection(
                shapely.geometry.Point(2, 3),
                shapely.geometry.Polygon.from_bounds(3, 5, 8, 13),
            ),
            approxify(
                {
                    "crs": "EPSG:32631",
                    "east": 1054970.528,
                    "north": 1442473.760,
                    "south": 331634.017,
                    "west": 388860.894,
                }
            ),
        ),
    ],
)
def test_aggregate_spatial_extent_handling_and_buffering(dry_run_env, dry_run_tracer, geometries, expected):
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "agg": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "lc"},
                "geometries": geometries,
                "reducer": {
                    "process_graph": {
                        "mean": {
                            "process_id": "mean",
                            "arguments": {"data": {"from_parameter": "data"}},
                            "result": True,
                        }
                    }
                },
            },
            "result": True,
        },
    }
    _ = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints["spatial_extent"] == expected


def test_aggregate_spatial_apply_dimension(dry_run_env, dry_run_tracer):
    polygon = {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]}
    pg = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {
                "bands": ["B04", "B08", "B11", "SCL"],
                "id": "S2_FOOBAR",
                "spatial_extent": None,
                "temporal_extent": ["2018-11-01", "2020-02-01"],
            },
        },
        "maskscldilation1": {
            "process_id": "mask_scl_dilation",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "scl_band_name": "SCL",
            },
        },
        "aggregatetemporalperiod1": {
            "process_id": "aggregate_temporal_period",
            "arguments": {
                "data": {"from_node": "maskscldilation1"},
                "period": "month",
                "reducer": {
                    "process_graph": {
                        "mean1": {
                            "process_id": "mean",
                            "arguments": {"data": {"from_parameter": "data"}},
                            "result": True,
                        }
                    }
                },
            },
        },
        "applydimension1": {
            "process_id": "apply_dimension",
            "arguments": {
                "data": {"from_node": "aggregatetemporalperiod1"},
                "dimension": "t",
                "process": {
                    "process_graph": {
                        "arrayinterpolatelinear1": {
                            "process_id": "array_interpolate_linear",
                            "arguments": {"data": {"from_parameter": "data"}},
                            "result": True,
                        }
                    }
                },
            },
        },
        "filtertemporal1": {
            "process_id": "filter_temporal",
            "arguments": {
                "data": {"from_node": "applydimension1"},
                "extent": ["2019-01-01", "2020-01-01"],
            },
        },
        "applydimension2": {
            "process_id": "apply_dimension",
            "arguments": {
                "data": {"from_node": "filtertemporal1"},
                "dimension": "bands",
                "process": {
                    "process_graph": {
                        "arrayelement1": {
                            "process_id": "array_element",
                            "arguments": {
                                "data": {"from_parameter": "data"},
                                "index": 1,
                            },
                        },
                        "arrayelement2": {
                            "process_id": "array_element",
                            "arguments": {
                                "data": {"from_parameter": "data"},
                                "index": 0,
                            },
                        },
                        "normalizeddifference1": {
                            "process_id": "normalized_difference",
                            "arguments": {
                                "x": {"from_node": "arrayelement1"},
                                "y": {"from_node": "arrayelement2"},
                            },
                        },
                        "arraymodify1": {
                            "process_id": "array_modify",
                            "arguments": {
                                "data": {"from_parameter": "data"},
                                "index": 0,
                                "values": {"from_node": "normalizeddifference1"},
                            },
                            "result": True,
                        },
                    }
                },
            },
        },
        "renamelabels1": {
            "process_id": "rename_labels",
            "arguments": {
                "data": {"from_node": "applydimension2"},
                "dimension": "bands",
                "target": ["NDVI", "B04", "B08"],
            },
        },
        "aggregatespatial1": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "renamelabels1"},
                "geometries": polygon,
                "reducer": {
                    "process_graph": {
                        "mean2": {
                            "process_id": "mean",
                            "arguments": {"data": {"from_parameter": "data"}},
                            "result": True,
                        }
                    }
                },
            },
            "result": True,
        },
    }

    cube = evaluate(pg, env=dry_run_env, do_dry_run=False)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", (), ("B04", "B08", "B11", "SCL")))
    assert constraints == {
        "spatial_extent": {
            "crs": "EPSG:32631",
            "east": 1056748.2872412915,
            "north": 552664.2968779367,
            "south": 0.0,
            "west": 166021.44308054057,
        },
        "process_type": [ProcessType.GLOBAL_TIME],
        "bands": ["B04", "B08", "B11", "SCL"],
        "custom_cloud_mask": {"method": "mask_scl_dilation", "scl_band_name": "SCL"},
        "aggregate_spatial": {"geometries": DriverVectorCube.from_geojson(polygon)},
        "temporal_extent": ("2018-11-01", "2020-02-01"),
        "weak_spatial_extent": {
            "crs": "EPSG:32631",
            "east": 1056748.2872412915,
            "north": 552664.2968779367,
            "south": 0.0,
            "west": 166021.44308054057,
        },
    }
    (geometries,) = dry_run_tracer.get_geometries()
    assert isinstance(geometries, DriverVectorCube)
    assert geometries.to_geojson() == DictSubSet(
        type="FeatureCollection",
        features=[
            DictSubSet(
                geometry={
                    "type": "Polygon",
                    "coordinates": (((0.0, 0.0), (3.0, 5.0), (8.0, 2.0), (0.0, 0.0)),),
                }
            ),
        ],
    )


def test_aggregate_spatial_and_filter_bbox(dry_run_env, dry_run_tracer):
    polygon = {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]}
    bbox = {"west": -1, "south": -1, "east": 9, "north": 9, "crs": "EPSG:4326"}
    cube = DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None)
    cube = cube.filter_bbox(bbox=bbox)
    cube = cube.aggregate_spatial(geometries=polygon, reducer="mean")

    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env, do_dry_run=False)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "spatial_extent": bbox,
        "aggregate_spatial": {"geometries": DriverVectorCube.from_geojson(polygon)},
        "weak_spatial_extent": {
            "crs": "EPSG:32631",
            "east": 1056748.2872412915,
            "north": 552664.2968779367,
            "south": 0.0,
            "west": 166021.44308054057,
        },
    }
    (geometries,) = dry_run_tracer.get_geometries()
    assert isinstance(geometries, DriverVectorCube)
    assert geometries.to_geojson() == DictSubSet(
        type="FeatureCollection",
        features=[
            DictSubSet(
                geometry={
                    "type": "Polygon",
                    "coordinates": (((0.0, 0.0), (3.0, 5.0), (8.0, 2.0), (0.0, 0.0)),),
                }
            ),
        ],
    )


def test_multiple_filter_spatial(dry_run_env, dry_run_tracer):
    polygon1 = {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]}
    polygon2 = {"type": "Polygon", "coordinates": [[(1, 1), (3, 5), (8, 2), (1, 1)]]}
    cube = DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None)
    cube = cube.filter_spatial(geometries=polygon1)
    cube = cube.resample_spatial(projection=4326, resolution=0.25)
    cube = cube.filter_spatial(geometries=polygon2)

    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env, do_dry_run=False)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    geometries = dry_run_tracer.get_last_geometry(operation="filter_spatial")
    assert constraints == {
        "spatial_extent": {
            "crs": "EPSG:32631",
            "east": 1056748.2872412915,
            "north": 552664.2968779367,
            "south": 0.0,
            "west": 166021.44308054057,
        },
        "filter_spatial": {"geometries": DummyVectorCube.from_geometry(shapely.geometry.shape(polygon1))},
        "resample": {"method": "near", "resolution": (0.25, 0.25), "target_crs": 4326},
        "weak_spatial_extent": {
            "crs": "EPSG:32631",
            "east": 1056748.2872412915,
            "north": 552664.2968779367,
            "south": 0.0,
            "west": 166021.44308054057,
        },
    }

    assert geometries == DummyVectorCube.from_geometry(shapely.geometry.shape(polygon2))


@pytest.mark.parametrize(
    ["path", "expected"],
    [
        ("geojson/Polygon01.json", (5.1, 51.2, 5.14, 51.23)),
        ("geojson/FeatureCollection01.json", (4.45, 51.1, 4.52, 51.2)),
    ],
)
def test_filter_spatial_delayed_vector(dry_run_env, dry_run_tracer, path, expected):
    cube = DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None)
    cube = cube.filter_spatial(geometries=get_path(path))
    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1

    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert isinstance(
        constraints["filter_spatial"]["geometries"],
        DummyVectorCube,
    )
    assert constraints["filter_spatial"]["geometries"].get_bounding_box() == expected


@pytest.mark.parametrize(
    ["url_path", "expected_bounds", "expected_crs"],
    [
        (
            "parquet/mol.pq",
            tuple(pytest.approx(x, abs=1) for x in (645146.7, 5672137.6, 648591.9, 5676210.3)),
            "EPSG:32631",
        ),
        (
            "parquet/mol-utm.pq",
            tuple(pytest.approx(x, abs=1) for x in (645146, 5672137, 648591, 5676210)),
            "EPSG:32631",
        ),
    ],
)
def test_filter_spatial_crs_handling(dry_run_env, dry_run_tracer, url_path, expected_bounds, expected_crs):
    """Test filter spatial with a Vector cube loaded from geoparquet, possibly in a non LonLat CRS."""
    with ephemeral_fileserver(TEST_DATA_ROOT) as fileserver_root:
        url = f"{fileserver_root}/{url_path}"

        cube = DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None)
        geometries = openeo.processes.load_url(url=url, format="Parquet")
        cube = cube.filter_spatial(geometries=geometries)
        pg = cube.flat_graph()
        res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1

    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    filter_spatial_geometries = constraints["filter_spatial"]["geometries"]
    assert isinstance(filter_spatial_geometries, DriverVectorCube)
    # assert filter_spatial_geometries.get_bounding_box() == expected_bounds
    # assert filter_spatial_geometries.get_crs().to_epsg() == expected_crs

    load_params = _extract_load_parameters(
        env=dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints}),
        source_id=("load_collection", ("S2_FOOBAR", ())),
    )
    # assert load_params.global_extent == BoundingBox.from_wsen_tuple([x.expected for x in expected_bounds],crs=expected_crs).as_dict()
    assert load_params.spatial_extent == dict(
        list(zip(["west", "south", "east", "north"], expected_bounds)) + [("crs", expected_crs)],
    )


def test_resample_filter_spatial(dry_run_env, dry_run_tracer):
    polygon = {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]}
    cube = DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None)
    cube = cube.filter_spatial(geometries=polygon)
    cube = cube.resample_spatial(projection=4326, resolution=0.25)

    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env, do_dry_run=False)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    (geometries,) = dry_run_tracer.get_geometries(operation="filter_spatial")
    assert constraints == {
        "spatial_extent": {
            "crs": "EPSG:32631",
            "east": 1056748.2872412915,
            "north": 552664.2968779367,
            "south": 0.0,
            "west": 166021.44308054057,
        },
        "filter_spatial": {"geometries": DummyVectorCube.from_geometry(shapely.geometry.shape(polygon))},
        "resample": {"method": "near", "resolution": (0.25, 0.25), "target_crs": 4326},
        "weak_spatial_extent": {
            "crs": "EPSG:32631",
            "east": 1056748.2872412915,
            "north": 552664.2968779367,
            "south": 0.0,
            "west": 166021.44308054057,
        },
    }
    assert isinstance(geometries, DummyVectorCube)
    assert shapely.geometry.mapping(geometries.to_multipolygon()) == {
        "type": "Polygon",
        "coordinates": (((0.0, 0.0), (3.0, 5.0), (8.0, 2.0), (0.0, 0.0)),),
    }


def test_resample_cube_spatial(dry_run_env, dry_run_tracer):
    polygon = {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]}
    cube = DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None)
    cube = cube.apply(lambda x: x + 1)
    cube = cube.resample_cube_spatial(
        target=DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None), method="max"
    )

    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env, do_dry_run=False)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 2
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))

    assert constraints == {"process_type": [ProcessType.FOCAL_SPACE]}


def test_auto_align(dry_run_env, dry_run_tracer):
    polygon = {"type": "Polygon", "coordinates": [[(0.1, 0.1), (3, 5), (8, 2), (0.1, 0.1)]]}
    cube = DataCube(PGNode("load_collection", id="ESA_WORLDCOVER_10M_2020_V1"), connection=None)
    cube = cube.filter_spatial(geometries=polygon)

    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("ESA_WORLDCOVER_10M_2020_V1", ()))
    (geometries,) = dry_run_tracer.get_geometries(operation="filter_spatial")
    assert constraints == {
        "spatial_extent": {"crs": "EPSG:4326", "east": 8.0, "north": 5.0, "south": 0.1, "west": 0.1},
        "filter_spatial": {"geometries": DummyVectorCube.from_geometry(shapely.geometry.shape(polygon))},
        "weak_spatial_extent": {"west": 0.1, "south": 0.1, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"},
    }
    assert isinstance(geometries, DummyVectorCube)
    assert shapely.geometry.mapping(geometries.to_multipolygon()) == {
        "type": "Polygon",
        "coordinates": (((0.1, 0.1), (3.0, 5.0), (8.0, 2.0), (0.1, 0.1)),),
    }
    # source_constraints = dry_run_tracer.get_source_constraints()
    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    load_params = _extract_load_parameters(
        dry_run_env, source_id=("load_collection", ("ESA_WORLDCOVER_10M_2020_V1", ()))
    )
    assert {
        "west": 0.09999999927961767,
        "east": 8.00008333258134,
        "south": 0.09999999971959994,
        "north": 5.000083333033345,
        "crs": "EPSG:4326",
    } == load_params.global_extent


def test_no_auto_align_when_resampling(dry_run_env, dry_run_tracer):
    polygon = {"type": "Polygon", "coordinates": [[(0.1, 0.1), (3, 5), (8, 2), (0.1, 0.1)]]}
    cube = DataCube(PGNode("load_collection", id="ESA_WORLDCOVER_10M_2020_V1"), connection=None)
    cube = cube.filter_spatial(geometries=polygon)
    cube = cube.resample_spatial(resolution = 0.001)

    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("ESA_WORLDCOVER_10M_2020_V1", ()))
    (geometries,) = dry_run_tracer.get_geometries(operation="filter_spatial")
    assert constraints == {
        'resample': {'method': 'near',
                     'resolution': (0.001, 0.001),
                     'target_crs': None},
        "spatial_extent": {"crs": "EPSG:4326", "east": 8.0, "north": 5.0, "south": 0.1, "west": 0.1},
        "filter_spatial": {"geometries": DummyVectorCube.from_geometry(shapely.geometry.shape(polygon))},
        "weak_spatial_extent": {"west": 0.1, "south": 0.1, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"},
    }
    assert isinstance(geometries, DummyVectorCube)
    assert shapely.geometry.mapping(geometries.to_multipolygon()) == {
        "type": "Polygon",
        "coordinates": (((0.1, 0.1), (3.0, 5.0), (8.0, 2.0), (0.1, 0.1)),),
    }
    # source_constraints = dry_run_tracer.get_source_constraints()
    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    load_params = _extract_load_parameters(
        dry_run_env, source_id=("load_collection", ("ESA_WORLDCOVER_10M_2020_V1", ()))
    )
    assert {'crs': 'EPSG:4326', 'east': 8.0, 'north': 5.0, 'south': 0.1, 'west': 0.1} == load_params.global_extent


def test_auto_align_without_explicit_resolution(dry_run_env, dry_run_tracer):
    """
    Test against a ZeroDivisionError regression. https://github.com/Open-EO/openeo-geotrellis-extensions/issues/506
    """
    polygon = {"type": "Polygon", "coordinates": [[(0.1, 0.1), (3, 5), (8, 2), (0.1, 0.1)]]}
    cube = DataCube(PGNode("load_collection", id="S2_FAPAR_CLOUDCOVER"), connection=None)
    cube = cube.filter_spatial(geometries=polygon)
    cube = cube.resample_spatial(projection="EPSG:32632")

    pg = cube.flat_graph()
    evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FAPAR_CLOUDCOVER", ()))
    (geometries,) = dry_run_tracer.get_geometries(operation="filter_spatial")
    assert constraints == {
        "resample": {"method": "near", "resolution": (0, 0), "target_crs": "EPSG:32632"},
        "spatial_extent": {
            "crs": "EPSG:32631",
            "east": 1056748.2872412915,
            "north": 552664.2968779367,
            "south": 11067.273501737884,
            "west": 177164.2952187257,
        },
        "filter_spatial": {"geometries": DummyVectorCube.from_geometry(shapely.geometry.shape(polygon))},
        "weak_spatial_extent": {
            "crs": "EPSG:32631",
            "east": 1056748.2872412915,
            "north": 552664.2968779367,
            "south": 11067.273501737884,
            "west": 177164.2952187257,
        },
    }
    assert isinstance(geometries, DummyVectorCube)
    assert shapely.geometry.mapping(geometries.to_multipolygon()) == {
        "type": "Polygon",
        "coordinates": (((0.1, 0.1), (3.0, 5.0), (8.0, 2.0), (0.1, 0.1)),),
    }
    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    load_params = _extract_load_parameters(dry_run_env, source_id=("load_collection", ("S2_FAPAR_CLOUDCOVER", ())))
    assert {
        "crs": "EPSG:32631",
        "east": 1056750,
        "north": 552670,
        "south": 11060,
        "west": 177160,
    } == load_params.global_extent


def test_global_bounds_from_weak_spatial_extent(dry_run_env, dry_run_tracer):
    # The global extent is the union from the spatial extent and the weak spatial extent.
    bbox = {"west": 1, "south": 1, "east": 3, "north": 3, "crs": "EPSG:4326"}
    polygon = {"type": "Polygon", "coordinates": [[(0.1, 0.1), (3, 5), (8, 2), (0.1, 0.1)]]}
    cube = DataCube(PGNode("load_collection", id="ESA_WORLDCOVER_10M_2020_V1"), connection=None)
    cube = cube.filter_bbox(bbox=bbox)
    cube = cube.filter_spatial(geometries=polygon)

    pg = cube.flat_graph()
    evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("ESA_WORLDCOVER_10M_2020_V1", ()))
    assert constraints == {
        "spatial_extent": {"west": 1, "south": 1, "east": 3, "north": 3, "crs": "EPSG:4326"},
        "weak_spatial_extent": {"crs": "EPSG:4326", "east": 8.0, "north": 5.0, "south": 0.1, "west": 0.1},
        "filter_spatial": {"geometries": DummyVectorCube.from_geometry(shapely.geometry.shape(polygon))},
    }
    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    load_params = _extract_load_parameters(
        dry_run_env, source_id=("load_collection", ("ESA_WORLDCOVER_10M_2020_V1", ()))
    )
    assert {
        "west": 0.9999999992760138,
        "east": 3.000083332601349,
        "south": 0.9999999997160103,
        "north": 3.0000833330413315,
        "crs": "EPSG:4326",
    } == load_params.global_extent


def test_aggregate_spatial_read_vector(dry_run_env, dry_run_tracer):
    geometry_path = str(get_path("geojson/GeometryCollection01.json"))
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "ESA_WORLDCOVER_10M_2020_V1"}},
        "vector": {"process_id": "read_vector", "arguments": {"filename": geometry_path}},
        "agg": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "lc"},
                "geometries": {"from_node": "vector"},
                "reducer": {
                    "process_graph": {
                        "mean": {
                            "process_id": "mean",
                            "arguments": {"data": {"from_parameter": "data"}},
                            "result": True,
                        }
                    }
                },
            },
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    (geometries,) = dry_run_tracer.get_geometries()
    assert src == ("load_collection", ("ESA_WORLDCOVER_10M_2020_V1", ()))

    assert constraints == {
        "spatial_extent": {"west": 5.05, "south": 51.21, "east": 5.15, "north": 51.3, "crs": "EPSG:4326"},
        "aggregate_spatial": {"geometries": geometries},
        "weak_spatial_extent": {"crs": "EPSG:4326", "east": 5.15, "north": 51.3, "south": 51.21, "west": 5.05},
    }
    assert isinstance(geometries, DriverVectorCube)


def test_aggregate_spatial_get_geometries_feature_collection(dry_run_env, dry_run_tracer):
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "ESA_WORLDCOVER_10M_2020_V1"}},
        "vector": {
            "process_id": "get_geometries",
            "arguments": {
                "feature_collection": {
                    "type": "FeatureCollection",
                    "name": "fields",
                    "crs": {
                        "type": "name",
                        "properties": {"name": "EPSG:4326"},
                    },
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]],
                            },
                            "properties": {"CODE_OBJ": "0000000000000001"},
                        }
                    ],
                }
            },
        },
        "agg": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "lc"},
                "geometries": {"from_node": "vector"},
                "reducer": {
                    "process_graph": {
                        "mean": {
                            "process_id": "mean",
                            "arguments": {"data": {"from_parameter": "data"}},
                            "result": True,
                        }
                    }
                },
            },
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("ESA_WORLDCOVER_10M_2020_V1", ()))

    expected_geometry_collection = DriverVectorCube.from_geojson(pg["vector"]["arguments"]["feature_collection"])
    assert constraints == {
        "spatial_extent": {
            "west": 0.0,
            "south": 0.0,
            "east": 8.0,
            "north": 5.0,
            "crs": "EPSG:4326",
        },
        "aggregate_spatial": {"geometries": expected_geometry_collection},
        "weak_spatial_extent": {"west": 0.0, "south": 0.0, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"},
    }
    (geometries,) = dry_run_tracer.get_geometries()
    assert isinstance(geometries, DriverVectorCube)


@pytest.mark.parametrize(
    ["arguments", "expected"],
    [
        (
            {},
            SarBackscatterArgs(
                coefficient="gamma0-terrain",
                elevation_model=None,
                mask=False,
                contributing_area=False,
                local_incidence_angle=False,
                ellipsoid_incidence_angle=False,
                noise_removal=True,
                options={},
            ),
        ),
        (
            {
                "coefficient": "gamma0-ellipsoid",
                "elevation_model": "SRTMGL1",
                "mask": True,
                "contributing_area": True,
                "local_incidence_angle": True,
                "ellipsoid_incidence_angle": True,
                "noise_removal": False,
                "options": {"tile_size": 1024},
            },
            SarBackscatterArgs(
                coefficient="gamma0-ellipsoid",
                elevation_model="SRTMGL1",
                mask=True,
                contributing_area=True,
                local_incidence_angle=True,
                ellipsoid_incidence_angle=True,
                noise_removal=False,
                options={"tile_size": 1024},
            ),
        ),
    ],
)
def test_evaluate_sar_backscatter(dry_run_env, dry_run_tracer, arguments, expected):
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "sar": {
            "process_id": "sar_backscatter",
            "arguments": dict(data={"from_node": "lc"}, **arguments),
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {"sar_backscatter": expected}


def test_load_collection_properties(dry_run_env, dry_run_tracer):
    def get_props(direction="DESCENDING"):
        return {
            "orbitDirection": {
                "process_graph": {
                    "od": {
                        "process_id": "eq",
                        "arguments": {"x": {"from_parameter": "value"}, "y": direction},
                        "result": True,
                    }
                }
            }
        }

    properties = get_props()
    asc_props = get_props("ASCENDING")
    pg = {
        "lc": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR", "properties": properties},
        },
        "lc2": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR", "properties": asc_props},
        },
        "merge": {
            "process_id": "merge_cubes",
            "arguments": {"cube1": {"from_node": "lc"}, "cube2": {"from_node": "lc2"}},
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)

    assert source_constraints == [
        (
            (
                "load_collection",
                (
                    "S2_FOOBAR",
                    (
                        (
                            "orbitDirection",
                            (("eq", "DESCENDING"),),
                        ),
                    ),
                ),
            ),
            {"properties": properties},
        ),
        (
            (
                "load_collection",
                (
                    "S2_FOOBAR",
                    (("orbitDirection", (("eq", "ASCENDING"),)),),
                ),
            ),
            {"properties": asc_props},
        ),
    ]


def test_load_stac_properties(dry_run_env, dry_run_tracer):
    properties = {
        "platform": {
            "process_graph": {
                "eq1": {"process_id": "eq", "arguments": {"x": {"from_parameter": "value"}, "y": "S2A"}, "result": True}
            }
        }
    }

    pg = {
        "ls": {
            "process_id": "load_stac",
            "arguments": {
                "url": "https://example.org/collections/S2",
                "properties": properties,
                "bands": ["B04", "B05"],
            },
            "result": True,
        }
    }

    evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)

    assert source_constraints == [
        (
            (
                "load_stac",
                (
                    "https://example.org/collections/S2",
                    (
                        (
                            "platform",
                            (("eq", "S2A"),),
                        ),
                    ),
                    ("B04", "B05"),
                ),
            ),
            {"bands": ["B04", "B05"], "properties": properties},
        ),
    ]


@pytest.mark.parametrize(
    ["spatial_extent", "expectation"],
    [
        ({"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0]]]}, nullcontext()),
        (
            {
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": [[[[0, 0], [1, 1], [1, 0]]]]},
                "properties": {},
            },
            nullcontext(),
        ),
        (
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [5, 50]},
                "properties": {},
            },
            pytest.raises(
                ProcessParameterInvalidException,
                match=r"unsupported GeoJSON; requires at least one Polygon or MultiPolygon$",
            ),
        ),
        (
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [5, 50]},
                        "properties": {},
                    },
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0]]]},
                        "properties": {},
                    },
                ],
            },
            pytest.raises(
                ProcessParameterInvalidException,
                match=r"unsupported GeoJSON; requires at least one Polygon or MultiPolygon$",
            ),
        ),
    ],
)
def test_load_stac_spatial_extent_requires_a_polygon(
    dry_run_tracer, backend_implementation, spatial_extent, expectation
):
    pg = {
        "loadstac1": {
            "process_id": "load_stac",
            "arguments": {
                "url": "https://stac.test",
                "spatial_extent": spatial_extent,
            },
            "result": True,
        }
    }

    dry_run_env = EvalEnv(
        {ENV_DRY_RUN_TRACER: dry_run_tracer, "backend_implementation": backend_implementation, "version": "2.0.0"}
    )

    with expectation:
        evaluate(pg, dry_run_env)


@pytest.mark.parametrize(
    ["arguments", "expected"],
    [
        (
            {},
            [
                {"rel": "atmospheric-scattering", "href": "https://remotesensing.vito.be/case/icor"},
                {
                    "href": "https://atmosphere.copernicus.eu/catalogue#/product/urn:x-wmo:md:int.ecmwf::copernicus:cams:prod:fc:total-aod:pid094",
                    "rel": "related",
                },
                {"href": "https://doi.org/10.7289/V52R3PMS", "rel": "elevation-model"},
                {"href": "https://doi.org/10.1109/LGRS.2016.2635942", "rel": "water-vapor"},
            ],
        ),
        (
            {"method": "SMAC"},
            [
                {"rel": "atmospheric-scattering", "href": "https://doi.org/10.1080/01431169408954055"},
                {
                    "href": "https://atmosphere.copernicus.eu/catalogue#/product/urn:x-wmo:md:int.ecmwf::copernicus:cams:prod:fc:total-aod:pid094",
                    "rel": "related",
                },
                {"href": "https://doi.org/10.7289/V52R3PMS", "rel": "elevation-model"},
                {"href": "https://doi.org/10.1109/LGRS.2016.2635942", "rel": "water-vapor"},
            ],
        ),
    ],
)
def test_evaluate_atmospheric_correction(dry_run_env, dry_run_tracer, arguments, expected):
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "sar": {
            "process_id": "atmospheric_correction",
            "arguments": dict(data={"from_node": "lc"}, **arguments),
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    metadata_links = dry_run_tracer.get_metadata_links()
    assert len(metadata_links) == 1
    src, links = metadata_links.popitem()
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert links == expected


def test_evaluate_predefined_property(backend_implementation):
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "TERRASCOPE_S2_FAPAR_V2"}, "result": True},
    }

    env = EvalEnv(dict(backend_implementation=backend_implementation, version="1.0.0"))
    evaluate(pg, do_dry_run=True, env=env)


def test_sources_are_subject_to_correct_constraints(dry_run_env, dry_run_tracer):
    pg = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {
                "bands": ["VV", "VH"],
                "id": "S2_FOOBAR",
                "spatial_extent": {
                    "west": 11.465226,
                    "east": 11.465435,
                    "south": 46.343118,
                    "north": 46.343281,
                    "crs": "EPSG:4326",
                },
                "temporal_extent": ["2018-01-01", "2018-01-01"],
            },
        },
        "sarbackscatter1": {
            "process_id": "sar_backscatter",
            "arguments": {
                "coefficient": "sigma0-ellipsoid",
                "contributing_area": False,
                "data": {"from_node": "loadcollection1"},
                "elevation_model": None,
                "ellipsoid_incidence_angle": False,
                "local_incidence_angle": False,
                "mask": False,
                "noise_removal": True,
            },
        },
        "renamelabels1": {
            "process_id": "rename_labels",
            "arguments": {
                "data": {"from_node": "sarbackscatter1"},
                "dimension": "bands",
                "source": ["VV", "VH"],
                "target": ["VV_sigma0", "VH_sigma0"],
            },
        },
        "sarbackscatter2": {
            "process_id": "sar_backscatter",
            "arguments": {
                "coefficient": "gamma0-terrain",
                "contributing_area": False,
                "data": {"from_node": "loadcollection1"},
                "elevation_model": None,
                "ellipsoid_incidence_angle": False,
                "local_incidence_angle": False,
                "mask": False,
                "noise_removal": True,
            },
        },
        "renamelabels2": {
            "process_id": "rename_labels",
            "arguments": {
                "data": {"from_node": "sarbackscatter2"},
                "dimension": "bands",
                "source": ["VV", "VH"],
                "target": ["VV_gamma0", "VH_gamma0"],
            },
        },
        "mergecubes1": {
            "process_id": "merge_cubes",
            "arguments": {"cube1": {"from_node": "renamelabels1"}, "cube2": {"from_node": "renamelabels2"}},
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 2

    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", (), ("VV", "VH")))
    assert constraints == {
        "temporal_extent": ("2018-01-01", "2018-01-01"),
        "spatial_extent": {
            "west": 11.465226,
            "east": 11.465435,
            "south": 46.343118,
            "north": 46.343281,
            "crs": "EPSG:4326",
        },
        "bands": ["VV", "VH"],
        "sar_backscatter": SarBackscatterArgs(
            coefficient="sigma0-ellipsoid",
            elevation_model=None,
            mask=False,
            contributing_area=False,
            local_incidence_angle=False,
            ellipsoid_incidence_angle=False,
            noise_removal=True,
            options={},
        ),
    }

    src, constraints = source_constraints[1]
    assert src == ("load_collection", ("S2_FOOBAR", (), ("VV", "VH")))
    assert constraints == {
        "temporal_extent": ("2018-01-01", "2018-01-01"),
        "spatial_extent": {
            "west": 11.465226,
            "east": 11.465435,
            "south": 46.343118,
            "north": 46.343281,
            "crs": "EPSG:4326",
        },
        "bands": ["VV", "VH"],
        "sar_backscatter": SarBackscatterArgs(
            coefficient="gamma0-terrain",
            elevation_model=None,
            mask=False,
            contributing_area=False,
            local_incidence_angle=False,
            ellipsoid_incidence_angle=False,
            noise_removal=True,
            options={},
        ),
    }


def test_pixel_buffer(dry_run_env, dry_run_tracer):
    pg = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {
                "bands": ["VV", "VH"],
                "id": "S2_FOOBAR",
                "spatial_extent": {
                    "west": 11.465226,
                    "east": 11.465435,
                    "south": 46.343118,
                    "north": 46.343281,
                    "crs": "EPSG:4326",
                },
                "temporal_extent": ["2018-01-01", "2018-01-01"],
            },
        },
        "apply_kernel": {
            "process_id": "apply_kernel",
            "arguments": {"kernel": [[1.0, 1.0], [1.0, 1.0]], "data": {"from_node": "loadcollection1"}},
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1

    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", (), ("VV", "VH")))
    assert constraints == {
        "temporal_extent": ("2018-01-01", "2018-01-01"),
        "spatial_extent": {
            "west": 11.465226,
            "east": 11.465435,
            "south": 46.343118,
            "north": 46.343281,
            "crs": "EPSG:4326",
        },
        "bands": ["VV", "VH"],
        "pixel_buffer": {"buffer_size": [1.0, 1.0]},
        "process_type": [ProcessType.FOCAL_SPACE],
    }


def test_filter_after_merge_cubes(dry_run_env, dry_run_tracer):
    """based on use case of https://jira.vito.be/browse/EP-3747"""
    pg = {
        "loadcollection1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR", "bands": ["B04", "B08"]}},
        "reducedimension1": {
            "process_id": "reduce_dimension",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "dimension": "bands",
                "reducer": {
                    "process_graph": {
                        "arrayelement1": {
                            "process_id": "array_element",
                            "arguments": {"data": {"from_parameter": "data"}, "index": 1},
                        },
                        "arrayelement2": {
                            "process_id": "array_element",
                            "arguments": {"data": {"from_parameter": "data"}, "index": 0},
                        },
                        "subtract1": {
                            "process_id": "subtract",
                            "arguments": {"x": {"from_node": "arrayelement1"}, "y": {"from_node": "arrayelement2"}},
                        },
                        "add1": {
                            "process_id": "add",
                            "arguments": {"x": {"from_node": "arrayelement1"}, "y": {"from_node": "arrayelement2"}},
                        },
                        "divide1": {
                            "process_id": "divide",
                            "arguments": {"x": {"from_node": "subtract1"}, "y": {"from_node": "add1"}},
                            "result": True,
                        },
                    }
                },
            },
        },
        "adddimension1": {
            "process_id": "add_dimension",
            "arguments": {
                "data": {"from_node": "reducedimension1"},
                "label": "s2_ndvi",
                "name": "bands",
                "type": "bands",
            },
        },
        "loadcollection2": {
            "process_id": "load_collection",
            "arguments": {
                "id": "PROBAV_L3_S10_TOC_NDVI_333M_V2",
                "bands": ["ndvi"],
            },
        },
        "resamplecubespatial1": {
            "process_id": "resample_cube_spatial",
            "arguments": {
                "data": {"from_node": "loadcollection2"},
                "method": "average",
                "target": {"from_node": "adddimension1"},
            },
        },
        "maskpolygon1": {
            "process_id": "mask_polygon",
            "arguments": {
                "data": {"from_node": "resamplecubespatial1"},
                "mask": {
                    "type": "Polygon",
                    "coordinates": [
                        [[5.03536, 51.219], [5.03586, 51.230], [5.01754, 51.231], [5.01704, 51.219], [5.03536, 51.219]]
                    ],
                },
            },
        },
        "mergecubes1": {
            "process_id": "merge_cubes",
            "arguments": {"cube1": {"from_node": "adddimension1"}, "cube2": {"from_node": "maskpolygon1"}},
        },
        "filtertemporal1": {
            "process_id": "filter_temporal",
            "arguments": {"data": {"from_node": "mergecubes1"}, "extent": ["2019-03-01", "2019-04-01"]},
        },
        "filterbbox1": {
            "process_id": "filter_bbox",
            "arguments": {
                "data": {"from_node": "filtertemporal1"},
                "extent": {
                    "west": 640860.0,
                    "east": 642140.0,
                    "north": 5677450.0,
                    "south": 5676170.0,
                    "crs": "EPSG:32631",
                },
            },
            "result": True,
        },
    }

    cube = evaluate(pg, env=dry_run_env, do_dry_run=False)
    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert source_constraints == [
        (
            ("load_collection", ("S2_FOOBAR", (), ("B04", "B08"))),
            {
                "bands": ["B04", "B08"],
                "spatial_extent": {
                    "crs": "EPSG:32631",
                    "east": 642140.0,
                    "north": 5677450.0,
                    "south": 5676170.0,
                    "west": 640860.0,
                },
                "temporal_extent": ("2019-03-01", "2019-04-01"),
                "weak_spatial_extent": {
                    "crs": "EPSG:4326",
                    "east": 5.03586,
                    "north": 51.231,
                    "south": 51.219,
                    "west": 5.01704,
                },
            },
        ),
        (
            ("load_collection", ("S2_FOOBAR", (), ("B04", "B08"))),
            {
                "bands": ["B04", "B08"],
                "spatial_extent": {
                    "crs": "EPSG:32631",
                    "east": 642140.0,
                    "north": 5677450.0,
                    "south": 5676170.0,
                    "west": 640860.0,
                },
                "temporal_extent": ("2019-03-01", "2019-04-01"),
            },
        ),
        (
            ("load_collection", ("PROBAV_L3_S10_TOC_NDVI_333M_V2", (), ("ndvi",))),
            {
                "bands": ["ndvi"],
                "process_type": [ProcessType.FOCAL_SPACE],
                "resample": {"method": "average", "resolution": (10, 10), "target_crs": CRS_UTM},
                "spatial_extent": {
                    "crs": "EPSG:32631",
                    "east": 642140.0,
                    "north": 5677450.0,
                    "south": 5676170.0,
                    "west": 640860.0,
                },
                "temporal_extent": ("2019-03-01", "2019-04-01"),
                "weak_spatial_extent": {
                    "crs": "EPSG:4326",
                    "east": 5.03586,
                    "north": 51.231,
                    "south": 51.219,
                    "west": 5.01704,
                },
            },
        ),
        (("load_collection", ("S2_FOOBAR", (), ("B04", "B08"))), {"bands": ["B04", "B08"]}),
    ]


def test_worldwater(dry_run_env, dry_run_tracer):
    ww_process = load_json("pg/1.0/worldwater_graph.json")
    evaluate(ww_process["process_graph"], env=dry_run_env, do_dry_run=False)
    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    # print(source_constraints)
    for c in source_constraints:
        if "WORLDCOVER" in str(c):
            assert "bands" not in c[1]


def test_CropSAR_aggregate_spatial_constraint(dry_run_env, dry_run_tracer):
    cropsar_process = load_json("pg/1.0/cropsar_graph.json")
    custom_process_from_process_graph(cropsar_process, namespace="test")

    try:
        geometry_path = str(get_path("geojson/thaipolys_ad.geojson"))

        pg = {
            "CropSAR1": {
                "process_id": "CropSAR",
                "arguments": {"file_polygons": geometry_path, "time_range": ["2019-07-01", "2019-08-31"]},
                "namespace": "test",
                "result": True,
            }
        }

        evaluate(pg, env=dry_run_env, do_dry_run=False)
        source_constraints = dry_run_tracer.get_source_constraints(merge=True)

        assert len(source_constraints) > 0

        constraints_with_geometry = [c for c in source_constraints if "aggregate_spatial" in c[1]]
        assert len(constraints_with_geometry) == 3

        for _, constraints in constraints_with_geometry:
            assert isinstance(constraints["aggregate_spatial"]["geometries"], DriverVectorCube)
    finally:
        del process_registry_100._processes["test", "CropSAR"]


@pytest.mark.parametrize(
    ["dimension_name", "expected"],
    [
        ("bands", ["x", "y", "t"]),
        ("t", ["x", "y", "bands"]),
    ],
)
def test_evaluate_drop_dimension(dry_run_env, dry_run_tracer, dimension_name, expected):
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "drop": {
            "process_id": "drop_dimension",
            "arguments": {"data": {"from_node": "lc"}, "name": dimension_name},
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)
    assert cube.metadata.dimension_names() == expected


def test_load_result_constraints(dry_run_env, dry_run_tracer):
    dry_run_env = dry_run_env.push({"openeo_api_version": "1.0.0"})
    pg = {
        "loadresult1": {
            "process_id": "load_result",
            "arguments": {
                "id": "https://oeo.net/openeo/1.1/jobs/j-f1ce01ef51d1481abaab7ceceb19c650/results",
                "spatial_extent": {"west": 0, "south": 50, "east": 5, "north": 55},
                "temporal_extent": ("2020-01-01", "2020-02-02"),
                "bands": ["VV", "VH"],
            },
            "result": True,
        }
    }

    evaluate(pg, env=dry_run_env)
    source_constraints = dry_run_tracer.get_source_constraints(merge=True)

    assert source_constraints == [
        (
            ("load_result", ("https://oeo.net/openeo/1.1/jobs/j-f1ce01ef51d1481abaab7ceceb19c650/results",)),
            {
                "bands": ["VV", "VH"],
                "spatial_extent": {
                    "crs": "EPSG:4326",
                    "east": 5,
                    "north": 55,
                    "south": 50,
                    "west": 0,
                },
                "temporal_extent": ("2020-01-01", "2020-02-02"),
            },
        )
    ]


def test_multiple_save_result(dry_run_env):
    pg = {
        "collection1": {"process_id": "load_collection", "arguments": {"id": "S2_FAPAR_CLOUDCOVER"}},
        "collection2": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "saveresult1": {
            "process_id": "save_result",
            "arguments": {"options": {}, "data": {"from_node": "collection2"}, "format": "GTiff"},
        },
        "mergecubes1": {
            "process_id": "merge_cubes",
            "arguments": {
                "cube1": {"from_node": "collection1"},
                "cube2": {"from_node": "saveresult1"},
                "overlap_resolver": {
                    "process_graph": {
                        "or1": {
                            "process_id": "or",
                            "arguments": {"x": {"from_parameter": "x"}, "y": {"from_parameter": "y"}},
                            "result": True,
                        }
                    }
                },
            },
            "result": False,
        },
        "saveresult2": {
            "result": True,
            "arguments": {"options": {}, "data": {"from_node": "mergecubes1"}, "format": "netCDF"},
            "process_id": "save_result",
        },
    }
    dry_run_env = dry_run_env.push({ENV_SAVE_RESULT: []})
    the_result = evaluate(pg, env=dry_run_env)
    save_result = dry_run_env.get(ENV_SAVE_RESULT)
    assert len(save_result) == 2
    assert len(the_result) == 2


def test_non_result_subtrees_are_evaluated(dry_run_env, caplog):
    pg = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FAPAR_CLOUDCOVER",
                "spatial_extent": {"west": 5, "south": 50, "east": 5.1, "north": 50.1},
                "temporal_extent": ["2024-07-11", "2024-07-21"],
                "bands": ["Flat:1"],
            },
        },
        "loadcollection2": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FOOBAR",
                "spatial_extent": {"west": 5, "south": 50, "east": 5.1, "north": 50.1},
                "temporal_extent": ["2024-07-11", "2024-07-21"],
                "bands": ["Flat:2"],
            },
        },
        "inspect1": {
            "process_id": "inspect",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "message": "intermediate result",
                "level": "warning",
            },
        },
        "mergecubes1": {
            "process_id": "merge_cubes",
            "arguments": {
                "cube1": {"from_node": "loadcollection1"},
                "cube2": {"from_node": "loadcollection2"},
            },
            "result": True,
        },
        "saveresult2": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "mergecubes1"}, "format": "netCDF"},
        },
    }

    result = evaluate(pg, env=dry_run_env)

    # side-effect 1: output asset
    assert result.format == "netCDF"

    # side-effect 2: inspect log
    assert "intermediate result" in caplog.messages


def test_invalid_latlon_in_geojson(dry_run_env):
    init_cube = DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None)

    polygon1 = {"type": "Polygon", "coordinates": [[(-361, 0), (3, 5), (8, 2), (0, 0)]]}
    cube = init_cube.filter_spatial(geometries=polygon1)
    with pytest.raises(OpenEOApiException) as e:
        evaluate(cube.flat_graph(), env=dry_run_env)
    assert e.value.message.startswith("Failed to parse Geojson. Invalid coordinate: (-361, 0)")

    polygon2 = {"type": "Polygon", "coordinates": [[(1, 1), (3, 5), (8, 101), (1, 1)]]}
    cube = init_cube.filter_spatial(geometries=polygon2)
    with pytest.raises(OpenEOApiException) as e:
        evaluate(cube.flat_graph(), env=dry_run_env)
    assert e.value.message.startswith("Failed to parse Geojson. Invalid coordinate: (8, 101)")

    polygon3 = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    (-180, -90.0),
                    (-180.0, 90.0),
                    (180.0, 90),
                    (180, -90.0),
                    (-180.0, -90.0),
                ]
            ]
        ],
    }
    cube = init_cube.filter_spatial(geometries=polygon3)
    evaluate(cube.flat_graph(), env=dry_run_env)

    geometrycollection = {
        "type": "GeometryCollection",
        "geometries": [
            {
                "type": "Polygon",
                "coordinates": [[[0.1, 0.1], [1.8, 0.1], [1.1, 1.8], [0.1, 0.1]]],
            },
            {
                "type": "Polygon",
                "coordinates": [[[2.99, -1.29], [2.279, 1.724], [0.725, -0.18], [0.725, -0.516]]],
            },
            {
                "type": "GeometryCollection",
                "geometries": [
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-361, -1.29],
                                [2.279, 1.724],
                                [0.725, -0.18],
                                [0.725, -0.516],
                            ]
                        ],
                    }
                ],
            },
        ],
    }
    cube = init_cube.filter_spatial(geometries=geometrycollection)
    with pytest.raises(OpenEOApiException) as e:
        evaluate(cube.flat_graph(), env=dry_run_env)
    assert e.value.message.startswith("Failed to parse Geojson. Invalid coordinate: [-361, -1.29]")

    point = {"type": "Point", "coordinates": [0, 0]}
    cube = init_cube.filter_spatial(geometries=point)
    evaluate(cube.flat_graph(), env=dry_run_env)

    multilinestring = {
        "type": "MultiLineString",
        "coordinates": [
            [[0, 0], [1, 1]],
            [[1, 0], [0, 1]],
        ],
    }
    cube = init_cube.filter_spatial(geometries=multilinestring)
    evaluate(cube.flat_graph(), env=dry_run_env)

    polygon_with_holes = {
        "type": "Polygon",
        "coordinates": [
            [[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]],
            [[100.8, 0.8], [100.8, 0.2], [100.2, 0.2], [100.2, 0.8], [100.8, 0.8]],
        ],
    }
    cube = init_cube.filter_spatial(geometries=polygon_with_holes)
    evaluate(cube.flat_graph(), env=dry_run_env)

    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": [-71.073283, -101]},
            }
        ],
    }
    cube = init_cube.filter_spatial(geometries=feature_collection)
    with pytest.raises(OpenEOApiException) as e:
        evaluate(cube.flat_graph(), env=dry_run_env)
    assert e.value.message.startswith("Failed to parse Geojson. Invalid coordinate: [-71.073283, -101]")

    multipoint_many_coordinates = {
        "type": "MultiPoint",
        "coordinates": [(x, y) for x in range(150, 200) for y in range(70, 101)],
    }
    cube = init_cube.filter_spatial(geometries=multipoint_many_coordinates)
    evaluate(cube.flat_graph(), env=dry_run_env)


@pytest.mark.parametrize("remove_original", [False, True])
def test_export_workspace(dry_run_tracer, backend_implementation, remove_original):
    mock_workspace_repository = mock.Mock(WorkspaceRepository)
    mock_workspace = mock_workspace_repository.get_by_id.return_value

    dry_run_env = EvalEnv(
        {ENV_DRY_RUN_TRACER: dry_run_tracer, "backend_implementation": backend_implementation, "version": "2.0.0"}
    )

    pg = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR"},
        },
        "saveresult1": {
            "process_id": "save_result",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "format": "GTiff",
            },
        },
        "exportworkspace1": {
            "process_id": "export_workspace",
            "arguments": {
                "data": {"from_node": "saveresult1"},
                "workspace": "some-workspace",
                "merge": "some/path",
            },
            "result": True,
        },
    }

    save_result = evaluate(pg, env=dry_run_env)

    assert save_result.is_format("GTiff")

    save_result.export_workspace(
        mock_workspace_repository,
        hrefs=["file:file1", "file:file2"],
        default_merge="/some/unique/path",
        remove_original=remove_original,
    )

    assert list(save_result.workspace_exports) == [
        SaveResult.WorkspaceExport(workspace_id="some-workspace", merge="some/path")
    ]

    mock_workspace.import_file.assert_has_calls(
        [
            mock.call(Path("file1"), "some/path", remove_original),
            mock.call(Path("file2"), "some/path", remove_original),
        ]
    )


@pytest.mark.parametrize("remove_original", [False, True])
def test_export_workspace_with_multiple_save_result(dry_run_tracer, backend_implementation, remove_original):
    mock_workspace_repository = mock.Mock(WorkspaceRepository)
    mock_workspace = mock_workspace_repository.get_by_id.return_value

    dry_run_env = EvalEnv(
        {ENV_DRY_RUN_TRACER: dry_run_tracer, "backend_implementation": backend_implementation, "version": "2.0.0"}
    )

    pg = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR"},
        },
        "saveresult1": {
            "process_id": "save_result",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "format": "netCDF",
            },
        },
        "exportworkspace1": {
            "process_id": "export_workspace",
            "arguments": {"data": {"from_node": "saveresult1"}, "workspace": "some-workspace", "merge": "some/path"},
        },
        "saveresult2": {
            "process_id": "save_result",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "format": "GTiff",
            },
        },
        "exportworkspace2": {
            "process_id": "export_workspace",
            "arguments": {
                "data": {"from_node": "saveresult2"},
                "workspace": "some-workspace",
            },
            "result": True,
        },
    }

    save_results = evaluate(pg, env=dry_run_env)
    assert len(save_results) == 2

    # collect evaluates end node "exportworkspace1" before "exportworkspace2"
    assert save_results[0].is_format("netCDF")
    assert list(save_results[0].workspace_exports) == [
        SaveResult.WorkspaceExport(workspace_id="some-workspace", merge="some/path"),
    ]

    assert save_results[1].is_format("GTiff")
    assert list(save_results[1].workspace_exports) == [
        SaveResult.WorkspaceExport(workspace_id="some-workspace", merge=None),
    ]

    for save_result in save_results:
        save_result.export_workspace(
            mock_workspace_repository,
            hrefs=[f"file:out.{save_result.format}"],
            default_merge="/some/unique/path",
            remove_original=remove_original,
        )

    mock_workspace.import_file.assert_has_calls(
        [
            mock.call(Path("out.netCDF"), "some/path", remove_original),
            mock.call(Path("out.GTiff"), "/some/unique/path", remove_original),
        ]
    )


def test_vector_to_raster(dry_run_env):
    geometry_path = str(get_path("geojson/GeometryCollection01.json"))
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "vector": {"process_id": "read_vector", "arguments": {"filename": geometry_path}},
        "agg": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "lc"},
                "geometries": {"from_node": "vector"},
                "reducer": {
                    "process_graph": {
                        "mean": {
                            "process_id": "mean",
                            "arguments": {"data": {"from_parameter": "data"}},
                            "result": True,
                        }
                    }
                },
            },
        },
        "raster": {
            "process_id": "vector_to_raster",
            "arguments": {"data": {"from_node": "agg"}, "target_data_cube": {"from_node": "lc"}},
        },
        "rename": {
            "process_id": "rename_labels",
            "arguments": {"data": {"from_node": "raster"}, "dimension": "bands", "target": ["score"]},
            "result": True,
        },
    }
    save_result = evaluate(pg, env=dry_run_env)
    assert isinstance(save_result, DryRunDataCube)


def test_ndvi_reduce(dry_run_env):
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "ndvi": {
            "process_id": "ndvi",
            "arguments": {"data": {"from_node": "lc"}},
            "result": True,
        },
    }
    save_result = evaluate(pg, env=dry_run_env)
    assert isinstance(save_result, DryRunDataCube)

    assert not save_result.metadata.has_band_dimension()


def test_complex_diamond_and_buffering(dry_run_env, dry_run_tracer):
    with ephemeral_fileserver(path=get_path("parquet")) as fileserver_root:
        url = f"{fileserver_root}/mol.pq"
        pg = load_json(
            "pg/1.0/sample_extract_diamond_buffering.json",
            preprocess=lambda s: s.replace("PLACEHOLDER_LOAD_URL", url),
        )
        save_result = evaluate(pg, env=dry_run_env)
    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    print(source_constraints)

    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    source_id_bands = (
        "load_collection",
        (
            "S2_FOOBAR",
            (("eo:cloud_cover", (("lte", 95),)), ("tileId", (("eq", "31UFP"),))),
            ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"),
        ),
    )
    source_id_scl = (
        "load_collection",
        ("S2_FOOBAR", (("eo:cloud_cover", (("lte", 95),)), ("tileId", (("eq", "31UFP"),))), ("SCL",)),
    )
    loadparams = _extract_load_parameters(dry_run_env, source_id_bands)

    print(loadparams)
    expected_extent = {"crs": "EPSG:32631", "east": 648990, "south": 5671740, "west": 644750, "north": 5676610}
    assert loadparams.global_extent == expected_extent
    assert loadparams.bands == [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "SCL",
    ]
    assert loadparams.pixel_buffer == None

    # extract next set of params
    loadparams = _extract_load_parameters(dry_run_env, source_id_scl)
    assert loadparams.bands == ["SCL"]
    assert loadparams.global_extent == expected_extent
    assert loadparams.pixel_buffer == [38.5, 38.5]

    loadparams = _extract_load_parameters(dry_run_env, source_id_scl)
    assert loadparams.global_extent == expected_extent
    assert loadparams.pixel_buffer == [38.5, 38.5]


def test_resampling_masking(dry_run_env, dry_run_tracer):
    pg = load_json("pg/1.0/resample_mask_merge.json")
    save_result = evaluate(pg, env=dry_run_env)
    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    print(source_constraints)

    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    source_id_bands = (
        "load_collection",
        (
            "S2_FOOBAR",
            (("eo:cloud_cover", (("lte", 95),)),),
            ( "B02", "B03", "B04"),
        ),
    )
    source_id_scl = (
        "load_collection",
        ("S2_FOOBAR", (("eo:cloud_cover", (("lte", 95),)),), ("SCL",)),
    )
    source_id_s1 = ('load_collection', ('SENTINEL1_GRD', (('sat:orbit_state', (('eq', 'DESCENDING'),)),), ('VH', 'VV')))
    loadparams = _extract_load_parameters(dry_run_env, source_id_bands)

    expected_extent = {'crs': 'EPSG:3035',
         'east': 3860390,
         'north': 2280390,
         'south': 2259610,
         'west': 3839610}
    assert loadparams.global_extent == expected_extent
    assert loadparams.bands == [
        "B02",
        "B03",
        "B04"
    ]
    assert loadparams.pixel_buffer == None
    assert loadparams.target_crs == 3035
    assert loadparams.target_resolution == (10,10)
    assert loadparams.spatial_extent == {
          "west": 3840000,
          "east": 3860000,
          "north": 2280000,
          "south": 2260000,
          "crs": "EPSG:3035"
        }

    # extract next set of params
    loadparams = _extract_load_parameters(dry_run_env, source_id_scl)
    assert loadparams.bands == ["SCL"]
    assert loadparams.global_extent == expected_extent
    assert loadparams.pixel_buffer == [38.5, 38.5]
    assert loadparams.target_crs == 3035
    assert loadparams.target_resolution == (10,10)
    assert loadparams.spatial_extent == {
        "west": 3840000,
        "east": 3860000,
        "north": 2280000,
        "south": 2260000,
        "crs": "EPSG:3035"
    }

    loadparams = _extract_load_parameters(dry_run_env, source_id_s1)
    assert loadparams.global_extent == expected_extent
    assert loadparams.pixel_buffer == None
    assert loadparams.target_crs == 3035
    assert loadparams.target_resolution == (10,10)
    assert loadparams.spatial_extent == {
        "west": 3840000,
        "east": 3860000,
        "north": 2280000,
        "south": 2260000,
        "crs": "EPSG:3035"
    }



def test_complex_extract_load_stac(dry_run_env, dry_run_tracer):
    pg = load_json("pg/1.0/complex_load_stac.json")
    save_result = evaluate(pg, env=dry_run_env)
    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    print(source_constraints)

    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    source_id_bands = (
        "load_collection",
        (
            "S2_FOOBAR",
            (("eo:cloud_cover", (("lte", 95),)),),
            ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"),
        ),
    )
    source_id_scl = ("load_collection", ("S2_FOOBAR", (("eo:cloud_cover", (("lte", 95),)),), ("SCL",)))
    source_id_stac = ("load_stac", ("https://stac.openeo.vito.be/collections/wenr_features", (), ()))
    loadparams = _extract_load_parameters(dry_run_env, source_id_bands)

    print(loadparams)
    expected_extent = {'west': 379840, 'south': 4679580, 'east': 400460, 'north': 4700510, 'crs': 'EPSG:32633'}
    assert loadparams.global_extent == expected_extent
    assert loadparams.bands == ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    assert loadparams.pixel_buffer == None

    # extract next set of params
    loadparams = _extract_load_parameters(dry_run_env, source_id_scl)
    assert loadparams.bands == ["SCL"]
    assert loadparams.global_extent == expected_extent
    assert loadparams.pixel_buffer == [38.5, 38.5]
    assert loadparams.target_resolution is None

    loadparams = _extract_load_parameters(dry_run_env, source_id_stac)
    assert loadparams.global_extent == expected_extent
    assert loadparams.target_resolution == (10, 10)


def test_normalize_geometries(dry_run_env, dry_run_tracer):
    geometry_path = str(get_path("geojson/points.geojson"))
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "vector": {"process_id": "read_vector", "arguments": {"filename": geometry_path}},
        "agg": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "lc"},
                "geometries": {"from_node": "vector"},
                "reducer": {
                    "process_graph": {
                        "mean": {
                            "process_id": "mean",
                            "arguments": {"data": {"from_parameter": "data"}},
                            "result": True,
                        }
                    }
                },
            },
        },
        "raster": {
            "process_id": "vector_to_raster",
            "arguments": {"data": {"from_node": "agg"}, "target_data_cube": {"from_node": "lc"}},
        },
        "rename": {
            "process_id": "rename_labels",
            "arguments": {"data": {"from_node": "raster"}, "dimension": "bands", "target": ["score"]},
            "result": True,
        },
    }
    save_result = evaluate(pg, env=dry_run_env)
    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    print(source_constraints)

    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    params = [_extract_load_parameters(dry_run_env, source_id) for source_id, _ in source_constraints]
    # extents = [BoundingBox.from_dict(param.spatial_extent).as_polygon() for param in params]

    # from shapely.ops import transform
    # import json
    # import pyproj
    # t = pyproj.Transformer.from_crs(pyproj.CRS.from_user_input(32632), pyproj.CRS.from_user_input(4326), always_xy=True)
    # extents += [transform(t.transform,BoundingBox.from_dict(param.global_extent).as_polygon().segmentize(1000)) for param in params]

    # print(json.dumps(dict(type="FeatureCollection",features=[dict(type="Feature",geometry=mapping(e),properties={}) for e in extents])))

    assert params[0].global_extent == {'crs': 'EPSG:32632',
     'east': 638890,
     'north': 5736520,
     'south': 5642750,
     'west': 84850
    }


def test_resample_cube_spatial_from_resampled_target(dry_run_env, dry_run_tracer):
    """
    Use `resample_cube_spatial` with a target that is result from a separate `resample_spatial`
    """
    target = DataCube.load_collection("SENTINEL1_GRD", connection=None)
    target = target.resample_spatial(resolution=(3, 5), projection=32631)
    assert target.metadata.spatial_dimensions == [
        SpatialDimension(name="x", extent=[None, None], crs=32631, step=3),
        SpatialDimension(name="y", extent=[None, None], crs=32631, step=5),
    ]

    cube = DataCube.load_collection("S2_FOOBAR", connection=None)
    assert cube.metadata is None
    cube = cube.resample_cube_spatial(target=target)
    assert cube.metadata.spatial_dimensions == [
        SpatialDimension(name="x", extent=[None, None], crs=32631, step=3),
        SpatialDimension(name="y", extent=[None, None], crs=32631, step=5),
    ]

    pg = cube.flat_graph()
    _ = evaluate(pg, env=dry_run_env, do_dry_run=False)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert source_constraints == [
        (
            ("load_collection", ("S2_FOOBAR", ())),
            {
                "process_type": [ProcessType.FOCAL_SPACE],
                "resample": {"method": "near", "resolution": (3, 5), "target_crs": 32631},
            },
        ),
        (
            ("load_collection", ("SENTINEL1_GRD", ())),
            {"resample": {"method": "near", "resolution": (3, 5), "target_crs": 32631}},
        ),
    ]


@pytest.mark.parametrize(
    ["s2_cube_dimensions", "expected_agera5_constraints", "expected_crs", "expected_resolution", "expected_logs"],
    [
        (
            # No datacube/dimension metadata
            None,
            {"process_type": [ProcessType.FOCAL_SPACE]},
            None,
            None,
            [
                (
                    # TODO #370/#396 get rid of this warning
                    "openeo_driver.stac.datacube",
                    logging.WARN,
                    "Forcing pystac datacube extension on possibly unsupported metadata",
                ),
                (
                    "openeo_driver.dry_run",
                    logging.WARNING,
                    dirty_equals.IsStr(
                        regex="Dry-run load_stac: failed to parse cube metadata from.*No datacube extension found in STAC object.*Falling back on generic metadata.*",
                    ),
                ),
            ],
        ),
        (
            # Invalid datacube/dimension metadata
            {"x": {"type-typo!?!": "spatial"}},
            {"process_type": [ProcessType.FOCAL_SPACE]},
            None,
            None,
            [
                (
                    # TODO #370/#396 get rid of this warning
                    "openeo_driver.stac.datacube",
                    logging.WARN,
                    "Forcing pystac datacube extension on possibly unsupported metadata",
                ),
                (
                    # TODO #370/#396 get rid of this warning
                    "openeo_driver.stac.datacube",
                    logging.WARN,
                    "Forcing pystac datacube extension on possibly unsupported metadata",
                ),
                (
                    "openeo_driver.dry_run",
                    logging.WARNING,
                    dirty_equals.IsStr(
                        regex="Dry-run load_stac: failed to parse cube metadata from.*RequiredPropertyMissing.*does not have required property type.*Falling back on generic metadata.*",
                    ),
                ),
            ],
        ),
        (
            # Simple spatial dimension metadata
            {
                "x": {"type": "spatial", "axis": "x", "extent": [1, 5]},
                "y": {"type": "spatial", "axis": "y", "extent": [40, 50]},
            },
            {"process_type": [ProcessType.FOCAL_SPACE]},
            None,
            None,
            [
                (
                    # TODO #370/#396 get rid of this warning
                    "openeo_driver.stac.datacube",
                    logging.WARN,
                    "Forcing pystac datacube extension on possibly unsupported metadata",
                ),
                (
                    # TODO #370/#396 get rid of this warning
                    "openeo_driver.stac.datacube",
                    logging.WARN,
                    "Forcing pystac datacube extension on possibly unsupported metadata",
                ),
            ],
        ),
        (
            # Spatial metadata includes CRS and resolution
            {
                "x": {"type": "spatial", "axis": "x", "extent": [1, 5], "reference_system": 32631, "step": 11},
                "y": {"type": "spatial", "axis": "y", "extent": [40, 50], "reference_system": 32631, "step": 22},
            },
            {
                "process_type": [ProcessType.FOCAL_SPACE],
                "resample": {"method": "near", "resolution": (11, 22), "target_crs": 32631},
            },
            32631,
            (11, 22),
            [
                (
                    # TODO #370/#396 get rid of this warning
                    "openeo_driver.stac.datacube",
                    logging.WARN,
                    "Forcing pystac datacube extension on possibly unsupported metadata",
                ),
                (
                    # TODO #370/#396 get rid of this warning
                    "openeo_driver.stac.datacube",
                    logging.WARN,
                    "Forcing pystac datacube extension on possibly unsupported metadata",
                ),
            ],
        ),
    ],
)
def test_load_stac_resample_cube_spatial(
    dry_run_env,
    dry_run_tracer,
    tmp_path,
    s2_cube_dimensions,
    expected_agera5_constraints,
    expected_crs,
    expected_resolution,
    expected_logs,
    caplog,
):
    """
    https://github.com/Open-EO/openeo-geopyspark-driver/issues/1114:

    use case:

        load_stac            load_stac
        s2_extractions       agera5
                \             /
               resample_cube_spatial
               resample agera5 to s2_extractions

    Resolution info from s2_extractions (if available)
    should be pushed as load parameters to load_stac of agera5
    """
    caplog.set_level(logging.WARN)

    s2_extractions_path = tmp_path / "s2_extractions.json"
    s2_extractions_path.write_text(
        json.dumps(
            StacDummyBuilder.collection(
                id="s2_extractions",
                cube_dimensions=s2_cube_dimensions,
            )
        )
    )
    agera5_path = tmp_path / "agera5.json"
    agera5_path.write_text(
        json.dumps(
            StacDummyBuilder.collection(
                id="agera5",
                cube_dimensions={
                    "x": {"type": "spatial", "axis": "x", "extent": [1, 5], "reference_system": 4326, "step": 1},
                    "y": {"type": "spatial", "axis": "y", "extent": [40, 50], "reference_system": 4326, "step": 1},
                },
            )
        )
    )

    pg = {
        "loadstac1": {"process_id": "load_stac", "arguments": {"url": str(s2_extractions_path)}},
        "loadstac2": {"process_id": "load_stac", "arguments": {"url": str(agera5_path)}},
        "resamplecubespatial1": {
            "process_id": "resample_cube_spatial",
            "arguments": {"data": {"from_node": "loadstac2"}, "target": {"from_node": "loadstac1"}},
        },
        "saveresult1": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "resamplecubespatial1"}, "format": "GTiff", "options": {}},
            "result": True,
        },
    }

    _ = evaluate(pg, env=dry_run_env, do_dry_run=False)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert source_constraints == [
        (("load_stac", (str(agera5_path), (), ())), expected_agera5_constraints),
        (("load_stac", (str(s2_extractions_path), (), ())), {}),
    ]

    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    load_params = _extract_load_parameters(dry_run_env, source_id=("load_stac", (str(agera5_path), (), ())))
    assert (load_params.target_crs, load_params.target_resolution) == (expected_crs, expected_resolution)

    assert caplog.record_tuples == expected_logs


@pytest.mark.parametrize(
    ["dimension", "expected"],
    [
        ("bands", ["t", "x", "y"]),
        ("t", ["bands", "x", "y"]),
    ],
)
def test_resample_cube_spatial_preserve_non_spatial(dry_run_env, dimension, expected):
    """
    https://github.com/Open-EO/openeo-python-driver/issues/397
    """
    pg = {
        "4d": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR"},
        },
        "3d": {
            "process_id": "reduce_dimension",
            "arguments": {
                "data": {"from_node": "4d"},
                "dimension": dimension,
                "reducer": {
                    "process_graph": {
                        "mean1": {"process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}}
                    }
                },
            },
        },
        "resample_cube_spatial": {
            "process_id": "resample_cube_spatial",
            "arguments": {"data": {"from_node": "3d"}, "target": {"from_node": "4d"}},
            "result": True,
        },
    }
    result = evaluate(pg, env=dry_run_env)
    assert isinstance(result, DryRunDataCube)
    assert result.metadata.dimension_names() == expected


@pytest.mark.skip("test taking too long (27 minutes)")
def test_very_large_graph(dry_run_env, dry_run_tracer):
    pg = load_json("pg/1.0/large_eugw_graph.json")["process_graph"]
    save_result = evaluate(pg, env=dry_run_env)
    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    print(source_constraints)
