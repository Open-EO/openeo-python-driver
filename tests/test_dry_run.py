import pytest
import shapely.geometry

from openeo.internal.graph_building import PGNode
from openeo.rest.datacube import DataCube

from openeo_driver.errors import OpenEOApiException
from openeo_driver.ProcessGraphDeserializer import evaluate, ENV_DRY_RUN_TRACER, _extract_load_parameters, \
    ENV_SOURCE_CONSTRAINTS, custom_process_from_process_graph, process_registry_100, ENV_SAVE_RESULT
from openeo_driver.datacube import DriverVectorCube
from openeo_driver.datastructs import SarBackscatterArgs
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.dry_run import DryRunDataTracer, DataSource, DataTrace, ProcessType
from openeo_driver.testing import DictSubSet, approxify
from openeo_driver.util.geometry import as_geojson_feature_collection
from openeo_driver.utils import EvalEnv
from tests.data import get_path, load_json


@pytest.fixture
def dry_run_tracer() -> DryRunDataTracer:
    return DryRunDataTracer()


@pytest.fixture
def dry_run_env(dry_run_tracer, backend_implementation) -> EvalEnv:
    return EvalEnv({
        ENV_DRY_RUN_TRACER: dry_run_tracer,
        "backend_implementation": backend_implementation,
        "version": "1.0.0"
    })


def test_source_load_collection():
    s1 = DataSource.load_collection(collection_id="FOOBAR")
    s2 = DataSource.load_collection(collection_id="FOOBAR")
    s3 = DataSource.load_collection(collection_id="FOOBARV2")
    assert s1.get_source_id() == s2.get_source_id()
    assert s1.get_source_id() != s3.get_source_id()


def test_source_load_disk_data():
    s1 = DataSource.load_disk_data(glob_pattern="foo*.tiff", format="GTiff", options={})
    s2 = DataSource.load_disk_data(glob_pattern="foo*.tiff", format="GTiff", options={})
    s3 = DataSource.load_disk_data(glob_pattern="foo*.tiff", format="GTiff", options={"meh": "xev"})
    assert s1.get_source_id() == s2.get_source_id()
    assert s1.get_source_id() != s3.get_source_id()


def test_source_load_result():
    s1 = DataSource.load_result(job_id='a1a6dabc-20ae-43b8-b7c1-f9c9bd3f6dfd')
    s2 = DataSource.load_result(job_id='a1a6dabc-20ae-43b8-b7c1-f9c9bd3f6dfd')
    s3 = DataSource.load_result(job_id='a16b4310-45de-49ae-824b-f501038057c6')
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
    assert [t.describe() for t in traces] == [
        "load_collection<-temporal_extent<-spatial_extent<-bands"
    ]


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
    assert source_constraints == [
        (("load_collection", ("S2_FOOBAR", ())), {})
    ]


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
        }
    }
    cube = evaluate(pg, env=dry_run_env)

    """
    source_constraints = dry_run_tracer.get_source_constraints(merge=False)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert sorted(constraints, key=str) == [
    """

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 2
    assert source_constraints == [(
        ("load_collection", ("S2_FOOBAR", ())),
        {
            "bands": ["grass"],
            "spatial_extent": {"west": 1, "east": 2, "south": 51, "north": 52, "crs": "EPSG:4326"}
        }), (
        ("load_collection", ("S2_FOOBAR", ())),
        {
            "bands": ["red"],
            "spatial_extent": {"west": 1, "east": 2, "south": 51, "north": 52, "crs": "EPSG:4326"}
        }),
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
                "bands": ["red", "green", "blue"]
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
                "extent": {"west": 1, "south": 51, "east": 3, "north": 53}
            },
        },
        "filter_bands": {
            "process_id": "filter_bands",
            "arguments": {"data": {"from_node": "filter_bbox"}, "bands": ["red"]},
            "result": True,
        }
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
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "temporal_extent": ("2020-01-01", "2020-10-10"),
        "spatial_extent": {"west": 0, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
        "bands": ["red", "green", "blue"],
    }


def test_inspect(dry_run_env, dry_run_tracer):
    """temporal/bbox/band extents in load_collection *and* filter_ processes"""
    pg = {
        "load": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FOOBAR",
                "spatial_extent": {"west": 0, "south": 50, "east": 5, "north": 55},
                "temporal_extent": ["2020-01-01", "2020-10-10"],
                "bands": ["red", "green", "blue"]
            },
        },
        "inspect": {
            "process_id": "inspect",
            "arguments": {"data": {"from_node": "load"}, "level": "error", "message":"logging a message"},
            "result":True
        },
    }
    cube = evaluate(pg, env=dry_run_env)



def test_evaluate_merge_collections(dry_run_env, dry_run_tracer):
    pg = {
        "load": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FOOBAR",
                "spatial_extent": {"west": 0, "south": 50, "east": 5, "north": 55},
                "temporal_extent": ["2020-01-01", "2020-10-10"],
                "bands": ["red", "green", "blue"]
            },
        },
        "load_s1": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FAPAR_CLOUDCOVER",
                "spatial_extent": {"west": -1, "south": 50, "east": 5, "north": 55},
                "temporal_extent": ["2020-01-01", "2020-10-10"],
                "bands": ["VV"]
            },
        },
        "merge": {
            "process_id": "merge_cubes",
            "arguments": {
                "cube1": {"from_node": "load"},
                "cube2": {"from_node": "load_s1"}
            },
            "result": True

        }
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 2

    source, constraints = source_constraints[0]

    assert source == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "temporal_extent": ("2020-01-01", "2020-10-10"),
        "spatial_extent": {"west": 0, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
        "bands": ["red", "green", "blue"],
    }

    source, constraints = source_constraints[1]

    assert source == ("load_collection", ("S2_FAPAR_CLOUDCOVER", ()))
    assert constraints == {
        "temporal_extent": ("2020-01-01", "2020-10-10"),
        "spatial_extent": {"west": -1, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
        "bands": ["VV"],
    }

    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    loadparams = _extract_load_parameters(dry_run_env, ("load_collection", ("S2_FOOBAR", ())))
    assert {"west": -1, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"} == loadparams.global_extent


def test_evaluate_load_collection_and_filter_extents_dynamic(dry_run_env, dry_run_tracer):
    """"Dynamic temporal/bbox/band extents in load_collection *and* filter_ processes"""
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
                "extent": {"west": {"from_node": "west2"}, "south": 51, "east": 3, "north": 53}
            },
        },
        "filter_bands": {
            "process_id": "filter_bands",
            "arguments": {"data": {"from_node": "filter_bbox"}, "bands": [{"from_node": "bandblue"}]},
            "result": True,
        }
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
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "temporal_extent": ("2020-01-01", "2020-10-10"),
        "spatial_extent": {"west": 1, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
        "bands": ["blue", "green", "red"],
    }


@pytest.mark.parametrize(["inside", "replacement", "expect_spatial_extent"], [
    (None, None, True),
    (False, None, True),
    (True, None, False),
    (None, 123, False),
])
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
            "spatial_extent": {"west": 0.0, "south": 0.0, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"}
        }
    else:
        expected = {}
    assert constraints == expected


def test_mask_polygon_and_load_collection_spatial_extent(dry_run_env, dry_run_tracer):
    polygon = {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]}
    cube = DataCube(PGNode(
        "load_collection", id="S2_FOOBAR",
        spatial_extent={"west": -1, "south": -1, "east": 10, "north": 10}
    ), connection=None)
    cube = cube.mask_polygon(mask=polygon)
    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "spatial_extent": {"west": -1, "south": -1, "east": 10, "north": 10, "crs": "EPSG:4326"}
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
    res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "spatial_extent": {"west": -1, "south": -1, "east": 9, "north": 9, "crs": "EPSG:4326"}
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
                            "process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}, "result": True
                        }
                    }
                }
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
        "spatial_extent": {"west": 0.0, "south": 0.0, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"},
        "aggregate_spatial": {"geometries": DriverVectorCube.from_geojson(polygon)},
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
                {
                    "west": 1.99991,
                    "south": 2.99991,
                    "east": 2.0000897,
                    "north": 3.0000897,
                    "crs": "EPSG:4326",
                }
            ),
        ),
        (
            as_geojson_feature_collection(
                shapely.geometry.Point(2, 3),
                shapely.geometry.Point(4, 5),
            ),
            approxify(
                {
                    "west": 1.99991,
                    "south": 2.99991,
                    "east": 4.0000897,
                    "north": 5.0000897,
                    "crs": "EPSG:4326",
                }
            ),
        ),
        (
            {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]},
            {"west": 0.0, "south": 0.0, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"},
        ),
        (
            as_geojson_feature_collection(
                shapely.geometry.Polygon.from_bounds(2, 3, 5, 8),
                shapely.geometry.Polygon.from_bounds(3, 5, 8, 13),
            ),
            {"west": 2.0, "south": 3.0, "east": 8.0, "north": 13.0, "crs": "EPSG:4326"},
        ),
        (
            as_geojson_feature_collection(
                shapely.geometry.Point(2, 3),
                shapely.geometry.Polygon.from_bounds(3, 5, 8, 13),
            ),
            approxify(
                {
                    "west": 1.99991,
                    "south": 2.99991,
                    "east": 8,
                    "north": 13,
                    "crs": "EPSG:4326",
                }
            ),
        ),
    ],
)
def test_aggregate_spatial_extent_handling_and_buffering(
    dry_run_env, dry_run_tracer, geometries, expected
):
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

    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "spatial_extent": {"west": 0.0, "south": 0.0, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"},
        "process_type": [ProcessType.GLOBAL_TIME],
        "bands": ["B04", "B08", "B11", "SCL"],
        "custom_cloud_mask": {"method": "mask_scl_dilation", 'scl_band_name': 'SCL'},
        "aggregate_spatial": {"geometries": DriverVectorCube.from_geojson(polygon)},
        "temporal_extent": ("2018-11-01", "2020-02-01")
    }
    geometries, = dry_run_tracer.get_geometries()
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
    res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "spatial_extent": bbox,
        "aggregate_spatial": {"geometries": DriverVectorCube.from_geojson(polygon)},
    }
    geometries, = dry_run_tracer.get_geometries()
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
    cube = cube.filter_spatial(geometries = polygon1)
    cube = cube.resample_spatial(projection=4326, resolution=0.25)
    cube = cube.filter_spatial(geometries=polygon2)

    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    geometries = dry_run_tracer.get_last_geometry(operation="filter_spatial")
    assert constraints == {
        "spatial_extent": {'crs': 'EPSG:4326','east': 8.0,'north': 5.0,'south': 0.0,'west': 0.0},
        "filter_spatial": {"geometries": shapely.geometry.shape(polygon1)},
        "resample": {'method': 'near', 'resolution': [0.25, 0.25], 'target_crs': 4326},
    }

    assert geometries == shapely.geometry.shape(polygon2)

def test_resample_filter_spatial(dry_run_env, dry_run_tracer):
    polygon = {"type": "Polygon", "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]}
    cube = DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None)
    cube = cube.filter_spatial(geometries = polygon)
    cube = cube.resample_spatial(projection=4326, resolution=0.25)

    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    geometries, = dry_run_tracer.get_geometries(operation="filter_spatial")
    assert constraints == {
        "spatial_extent": {'crs': 'EPSG:4326','east': 8.0,'north': 5.0,'south': 0.0,'west': 0.0},
        "filter_spatial": {"geometries": shapely.geometry.shape(polygon)},
        "resample": {'method': 'near', 'resolution': [0.25, 0.25], 'target_crs': 4326},
    }
    assert isinstance(geometries, shapely.geometry.Polygon)
    assert shapely.geometry.mapping(geometries) == {
        "type": "Polygon",
        "coordinates": (((0.0, 0.0), (3.0, 5.0), (8.0, 2.0), (0.0, 0.0)),)
    }


def test_auto_align(dry_run_env, dry_run_tracer):
    polygon = {"type": "Polygon", "coordinates": [[(0.1, 0.1), (3, 5), (8, 2), (0.1, 0.1)]]}
    cube = DataCube(PGNode("load_collection", id="ESA_WORLDCOVER_10M_2020_V1"), connection=None)
    cube = cube.filter_spatial(geometries = polygon)

    pg = cube.flat_graph()
    res = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("ESA_WORLDCOVER_10M_2020_V1", ()))
    geometries, = dry_run_tracer.get_geometries(operation="filter_spatial")
    assert constraints == {
        "spatial_extent": {'crs': 'EPSG:4326','east': 8.0,'north': 5.0,'south': 0.1,'west': 0.1},
        "filter_spatial": {"geometries": shapely.geometry.shape(polygon)},
    }
    assert isinstance(geometries, shapely.geometry.Polygon)
    assert shapely.geometry.mapping(geometries) == {
        "type": "Polygon",
        "coordinates": (((0.1, 0.1), (3.0, 5.0), (8.0, 2.0), (0.1, 0.1)),)
    }
    #source_constraints = dry_run_tracer.get_source_constraints()
    dry_run_env = dry_run_env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
    load_params = _extract_load_parameters(dry_run_env, source_id= ("load_collection", ("ESA_WORLDCOVER_10M_2020_V1", ())))
    assert {'west': 0.09999999927961767, 'east': 8.00008333258134, 'south': 0.09999999971959994, 'north': 5.000083333033345, 'crs': 'EPSG:4326'} == load_params.global_extent



def test_aggregate_spatial_read_vector(dry_run_env, dry_run_tracer):
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
                            "process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}, "result": True
                        }
                    }
                }
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
        "spatial_extent": {"west": 5.05, "south": 51.21, "east": 5.15, "north": 51.3, "crs": "EPSG:4326"},
        "aggregate_spatial": {"geometries": DelayedVector(geometry_path)},
    }
    geometries, = dry_run_tracer.get_geometries()
    assert isinstance(geometries, DelayedVector)


def test_aggregate_spatial_get_geometries_feature_collection(
    dry_run_env, dry_run_tracer
):
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "vector": {
            "process_id": "get_geometries",
            "arguments": {
                "feature_collection": {
                    "type": "FeatureCollection",
                    "name": "fields",
                    "crs": {
                        "type": "name",
                        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
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
                            "process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}, "result": True
                        }
                    }
                }
            },
            "result": True,
        },
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))

    expected_geometry_collection = DriverVectorCube.from_geojson(
        pg["vector"]["arguments"]["feature_collection"]
    )
    assert constraints == {
        "spatial_extent": {
            "west": 0.0,
            "south": 0.0,
            "east": 8.0,
            "north": 5.0,
            "crs": "EPSG:4326",
        },
        "aggregate_spatial": {"geometries": expected_geometry_collection},
    }
    (geometries,) = dry_run_tracer.get_geometries()
    assert isinstance(geometries, DriverVectorCube)


@pytest.mark.parametrize(
    ["arguments", "expected"],
    [
        (
            {},
            SarBackscatterArgs(coefficient="gamma0-terrain", elevation_model=None, mask=False, contributing_area=False,
                               local_incidence_angle=False, ellipsoid_incidence_angle=False, noise_removal=True,
                               options={})
    ),
    (
            {
                "coefficient": "gamma0-ellipsoid", "elevation_model": "SRTMGL1", "mask": True,
                "contributing_area": True, "local_incidence_angle": True, "ellipsoid_incidence_angle": True,
                "noise_removal": False, "options": {"tile_size": 1024}
            },
            SarBackscatterArgs(
                coefficient="gamma0-ellipsoid", elevation_model="SRTMGL1", mask=True, contributing_area=True,
                local_incidence_angle=True, ellipsoid_incidence_angle=True, noise_removal=False,
                options={"tile_size": 1024}
            )
    )
])
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
                        "arguments": {
                            "x": {"from_parameter": "value"},
                            "y": direction
                        },
                        "result": True
                    }
                }
            }
        }

    properties = get_props()
    asc_props = get_props("ASCENDING")
    pg = {
        "lc": {
            "process_id": "load_collection", "arguments": {"id": "S2_FOOBAR", "properties": properties},
        },
        "lc2": {
            "process_id": "load_collection", "arguments": {"id": "S2_FOOBAR", "properties": asc_props},
        },
        "merge": {
            "process_id": "merge_cubes",
            "arguments": {"cube1": {"from_node": "lc"}, "cube2": {"from_node": "lc2"}},
            "result": True
        }
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)

    assert source_constraints == [
        (("load_collection", ("S2_FOOBAR", (("orbitDirection", (("eq", "DESCENDING"),),),))), {"properties": properties}),
        (("load_collection", ("S2_FOOBAR", (("orbitDirection", (("eq", "ASCENDING"),)),),)), {"properties": asc_props})
    ]


@pytest.mark.parametrize(["arguments", "expected"], [
    (
            {},
            [
                {"rel": "atmospheric-scattering", "href": "https://remotesensing.vito.be/case/icor"},
                {
                    'href': 'https://atmosphere.copernicus.eu/catalogue#/product/urn:x-wmo:md:int.ecmwf::copernicus:cams:prod:fc:total-aod:pid094',
                    'rel': 'related'},
                {'href': 'https://doi.org/10.7289/V52R3PMS', 'rel': 'elevation-model'},
                {'href': 'https://doi.org/10.1109/LGRS.2016.2635942', 'rel': 'water-vapor'}
            ]
    ),
    (
            {"method": "SMAC"},
            [
                {"rel": "atmospheric-scattering", "href": "https://doi.org/10.1080/01431169408954055"},
                {
                    'href': 'https://atmosphere.copernicus.eu/catalogue#/product/urn:x-wmo:md:int.ecmwf::copernicus:cams:prod:fc:total-aod:pid094',
                    'rel': 'related'},
                {'href': 'https://doi.org/10.7289/V52R3PMS', 'rel': 'elevation-model'},
                {'href': 'https://doi.org/10.1109/LGRS.2016.2635942', 'rel': 'water-vapor'}
            ]
    ),
])
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

    env = EvalEnv(dict(backend_implementation=backend_implementation))
    evaluate(pg, do_dry_run=True, env=env)


def test_sources_are_subject_to_correct_constraints(dry_run_env, dry_run_tracer):
    pg = {
        'loadcollection1': {'process_id': 'load_collection',
                            'arguments': {'bands': ['VV', 'VH'], 'id': 'S2_FOOBAR',
                                          'spatial_extent': {'west': 11.465226, 'east': 11.465435, 'south': 46.343118,
                                                             'north': 46.343281, 'crs': 'EPSG:4326'},
                                          'temporal_extent': ['2018-01-01', '2018-01-01']}},
        'sarbackscatter1': {'process_id': 'sar_backscatter',
                            'arguments': {'coefficient': 'sigma0-ellipsoid', 'contributing_area': False,
                                          'data': {'from_node': 'loadcollection1'}, 'elevation_model': None,
                                          'ellipsoid_incidence_angle': False, 'local_incidence_angle': False,
                                          'mask': False, 'noise_removal': True}},
        'renamelabels1': {'process_id': 'rename_labels',
                          'arguments': {'data': {'from_node': 'sarbackscatter1'}, 'dimension': 'bands',
                                        'source': ['VV', 'VH'], 'target': ['VV_sigma0', 'VH_sigma0']}},
        'sarbackscatter2': {'process_id': 'sar_backscatter',
                            'arguments': {'coefficient': 'gamma0-terrain', 'contributing_area': False,
                                          'data': {'from_node': 'loadcollection1'}, 'elevation_model': None,
                                          'ellipsoid_incidence_angle': False, 'local_incidence_angle': False,
                                          'mask': False, 'noise_removal': True}},
        'renamelabels2': {'process_id': 'rename_labels',
                          'arguments': {'data': {'from_node': 'sarbackscatter2'}, 'dimension': 'bands',
                                        'source': ['VV', 'VH'], 'target': ['VV_gamma0', 'VH_gamma0']}},
        'mergecubes1': {'process_id': 'merge_cubes', 'arguments': {'cube1': {'from_node': 'renamelabels1'},
                                                                   'cube2': {'from_node': 'renamelabels2'}},
                        'result': True}
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 2

    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "temporal_extent": ('2018-01-01', '2018-01-01'),
        "spatial_extent": {'west': 11.465226, 'east': 11.465435, 'south': 46.343118, 'north': 46.343281,
                           'crs': 'EPSG:4326'},
        'bands': ['VV', 'VH'],
        'sar_backscatter': SarBackscatterArgs(coefficient='sigma0-ellipsoid', elevation_model=None, mask=False,
                                              contributing_area=False, local_incidence_angle=False,
                                              ellipsoid_incidence_angle=False, noise_removal=True, options={})
    }

    src, constraints = source_constraints[1]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "temporal_extent": ('2018-01-01', '2018-01-01'),
        "spatial_extent": {'west': 11.465226, 'east': 11.465435, 'south': 46.343118, 'north': 46.343281,
                           'crs': 'EPSG:4326'},
        'bands': ['VV', 'VH'],
        'sar_backscatter': SarBackscatterArgs(coefficient='gamma0-terrain', elevation_model=None, mask=False,
                                              contributing_area=False, local_incidence_angle=False,
                                              ellipsoid_incidence_angle=False, noise_removal=True, options={})
    }


def test_pixel_buffer(dry_run_env, dry_run_tracer):
    pg = {
        'loadcollection1': {'process_id': 'load_collection',
                            'arguments': {'bands': ['VV', 'VH'], 'id': 'S2_FOOBAR',
                                          'spatial_extent': {'west': 11.465226, 'east': 11.465435, 'south': 46.343118,
                                                             'north': 46.343281, 'crs': 'EPSG:4326'},
                                          'temporal_extent': ['2018-01-01', '2018-01-01']}},
        'apply_kernel': {'process_id': 'apply_kernel',
                            'arguments': {'kernel': [[1.0,1.0],[1.0,1.0]],
                                          'data': {'from_node': 'loadcollection1'}},
                         'result':True
                         }

    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1

    src, constraints = source_constraints[0]
    assert src == ("load_collection", ("S2_FOOBAR", ()))
    assert constraints == {
        "temporal_extent": ('2018-01-01', '2018-01-01'),
        "spatial_extent": {'west': 11.465226, 'east': 11.465435, 'south': 46.343118, 'north': 46.343281,
                           'crs': 'EPSG:4326'},
        'bands': ['VV', 'VH'],
        'pixel_buffer': {'buffer_size': [1.0,1.0]},
        'process_type': [ ProcessType.FOCAL_SPACE],
    }




def test_filter_after_merge_cubes(dry_run_env, dry_run_tracer):
    """based on use case of https://jira.vito.be/browse/EP-3747"""
    pg = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR", "bands": ["B04", "B08"]}
        },
        "reducedimension1": {
            "process_id": "reduce_dimension",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "dimension": "bands",
                "reducer": {
                    "process_graph": {
                        "arrayelement1": {
                            "process_id": "array_element",
                            "arguments": {"data": {"from_parameter": "data"}, "index": 1}
                        },
                        "arrayelement2": {
                            "process_id": "array_element",
                            "arguments": {"data": {"from_parameter": "data"}, "index": 0}
                        },
                        "subtract1": {
                            "process_id": "subtract",
                            "arguments": {"x": {"from_node": "arrayelement1"}, "y": {"from_node": "arrayelement2"}}
                        },
                        "add1": {
                            "process_id": "add",
                            "arguments": {"x": {"from_node": "arrayelement1"}, "y": {"from_node": "arrayelement2"}}
                        },
                        "divide1": {
                            "process_id": "divide",
                            "arguments": {"x": {"from_node": "subtract1"}, "y": {"from_node": "add1"}},
                            "result": True
                        }
                    }
                }
            }
        },
        "adddimension1": {
            "process_id": "add_dimension",
            "arguments": {
                "data": {"from_node": "reducedimension1"}, "label": "s2_ndvi", "name": "bands",
                "type": "bands"}
        },
        "loadcollection2": {
            "process_id": "load_collection",
            "arguments": {"id": "PROBAV_L3_S10_TOC_NDVI_333M_V2", "bands": ["ndvi"], }
        },
        "resamplecubespatial1": {
            "process_id": "resample_cube_spatial",
            "arguments": {
                "data": {"from_node": "loadcollection2"},
                "method": "average",
                "target": {"from_node": "adddimension1"}
            }
        },
        "maskpolygon1": {
            "process_id": "mask_polygon",
            "arguments": {
                "data": {"from_node": "resamplecubespatial1"},
                "mask": {
                    "type": "Polygon",
                    "coordinates": [[
                        [5.03536, 51.219], [5.03586, 51.230], [5.01754, 51.231], [5.01704, 51.219], [5.03536, 51.219]
                    ]]
                }
            }
        },
        "mergecubes1": {
            "process_id": "merge_cubes",
            "arguments": {
                "cube1": {"from_node": "adddimension1"},
                "cube2": {"from_node": "maskpolygon1"}
            }
        },
        "filtertemporal1": {
            "process_id": "filter_temporal",
            "arguments": {
                "data": {"from_node": "mergecubes1"},
                "extent": ["2019-03-01", "2019-04-01"]
            }
        },
        "filterbbox1": {
            "process_id": "filter_bbox",
            "arguments": {
                "data": {"from_node": "filtertemporal1"},
                "extent": {
                    "west": 640860.0, "east": 642140.0, "north": 5677450.0, "south": 5676170.0, "crs": "EPSG:32631"
                }
            },
            "result": True
        }
    }

    cube = evaluate(pg, env=dry_run_env)
    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert source_constraints == [
        (
            ('load_collection', ('S2_FOOBAR', ())),
            {
                'bands': ['B04', 'B08'],
                'spatial_extent': {
                    'crs': 'EPSG:32631', 'east': 642140.0, 'north': 5677450.0, 'south': 5676170.0, 'west': 640860.0,
                },
                'temporal_extent': ('2019-03-01', '2019-04-01')}
        ),
        (
            ('load_collection', ('PROBAV_L3_S10_TOC_NDVI_333M_V2', ())),
            {
                'bands': ['ndvi'],
                'process_type': [ProcessType.FOCAL_SPACE],
                'resample': {'method': 'average','resolution': [10, 10], 'target_crs': 'AUTO:42001'},
                'spatial_extent': {
                    'crs': 'EPSG:32631', 'east': 642140.0, 'north': 5677450.0, 'south': 5676170.0, 'west': 640860.0,
                },
                'temporal_extent': ('2019-03-01', '2019-04-01')}
        ),
        (
            ('load_collection', ('S2_FOOBAR', ())),
            {
                'bands': ['B04', 'B08'],
                'spatial_extent': {
                    'crs': 'EPSG:32631', 'east': 642140.0, 'north': 5677450.0, 'south': 5676170.0, 'west': 640860.0,
                },
                'temporal_extent': ('2019-03-01', '2019-04-01')}
        )
    ]


def test_CropSAR_aggregate_spatial_constraint(dry_run_env, dry_run_tracer):
    cropsar_process = load_json("pg/1.0/cropsar_graph.json")
    custom_process_from_process_graph(cropsar_process, namespace="test")

    try:
        geometry_path = str(get_path("geojson/thaipolys_ad.geojson"))

        pg = {
            "CropSAR1": {
                "process_id": "CropSAR",
                "arguments": {
                    "file_polygons": geometry_path,
                    "time_range": [
                        "2019-07-01",
                        "2019-08-31"
                    ]
                },
                "namespace": "test",
                "result": True
            }
        }

        evaluate(pg, env=dry_run_env)
        source_constraints = dry_run_tracer.get_source_constraints(merge=True)

        assert len(source_constraints) > 0

        for _, constraints in source_constraints:
            assert constraints['aggregate_spatial']['geometries'].path == geometry_path
    finally:
        del process_registry_100._processes['test', 'CropSAR']


@pytest.mark.parametrize(["dimension_name", "expected"], [
    ("bands", ["x", "y", "t"]),
    ("t", ["x", "y", "bands"]),
])
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
    pg = {
        "loadresult1": {
            "process_id": "load_result",
            "arguments": {
                "id": "https://oeo.net/openeo/1.1/jobs/j-f1ce01ef51d1481abaab7ceceb19c650/results",
                "spatial_extent": {"west": 0, "south": 50, "east": 5, "north": 55},
                "temporal_extent": ("2020-01-01", "2020-02-02"),
                "bands": ["VV", "VH"]
            },
            "result": True
        }
    }

    evaluate(pg, env=dry_run_env)
    source_constraints = dry_run_tracer.get_source_constraints(merge=True)

    assert source_constraints == [
        (
            ('load_result', ('https://oeo.net/openeo/1.1/jobs/j-f1ce01ef51d1481abaab7ceceb19c650/results',)),
            {
                'bands': ['VV', 'VH'],
                'spatial_extent': {
                    'crs': 'EPSG:4326', 'east': 5, 'north': 55, 'south': 50, 'west': 0,
                },
                'temporal_extent': ('2020-01-01', '2020-02-02')
            }
        )
    ]

def test_multiple_save_result(dry_run_env):
    pg = {
      "collection1": {
        "process_id": "load_collection",
        "arguments": {
          "id": "S2_FAPAR_CLOUDCOVER"
        }
      },
      "collection2": {
        "process_id": "load_collection",
        "arguments": {
          "id": "S2_FOOBAR"
        }
      },
      "saveresult1": {
        "process_id": "save_result",
        "arguments": {
          "options": {},
          "data": {
            "from_node": "collection2"
          },
          "format": "GTiff"
        }
      },
      "mergecubes1": {
        "process_id": "merge_cubes",
        "arguments": {
          "cube1": {
            "from_node": "collection1"
          },
          "cube2": {
            "from_node": "saveresult1"
          },
          "overlap_resolver": {
            "process_graph": {
              "or1": {
                "process_id": "or",
                "arguments": {
                  "x": {
                    "from_parameter": "x"
                  },
                  "y": {
                    "from_parameter": "y"
                  }
                },
                "result": True
              }
            }
          }
        },
        "result": False
      },
      "saveresult2": {
        "result": True,
        "arguments": {
          "options": {},
          "data": {
            "from_node": "mergecubes1"
          },
          "format": "netCDF"
        },
        "process_id": "save_result"
      }
    }
    dry_run_env = dry_run_env.push({ENV_SAVE_RESULT: []})
    the_result = evaluate(pg, env=dry_run_env)
    save_result = dry_run_env.get(ENV_SAVE_RESULT)
    assert  len(save_result) == 2
    assert len(the_result) == 2


def test_invalid_latlon_in_geojson(dry_run_env):
    init_cube = DataCube(PGNode("load_collection", id="S2_FOOBAR"), connection=None)

    polygon1 = {"type": "Polygon", "coordinates": [[(-361, 0), (3, 5), (8, 2), (0, 0)]]}
    cube = init_cube.filter_spatial(geometries=polygon1)
    with pytest.raises(OpenEOApiException) as e:
        evaluate(cube.flat_graph(), env=dry_run_env)
    assert e.value.message.startswith(
        "Failed to parse Geojson. Invalid coordinate: (-361, 0)"
    )

    polygon2 = {"type": "Polygon", "coordinates": [[(1, 1), (3, 5), (8, 101), (1, 1)]]}
    cube = init_cube.filter_spatial(geometries=polygon2)
    with pytest.raises(OpenEOApiException) as e:
        evaluate(cube.flat_graph(), env=dry_run_env)
    assert e.value.message.startswith(
        "Failed to parse Geojson. Invalid coordinate: (8, 101)"
    )

    polygon3 = {
        "type": "MultiPolygon",
        "coordinates": [
            [[(-360, -100), (-360, 100), (360, 100), (360, -100), (-360, -100)]]
        ],
    }
    cube = init_cube.filter_spatial(geometries=polygon3)
    evaluate(cube.flat_graph(), env=dry_run_env)
