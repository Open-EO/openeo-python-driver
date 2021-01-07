import pytest
import shapely.geometry

from openeo_driver.ProcessGraphDeserializer import evaluate, ENV_DRY_RUN_TRACER
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.dry_run import DryRunDataTracer, DataSource, DataTrace
from openeo_driver.testing import IgnoreOrder
from openeo_driver.utils import EvalEnv
from tests.data import get_path


@pytest.fixture
def dry_run_tracer() -> DryRunDataTracer:
    return DryRunDataTracer()


@pytest.fixture
def dry_run_env(dry_run_tracer) -> EvalEnv:
    return EvalEnv({
        ENV_DRY_RUN_TRACER: dry_run_tracer,
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


def test_data_source_hashable():
    s1 = DataSource("load_foo", arguments={"a": "b", "c": "d", "e": "f", "g": "h"})
    s2 = DataSource("load_foo", arguments={"g": "h", "e": "f", "c": "d", "a": "b"})
    assert s1.get_source_id() == s2.get_source_id()


def test_data_trace():
    source = DataSource.load_collection("S2")
    trace = DataTrace(parent=source, operation="filter_bbox", arguments={"bbox": "Belgium"})
    trace = DataTrace(parent=trace, operation="filter_bbox", arguments={"bbox": "Mol"})
    trace = DataTrace(parent=trace, operation="ndvi", arguments={"red": "B04"})
    assert trace.get_source() is source
    assert trace.get_arguments_by_operation("filter_bbox") == [{"bbox": "Belgium"}, {"bbox": "Mol"}]
    assert list(trace.get_arguments_by_operation("ndvi")) == [{"red": "B04"}]


def test_dry_run_data_tracer():
    tracer = DryRunDataTracer()
    source = DataSource.load_collection("S2")
    trace = DataTrace(parent=source, operation="ndvi", arguments={})
    res = tracer.add_trace(trace)
    assert res is trace
    assert tracer.get_trace_leaves() == {trace}


def test_dry_run_data_tracer_process_traces():
    tracer = DryRunDataTracer()
    source = DataSource.load_collection("S2")
    trace1 = DataTrace(parent=source, operation="ndvi", arguments={})
    tracer.add_trace(trace1)
    trace2 = DataTrace(parent=source, operation="evi", arguments={})
    tracer.add_trace(trace2)
    assert tracer.get_trace_leaves() == {trace1, trace2}
    traces = tracer.process_traces([trace1, trace2], operation="filter_bbox", arguments={"bbox": "mol"})
    assert tracer.get_trace_leaves() == set(traces)
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
    assert source_constraints == {}


def test_evaluate_basic_load_collection(dry_run_env, dry_run_tracer):
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}, "result": True},
    }
    cube = evaluate(pg, env=dry_run_env)

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert source_constraints == {
        ("load_collection", ("S2_FOOBAR",)): {}
    }


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

    source_constraints = dry_run_tracer.get_source_constraints(merge=False)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
    assert constraints == [{"temporal_extent": [("2020-02-02", "2020-03-03")]}]

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
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
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
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
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
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

    source_constraints = dry_run_tracer.get_source_constraints(merge=False)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
    assert sorted(constraints, key=str) == [
        {
            "bands": [["grass"]],
            "spatial_extent": [{"west": 1, "east": 2, "south": 51, "north": 52, "crs": "EPSG:4326"}]
        },
        {
            "bands": [["red"]],
            "spatial_extent": [{"west": 1, "east": 2, "south": 51, "north": 52, "crs": "EPSG:4326"}]
        },
    ]

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
    assert constraints == {
        "bands": IgnoreOrder(["red", "grass"]),
        "spatial_extent": {"west": 1, "east": 2, "south": 51, "north": 52, "crs": "EPSG:4326"}
    }


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

    source_constraints = dry_run_tracer.get_source_constraints(merge=False)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
    assert constraints == [{
        "temporal_extent": [("2020-01-01", "2020-10-10"), ("2020-02-02", "2020-03-03"), ],
        "spatial_extent": [
            {"west": 0, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
            {"west": 1, "south": 51, "east": 3, "north": 53, "crs": "EPSG:4326"}
        ],
        "bands": [["red", "green", "blue"], ["red"]],
    }]

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
    assert constraints == {
        "temporal_extent": ("2020-01-01", "2020-10-10"),
        "spatial_extent": {"west": 0, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
        "bands": ["red", "green", "blue"],
    }


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

    source_constraints = dry_run_tracer.get_source_constraints(merge=False)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
    assert constraints == [{
        "temporal_extent": [("2020-01-01", "2020-10-10"), ("2020-02-02", "2020-03-03")],
        "spatial_extent": [
            {"west": 1, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
            {"west": 2.0, "south": 51, "east": 3, "north": 53, "crs": "EPSG:4326"}
        ],
        "bands": [["blue", "green", "red"], ["blue"]],
    }]

    source_constraints = dry_run_tracer.get_source_constraints(merge=True)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
    assert constraints == {
        "temporal_extent": ("2020-01-01", "2020-10-10"),
        "spatial_extent": {"west": 1, "south": 50, "east": 5, "north": 55, "crs": "EPSG:4326"},
        "bands": ["blue", "green", "red"],
    }


def test_aggregate_spatial(dry_run_env, dry_run_tracer):
    polygon = {
        "type": "Polygon",
        "coordinates": [[(0, 0), (3, 5), (8, 2), (0, 0)]]
    }
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
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
    assert constraints == {
        "spatial_extent": {"west": 0.0, "south": 0.0, "east": 8.0, "north": 5.0, "crs": "EPSG:4326"},
        "aggregate_spatial": {"geometries": shapely.geometry.shape(polygon)},
    }
    geometries, = dry_run_tracer.get_geometries()
    assert isinstance(geometries, shapely.geometry.Polygon)
    assert shapely.geometry.mapping(geometries) == {
        "type": "Polygon",
        "coordinates": (((0.0, 0.0), (3.0, 5.0), (8.0, 2.0), (0.0, 0.0)),)
    }


def test_aggregate_spatial_read_vector(dry_run_env, dry_run_tracer):
    geometry_path = str(get_path("GeometryCollection.geojson"))
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
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
    assert constraints == {
        "spatial_extent": {"west": 5.05, "south": 51.21, "east": 5.15, "north": 51.3, "crs": "EPSG:4326"},
        "aggregate_spatial": {"geometries": DelayedVector(geometry_path)},
    }
    geometries, = dry_run_tracer.get_geometries()
    assert isinstance(geometries, DelayedVector)


@pytest.mark.parametrize(["arguments", "expected"], [
    (
            {},
            {
                "backscatter_coefficient": "gamma0", "orthorectify": False, "elevation_model": None,
                "options": {},
            }
    ),
    (
            {
                "backscatter_coefficient": "sigma0", "orthorectify": True, "elevation_model": "dem",
                "options": {"dem_zoom": 9},
            },
            {
                "backscatter_coefficient": "sigma0", "orthorectify": True, "elevation_model": "dem",
                "options": {"dem_zoom": 9},
            }
    ),
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

    source_constraints = dry_run_tracer.get_source_constraints(merge=False)
    assert len(source_constraints) == 1
    src, constraints = source_constraints.popitem()
    assert src == ("load_collection", ("S2_FOOBAR",))
    assert constraints == [{"sar_backscatter": [expected]}]
