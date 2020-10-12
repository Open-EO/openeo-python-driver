from openeo_driver.ProcessGraphDeserializer import evaluate
from openeo_driver.dry_run import DryRunDataCube
from openeo_driver.utils import EvalEnv


def dry_run_evaluate(process_graph) -> DryRunDataCube:
    env = EvalEnv({"dry-run": True, "version": "1.0.0"})
    cube = evaluate(process_graph, env=env)
    return cube


def test_basic_filter_temporal():
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "ft": {
            "process_id": "filter_temporal",
            "arguments": {"data": {"from_node": "lc"}, "extent": ["2020-02-02", "2020-03-03"]},
            "result": True,
        },
    }
    cube = dry_run_evaluate(pg)
    assert len(cube.journals) == 1
    assert cube.journals[0].creation == ('load_collection', 'S2_FOOBAR')
    assert cube.journals[0].get("temporal_extent") == [('2020-02-02', '2020-03-03')]


def test_temporal_extent_dynamic():
    pg = {
        "load": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "extent": {"process_id": "constant", "arguments": {"x": ["2020-01-01", "2020-02-02"]}},
        "filtertemporal": {
            "process_id": "filter_temporal",
            "arguments": {"data": {"from_node": "load"}, "extent": {"from_node": "extent"}},
            "result": True,
        },
    }
    cube = dry_run_evaluate(pg)
    assert len(cube.journals) == 1
    assert cube.journals[0].creation == ('load_collection', 'S2_FOOBAR')
    assert cube.journals[0].get("temporal_extent") == [("2020-01-01", "2020-02-02")]


def test_temporal_extent_dynamic_item():
    pg = {
        "load": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "start": {"process_id": "constant", "arguments": {"x": "2020-01-01"}},
        "filtertemporal": {
            "process_id": "filter_temporal",
            "arguments": {"data": {"from_node": "load"}, "extent": [{"from_node": "start"}, "2020-02-02"]},
            "result": True,
        },
    }
    cube = dry_run_evaluate(pg)
    assert len(cube.journals) == 1
    assert cube.journals[0].creation == ('load_collection', 'S2_FOOBAR')
    assert cube.journals[0].get("temporal_extent") == [("2020-01-01", "2020-02-02")]


def test_graph_diamond():
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
    cube = dry_run_evaluate(pg)
    assert len(cube.journals) == 2
    assert cube.journals[0].creation == ('load_collection', 'S2_FOOBAR')
    assert cube.journals[0].get("bands") == [["red"]]
    assert cube.journals[0].get("spatial_extent") == [{"west": 1, "east": 2, "south": 51, "north": 52, "crs": "EPSG:4326"}]
    assert cube.journals[1].creation == ('load_collection', 'S2_FOOBAR')
    assert cube.journals[1].get("bands") == [["grass"]]
    assert cube.journals[1].get("spatial_extent") == [{"west": 1, "east": 2, "south": 51, "north": 52, "crs": "EPSG:4326"}]


def test_load_collection_and_filter_extents():
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
    cube = dry_run_evaluate(pg)
    assert len(cube.journals) == 1
    assert cube.journals[0].creation == ('load_collection', 'S2_FOOBAR')
    assert cube.journals[0].get("temporal_extent") == [('2020-01-01', '2020-10-10'), ('2020-02-02', '2020-03-03')]
    assert cube.journals[0].get("spatial_extent") == [
        {"west": 0, "south": 50, "east": 5, "north": 55, 'crs': 'EPSG:4326'},
        {"west": 1, "south": 51, "east": 3, "north": 53, 'crs': 'EPSG:4326'},
    ]
    assert cube.journals[0].get("bands") == [["red", "green", "blue"], ["red"]]


def test_load_collection_and_filter_extents_dynamic():
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
    cube = dry_run_evaluate(pg)
    assert len(cube.journals) == 1
    assert cube.journals[0].creation == ('load_collection', 'S2_FOOBAR')
    assert cube.journals[0].get("temporal_extent") == [('2020-01-01', '2020-10-10'), ('2020-02-02', '2020-03-03')]
    assert cube.journals[0].get("spatial_extent") == [
        {"west": 1, "south": 50, "east": 5, "north": 55, 'crs': 'EPSG:4326'},
        {"west": 2.0, "south": 51, "east": 3, "north": 53, 'crs': 'EPSG:4326'},
    ]
    assert cube.journals[0].get("bands") == [["blue", "green", "red"], ["blue"]]
