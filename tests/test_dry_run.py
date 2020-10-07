from openeo_driver.ProcessGraphDeserializer import evaluate
from openeo_driver.dry_run import DryRunDataCube
from openeo_driver.utils import EvalEnv


def test_filter_temporal():
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2"}},
        "ft": {
            "process_id": "filter_temporal",
            "arguments": {"data": {"from_node": "lc"}, "extent": ["2020-02-02", "2020-03-03"]},
            "result": True,
        },
    }

    env = EvalEnv({"dry-run": True, "version": "1.0.0"})
    cube: DryRunDataCube = evaluate(pg, env=env)
    assert len(cube.journals) == 1
    assert cube.journals[0].get("load_collection") == [{"collection_id": "S2"}]
    assert cube.journals[0].get("filter_temporal") == [('2020-02-02', '2020-03-03')]


def test_split_merge():
    pg = {
        "load": {"process_id": "load_collection", "arguments": {"id": "S2"}},
        "band_red": {
            "process_id": "filter_bands",
            "arguments": {"data": {"from_node": "load"}, "bands": ["red"]},
        },
        "band_grass": {
            "process_id": "filter_bands",
            "arguments": {"data": {"from_node": "load"}, "bands": ["mask"]},

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

    env = EvalEnv({"dry-run": True, "version": "1.0.0"})
    cube: DryRunDataCube = evaluate(pg, env=env)
    assert len(cube.journals) == 2
    assert cube.journals[0].get("load_collection") == [{"collection_id": "S2"}]
    assert cube.journals[0].get("filter_bands") == [["red"]]
    assert cube.journals[0].get("filter_bbox") == [{"west": 1, "east": 2, "south": 51, "north": 52, "crs": "EPSG:4326"}]
    assert cube.journals[1].get("load_collection") == [{"collection_id": "S2"}]
    assert cube.journals[1].get("filter_bands") == [["mask"]]
    assert cube.journals[1].get("filter_bbox") == [{"west": 1, "east": 2, "south": 51, "north": 52, "crs": "EPSG:4326"}]


def test_load_collection_extents():
    pg = {
        "load": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2",
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

    env = EvalEnv({"dry-run": True, "version": "1.0.0"})
    cube: DryRunDataCube = evaluate(pg, env=env)
    assert len(cube.journals) == 1
    assert cube.journals[0].get("load_collection") == [{"collection_id": "S2"}]
    assert cube.journals[0].get("filter_temporal") == [('2020-01-01', '2020-10-10'), ('2020-02-02', '2020-03-03')]
    assert cube.journals[0].get("filter_bbox") == [
        {"west": 0, "south": 50, "east": 5, "north": 55, 'crs': 'EPSG:4326'},
        {"west": 1, "south": 51, "east": 3, "north": 53, 'crs': 'EPSG:4326'},
    ]
    assert cube.journals[0].get("filter_bands") == [["red", "green", "blue"], ["red"]]
