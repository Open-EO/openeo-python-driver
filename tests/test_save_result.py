import numpy as np
import pytest
from shapely.geometry import GeometryCollection, Polygon

from openeo_driver.save_result import AggregatePolygonResult, SaveResult
from .data import load_json, json_normalize


def test_is_format():
    r = SaveResult("GTiff")
    assert r.is_format("gtiff")
    assert r.is_format("gtiff", "geotiff")
    assert not r.is_format("geotiff")


def test_aggregate_polygon_result_basic():
    timeseries = {
        "2019-10-15T08:15:45Z": [[1, 2, 3], [4, 5, 6]],
        "2019-11-11T01:11:11Z": [[7, 8, 9], [10, 11, 12]],
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    result = AggregatePolygonResult(timeseries, regions=regions)
    result.set_format("covjson")

    data = result.get_data()
    assert json_normalize(data) == load_json("aggregate_polygon_result_basic_covjson.json")


def test_aggregate_polygon_result_nan_values():
    timeseries = {
        "2019-10-15T08:15:45Z": [[1, 2, 3], [4, np.nan, 6]],
        "2019-11-11T01:11:11Z": [[7, 8, 9], [np.nan, np.nan, np.nan]],
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    result = AggregatePolygonResult(timeseries, regions=regions)
    result.set_format("covjson")

    data = json_normalize(result.prepare_for_json())
    assert data["ranges"]["band0"]["values"] == [1, 4, 7, None]
    assert data["ranges"]["band1"]["values"] == [2, None, 8, None]
    assert data["ranges"]["band2"]["values"] == [3, 6, 9, None]


def test_aggregate_polygon_result_empty_ts_data():
    timeseries = {
        "2019-01-01T12:34:56Z": [[1, 2], [3, 4]],
        "2019-02-01T12:34:56Z": [[5, 6], []],
        "2019-03-01T12:34:56Z": [[], []],
        "2019-04-01T12:34:56Z": [[], [7, 8]],
        "2019-05-01T12:34:56Z": [[9, 10], [11, 12]],
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    result = AggregatePolygonResult(timeseries, regions=regions)
    result.set_format("covjson")

    data = json_normalize(result.prepare_for_json())
    assert data["domain"]["axes"]["t"]["values"] == [
        "2019-01-01T12:34:56Z",
        "2019-02-01T12:34:56Z",
        "2019-04-01T12:34:56Z",
        "2019-05-01T12:34:56Z",
    ]
    assert data["ranges"] == {
        "band0": {
            "type": "NdArray", "dataType": "float", "axisNames": ["t", "composite"], "shape": [4, 2],
            "values": [1, 3, 5, None, None, 7, 9, 11],
        },
        "band1": {
            "type": "NdArray", "dataType": "float", "axisNames": ["t", "composite"], "shape": [4, 2],
            "values": [2, 4, 6, None, None, 8, 10, 12],
        },
    }


def test_aggregate_polygon_result_inconsistent_bands():
    timeseries = {
        "2019-01-01T12:34:56Z": [[1, 2], [3, 4, 5]],
        "2019-02-01T12:34:56Z": [[6], []],
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    result = AggregatePolygonResult(timeseries, regions=regions)
    result.set_format("covjson")

    with pytest.raises(ValueError):
        result.prepare_for_json()
