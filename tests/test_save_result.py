import datetime
import json
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import GeometryCollection, Polygon
import dirty_equals

from openeo.metadata import CollectionMetadata
from openeo_driver.datacube import DriverVectorCube
from openeo_driver.save_result import AggregatePolygonResult, SaveResult, AggregatePolygonSpatialResult, \
    AggregatePolygonResultCSV, JSONResult
from openeo_driver.workspacerepository import WorkspaceRepository
from .data import load_json, json_normalize, get_path


regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])


def test_is_format():
    r = SaveResult("GTiff")
    assert r.is_format("gtiff")
    assert r.is_format("gtiff", "geotiff")
    assert not r.is_format("geotiff")


def test_with_format():
    g = SaveResult("GTiff", options={"ZLEVEL": 9})
    n = g.with_format("netCDF", options={})

    assert (g.format, g.options) == ("GTiff", {"ZLEVEL": 9})
    assert (n.format, n.options) == ("netCDF", {})


@pytest.mark.parametrize(["merge", "expected_workspace_path"], [
    ("some/path", "some/path"),
    ("", "."),
    (None, "/some/unique/path")
])
@pytest.mark.parametrize("remove_original", [False, True])
def test_export_workspace(merge, expected_workspace_path, remove_original):
    mock_workspace_repository = mock.Mock(spec=WorkspaceRepository)
    mock_workspace = mock_workspace_repository.get_by_id.return_value

    r = SaveResult()
    r.add_workspace_export(workspace_id="some-workspace", merge=merge)
    r.export_workspace(
        workspace_repository=mock_workspace_repository,
        hrefs=["/some/file"],
        default_merge="/some/unique/path",
        remove_original=remove_original,
    )

    assert list(r.workspace_exports) == [SaveResult.WorkspaceExport(workspace_id="some-workspace", merge=merge)]

    mock_workspace.import_file.assert_called_with(Path("/some/file"), expected_workspace_path, remove_original)


@pytest.mark.parametrize(
    ["merge", "expected_workspace_path"], [("some/path", "some/path"), ("", "."), (None, "/some/unique/path")]
)
@pytest.mark.parametrize("remove_original", [False, True])
def test_export_workspace_s3(merge, expected_workspace_path, remove_original):
    mock_workspace_repository = mock.Mock(spec=WorkspaceRepository)
    mock_workspace = mock_workspace_repository.get_by_id.return_value

    r = SaveResult()
    r.add_workspace_export(workspace_id="some-workspace", merge=merge)
    r.export_workspace(
        workspace_repository=mock_workspace_repository,
        hrefs=["s3://some_bucket/some/key"],
        default_merge="/some/unique/path",
        remove_original=remove_original,
    )

    assert list(r.workspace_exports) == [SaveResult.WorkspaceExport(workspace_id="some-workspace", merge=merge)]

    mock_workspace.import_object.assert_called_with(
        "s3://some_bucket/some/key", expected_workspace_path, remove_original
    )


def test_aggregate_polygon_result_basic():
    timeseries = {
        "2019-10-15T08:15:45Z": [[1, 2, 3], [4, 5, 6]],
        "2019-11-11T01:11:11Z": [[7, 8, 9], [10, 11, 12]],
    }


    result = AggregatePolygonResult(timeseries, regions=regions)
    result.set_format("covjson")

    data = result.get_data()
    assert json_normalize(data) == load_json("aggregate_polygon_result_basic_covjson.json")


def test_aggregate_polygon_covjson_result_vector_cube():
    timeseries = {
        "2019-10-15T08:15:45Z": [[1, 2, 3], [4, 5, 6]],
        "2019-11-11T01:11:11Z": [[7, 8, 9], [10, 11, 12]],
    }

    gdf = gpd.read_file(str(get_path("geojson/FeatureCollection02.json")))
    regions = DriverVectorCube(gdf)

    result = AggregatePolygonResult(timeseries, regions=regions)
    result.set_format("covjson")

    data = result.get_data()

    assert json_normalize(data) == load_json("aggregate_polygon_covjson_result_vector_cube.json")


def test_aggregate_polygon_result_nan_values():
    timeseries = {
        "2019-10-15T08:15:45Z": [[1, 2, 3], [4, np.nan, 6]],
        "2019-11-11T01:11:11Z": [[7, 8, 9], [np.nan, np.nan, np.nan]],
    }


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


    result = AggregatePolygonResult(timeseries, regions=regions)
    result.set_format("covjson")

    with pytest.raises(ValueError):
        result.prepare_for_json()


def test_aggregate_polygon_result_CSV(tmp_path):
    metadata = CollectionMetadata({
        "cube:dimensions": {
            "x": {"type": "spatial"},
            "b": {"type": "bands", "values": ["red", "green","blue"]}
        }
    })

    regions_with_nonexistant = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    result = AggregatePolygonResultCSV(csv_dir=Path(__file__).parent / "data" /"aggregate_spatial_spacetime_cube", regions=regions_with_nonexistant, metadata=metadata)
    result.set_format("json")

    assets = result.write_assets(tmp_path / "ignored")
    [(_, asset_metadata)] = assets.items()

    assert "application/json" == asset_metadata["type"]
    assert asset_metadata["bands"] == [
        dirty_equals.IsPartialDict(name="red"),
        dirty_equals.IsPartialDict(name="green"),
        dirty_equals.IsPartialDict(name="blue"),
    ]
    assert asset_metadata["raster:bands"] == [
        {
            "statistics": {
                "minimum": pytest.approx(4646, rel=0.1),
                "mean": pytest.approx(4646, rel=0.1),
                "maximum": pytest.approx(4646, rel=0.1),
                "stddev": pytest.approx(0.31, rel=0.1),
                "valid_percent": 100.0,
            }
        },
        {
            "statistics": {
                "minimum": pytest.approx(4865, rel=0.1),
                "mean": pytest.approx(4865, rel=0.1),
                "maximum": pytest.approx(4865, rel=0.1),
                "stddev": pytest.approx(0.265, rel=0.1),
                "valid_percent": 100.0,
            }
        },
        {
            "statistics": {
                "minimum": pytest.approx(5178, rel=0.1),
                "mean": pytest.approx(5178, rel=0.1),
                "maximum": pytest.approx(5178, rel=0.1),
                "stddev": pytest.approx(0.41, rel=0.1),
                "valid_percent": 100.0,
            }
        },
    ]

    assert "file:size" in asset_metadata

    filename = asset_metadata["href"]
    expected = {
        "2017-09-05T00:00:00Z": [
            [4646.262612301313, 4865.926572218383, 5178.517363510712],
            [None, None, None],
            [4645.719597475695, 4865.467252259935, 5177.803342998465],
        ],
        "2017-09-06T00:00:00Z": [
            [None, None, None],
            [None, None, None],
            [4645.719597475695, 4865.467252259935, 5177.803342998465],
        ],
    }
    with open(filename) as f:
        timeseries_ds = json.load(f)
    assert timeseries_ds == expected

class TestAggregatePolygonSpatialResult:

    def test_load_csv(self):
        csv_dir = get_path("aggregate_spatial_spatial_cube")
        regions = GeometryCollection([
            Polygon([(0, 0), (5, 1), (1, 4)]),
            Polygon([(6, 1), (1, 7), (9, 9)])
        ])
        res = AggregatePolygonSpatialResult(csv_dir=csv_dir, regions=regions)
        assert res.prepare_for_json() == [
            [4646.262612301313, 4865.926572218383, 5178.517363510712],
            [4645.719597475695, 4865.467252259935, 5177.803342998465],
        ]

def test_jsonresult(tmp_path):

    the_data = {
        "bla":"bla",
        "bla2":np.nan,
        "bla3":{'bla':'bla'},
        "bla4":datetime.datetime.fromisocalendar(2020,3,1)
    }
    JSONResult(the_data).write_assets(tmp_path/"bla")

    the_path = tmp_path.joinpath("result.json")
    actual_data = json.load(the_path.open())
    assert { 'bla': 'bla',
             'bla2': None,
             'bla3': {'bla': 'bla'},
             'bla4': '2020-01-13T00:00:00'} == actual_data
