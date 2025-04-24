import pandas as pd
import geopandas as gpd
import pytest
from numpy import nan
from shapely.geometry import GeometryCollection, Polygon

from openeo_driver.datacube import DriverVectorCube
from openeo_driver.save_result import AggregatePolygonResult
from .data import get_path


def test_aggregate_polygon_result_basic(tmp_path):
    time_series = {
        "2019-10-15T08:15:45Z": [[1, 2, 3], [4, 5, 6]],
        "2019-11-11T01:11:11Z": [[7, 8, 9], [10, 11, 12]],
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    result = AggregatePolygonResult(time_series, regions=regions)
    result.set_format("csv")

    filename = result.to_csv(tmp_path / 'timeseries.csv')

    expected_results = {
        '2019-10-15T08:15:45Z__01': [1, 4],
        '2019-10-15T08:15:45Z__02': [2, 5],
        '2019-10-15T08:15:45Z__03': [3, 6],
        '2019-11-11T01:11:11Z__01': [7, 10],
        '2019-11-11T01:11:11Z__02': [8, 11],
        '2019-11-11T01:11:11Z__03': [9, 12]
    }

    assert pd.DataFrame.from_dict(expected_results).equals(pd.read_csv(filename))


def test_aggregate_polygon_result_empty_ts_data(tmp_path):
    time_series = {
        "2019-01-01T12:34:56Z": [[], [3, 4]],
        "2019-02-01T12:34:56Z": [[5, 6], []],
        "2019-03-01T12:34:56Z": [[], []],
        "2019-04-01T12:34:56Z": [[], [7, 8]],
        "2019-05-01T12:34:56Z": [[9, 10], [11, 12]],
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    result = AggregatePolygonResult(time_series, regions=regions)
    result.set_format("csv")

    filename = result.to_csv(tmp_path / 'timeseries_empty.csv')

    expected_results = {
        '2019-01-01T12:34:56Z__01': [nan, 3],
        '2019-01-01T12:34:56Z__02': [nan, 4],
        '2019-02-01T12:34:56Z__01': [5, nan],
        '2019-02-01T12:34:56Z__02': [6, nan],
        '2019-03-01T12:34:56Z__01': [nan, nan],
        '2019-03-01T12:34:56Z__02': [nan, nan],
        '2019-04-01T12:34:56Z__01': [nan, 7],
        '2019-04-01T12:34:56Z__02': [nan, 8],
        '2019-05-01T12:34:56Z__01': [9, 11],
        '2019-05-01T12:34:56Z__02': [10, 12]
    }

    assert pd.DataFrame.from_dict(expected_results).equals(pd.read_csv(filename))


def test_aggregate_polygon_result_empty_single_polygon_single_band(tmp_path):
    """https://github.com/Open-EO/openeo-python-driver/issues/70"""
    time_series = {
        "2019-01-01T12:34:56Z": [[]],
        "2019-02-01T12:34:56Z": [[2]],
        "2019-03-01T12:34:56Z": [[]],
        "2019-04-01T12:34:56Z": [[8]],
        "2019-05-01T12:34:56Z": [[9]],
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
    ])

    result = AggregatePolygonResult(time_series, regions=regions)
    result.set_format("csv")

    filename = result.to_csv(tmp_path / 'timeseries_empty.csv')

    expected_results = {
        '2019-01-01T12:34:56Z': [nan],
        '2019-02-01T12:34:56Z': [2],
        '2019-03-01T12:34:56Z': [nan],
        '2019-04-01T12:34:56Z': [8],
        '2019-05-01T12:34:56Z': [9],
    }

    assert pd.DataFrame.from_dict(expected_results).equals(pd.read_csv(filename))


def test_aggregate_polygon_result_inconsistent_bands(tmp_path):
    time_series = {
        "2019-01-01T12:34:56Z": [[1, 2], [3, 4, 5]],
        "2019-02-01T12:34:56Z": [[6], []],
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    result = AggregatePolygonResult(time_series, regions=regions)
    result.set_format("csv")

    with pytest.raises(IndexError):
        result.to_csv(tmp_path / 'timeseries_invalid.csv')

def test_write_driver_vector_cube_to_csv(tmp_path):
    vector_cube = DriverVectorCube.from_fiona([str(get_path("geojson/FeatureCollection02.json"))],driver="geojson")
    vector_cube.write_assets(tmp_path / "dummy", format="CSV")

    actual_gdf = pd.read_csv(tmp_path / "vectorcube.csv")
    assert actual_gdf.shape == (2, 4)