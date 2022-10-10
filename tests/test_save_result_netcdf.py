from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal, assert_allclose
from openeo_driver.save_result import AggregatePolygonResult, AggregatePolygonResultCSV
from shapely.geometry import GeometryCollection, Polygon

from openeo.metadata import CollectionMetadata


def test_aggregate_polygon_result_basic(tmp_path):
    timeseries = {
        "2019-11-11T01:11:11Z": [[], [10, 11, 12]],
        "2019-10-15T08:15:45Z": [[1, 2, 3], [4, np.nan, 6]],
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    metadata = CollectionMetadata({
        "cube:dimensions": {
            "x": {"type": "spatial"},
            "b": {"type": "bands", "values": ["red", "green","blue"]}
        }
    })

    result = AggregatePolygonResult(timeseries, regions=regions, metadata=metadata)
    result.set_format("netcdf")

    assets = result.write_assets(tmp_path / "ignored")
    theAsset = assets.popitem()[1]
    filename = theAsset['href']

    assert 'application/x-netcdf' == theAsset['type']
    assert ["red", "green", "blue"] == [b['name'] for b in theAsset['bands']]

    timeseries_ds = xr.open_dataset(filename)
    print(timeseries_ds)
    assert_array_equal(timeseries_ds.red.coords['t'].data, np.asarray([ np.datetime64('2019-10-15T08:15:45'),np.datetime64('2019-11-11T01:11:11')]))
    timeseries_ds.red.sel(feature=1)
    timeseries_ds.red.sel( t='2019-10-16')
    print(timeseries_ds)
    assert_array_equal( 4, timeseries_ds.red.sel(feature=1).sel( t="2019-10-15T08:15:45Z").data)


def test_aggregate_polygon_result_nan_values(tmp_path):
    timeseries = {
        "2019-10-15T08:15:45Z": [[1, 2, 3], [4, np.nan, 6]],
        "2019-11-11T01:11:11Z": [[7, 8, 9], [np.nan, np.nan, np.nan]],
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    result = AggregatePolygonResult(timeseries, regions=regions)

    filename = result.to_netcdf(tmp_path / 'timeseries_xarray_nodata.nc')
    timeseries_ds = xr.open_dataset(filename)
    assert_array_equal(timeseries_ds.band_0.sel(feature=0).data, [1,7])
    assert_array_equal(timeseries_ds.band_2.sel(feature=0).data, [3, 9])
    assert_array_equal(timeseries_ds.band_2.sel(feature=1).data, [6, np.nan])


def test_aggregate_polygon_result_empty_ts_data(tmp_path):
    timeseries = {
        "2019-04-01T12:34:56Z": [[], [7, 8]],
        "2019-01-01T12:34:56Z": [[1, 2], [3, 4]],
        "2019-02-01T12:34:56Z": [[5, 6], []],
        "2019-05-01T12:34:56Z": [[9, 10], [11, 12]],
        "2019-03-01T12:34:56Z": [[], []]
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    result = AggregatePolygonResult(timeseries, regions=regions)
    result.set_format("netcdf")

    filename = result.to_netcdf(tmp_path / 'timeseries_xarray_empty.nc')
    timeseries_ds = xr.open_dataset(filename)

    assert_array_equal(timeseries_ds.t.data, [
        np.datetime64("2019-01-01T12:34:56Z"),
        np.datetime64("2019-02-01T12:34:56Z"),
        np.datetime64("2019-04-01T12:34:56Z"),
        np.datetime64("2019-05-01T12:34:56Z"),
    ])
    assert_array_equal( timeseries_ds.band_0.sel(feature=0).data, [1,  5, np.nan,  9])
    assert_array_equal( timeseries_ds.band_1.sel(feature=0).data, [2, 6,  np.nan, 10])



def test_aggregate_polygon_result_inconsistent_bands(tmp_path):
    timeseries = {
        "2019-01-01T12:34:56Z": [[1, 2], [3, 4, 5]],
        "2019-02-01T12:34:56Z": [[6], []],
    }
    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    result = AggregatePolygonResult(timeseries, regions=regions)
    result.set_format("netcdf")

    with pytest.raises(Exception):
        filename = result.to_netcdf(tmp_path / 'timeseries_xarray_invalid.nc')


def test_aggregate_polygon_result_CSV(tmp_path):

    regions = GeometryCollection([
        Polygon([(0, 0), (5, 1), (1, 4)]),
        Polygon([(6, 1), (1, 7), (9, 9)])
    ])

    metadata = CollectionMetadata({
        "cube:dimensions": {
            "x": {"type": "spatial"},
            "b": {"type": "bands", "values": ["red", "green","blue"]}
        }
    })

    result = AggregatePolygonResultCSV(csv_dir=Path(__file__).parent / "data" /"aggregate_spatial_spacetime_cube", regions=regions, metadata=metadata)
    result.set_format("netcdf")

    assets = result.write_assets(tmp_path / "ignored")
    theAsset = assets.popitem()[1]
    filename = theAsset['href']

    assert 'application/x-netcdf' == theAsset['type']
    assert ["red", "green", "blue"] == [b['name'] for b in theAsset['bands']]
    assert 'raster:bands' in theAsset
    assert 'file:size' in theAsset

    assert 'mean' in theAsset['raster:bands'][0]
    assert 'minimum' in theAsset['raster:bands'][0]
    assert 100.0 == theAsset['raster:bands'][0]['valid_percent']

    timeseries_ds = xr.open_dataset(filename)
    print(timeseries_ds)

    assert_array_equal(timeseries_ds.red.coords['t'].data, np.asarray([ np.datetime64('2017-09-05T00:00:00'),np.datetime64('2017-09-06T00:00:00')]))

    timeseries_ds.red.sel( t='2017-09-05')

    assert_allclose( timeseries_ds.red.sel(feature=1).sel( t='2017-09-06').data,4645.719597)