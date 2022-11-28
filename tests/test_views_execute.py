
import json
import math
import re
import sys
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Iterable
from unittest import mock
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry

from openeo_driver.ProcessGraphDeserializer import custom_process_from_process_graph
from openeo_driver.datacube import DriverDataCube, DriverVectorCube
from openeo_driver.datastructs import SarBackscatterArgs, ResolutionMergeArgs
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.dry_run import ProcessType
from openeo_driver.dummy import dummy_backend
from openeo_driver.dummy.dummy_backend import DummyVisitor
from openeo_driver.errors import (
    ProcessGraphMissingException,
    ProcessGraphInvalidException,
)
from openeo_driver.testing import (
    ApiTester,
    preprocess_check_and_replace,
    TEST_USER,
    TEST_USER_BEARER_TOKEN,
    preprocess_regex_check_and_replace,
    generate_unique_test_process_id,
    RegexMatcher,
    DictSubSet,
)
from openeo_driver.util.geometry import (
    as_geojson_feature,
    as_geojson_feature_collection,
)
from openeo_driver.util.ioformats import IOFORMATS
from openeo_driver.utils import EvalEnv
from .data import get_path, TEST_DATA_ROOT, load_json


@pytest.fixture(params=["1.0.0"])
def api_version(request):
    return request.param


@pytest.fixture
def api(api_version, client, backend_implementation) -> ApiTester:
    dummy_backend.reset(backend_implementation)

    data_root = TEST_DATA_ROOT / "pg" / (".".join(api_version.split(".")[:2]))
    return ApiTester(api_version=api_version, client=client, data_root=data_root)


@pytest.fixture
def api040(client,backend_implementation) -> ApiTester:
    dummy_backend.reset(backend_implementation)
    data_root = TEST_DATA_ROOT / "pg" / "0.4"
    return ApiTester(api_version="0.4.0", client=client, data_root=data_root)


@pytest.fixture
def api100(client,backend_implementation) -> ApiTester:
    dummy_backend.reset(backend_implementation)
    data_root = TEST_DATA_ROOT / "pg" / "1.0"
    return ApiTester(api_version="1.0.0", client=client, data_root=data_root)


# Major.minor version of current python
CURRENT_PY3x = f"{sys.version_info.major}.{sys.version_info.minor}"


def test_udf_runtimes(api):
    runtimes = api.get('/udf_runtimes').assert_status_code(200).json
    assert runtimes == DictSubSet({
        "Python": DictSubSet({
            "title": RegexMatcher("Python"),
            "type": "language",
            "default": "3",
            "versions": DictSubSet({
                "3": {"libraries": DictSubSet({"numpy": {"version": RegexMatcher(r"\d+\.\d+\.\d+")}})},
                CURRENT_PY3x: {"libraries": DictSubSet({"numpy": {"version": RegexMatcher(r"\d+\.\d+\.\d+")}})},
            })
        })
    })


def test_execute_simple_download(api):
    resp = api.check_result("basic.json")
    assert re.match(r"GTiff:save_result\(.*DummyDataCube", resp.text, flags=re.IGNORECASE)


def test_load_collection(api):
    api.check_result({
        'collection': {
            'process_id': 'load_collection',
            'arguments': {'id': 'S2_FAPAR_CLOUDCOVER'},
            'result': True
        }
    })


def test_load_collection_date_shift(api):
    api.check_result({
        'dateshift1': {'arguments': {'date': '2020-01-01',
                                     'unit': 'day',
                                     'value': -30},
                       'process_id': 'date_shift'
                       },
        'loadcollection1': {'arguments': {'id': 'S2_FAPAR_CLOUDCOVER',
                                   'temporal_extent': [{'from_node': 'dateshift1'},
                                                       '2021-01-01']},
                     'process_id': 'load_collection',
                     'result': True}})

    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params["temporal_extent"] == ('2019-12-02', '2021-01-01')


def test_execute_filter_temporal(api):
    api.check_result({
        'loadcollection1': {
            'process_id': 'load_collection',
            'arguments': {'id': 'S2_FAPAR_CLOUDCOVER'}
        },
        'filtertemporal1': {
            'process_id': 'filter_temporal',
            'arguments': {
                'data': {'from_node': 'loadcollection1'},
                'extent': ['2018-01-01', '2018-12-31']
            },
            'result': True
        },
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params["temporal_extent"] == ("2018-01-01", "2018-12-31")


def test_execute_filter_temporal_extent_reversed(api):
    resp = api.result({
        'loadcollection1': {
            'process_id': 'load_collection',
            'arguments': {'id': 'S2_FAPAR_CLOUDCOVER'}
        },
        'filtertemporal1': {
            'process_id': 'filter_temporal',
            'arguments': {
                'data': {'from_node': 'loadcollection1'},
                'extent': ['2021-12-07', '2021-12-06']
            },
            'result': True
        },
    })

    resp.assert_error(400, "ProcessParameterInvalid",
                      message="The value passed for parameter 'extent' in process 'filter_temporal' is invalid:"
                              " end '2021-12-06' is before start '2021-12-07'")


def test_execute_filter_temporal_extent_reversed(api):
    resp = api.result({
        'loadcollection1': {
            'process_id': 'load_collection',
            'arguments': {'id': 'S2_FAPAR_CLOUDCOVER'}
        },
        'filtertemporal1': {
            'process_id': 'filter_temporal',
            'arguments': {
                'data': {'from_node': 'loadcollection1'},
                'extent': [None, None]
            },
            'result': True
        },
    })

    resp.assert_error(400, "ProcessParameterInvalid",
                      message="The value passed for parameter 'extent' in process 'filter_temporal' is invalid:"
                              " both start and end are null")


def test_execute_filter_bbox(api):
    api.check_result({
        'loadcollection1': {
            'process_id': 'load_collection',
            'arguments': {'id': 'S2_FAPAR_CLOUDCOVER'}
        },
        'filterbbox1': {
            'process_id': 'filter_bbox',
            'arguments': {
                'data': {'from_node': 'loadcollection1'},
                'extent': {
                    "west": 3, "east": 5,
                    "south": 50, "north": 51,
                    "crs": "EPSG:4326",
                }
            },
            'result': True
        },
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params["spatial_extent"] == {"west": 3, "east": 5, "south": 50, "north": 51, "crs": "EPSG:4326", }


def test_execute_filter_bbox_integer_crs(api):
    api.check_result({
        'loadcollection1': {
            'process_id': 'load_collection',
            'arguments': {'id': 'S2_FAPAR_CLOUDCOVER'}
        },
        'filterbbox1': {
            'process_id': 'filter_bbox',
            'arguments': {
                'data': {'from_node': 'loadcollection1'},
                'extent': {
                    "west": 3, "east": 5,
                    "south": 50, "north": 51,
                    "crs": 4326,
                }
            },
            'result': True
        },
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params["spatial_extent"] == {"west": 3, "east": 5, "south": 50, "north": 51, "crs": "EPSG:4326", }


def test_execute_filter_bands(api):
    api.check_result({
        'loadcollection1': {
            'process_id': 'load_collection',
            'arguments': {'id': 'S2_FOOBAR'},
        },
        'filterbands1': {
            'process_id': 'filter_bands',
            'arguments': {
                'data': {'from_node': 'loadcollection1'},
                'bands': ["B02", "B03"],
            },
            'result': True
        },
    })
    load_params = dummy_backend.last_load_collection_call("S2_FOOBAR")
    assert load_params["bands"] == ["B02", "B03"]


def test_execute_apply_kernel(api):
    kernel_list = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    api.check_result("apply_kernel.json")
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    assert dummy.apply_kernel.call_count == 1
    np_kernel = dummy.apply_kernel.call_args[1]["kernel"]
    assert np_kernel.tolist() == kernel_list
    assert dummy.apply_kernel.call_args[1]["factor"] == 3


def test_load_collection_filter(api):
    api.check_result({
        'collection': {
            'process_id': 'load_collection',
            'arguments': {
                'id': 'S2_FAPAR_CLOUDCOVER',
                'spatial_extent': {
                    'west': 5.027, 'east': 5.0438, 'north': 51.2213,
                    'south': 51.1974, 'crs': 'EPSG:4326'
                },
                'temporal_extent': ['2018-01-01', '2018-12-31'],
                'featureflags': {'experimental': True}
            },
            'result': True
        }
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params["temporal_extent"] == ('2018-01-01', '2018-12-31')
    assert params["spatial_extent"] == {
        'west': 5.027, 'south': 51.1974, 'east': 5.0438, 'north': 51.2213, 'crs': 'EPSG:4326',
    }
    assert params["featureflags"] == {'experimental': True}


@pytest.mark.parametrize(
    ["spatial_extent", "expected"],
    [
        (
            load_json("geojson/Polygon01.json"),
            {
                "west": 5.1,
                "south": 51.2,
                "east": 5.14,
                "north": 51.23,
                "crs": "EPSG:4326",
            },
        ),
        (
            load_json("geojson/MultiPolygon01.json"),
            {
                "west": 5.1,
                "south": 51.2,
                "east": 5.14,
                "north": 51.24,
                "crs": "EPSG:4326",
            },
        ),
        (
            load_json("geojson/Feature01.json"),
            {"west": 1.0, "south": 1.0, "east": 3.0, "north": 3.0, "crs": "EPSG:4326"},
        ),
        (
            load_json("geojson/Feature03-null-properties.json"),
            {"west": 1.0, "south": 1.0, "east": 3.0, "north": 3.0, "crs": "EPSG:4326"},
        ),
        (
            as_geojson_feature(shapely.geometry.Polygon.from_bounds(11, 22, 33, 44)),
            {"west": 11, "south": 22, "east": 33, "north": 44, "crs": "EPSG:4326"},
        ),
        (
            load_json("geojson/FeatureCollection01.json"),
            {
                "west": 4.45,
                "south": 51.1,
                "east": 4.52,
                "north": 51.2,
                "crs": "EPSG:4326",
            },
        ),
        (
            as_geojson_feature_collection(
                as_geojson_feature(
                    shapely.geometry.Polygon.from_bounds(11, 22, 33, 44),
                    properties=None,
                ),
                as_geojson_feature(
                    shapely.geometry.Polygon.from_bounds(22, 33, 44, 55),
                    properties={"color": "red"},
                ),
            ),
            {"west": 11, "south": 22, "east": 44, "north": 55, "crs": "EPSG:4326"},
        ),
        (
            as_geojson_feature_collection(
                shapely.geometry.Point(2, 3),
                shapely.geometry.Point(4, 5),
            ),
            {"west": 2, "south": 3, "east": 4, "north": 5, "crs": "EPSG:4326"},
        ),
    ],
)
def test_load_collection_spatial_extent_geojson(api, spatial_extent, expected):
    api.check_result(
        {
            "collection": {
                "process_id": "load_collection",
                "arguments": {
                    "id": "S2_FAPAR_CLOUDCOVER",
                    "spatial_extent": spatial_extent,
                    "temporal_extent": ["2018-01-01", "2018-12-31"],
                },
                "result": True,
            }
        }
    )
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params["temporal_extent"] == ("2018-01-01", "2018-12-31")
    assert params["spatial_extent"] == expected


def test_execute_apply_unary_040(api040):
    api040.check_result("apply_unary.json")
    assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").apply.call_count == 2


def test_execute_apply_unary(api100):
    api100.check_result("apply_unary.json")
    assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").apply.call_count == 1


def test_execute_apply_unary_parent_scope(api100):
    api100.check_result(
        "apply_unary.json",
        preprocess=preprocess_check_and_replace('"from_parameter": "x"', '"from_parameter": "data"')
    )


#
@pytest.mark.skip('parameter checking of callback graphs now happens somewhere else')
def test_execute_apply_unary_invalid_from_parameter(api100):
    resp = api100.result(
        "apply_unary.json",
        preprocess=preprocess_check_and_replace('"from_parameter": "x"', '"from_parameter": "1nv8l16"')
    )
    resp.assert_error(400, "ProcessParameterRequired")


def test_execute_apply_run_udf_040(api040):
    api040.check_result("apply_run_udf.json")
    assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").apply_tiles.call_count == 1


def test_execute_apply_run_udf_100(api100):
    api100.check_result("apply_run_udf.json")
    assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").apply.call_count == 1


def test_reduce_temporal_run_udf(api):
    api.check_result("reduce_temporal_run_udf.json")
    if api.api_version_compare.at_least("1.0.0"):
        loadCall = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
        assert loadCall.process_types == set([ProcessType.GLOBAL_TIME])
        assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").reduce_dimension.call_count == 1
    else:
        assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").apply_tiles_spatiotemporal.call_count == 1


def test_reduce_temporal_run_udf_legacy_client(api):
    api.check_result(
        "reduce_temporal_run_udf.json",
        preprocess=preprocess_check_and_replace('"dimension": "t"', '"dimension": "temporal"')
    )
    if api.api_version_compare.at_least("1.0.0"):
        assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").reduce_dimension.call_count == 1
    else:
        assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").apply_tiles_spatiotemporal.call_count == 1


def test_reduce_temporal_run_udf_invalid_dimension(api):
    resp = api.result(
        "reduce_temporal_run_udf.json",
        preprocess=preprocess_check_and_replace('"dimension": "t"', '"dimension": "tempo"')
    )
    resp.assert_error(
        400, "ProcessParameterInvalid",
        message="The value passed for parameter 'dimension' in process '{p}' is invalid: got 'tempo', but should be one of ['x', 'y', 't']".format(
            p="reduce_dimension" if api.api_version_compare.at_least("1.0.0") else "reduce"
        )
    )


def test_reduce_bands_run_udf(api):
    api.check_result("reduce_bands_run_udf.json")
    if api.api_version_compare.at_least("1.0.0"):
        assert dummy_backend.get_collection("S2_FOOBAR").reduce_dimension.call_count == 1
    else:
        assert dummy_backend.get_collection("S2_FOOBAR").apply_tiles.call_count == 1


def test_reduce_bands_run_udf_legacy_client(api):
    api.check_result(
        "reduce_bands_run_udf.json",
        preprocess=preprocess_check_and_replace('"dimension": "bands"', '"dimension": "spectral_bands"')
    )
    if api.api_version_compare.at_least("1.0.0"):
        assert dummy_backend.get_collection("S2_FOOBAR").reduce_dimension.call_count == 1
    else:
        assert dummy_backend.get_collection("S2_FOOBAR").apply_tiles.call_count == 1


def test_reduce_bands_run_udf_invalid_dimension(api):
    resp = api.result(
        "reduce_bands_run_udf.json",
        preprocess=preprocess_check_and_replace('"dimension": "bands"', '"dimension": "layers"')
    )
    resp.assert_error(
        400, 'ProcessParameterInvalid',
        message="The value passed for parameter 'dimension' in process '{p}' is invalid: got 'layers', but should be one of ['x', 'y', 't', 'bands']".format(
            p="reduce_dimension" if api.api_version_compare.at_least("1.0.0") else "reduce"
        )
    )


def test_apply_dimension_temporal_run_udf(api):
    api.check_result("apply_dimension_temporal_run_udf.json")
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    assert dummy.apply_dimension.call_count == 1
    args,kwargs = dummy.apply_dimension.call_args
    if api.api_version_compare.at_least("1.0.0"):
        callback = args[0]
        #check if callback is valid
        DummyVisitor().accept_process_graph(callback)
        dummy.rename_dimension.assert_called_with('t', 'new_time_dimension')
        load_parameters = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
        assert load_parameters.process_types == set([ProcessType.GLOBAL_TIME])


def test_apply_dimension_temporal_run_udf_legacy_client(api):
    api.check_result(
        "apply_dimension_temporal_run_udf.json",
        preprocess=preprocess_check_and_replace('"dimension": "t"', '"dimension": "temporal"')
    )
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    assert dummy.apply_dimension.call_count == 1


def test_apply_dimension_temporal_run_udf_invalid_temporal_dimension(api):
    resp = api.result(
        "apply_dimension_temporal_run_udf.json",
        preprocess=preprocess_check_and_replace('"dimension": "t"', '"dimension": "letemps"')
    )
    resp.assert_error(
        400, 'ProcessParameterInvalid',
        message="The value passed for parameter 'dimension' in process 'apply_dimension' is invalid: got 'letemps', but should be one of ['x', 'y', 't']"
    )


def test_apply_neighborhood(api100):
    api100.check_result(
        "apply_neighborhood.json"
    )
    load_parameters = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert load_parameters.process_types == set([ProcessType.GLOBAL_TIME])


def test_reduce_max_t(api):
    api.check_result("reduce_max.json", preprocess=preprocess_check_and_replace("PLACEHOLDER", "t"))


def test_reduce_max_x(api):
    api.check_result("reduce_max.json", preprocess=preprocess_check_and_replace("PLACEHOLDER", "x"))


def test_reduce_max_y(api):
    api.check_result("reduce_max.json", preprocess=preprocess_check_and_replace("PLACEHOLDER", "y"))


def test_reduce_max_bands(api):
    api.check_result("reduce_max.json", preprocess=preprocess_check_and_replace("PLACEHOLDER", "bands"))


def test_reduce_max_bands_legacy_style(api):
    api.check_result("reduce_max.json", preprocess=preprocess_check_and_replace("PLACEHOLDER", "spectral_bands"))


def test_reduce_max_invalid_dimension(api):
    res = api.result("reduce_max.json", preprocess=preprocess_check_and_replace("PLACEHOLDER", "orbit"))
    res.assert_error(
        400, 'ProcessParameterInvalid',
        message="The value passed for parameter 'dimension' in process '{p}' is invalid: got 'orbit', but should be one of ['x', 'y', 't', 'bands']".format(
            p="reduce_dimension" if api.api_version_compare.at_least("1.0.0") else "reduce"
        )
    )


def test_execute_merge_cubes(api):
    api.check_result("merge_cubes.json")
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    assert dummy.merge_cubes.call_count == 1
    args, kwargs = dummy.merge_cubes.call_args
    assert args[1:] == ('or',)


def test_execute_resample_and_merge_cubes(api100):
    api100.check_result("resample_and_merge_cubes.json")
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    last_load_collection_call = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert last_load_collection_call.target_crs == "AUTO:42001"
    assert last_load_collection_call.target_resolution == [10, 10]
    assert dummy.merge_cubes.call_count == 1
    assert dummy.resample_cube_spatial.call_count == 1
    args, kwargs = dummy.merge_cubes.call_args
    assert args[1:] == ('or',)


def test_execute_merge_cubes_and_reduce(api100):
    api100.check_result("merge_cubes_and_reduce.json")
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    assert dummy.reduce_dimension.call_count == 1
    args, kwargs = dummy.reduce_dimension.call_args
    assert args == ()
    assert kwargs["dimension"] == "t"


def test_reduce_bands(api):
    api.check_result("reduce_bands.json")
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    if api.api_version_compare.at_least("1.0.0"):
        reduce_bands = dummy.reduce_dimension
    else:
        reduce_bands = dummy.reduce_bands
    reduce_bands.assert_called_once()
    if api.api_version_compare.below("1.0.0"):
        visitor = reduce_bands.call_args_list[0][0][0]
        assert isinstance(visitor, dummy_backend.DummyVisitor)
        assert set(p[0] for p in visitor.processes) == {"sum", "subtract", "divide"}


def test_reduce_bands_legacy_client(api):
    api.check_result(
        "reduce_bands.json",
        preprocess=preprocess_check_and_replace('"dimension": "bands"', '"dimension": "spectral_bands"')
    )
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    if api.api_version_compare.at_least("1.0.0"):
        reduce_bands = dummy.reduce_dimension
    else:
        reduce_bands = dummy.reduce_bands

    reduce_bands.assert_called_once()
    if api.api_version_compare.below("1.0.0"):
        visitor = reduce_bands.call_args_list[0][0][0]
        assert isinstance(visitor, dummy_backend.DummyVisitor)
        assert set(p[0] for p in visitor.processes) == {"sum", "subtract", "divide"}


def test_reduce_bands_invalid_dimension(api):
    res = api.result("reduce_bands.json",
                     preprocess=preprocess_check_and_replace('"dimension": "bands"', '"dimension": "layor"'))
    res.assert_error(
        400, "ProcessParameterInvalid",
        message="The value passed for parameter 'dimension' in process '{p}' is invalid: got 'layor', but should be one of ['x', 'y', 't', 'bands']".format(
            p="reduce_dimension" if api.api_version_compare.at_least("1.0.0") else "reduce"
        )
    )


def test_execute_mask(api):
    api.check_result("mask.json")
    assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").mask.call_count == 1

    expected_spatial_extent = {
        "west": 7.02,
        "south": 51.2,
        "east": 7.65,
        "north": 51.7,
        "crs": "EPSG:4326",
    }
    expected_geometry = DriverVectorCube.from_geojson(
        {
            "type": "Polygon",
            "coordinates": [
                [[7.02, 51.7], [7.65, 51.7], [7.65, 51.2], [7.04, 51.3], [7.02, 51.7]],
            ],
        }
    )

    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == expected_spatial_extent
    assert params["aggregate_spatial_geometries"] == expected_geometry

    params = dummy_backend.last_load_collection_call('S2_FAPAR_CLOUDCOVER')
    assert params["spatial_extent"] == expected_spatial_extent


def test_execute_diy_mask(api):
    api.check_result("scl_mask_custom.json")
    assert dummy_backend.get_collection("TERRASCOPE_S2_FAPAR_V2").mask.call_count == 1

    load_collections = dummy_backend.all_load_collection_calls("TERRASCOPE_S2_FAPAR_V2")
    assert len(load_collections) == 4
    assert load_collections[0].pixel_buffer == [8.5,8.5]
    assert load_collections[0].bands == ['SCENECLASSIFICATION_20M']
    assert load_collections[1].pixel_buffer == [100.5, 100.5]
    assert load_collections[1].bands == ['SCENECLASSIFICATION_20M']
    assert load_collections[2].bands == ['SCENECLASSIFICATION_20M'] #due to use of resample_cube_spatial
    assert load_collections[3].bands == ['FAPAR_10M']



def test_execute_mask_optimized_loading(api):
    api.check_result("mask.json",
                     preprocess=preprocess_check_and_replace('"10"', 'null')
                     )
    assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").mask.call_count == 1

    expected_spatial_extent = {
        "west": 7.02,
        "south": 51.2,
        "east": 7.65,
        "north": 51.7,
        "crs": "EPSG:4326",
    }
    expected_geometry = DriverVectorCube.from_geojson(
        {
            "type": "Polygon",
            "coordinates": [
                [[7.02, 51.7], [7.65, 51.7], [7.65, 51.2], [7.04, 51.3], [7.02, 51.7]],
            ],
        }
    )

    params = dummy_backend.last_load_collection_call('S2_FAPAR_CLOUDCOVER')
    assert params["spatial_extent"] == expected_spatial_extent
    assert isinstance(params.data_mask, DriverDataCube)

    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == expected_spatial_extent
    assert params["aggregate_spatial_geometries"] == expected_geometry

def test_execute_mask_same_source(api):
    api.check_result({
      "process_graph": {
        "loadcollection1": {
          "arguments": {
            "bands": [
              "B02",
              "B03"
            ],
            "id": "S2_FOOBAR",
            "properties": {
              "polarization": {
                "process_graph": {
                  "eq1": {
                    "arguments": {
                      "x": {
                        "from_parameter": "value"
                      },
                      "y": "DV"
                    },
                    "process_id": "eq",
                    "result": True
                  }
                }
              }
            },
            "spatial_extent": {
              "crs": "epsg:4326",
              "east": -73.90597343,
              "north": 4.724080996,
              "south": 4.68986451,
              "west": -74.0681076
            },
            "temporal_extent": [
              "2021-06-01",
              "2021-09-01"
            ]
          },
          "process_id": "load_collection"
        },
        "mask1": {
          "arguments": {
            "data": {
              "from_node": "renamelabels1"
            },
            "mask": {
              "from_node": "reducedimension1"
            }
          },
          "process_id": "mask"
        },
        "reducedimension1": {
          "arguments": {
            "data": {
              "from_node": "renamelabels1"
            },
            "dimension": "bands",
            "reducer": {
              "process_graph": {
                "arrayelement1": {
                  "arguments": {
                    "data": {
                      "from_parameter": "data"
                    },
                    "index": 2
                  },
                  "process_id": "array_element"
                },
                "eq2": {
                  "arguments": {
                    "x": {
                      "from_node": "arrayelement1"
                    },
                    "y": 2
                  },
                  "process_id": "eq",
                  "result": True
                }
              }
            }
          },
          "process_id": "reduce_dimension"
        },
        "renamelabels1": {
          "arguments": {
            "data": {
              "from_node": "sarbackscatter1"
            },
            "dimension": "bands",
            "target": [
              "VH",
              "VV",
              "mask",
              "incidence_angle"
            ]
          },
          "process_id": "rename_labels"
        },
        "sarbackscatter1": {
          "arguments": {
            "coefficient": "gamma0-terrain",
            "contributing_area": False,
            "data": {
              "from_node": "loadcollection1"
            },
            "elevation_model": "COPERNICUS_30",
            "ellipsoid_incidence_angle": False,
            "local_incidence_angle": False,
            "mask": True,
            "noise_removal": True
          },
          "process_id": "sar_backscatter"
        },
        "saveresult1": {
          "arguments": {
            "data": {
              "from_node": "mask1"
            },
            "format": "netCDF",
            "options": {}
          },
          "process_id": "save_result",
          "result": True
        }
      }
    })
    load_collections = dummy_backend.all_load_collection_calls("S2_FOOBAR")
    assert len(load_collections) == 1
    assert dummy_backend.get_collection("S2_FOOBAR").mask.call_count == 1

def test_complex_graph(api):
    api.check_result("complex_graph.json")
    load_collections = dummy_backend.all_load_collection_calls("SENTINEL1_GRD")
    load_collections_s2 = dummy_backend.all_load_collection_calls("SENTINEL2_L2A_SENTINELHUB")
    assert len(load_collections) == 1
    assert len(load_collections_s2) == 2

def test_mask_polygon(api):
    api.check_result("mask_polygon.json")
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    assert dummy.mask_polygon.call_count == 1
    args, kwargs = dummy.mask_polygon.call_args
    assert isinstance(kwargs['mask'], shapely.geometry.Polygon)


@pytest.mark.parametrize(["mask", "expected"], [
    ({"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 0)]]}, shapely.geometry.Polygon),
    (
            {
                "type": "MultiPolygon",
                "coordinates": [
                    [[(0, 0), (1, 0), (1, 1), (0, 0)]],
                    [[(2, 0), (3, 0), (3, 1), (2, 0)]],
                ]
            },
            shapely.geometry.MultiPolygon
    ),
    (
            {
                "type": "GeometryCollection",
                "geometries": [
                    {"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 0)]]},
                    {"type": "Polygon", "coordinates": [[(2, 0), (3, 0), (3, 1), (2, 0)]]},
                ]
            },
            shapely.geometry.MultiPolygon
    ),
    (
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 0)]]}, },
            shapely.geometry.Polygon
    ),
    (
            {
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": [
                    [[(0, 0), (1, 0), (1, 1), (0, 0)]],
                    [[(2, 0), (3, 0), (3, 1), (2, 0)]],
                ]},
            },
            shapely.geometry.MultiPolygon,
    ),
    (
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 0)]]},
                    },
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "MultiPolygon",
                            "coordinates": [
                                [[(0, 0), (1, 0), (1, 1), (0, 0)]],
                                [[(2, 0), (3, 0), (3, 1), (2, 0)]]
                            ]
                        }
                    }
                ]
            },
            shapely.geometry.MultiPolygon
    ),
])
def test_mask_polygon_types(api100, mask, expected):
    pg = {
        "lc1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "mask1": {"process_id": "mask_polygon", "arguments": {
            "data": {"from_node": "lc1"},
            "mask": mask
        }, "result": True}
    }
    api100.check_result(pg)
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.mask_polygon.call_count == 1
    args, kwargs = dummy.mask_polygon.call_args
    assert isinstance(kwargs['mask'], expected)


def test_mask_polygon_vector_cube(api100):
    path = str(get_path("geojson/FeatureCollection02.json"))
    pg = {
        "lc1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "lf": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": [path], "format": "GeoJSON"},
        },
        "mask": {
            "process_id": "mask_polygon",
            "arguments": {"data": {"from_node": "lc1"}, "mask": {"from_node": "lf"}},
            "result": True
        }
    }
    api100.check_result(pg)
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.mask_polygon.call_count == 1
    args, kwargs = dummy.mask_polygon.call_args
    assert isinstance(kwargs['mask'], shapely.geometry.MultiPolygon)


def test_aggregate_temporal_period(api100):
    api100.check_result("aggregate_temporal_period_max.json")


def test_aggregate_temporal_max(api):
    api.check_result("aggregate_temporal_max.json")


def test_aggregate_temporal_max_legacy_client(api):
    api.check_result(
        "aggregate_temporal_max.json",
        preprocess=preprocess_check_and_replace('"dimension": "t"', '"dimension": "temporal"')
    )


def test_aggregate_temporal_max_invalid_temporal_dimension(api):
    resp = api.result(
        "aggregate_temporal_max.json",
        preprocess=preprocess_check_and_replace('"dimension": "t"', '"dimension": "detijd"')
    )
    resp.assert_error(
        400, 'ProcessParameterInvalid',
        message="The value passed for parameter 'dimension' in process 'aggregate_temporal' is invalid: got 'detijd', but should be one of ['x', 'y', 't']"
    )


def test_aggregate_temporal_max_no_dimension(api):
    api.check_result(
        "aggregate_temporal_max.json",
        preprocess=preprocess_check_and_replace('"dimension": "t"', '"dimension": null')
    )


def test_aggregate_spatial(api):
    resp = api.check_result("aggregate_spatial.json")
    assert resp.json == {
        "2015-07-06T00:00:00Z": [[2.345]],
        "2015-08-22T00:00:00Z": [[None]]
    }
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params["spatial_extent"] == {
        "west": 7.02,
        "south": 51.29,
        "east": 7.65,
        "north": 51.75,
        "crs": "EPSG:4326",
    }
    assert params["aggregate_spatial_geometries"] == DriverVectorCube.from_geojson(
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [7.02, 51.75],
                    [7.65, 51.74],
                    [7.65, 51.29],
                    [7.04, 51.31],
                    [7.02, 51.75],
                ]
            ],
        }
    )


def test_execute_aggregate_spatial_spatial_cube(api100):
    resp = api100.check_result("aggregate_spatial_spatial_cube.json")
    assert resp.json == [[2.345, None], [2.0, 3.0]]


@pytest.mark.parametrize(["geometries", "expected"], [
    ("some text", "Invalid type: <class 'str'> ('some text')"),
    (1234, "Invalid type: <class 'int'> (1234)"),
    (["a", "list"], "Invalid type: <class 'list'> (['a', 'list'])")
])
def test_aggregate_spatial_invalid_geometry(api100, geometries, expected):
    pg = api100.load_json("aggregate_spatial.json")
    assert pg["aggregate_spatial"]["arguments"]["geometries"]
    pg["aggregate_spatial"]["arguments"]["geometries"] = geometries
    _ = api100.result(pg).assert_error(400, "ProcessParameterInvalid", expected)


@pytest.mark.parametrize(["feature_collection_test_path"], [
    ["geojson/FeatureCollection02.json"],
    ["geojson/FeatureCollection05.json"]
])
def test_aggregate_spatial_vector_cube_basic(api100, feature_collection_test_path):
    path = get_path(feature_collection_test_path)
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR", "bands": ["B02", "B03", "B04"]}},
        "lf": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": [str(path)], "format": "GeoJSON"},
        },
        "ag": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "lc"},
                "geometries": {"from_node": "lf"},
                "reducer": {"process_graph": {
                    "mean": {"process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}, "result": True}}
                }
            },
            "result": True
        }
    }
    res = api100.check_result(pg)

    params = dummy_backend.last_load_collection_call("S2_FOOBAR")
    assert params["spatial_extent"] == {"west": 1, "south": 1, "east": 5, "north": 4, "crs": "EPSG:4326"}
    assert isinstance(params["aggregate_spatial_geometries"], DriverVectorCube)

    assert res.json == DictSubSet(
        {
            "type": "FeatureCollection",
            "features": [
                DictSubSet(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[1, 1], [3, 1], [2, 3], [1, 1]]],
                        },
                        "properties": {
                            "id": "first",
                            "pop": 1234,
                            "agg~2015-07-06T00:00:00Z~B02": 2.345,
                            "agg~2015-07-06T00:00:00Z~B03": None,
                            "agg~2015-07-06T00:00:00Z~B04": 2.0,
                            "agg~2015-08-22T00:00:00Z~B02": 3.0,
                            "agg~2015-08-22T00:00:00Z~B03": 4.0,
                            "agg~2015-08-22T00:00:00Z~B04": 5.0,
                        },
                    }
                ),
                DictSubSet(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[4, 2], [5, 4], [3, 4], [4, 2]]],
                        },
                        "properties": {
                            "id": "second",
                            "pop": 5678,
                            "agg~2015-07-06T00:00:00Z~B02": 6.0,
                            "agg~2015-07-06T00:00:00Z~B03": 7.0,
                            "agg~2015-07-06T00:00:00Z~B04": 8.0,
                            "agg~2015-08-22T00:00:00Z~B02": 9.0,
                            "agg~2015-08-22T00:00:00Z~B03": 10.0,
                            "agg~2015-08-22T00:00:00Z~B04": 11.0,
                        },
                    }
                ),
            ],
        }
    )


@pytest.mark.parametrize(["info", "preprocess_pg", "aggregate_data", "p1_properties", "p2_properties"], [
    (
            "time-and-bands",
            {},
            "lc",
            {
                "id": "first",
                "pop": 1234,
                "agg~2015-07-06T00:00:00Z~B02": 2.345,
                "agg~2015-07-06T00:00:00Z~B03": None,
                "agg~2015-07-06T00:00:00Z~B04": 2,
                "agg~2015-08-22T00:00:00Z~B02": 3,
                "agg~2015-08-22T00:00:00Z~B03": 4,
                "agg~2015-08-22T00:00:00Z~B04": 5,
            },
            {
                "id": "second", "pop": 5678,
                "agg~2015-07-06T00:00:00Z~B02": 6, "agg~2015-07-06T00:00:00Z~B03": 7, "agg~2015-07-06T00:00:00Z~B04": 8,
                "agg~2015-08-22T00:00:00Z~B02": 9, "agg~2015-08-22T00:00:00Z~B03": 10, "agg~2015-08-22T00:00:00Z~B04": 11,
            },
    ),
    (
            "no-time",
            {
                "r": {"process_id": "reduce_dimension", "arguments": {
                    "data": {"from_node": "lc"},
                    "dimension": "t",
                    "reducer": {"process_graph": {"mean": {
                        "process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}, "result": True,
                    }}},
                }},
            },
            "r",
            {
                "id": "first",
                "pop": 1234,
                "agg~B02": 2.345,
                "agg~B03": None,
                "agg~B04": 2,
            },
            {"id": "second", "pop": 5678, "agg~B02": 3, "agg~B03": 4, "agg~B04": 5},
    ),
    (
            "no-bands",
            {
                "r": {"process_id": "reduce_dimension", "arguments": {
                    "data": {"from_node": "lc"},
                    "dimension": "bands",
                    "reducer": {"process_graph": {"mean": {
                        "process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}, "result": True,
                    }}},
                }}
            },
            "r",
            {
                "id": "first",
                "pop": 1234,
                "agg~2015-07-06T00:00:00Z": 2.345,
                "agg~2015-08-22T00:00:00Z": None,
            },
            {
                "id": "second",
                "pop": 5678,
                "agg~2015-07-06T00:00:00Z": 2,
                "agg~2015-08-22T00:00:00Z": 3,
            },
        ),
        (
            "no-time-nor-bands",
            {
                "r1": {"process_id": "reduce_dimension", "arguments": {
                    "data": {"from_node": "lc"},
                    "dimension": "t",
                    "reducer": {"process_graph": {"mean": {
                        "process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}, "result": True,
                    }}},
                }},
                "r2": {"process_id": "reduce_dimension", "arguments": {
                    "data": {"from_node": "r1"},
                    "dimension": "bands",
                    "reducer": {"process_graph": {"mean": {
                        "process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}, "result": True,
                    }}},
                }},
            },
            "r2",
            {"id": "first", "pop": 1234, "agg": 2.345},
            {"id": "second", "pop": 5678, "agg": None},
    ),
])
def test_aggregate_spatial_vector_cube_dimensions(
        api100, info, preprocess_pg, aggregate_data, p1_properties, p2_properties
):
    path = get_path("geojson/FeatureCollection02.json")
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR", "bands": ["B02", "B03", "B04"]}},
        "lf": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": [str(path)], "format": "GeoJSON"},
        },
        "ag": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": aggregate_data},
                "geometries": {"from_node": "lf"},
                "reducer": {"process_graph": {
                    "mean": {"process_id": "mean", "arguments": {"data": {"from_parameter": "data"}}, "result": True}}
                }
            },
            "result": True
        }
    }
    pg.update(preprocess_pg)
    res = api100.check_result(pg)

    params = dummy_backend.last_load_collection_call("S2_FOOBAR")
    assert params["spatial_extent"] == {"west": 1, "south": 1, "east": 5, "north": 4, "crs": "EPSG:4326"}
    assert isinstance(params["aggregate_spatial_geometries"], DriverVectorCube)

    assert res.json == DictSubSet({
        "type": "FeatureCollection",
        "features": [
            DictSubSet({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[1, 1], [3, 1], [2, 3], [1, 1]]]},
                "properties": p1_properties,
            }),
            DictSubSet({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[4, 2], [5, 4], [3, 4], [4, 2]]]},
                "properties": p2_properties,
            }),
        ]
    })


def test_create_wmts_040(api040):
    api040.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    process_graph = api040.load_json("filter_temporal.json")
    post_data = {
        "type": 'WMTS',
        "process_graph": process_graph,
        "custom_param": 45,
        "title": "My Service",
        "description": "Service description"
    }
    resp = api040.post('/services', json=post_data).assert_status_code(201)
    assert resp.headers['OpenEO-Identifier'] == 'c63d6c27-c4c2-4160-b7bd-9e32f582daec'
    assert resp.headers['Location'].endswith("/services/c63d6c27-c4c2-4160-b7bd-9e32f582daec")


def test_create_wmts_100(api100):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    process_graph = api100.load_json("filter_temporal.json")
    post_data = {
        "type": 'WMTS',
        "process": {
            "process_graph": process_graph,
            "id": "filter_temporal_wmts"
        },
        "custom_param": 45,
        "title": "My Service",
        "description": "Service description"
    }
    resp = api100.post('/services', json=post_data).assert_status_code(201)
    assert resp.headers['OpenEO-Identifier'] == 'c63d6c27-c4c2-4160-b7bd-9e32f582daec'
    assert resp.headers['Location'].endswith("/services/c63d6c27-c4c2-4160-b7bd-9e32f582daec")


def test_read_vector(api):
    geometry_filename = str(get_path("geojson/GeometryCollection01.json"))
    process_graph = api.load_json(
        "read_vector.json",
        preprocess=preprocess_check_and_replace("PLACEHOLDER", geometry_filename)
    )
    resp = api.check_result(process_graph)
    assert b'NaN' not in resp.data
    assert resp.json == {"2015-07-06T00:00:00Z": [[2.345], [2.345]], "2015-08-22T00:00:00Z": [[None], [None]]}
    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == {"west": 5, "south": 51, "east": 6, "north": 52, "crs": 'EPSG:4326'}
    assert params["temporal_extent"] == ('2017-11-21', '2017-12-21')
    assert params["aggregate_spatial_geometries"] == DelayedVector(geometry_filename)


def test_read_vector_no_load_collection_spatial_extent(api):
    geometry_filename = str(get_path("geojson/GeometryCollection01.json"))
    preprocess1 = preprocess_check_and_replace("PLACEHOLDER", geometry_filename)
    preprocess2 = preprocess_regex_check_and_replace(r'"spatial_extent"\s*:\s*\{.*?\},', replacement='')
    process_graph = api.load_json(
        "read_vector.json", preprocess=lambda s: preprocess2(preprocess1(s))
    )
    resp = api.check_result(process_graph)
    assert b'NaN' not in resp.data
    assert resp.json == {"2015-07-06T00:00:00Z": [[2.345], [2.345]], "2015-08-22T00:00:00Z": [[None], [None]]}
    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == {"west": 5.05, "south": 51.21, "east": 5.15, "north": 51.3, "crs": 'EPSG:4326'}
    assert params["temporal_extent"] == ('2017-11-21', '2017-12-21')
    assert params["aggregate_spatial_geometries"] == DelayedVector(geometry_filename)


@pytest.mark.parametrize("udf_code", [
    """
        from openeo_udf.api.datacube import DataCube  # Old style openeo_udf API
        def fct_buffer(udf_data: UdfData):
            return udf_data
    """,
    """
        from openeo.udf import UdfData
        def fct_buffer(udf_data: UdfData):
            return udf_data
    """,
])
def test_run_udf_on_vector(api100, udf_code):
    udf_code = textwrap.dedent(udf_code)
    process_graph = {
        "geojson_file": {
            "process_id": "read_vector",
            "arguments": {"filename": str(get_path("geojson/GeometryCollection01.json"))},
        },
        "udf": {
            "process_id": "run_udf",
            "arguments": {
                "data": {"from_node": "geojson_file"},
                "udf": udf_code,
                "runtime": "Python",
            },
            "result": "true"
        }
    }
    resp = api100.check_result(process_graph)
    print(resp.json)
    assert len(resp.json) == 2
    assert resp.json[0]['type'] == 'Polygon'


@pytest.mark.parametrize("udf_code", [
    """
        from openeo_udf.api.udf_data import UdfData  # Old style openeo_udf API
        from openeo_udf.api.structured_data import StructuredData  # Old style openeo_udf API
        def fct_buffer(udf_data: UdfData):
            data = udf_data.get_structured_data_list()
            udf_data.set_structured_data_list([
                StructuredData(description='res', data={'len': len(s.data), 'keys': s.data.keys(), 'values': s.data.values()}, type='dict') 
                for s in data
            ])
    """,
    """
        from openeo.udf import UdfData, StructuredData
        def fct_buffer(udf_data: UdfData):
            data = udf_data.get_structured_data_list()
            udf_data.set_structured_data_list([
                StructuredData(description='res', data={'len': len(s.data), 'keys': s.data.keys(), 'values': s.data.values()}, type='dict') 
                for s in data
            ])
    """,
])
def test_run_udf_on_json(api100, udf_code):
    udf_code = textwrap.dedent(udf_code)
    process_graph = api100.load_json(
        "run_udf_on_timeseries.json",
        preprocess=lambda s: s.replace('"PLACEHOLDER_UDF"', repr(udf_code))
    )
    resp = api100.check_result(process_graph)
    assert resp.json == {
        "len": 2,
        "keys": ["2015-07-06T00:00:00Z", "2015-08-22T00:00:00Z"],
        "values": [[[2.345, None]], [[2.0, 3.0]]],
    }


@pytest.mark.parametrize("udf_code", [
    """
        from openeo_udf.api.udf_data import UdfData  # Old style openeo_udf API
        from openeo_udf.api.structured_data import StructuredData   # Old style openeo_udf API
        def transform(data: UdfData) -> UdfData:
            res = [
                StructuredData(description="res", data=[x * x for x in sd.data], type="list")
                for sd in data.get_structured_data_list()
            ]
            data.set_structured_data_list(res)
    """,
    """
        from openeo.udf import UdfData, StructuredData
        def transform(data: UdfData) -> UdfData:
            res = [
                StructuredData(description="res", data=[x * x for x in sd.data], type="list")
                for sd in data.get_structured_data_list()
            ]
            data.set_structured_data_list(res)
    """,
])
def test_run_udf_on_list(api100, udf_code):
    udf_code = textwrap.dedent(udf_code)
    process_graph = {
        "udf": {
            "process_id": "run_udf",
            "arguments": {
                "data": [1, 2, 3, 5, 8],
                "udf": udf_code,
                "runtime": "Python"
            },
            "result": True
        }
    }
    resp = api100.check_result(process_graph)
    assert resp.json == [1, 4, 9, 25, 64]


@pytest.mark.parametrize(["runtime", "version", "failure"], [
    ("Python", None, None),
    ("pYthOn", None, None),
    ("Python", "3", None),
    ("Python", CURRENT_PY3x, None),
    ("Python", "2", (
            "InvalidVersion",
            re.compile(r"Unsupported UDF runtime version Python '2'. Should be one of \['3', '3\.\d+'.* or null")
    )),
    ("Python", "1.2.3", (
            "InvalidVersion",
            re.compile(r"Unsupported UDF runtime version Python '1.2.3'. Should be one of \['3', '3\.\d+'.* or null"),
    )),
    ("Python-Jep", None, None),
    ("Python-Jep", "3", None),
    ("meh", CURRENT_PY3x, ("InvalidRuntime", "Unsupported UDF runtime 'meh'. Should be one of ['Python', 'Python-Jep']")),
    (None, CURRENT_PY3x, ("InvalidRuntime", "Unsupported UDF runtime None. Should be one of ['Python', 'Python-Jep']"),
     ),
])
def test_run_udf_on_list_runtimes(api100, runtime, version, failure):
    udf_code = textwrap.dedent("""
        from openeo.udf import UdfData, StructuredData
        def transform(data: UdfData) -> UdfData:
            res = [
                StructuredData(description="res", data=[x * x for x in sd.data], type="list")
                for sd in data.get_structured_data_list()
            ]
            data.set_structured_data_list(res)
    """)
    process_graph = {
        "udf": {
            "process_id": "run_udf",
            "arguments": {
                "data": [1, 2, 3, 5, 8],
                "udf": udf_code,
                "runtime": runtime,
                "version": version
            },
            "result": True
        }
    }
    resp = api100.result(process_graph)
    if failure:
        error_code, message = failure
        resp.assert_error(400, error_code=error_code, message=message)
    else:
        assert resp.assert_status_code(200).json == [1, 4, 9, 25, 64]



def test_process_reference_as_argument(api100):
    process_graph = api100.load_json(
        "process_reference_as_argument.json"
    )
    resp = api100.check_result(process_graph)
    print(resp.json)


def test_load_collection_without_spatial_extent_incorporates_read_vector_extent(api):
    process_graph = api.load_json(
        "read_vector_spatial_extent.json",
        preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("geojson/GeometryCollection01.json")))
    )
    resp = api.check_result(process_graph)
    assert b'NaN' not in resp.data
    assert resp.json == {
        "2015-07-06T00:00:00Z": [[2.345], [2.345]],
        "2015-08-22T00:00:00Z": [[None], [None]]
    }
    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == {"west": 5.05, "south": 51.21, "east": 5.15, "north": 51.3, "crs": 'EPSG:4326'}


def test_read_vector_from_feature_collection(api):
    process_graph = api.load_json(
        "read_vector_feature_collection.json",
        preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("geojson/FeatureCollection01.json")))
    )
    resp = api.check_result(process_graph)
    assert b'NaN' not in resp.data
    assert resp.json == {
        "2015-07-06T00:00:00Z": [[2.345], [2.345]],
        "2015-08-22T00:00:00Z": [[None], [None]]
    }
    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == {"west": 5, "south": 51, "east": 6, "north": 52, "crs": 'EPSG:4326'}


class TestVectorCubeLoading:

    @pytest.mark.parametrize("add_save_result", [False, True])
    def test_geojson_feature_collection(self, api, add_save_result):
        """Load vector cube from local feature collection GeoJSON file."""
        path = str(get_path("geojson/FeatureCollection02.json"))
        pg = {"lf": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": [path], "format": "GeoJSON"},
        }}
        if add_save_result:
            pg["sr"] = {
                "process_id": "save_result",
                "arguments": {"data": {"from_node": "lf"}, "format": "GeoJSON"},
            }
        pg["sr" if add_save_result else "lf"]["result"] = True
        resp = api.check_result(pg)
        assert resp.headers["Content-Type"] == "application/geo+json"
        assert resp.json == DictSubSet({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature", "properties": {"id": "first", "pop": 1234},
                    "geometry": {"type": "Polygon", "coordinates": [[[1, 1], [3, 1], [2, 3], [1, 1]]]}
                },
                {
                    "type": "Feature", "properties": {"id": "second", "pop": 5678},
                    "geometry": {"type": "Polygon", "coordinates": [[[4, 2], [5, 4], [3, 4], [4, 2]]]}
                },
            ]
        })

    @pytest.mark.parametrize(["path", "expected_features"], [
        (
                "geojson/Polygon01.json",
                [DictSubSet({"type": "Feature", "geometry": DictSubSet({"type": "Polygon"})})],
        ),
        (
                "geojson/MultiPolygon01.json",
                [DictSubSet({"type": "Feature", "geometry": DictSubSet({"type": "MultiPolygon"})})],
        ),
        (
                "geojson/GeometryCollection01.json",
                [DictSubSet({"type": "Feature", "geometry": DictSubSet({"type": "GeometryCollection"})})],
        ),
        (
                "geojson/Feature01.json",
                [DictSubSet({"type": "Feature", "geometry": DictSubSet({"type": "Polygon"})})],
        ),
        (
                "geojson/FeatureCollection01.json", [
                    DictSubSet({"type": "Feature", "geometry": DictSubSet({"type": "Polygon"})}),
                    DictSubSet({"type": "Feature", "geometry": DictSubSet({"type": "Polygon"})}),
                ],
        ),
    ])
    def test_geojson_types(self, api, path, expected_features):
        """Load vector cube from GeoJSON types"""
        pg = {"lf": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": [str(get_path(path))], "format": "GeoJSON"},
            "result": True,
        }}
        resp = api.check_result(pg)
        assert resp.headers["Content-Type"] == "application/geo+json"
        assert resp.json == DictSubSet({
            "type": "FeatureCollection",
            "features": expected_features
        })

    def test_geojson_url(self, api, urllib_mock):
        """Load vector cube from GeoJSON URL"""
        urllib_mock.get(
            "https://a.test/features.geojson",
            data=get_path("geojson/FeatureCollection02.json").read_bytes()
        )

        pg = {"lf": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": ["https://a.test/features.geojson"], "format": "GeoJSON"},
            "result": True,
        }}
        resp = api.check_result(pg)
        assert resp.headers["Content-Type"] == "application/geo+json"
        assert resp.json == DictSubSet({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature", "properties": {"id": "first", "pop": 1234},
                    "geometry": {"type": "Polygon", "coordinates": [[[1, 1], [3, 1], [2, 3], [1, 1]]]}
                },
                {
                    "type": "Feature", "properties": {"id": "second", "pop": 5678},
                    "geometry": {"type": "Polygon", "coordinates": [[[4, 2], [5, 4], [3, 4], [4, 2]]]}
                },
            ]
        })

    @pytest.mark.parametrize(["path", "format"], [
        ("geojson/mol.json", "GeoJSON"),
        ("shapefile/mol.shp", "ESRI Shapefile"),
        ("gpkg/mol.gpkg", "GPKG"),
    ])
    def test_local_vector_file(self, api, path, format):
        """Load vector cube from local vector file"""
        path = str(get_path(path))
        pg = {"lf": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": [path], "format": format},
            "result": True,
        }}
        resp = api.check_result(pg)
        assert resp.headers["Content-Type"] == "application/geo+json"
        assert resp.json == DictSubSet({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature", "properties": {"id": 23, "name": "Mol", "class": 4},
                    "geometry": DictSubSet({"type": "Polygon"}),
                },
                {
                    "type": "Feature", "properties": {"id": 58, "name": "TAP", "class": 5},
                    "geometry": DictSubSet({"type": "Polygon"}),
                },
            ]
        })

    @pytest.mark.parametrize(["path", "format"], [
        ("geojson/mol.json", "GeoJSON"),
        ("gpkg/mol.gpkg", "GPKG"),
    ])
    def test_vector_url(self, api, path, format, urllib_mock):
        """Load vector cube from URL"""
        path = get_path(path)
        url = f"https://a.test/{path.name}"
        urllib_mock.get(url, data=path.read_bytes())

        pg = {"lf": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": [url], "format": format},
            "result": True,
        }}
        resp = api.check_result(pg)
        assert resp.headers["Content-Type"] == "application/geo+json"
        assert resp.json == DictSubSet({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature", "properties": {"id": 23, "name": "Mol", "class": 4},
                    "geometry": DictSubSet({"type": "Polygon"}),
                },
                {
                    "type": "Feature", "properties": {"id": 58, "name": "TAP", "class": 5},
                    "geometry": DictSubSet({"type": "Polygon"}),
                },
            ]
        })

    def _zip_content(self, paths: Iterable[Path]) -> bytes:
        """Zip given files to an in-memory ZIP file"""
        with BytesIO() as bytes_io:
            with ZipFile(bytes_io, mode="w") as zip_file:
                for path in paths:
                    zip_file.writestr(path.name, path.read_bytes())
            return bytes_io.getvalue()

    def test_shapefile_url(self, api, urllib_mock):
        """Load vector cube from shapefile (zip) URL"""
        zip_bytes = self._zip_content(get_path("shapefile").glob("mol.*"))
        urllib_mock.get(f"https://a.test/geom.shp.zip", data=zip_bytes)
        pg = {"lf": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": ["https://a.test/geom.shp.zip"], "format": "ESRI Shapefile"},
            "result": True,
        }}
        resp = api.check_result(pg)
        assert resp.headers["Content-Type"] == "application/geo+json"
        assert resp.json == DictSubSet({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature", "properties": {"id": 23, "name": "Mol", "class": 4},
                    "geometry": DictSubSet({"type": "Polygon"}),
                },
                {
                    "type": "Feature", "properties": {"id": 58, "name": "TAP", "class": 5},
                    "geometry": DictSubSet({"type": "Polygon"}),
                },
            ]
        })

    @pytest.mark.parametrize(["output_format", "content_type", "data_prefix"], [
        ("GeoJson", "application/geo+json", b"{"),
        ("ESRI Shapefile", "application/zip", b"PK\x03\x04"),
        ("GPKG", "application/geopackage+sqlite3", b"SQLite format 3"),
    ])
    def test_vector_save_result(self, api, output_format, content_type, data_prefix, tmp_path):
        path = str(get_path("geojson/FeatureCollection02.json"))
        pg = {
            "lf": {
                "process_id": "load_uploaded_files",
                "arguments": {"paths": [path], "format": "GeoJSON"},
            },
            "sr": {
                "process_id": "save_result",
                "arguments": {"data": {"from_node": "lf"}, "format": output_format,
                              "options": {"zip_multi_file": True}},
                "result": True,
            }
        }
        resp = api.check_result(pg)
        format_info = IOFORMATS.get(output_format)
        assert resp.headers["Content-Type"] == content_type
        assert resp.data.startswith(data_prefix)

        download = tmp_path / f"download.{format_info.extension}{'.zip' if 'zip' in content_type else ''}"
        download.write_bytes(resp.data)

        df = gpd.read_file(download, driver=format_info.fiona_driver)
        assert list(df.columns) == ["id", "pop", "geometry"]
        assert list(df["id"]) == ["first", "second"]
        for geometry, expected in zip(df["geometry"], [(1, 1, 3, 3), (3, 2, 5, 4)]):
            assert isinstance(geometry, shapely.geometry.Polygon)
            assert geometry.bounds == expected


def test_no_nested_JSONResult(api):
    api.set_auth_bearer_token()
    api.post(
        path="/result",
        json=api.load_json("no_nested_json_result.json"),
    ).assert_status_code(200).assert_content()


def test_load_disk_data(api):
    api.check_result("load_disk_data.json")
    params = dummy_backend.last_load_collection_call(
        "/data/MTDA/CGS_S2/CGS_S2_FAPAR/2019/04/24/*/*/10M/*_FAPAR_10M_V102.tif"
    )
    assert params["spatial_extent"] == {"west": 3, "south": 50, "east": 6, "north": 51, "crs": "EPSG:4326"}


def test_mask_with_vector_file(api):
    process_graph = api.load_json(
        "mask_with_vector_file.json",
        preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("geojson/MultiPolygon02.json")))
    )
    api.check_result(process_graph)


def test_aggregate_feature_collection(api):
    api.check_result("aggregate_feature_collection.json")
    params = dummy_backend.last_load_collection_call('S2_FOOBAR')
    assert params["spatial_extent"] == {"west": 5, "south": 51, "east": 6, "north": 52, "crs": 'EPSG:4326'}


def test_aggregate_feature_collection_no_load_collection_spatial_extent(api):
    preprocess = preprocess_regex_check_and_replace(r'"spatial_extent"\s*:\s*\{.*?\},', replacement='')
    api.check_result("aggregate_feature_collection.json", preprocess=preprocess)
    params = dummy_backend.last_load_collection_call('S2_FOOBAR')
    assert params["spatial_extent"] == {
        "west": 5.076, "south": 51.21, "east": 5.166, "north": 51.26, "crs": 'EPSG:4326'
    }


@pytest.mark.parametrize("auth", [True, False])
def test_post_result_process_100(api100, client, auth):
    if auth:
        api100.set_auth_bearer_token()
    response = api100.post(
        path='/result',
        json=api100.get_process_graph_dict(api100.load_json("basic.json")),
    )
    if auth:
        response.assert_status_code(200).assert_content()
    else:
        response.assert_error(401, "AuthenticationRequired")


@pytest.mark.parametrize("body", [
    {"foo": "meh"},
    {"process": "meh"},
    {"process_graph": "meh"},
])
def test_missing_process_graph(api, body):
    api.set_auth_bearer_token()
    response = api.post(path='/result', json=body)
    response.assert_error(status_code=ProcessGraphMissingException.status_code, error_code='ProcessGraphMissing')


@pytest.mark.parametrize("body", [
    {"process": {"process_graph": "meh"}},
    {"process": {"process_graph": [1, 2, 3]}},
])
def test_invalid_process_graph(api, body):
    api.set_auth_bearer_token()
    response = api.post(path='/result', json=body)
    response.assert_error(status_code=ProcessGraphInvalidException.status_code, error_code='ProcessGraphInvalid')


def test_fuzzy_mask(api):
    api.check_result("fuzzy_mask.json")


def test_fuzzy_mask_parent_scope(api100):
    api100.check_result(
        "fuzzy_mask.json",
        preprocess=preprocess_check_and_replace('"from_parameter": "x"', '"from_parameter": "data"')
    )


def test_fuzzy_mask_add_dim(api):
    api.check_result("fuzzy_mask_add_dim.json")


def test_rename_labels(api100):
    api100.check_result("rename_labels.json")


@pytest.mark.parametrize("namespace", ["user", None, "_undefined"])
def test_user_defined_process_bbox_mol_basic(api100, namespace, udp_registry):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    bbox_mol_spec = api100.load_json("udp/bbox_mol.json")
    udp_registry.save(user_id=TEST_USER, process_id="bbox_mol", spec=bbox_mol_spec)
    pg = api100.load_json("udp_bbox_mol_basic.json")
    if namespace != "_undefined":
        pg["bboxmol1"]["namespace"] = namespace
    elif "namespace" in pg["bboxmol1"]:
        del pg["bboxmol1"]["namespace"]
    api100.check_result(pg)
    params = dummy_backend.last_load_collection_call('S2_FOOBAR')
    assert params["spatial_extent"] == {"west": 5.05, "south": 51.2, "east": 5.1, "north": 51.23, "crs": 'EPSG:4326'}


@pytest.mark.parametrize("namespace", ["backend", "foobar"])
def test_user_defined_process_bbox_mol_basic_other_namespace(api100, udp_registry, namespace):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    bbox_mol_spec = api100.load_json("udp/bbox_mol.json")
    udp_registry.save(user_id=TEST_USER, process_id="bbox_mol", spec=bbox_mol_spec)
    pg = api100.load_json("udp_bbox_mol_basic.json")
    pg["bboxmol1"]["namespace"] = namespace
    api100.result(pg).assert_error(status_code=400, error_code="ProcessUnsupported")


@pytest.mark.parametrize(["udp_args", "expected_start_date", "expected_end_date"], [
    ({}, "2019-01-01", None),
    ({"start_date": "2019-12-12"}, "2019-12-12", None),
    ({"end_date": "2019-12-12"}, "2019-01-01", "2019-12-12"),
    ({"start_date": "2019-08-08", "end_date": "2019-12-12"}, "2019-08-08", "2019-12-12"),
])
def test_user_defined_process_date_window(
        api100, udp_registry, udp_args, expected_start_date, expected_end_date
):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api100.load_json("udp/date_window.json")
    udp_registry.save(user_id=TEST_USER, process_id="date_window", spec=spec)

    pg = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR"}
        },
        "datewindow": {
            "process_id": "date_window",
            "namespace": "user",
            "arguments": dict(
                data={"from_node": "loadcollection1"},
                **udp_args
            ),
            "result": True
        }
    }

    api100.check_result(pg)
    params = dummy_backend.last_load_collection_call('S2_FOOBAR')
    assert params["temporal_extent"] == (expected_start_date, expected_end_date)


def test_user_defined_process_required_parameter(api100, udp_registry):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api100.load_json("udp/date_window.json")
    udp_registry.save(user_id=TEST_USER, process_id="date_window", spec=spec)

    pg = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR"}
        },
        "bboxmol1": {
            "process_id": "date_window",
            "namespace": "user",
            "arguments": {"start_date": "2019-08-08", "end_date": "2019-12-12"},
            "result": True
        }
    }

    response = api100.result(pg)
    response.assert_error(400, "ProcessParameterRequired", message="parameter 'data' is required")


@pytest.mark.parametrize("set_parameter", [False, True])
def test_udp_udf_reduce_dimension(api100, udp_registry, set_parameter):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api100.load_json("udp/udf_reduce_dimension.json")
    udp_registry.save(user_id=TEST_USER, process_id="udf_reduce_dimension", spec=spec)

    udp_args = {"data": {"from_node": "loadcollection1"}}
    if set_parameter:
        udp_args["udfparam"] = "test_the_udfparam"
    pg = {
        "loadcollection1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "udfreducedimension1": {
            "process_id": "udf_reduce_dimension", "namespace": "user", "arguments": udp_args, "result": True
        }
    }

    response = api100.result(pg).assert_status_code(200)
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.reduce_dimension.call_count == 1
    dummy.reduce_dimension.assert_called_with(reducer=mock.ANY, dimension="bands", context=None, env=mock.ANY)
    args, kwargs = dummy.reduce_dimension.call_args
    assert "runudf1" in kwargs["reducer"]
    env: EvalEnv = kwargs["env"]
    expected_param = "test_the_udfparam" if set_parameter else "udfparam_default"
    assert env.collect_parameters()["udfparam"] == expected_param


@pytest.mark.parametrize("set_parameter", [False, True])
def test_udp_apply_neighborhood(api100, udp_registry, set_parameter):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api100.load_json("udp/udf_apply_neighborhood.json")
    udp_registry.save(user_id=TEST_USER, process_id="udf_apply_neighborhood", spec=spec)

    udp_args = {"data": {"from_node": "loadcollection1"}}
    if set_parameter:
        udp_args["udfparam"] = "test_the_udfparam"
    pg = {
        "loadcollection1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "udfapplyneighborhood1": {
            "process_id": "udf_apply_neighborhood", "namespace": "user", "arguments": udp_args, "result": True
        }
    }

    response = api100.result(pg).assert_status_code(200)
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.apply_neighborhood.call_count == 1
    dummy.apply_neighborhood.assert_called_with(process=mock.ANY, size=mock.ANY, overlap=mock.ANY, env=mock.ANY)
    args, kwargs = dummy.apply_neighborhood.call_args
    assert "runudf1" in kwargs["process"]
    env: EvalEnv = kwargs["env"]
    expected_param = "test_the_udfparam" if set_parameter else "udfparam_default"
    assert env.collect_parameters()["udfparam"] == expected_param


def test_user_defined_process_udp_vs_pdp_priority(api100, udp_registry):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    # First without a defined "ndvi" UDP
    api100.check_result("udp_ndvi.json")
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.ndvi.call_count == 1
    dummy.ndvi.assert_called_with(nir=None, red=None, target_band=None)
    assert dummy.reduce_dimension.call_count == 0

    # Overload ndvi with UDP.
    udp_registry.save(user_id=TEST_USER, process_id="ndvi", spec=api100.load_json("udp/myndvi.json"))
    api100.check_result("udp_ndvi.json")
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.ndvi.call_count == 1
    assert dummy.reduce_dimension.call_count == 1
    dummy.reduce_dimension.assert_called_with(reducer=mock.ANY, dimension="bands", context=None, env=mock.ANY)
    args, kwargs = dummy.reduce_dimension.call_args
    assert "red" in kwargs["reducer"]
    assert "nir" in kwargs["reducer"]


def test_execute_03_style_filter_bbox(api):
    res = api.result({
    "loadcollection1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
    "filterbbox1": {
        "process_id": "filter_bbox",
        "arguments": {
            "data": {"from_node": "loadcollection1"},
            "west": 4.6511, "east": 4.6806, "north": 51.20859, "south": 51.18997, "crs": "epsg:4326"
        },
        "result": True}
    })
    res.assert_error(
        status_code=400, error_code="ProcessParameterRequired",
        message="Process 'filter_bbox' parameter 'extent' is required"
    )


def test_execute_03_style_filter_temporal(api):
    res = api.result({
    "loadcollection1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
    "filtertemporal1": {
        "process_id": "filter_temporal",
        "arguments": {
            "data": {"from_node": "loadcollection1"},
            "from": "2019-10-28", "to": "2019-10-28"
        },
        "result": True
    }})
    res.assert_error(
        status_code=400, error_code="ProcessParameterRequired",
        message="Process 'filter_temporal' parameter 'extent' is required"
    )


def test_sleep(api):
    with mock.patch("time.sleep") as sleep:
        r = api.check_result({
            "loadcollection1": {
                "process_id": "load_collection",
                "arguments": {"id": "S2_FOOBAR"}
            },
            "sleep1": {
                "process_id": "sleep",
                "arguments": {"data": {"from_node": "loadcollection1"}, "seconds": 5},
                "result": True
            }
        })
    sleep.assert_called_with(5)


def test_discard_result(api):
    res = api.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR"}
        },
        "discardresult1": {
            "process_id": "discard_result",
            "arguments": {"data": {"from_node": "loadcollection1"}},
            "result": True
        }
    })

    assert res.json is None


@pytest.mark.parametrize(["namespace", "url_mocks", "expected_error"], [
    (
            "https://oeo.test/u/42/udp/bbox_mol",
            {"https://oeo.test/u/42/udp/bbox_mol": "udp/bbox_mol.json"},
            None,
    ),
    (
            "https://oeo.test/u/42/udp/",
            {"https://oeo.test/u/42/udp/bbox_mol": "udp/bbox_mol.json"},
            None,
    ),
    (
            "https://oeo.test/u/42/udp/",
            {
                "https://oeo.test/u/42/udp/bbox_mol": 404,
                "https://oeo.test/u/42/udp/bbox_mol.json": "udp/bbox_mol.json",
            },
            None,
    ),
    (
            "https://share.example/u42/bbox_mol.json",
            {"https://share.example/u42/bbox_mol.json": "udp/bbox_mol.json"},
            None,
    ),
    (
            "https://share.test/u42/bbox_mol.json",
            {
                "https://share.test/u42/bbox_mol.json": (302, "https://shr976.test/45435"),
                "https://shr976.test/45435": "udp/bbox_mol.json",
            },
            None,
    ),
    (
            "https://oeo.test/u/42/udp/bbox_mol",
            {"https://oeo.test/u/42/udp/bbox_mol": 404},
            (
                    400, "ProcessUnsupported",
                    "'bbox_mol' is not available in namespace 'https://oeo.test/u/42/udp/bbox_mol'."
            ),
    ),
    (
            "https://oeo.test/u/42/udp/",
            {
                "https://oeo.test/u/42/udp/bbox_mol": 404,
                "https://oeo.test/u/42/udp/bbox_mol.json": 404,
            },
            (
                    400, "ProcessUnsupported",
                    "'bbox_mol' is not available in namespace 'https://oeo.test/u/42/udp/'."
            ),
    ),
    (
            "https://oeo.test/u/42/udp/bbox_mol",
            {"https://oeo.test/u/42/udp/bbox_mol": {"foo": "bar"}},
            (400, "ProcessGraphInvalid", "Invalid process graph specified."),
    ),
    (
            "https://oeo.test/u/42/udp/bbox_mol",
            {"https://oeo.test/u/42/udp/bbox_mol": '{"foo": invalid json'},
            (400, "ProcessGraphInvalid", "Invalid process graph specified."),
    ),
])
def test_evaluate_process_from_url(api100, requests_mock, namespace, url_mocks, expected_error):
    for url, value in url_mocks.items():
        if isinstance(value, str):
            if value.endswith(".json"):
                bbox_mol_spec = api100.load_json(value)
                requests_mock.get(url, json=bbox_mol_spec)
            else:
                requests_mock.get(url, text=value)
        elif isinstance(value, dict):
            requests_mock.get(url, json=value)
        elif value in [404, 500]:
            requests_mock.get(url, status_code=value)
        elif isinstance(value, tuple) and value[0] in [302]:
            status_code, target = value
            requests_mock.get(url, status_code=status_code, headers={"Location": target})
        else:
            raise ValueError(value)

    # Evaluate process graph (with URL namespace)
    pg = api100.load_json("udp_bbox_mol_basic.json")
    assert pg["bboxmol1"]["process_id"] == "bbox_mol"
    pg["bboxmol1"]["namespace"] = namespace

    res = api100.result(pg)
    if expected_error:
        status_code, error_code, message = expected_error
        res.assert_error(status_code=status_code, error_code=error_code, message=message)
    else:
        res.assert_status_code(200)
        params = dummy_backend.last_load_collection_call('S2_FOOBAR')
        assert params["spatial_extent"] == {"west": 5.05, "south": 51.2, "east": 5.1, "north": 51.23, "crs": 'EPSG:4326'}


def test_execute_no_cube_1_plus_2(api100):
    # Calculator as a service!
    res = api100.result({
        "add1": {"process_id": "add", "arguments": {"x": 1, "y": 2}, "result": True}
    })
    assert res.assert_status_code(200).json == 3


@pytest.mark.parametrize(["process_graph", "expected"], [
    ({"add1": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}}, 8),
    ({"mul1": {"process_id": "multiply", "arguments": {"x": 3, "y": 5}, "result": True}}, 15),
    ({"sum1": {"process_id": "sum", "arguments": {"data": [1, 2, 3, 4, 5]}, "result": True}}, 15),
    ({"log1": {"process_id": "log", "arguments": {"x": 10000, "base": 10}, "result": True}}, 4.0),
    ({"flr1": {"process_id": "floor", "arguments": {"x": 12.34}, "result": True}}, 12),
    ({"cos1": {"process_id": "cos", "arguments": {"x": np.pi}, "result": True}}, -1),
    ({"max1": {"process_id": "max", "arguments": {"data": [2, 8, 5, 3]}, "result": True}}, 8),
    ({"med1": {"process_id": "median", "arguments": {"data": [2, 8, 5, 3, 11]}, "result": True}}, 5),
    ({"var1": {"process_id": "variance", "arguments": {"data": [2, 8, 5, 3]}, "result": True}}, 7.0),
    ({"cnt1": {"process_id": "count", "arguments": {"data": [2, 8, None, 3]}, "result": True}}, 3),
    ({"pi1": {"process_id": "pi", "arguments": {}, "result": True}}, math.pi),
    ({"e1": {"process_id": "e", "arguments": {}, "result": True}}, math.e),
])
def test_execute_no_cube_just_math(api100, process_graph, expected):
    assert api100.result(process_graph).assert_status_code(200).json == pytest.approx(expected,0.0001)


@pytest.mark.parametrize(["process_graph", "expected"], [
    ({"if1": {"process_id": "if", "arguments": {"value": True, "accept": 2}, "result": True}}, 2),
    ({"if1": {"process_id": "if", "arguments": {"value": False, "accept": 2, "reject": 3}, "result": True}}, 3),
    (
            # Extra effort to check that `if` output is null
            {
                "if1": {"process_id": "if", "arguments": {"value": False, "accept": 2}},
                "isnodata": {"process_id": "is_nodata", "arguments": {"x": {"from_node": "if1"}}, "result": True},
            },
            True
    ),
    # any/all implementation is bit weird at the moment https://github.com/Open-EO/openeo-processes-python/issues/16
    # ({"any1": {"process_id": "any", "arguments": {"data": [False, True, False]}, "result": True}}, True),
    # ({"all1": {"process_id": "all", "arguments": {"data": [False, True, False]}, "result": True}}, False),
])
def test_execute_no_cube_logic(api100, process_graph, expected):
    assert api100.result(process_graph).assert_status_code(200).json == expected


@pytest.mark.parametrize(["process_id", "arguments", "expected"], [
    ("text_begins", {"data": "FooBar", "pattern": "Foo"}, True),
    ("text_begins", {"data": "FooBar", "pattern": "Bar"}, False),
    ("text_begins", {"data": "FooBar", "pattern": "fOo"}, False),
    ("text_begins", {"data": "FooBar", "pattern": "fOo", "case_sensitive": False}, True),
    ("text_contains", {"data": "FooBar", "pattern": "oB"}, True),
    ("text_contains", {"data": "FooBar", "pattern": "ob"}, False),
    ("text_contains", {"data": "FooBar", "pattern": "ob", "case_sensitive": False}, True),
    ("text_ends", {"data": "FooBar", "pattern": "Bar"}, True),
    ("text_ends", {"data": "FooBar", "pattern": "Foo"}, False),
    ("text_ends", {"data": "FooBar", "pattern": "bar"}, False),
    ("text_ends", {"data": "FooBar", "pattern": "bar", "case_sensitive": False}, True),
    ("text_merge", {"data": ["foo", "bar"]}, "foobar"),
    ("text_merge", {"data": ["foo", "bar"], "separator": "--"}, "foo--bar"),
    ("text_merge", {"data": [1, 2, 3], "separator": "/"}, "1/2/3"),
    ("text_merge", {"data": [1, "b"], "separator": 0}, "10b"),
])
def test_text_processes(api100, process_id, arguments, expected):
    # TODO: null propagation (`text_begins(data=null,...) -> null`) can not be tested at the moment
    pg = {"t": {"process_id": process_id, "arguments": arguments, "result":True}}
    assert api100.result(pg).assert_status_code(200).json == expected


@pytest.mark.parametrize(["process_graph", "expected"], [
    ({"arc1": {"process_id": "array_contains", "arguments": {"data": [2, 8, 5, 3], "value": 5}, "result": True}}, True),
    ({"arf1": {"process_id": "array_find", "arguments": {"data": [2, 8, 5, 3], "value": 5}, "result": True}}, 2),
    ({"srt1": {"process_id": "sort", "arguments": {"data": [2, 8, 5, 3]}, "result": True}}, [2, 3, 5, 8]),
    (
            {"arc1": {"process_id": "array_concat", "arguments": {"array1": [2, 8], "array2": [5, 3]}, "result": True}},
            [2, 8, 5, 3]
    ),
    (
            {
                "ar1": {"process_id": "array_create", "arguments": {"data": [2, 8]}},
                "ar2": {"process_id": "array_create", "arguments": {"data": [5, 3]}},
                "arc1": {
                    "process_id": "array_concat",
                    "arguments": {"array1": {"from_node": "ar1"}, "array2": {"from_node": "ar2"}}, "result": True
                }
            },
            [2, 8, 5, 3]
    ),
    (
            {"arc1": {"process_id": "array_create", "arguments": {"data": [2, 8], "repeat": 3}, "result": True}},
            [2, 8, 2, 8, 2, 8]
    ),
])
def test_execute_no_cube_just_arrays(api100, process_graph, expected):
    assert api100.result(process_graph).assert_status_code(200).json == expected


def test_execute_no_cube_dynamic_args(api100):
    pg = {
        "loadcollection1": {'process_id': 'load_collection', 'arguments': {'id': 'S2_FOOBAR'}},
        "add1": {"process_id": "add", "arguments": {"x": 2.5, "y": 5.25}},
        "applykernel1": {
            "process_id": "apply_kernel",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                # TODO: test using "from_node" as kernel component?
                "kernel": [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
                "factor": {"from_node": "add1"},
            },
            "result": True
        }
    }
    api100.check_result(pg)
    apply_kernel_mock = dummy_backend.get_collection("S2_FOOBAR").apply_kernel
    args, kwargs = apply_kernel_mock.call_args
    assert kwargs["factor"] == 7.75


@pytest.mark.parametrize(["border", "expected"], [(0, 0), ("0", 0), ])
def test_execute_apply_kernel_border(api100, border, expected):
    pg = {
        "lc1": {'process_id': 'load_collection', 'arguments': {'id': 'S2_FOOBAR'}},
        "ak1": {
            "process_id": "apply_kernel",
            "arguments": {
                "data": {"from_node": "lc1"},
                "kernel": [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
                "border": border,
            },
            "result": True
        }
    }
    api100.check_result(pg)
    apply_kernel_mock = dummy_backend.get_collection("S2_FOOBAR").apply_kernel
    args, kwargs = apply_kernel_mock.call_args
    assert kwargs["border"] == expected


# TODO: test using dynamic arguments in bbox_filter (not possible yet: see EP-3509)


def test_execute_EP3509_process_order(api100):
    pg = {
        "loadcollection1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "start": {"process_id": "constant", "arguments": {"x": "2020-02-02"}},
        "end": {"process_id": "constant", "arguments": {"x": "2020-03-03"}},
        "filtertemporal1": {"process_id": "filter_temporal", "arguments": {
            "data": {"from_node": "loadcollection1"},
            "extent": [{"from_node": "start"}, {"from_node": "end"}],
        }},
        "west": {"process_id": "add", "arguments": {"x": 3, "y": 2}},
        "east": {"process_id": "add", "arguments": {"x": 3, "y": 3}},
        "filterbbox1": {"process_id": "filter_bbox", "arguments": {
            "data": {"from_node": "filtertemporal1"},
            "extent": {"west": {"from_node": "west"}, "east": {"from_node": "east"}, "south": 50, "north": 51}
        }},
        "bands": {"process_id": "constant", "arguments": {"x": ["B02", "B03"]}},
        "filterbands1": {"process_id": "filter_bands", "arguments": {
            "data": {"from_node": "filterbbox1"},
            "bands": {"from_node": "bands"}
        }},
        "applykernel": {"process_id": "apply_kernel", "arguments": {
            "data": {"from_node": "filterbands1"}, "kernel": [1]
        }, "result": True}
    }
    api100.check_result(pg)
    params = dummy_backend.last_load_collection_call("S2_FOOBAR")
    assert params["temporal_extent"] == ("2020-02-02", "2020-03-03")
    assert params["spatial_extent"] == {"west": 5, "east": 6, "south": 50, "north": 51, "crs": "EPSG:4326"}
    assert params["bands"] == ["B02", "B03"]


@pytest.mark.parametrize(["pg", "ndvi_expected", "mask_expected"], [
    (
            {
                "load1": {"process_id": "load_collection", "arguments": {"id": "PROBAV_L3_S10_TOC_NDVI_333M_V2"}},
                "load2": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
                "bands1": {
                    "process_id": "filter_bands",
                    "arguments": {"data": {"from_node": "load1"}, "bands": ["ndvi"]}
                },
                "bands2": {
                    "process_id": "filter_bands",
                    "arguments": {"data": {"from_node": "load2"}, "bands": ["B02"]}
                },
                "mask": {"process_id": "mask", "arguments": {
                    "data": {"from_node": "bands1"},
                    "mask": {"from_node": "bands2"},
                }, "result": True}
            },
            ["ndvi"], ["B02"]
    ),
    (
            {
                "load1": {"process_id": "load_collection", "arguments": {"id": "PROBAV_L3_S10_TOC_NDVI_333M_V2"}},
                "load2": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
                "bands1": {
                    "process_id": "filter_bands",
                    "arguments": {"data": {"from_node": "load1"}, "bands": ["ndvi"]}
                },
                "mask": {"process_id": "mask", "arguments": {
                    "data": {"from_node": "bands1"},
                    "mask": {"from_node": "load2"},
                }, "result": True}
            },
            ["ndvi"], None
    ),
    (
            {
                "load1": {"process_id": "load_collection", "arguments": {"id": "PROBAV_L3_S10_TOC_NDVI_333M_V2"}},
                "load2": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
                "bands2": {
                    "process_id": "filter_bands",
                    "arguments": {"data": {"from_node": "load2"}, "bands": ["B02"]}
                },
                "mask": {"process_id": "mask", "arguments": {
                    "data": {"from_node": "load1"},
                    "mask": {"from_node": "bands2"},
                }, "result": True}
            },
            None, ["B02"]
    ),
])
def test_execute_EP3509_issue38_leaking_band_filter(api, pg, ndvi_expected, mask_expected):
    """https://github.com/Open-EO/openeo-python-driver/issues/38"""
    api.check_result(pg)
    assert dummy_backend.last_load_collection_call("PROBAV_L3_S10_TOC_NDVI_333M_V2").get("bands") == ndvi_expected
    assert dummy_backend.last_load_collection_call("S2_FOOBAR").get("bands") == mask_expected


def test_reduce_add_reduce_dim(api100):
    """Test reduce_dimension -> add_dimension -> reduce_dimension"""
    content = api100.check_result("reduce_add_reduce_dimension.json")
    dummy = dummy_backend.get_collection("S2_FOOBAR")

    assert dummy.reduce_dimension.call_count == 1

    dims = content.json["cube:dimensions"]
    names = [k for k in dims]
    assert names == ["x", "y", "t"]


def test_reduce_drop_dimension(api100):
    content = api100.check_result({
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "drop": {
            "process_id": "drop_dimension",
            "arguments": {"data": {"from_node": "lc"}, "name": "bands"},
            "result": False,
        },
        "save": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "drop"}, "format": "JSON"},
            "result": True,
        }
    })
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.drop_dimension.call_count == 1
    dims = content.json["cube:dimensions"]
    names = [k for k in dims]

    assert names == ["x", "y", "t"]


def test_reduce_dimension_labels(api100):
    res = api100.check_result({
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "drop": {
            "process_id": "dimension_labels",
            "arguments": {"data": {"from_node": "lc"}, "dimension": "bands"},
            "result": True,
        },
    })
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.dimension_labels.call_count == 1
    assert res.json == ["x", "y", "t", "bands"]


@pytest.mark.parametrize(["arguments", "expected"], [
    ({"data": {"from_node": "l"}, "name": "foo", "label": "bar"}, "other"),
    ({"data": {"from_node": "l"}, "name": "foo", "label": "bar", "type": "bands"}, "bands"),
])
def test_add_dimension_type_argument(api100, arguments, expected):
    api100.check_result({
        "l": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "s": {"process_id": "add_dimension", "arguments": arguments, "result": True}
    })
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    dummy.add_dimension.assert_called_with(name="foo", label="bar", type=expected)


def test_add_dimension_duplicate(api100):
    res = api100.result({
        "l": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "a1": {
            "process_id": "add_dimension",
            "arguments": {"data": {"from_node": "l"}, "name": "bandz", "label": "bar", "type": "bands"},
        },
        "a2": {
            "process_id": "add_dimension",
            "arguments": {"data": {"from_node": "a1"}, "name": "bandz", "label": "bar", "type": "bands"},
            "result": True,
        },
    })
    res.assert_error(400, "DimensionExists")


@pytest.mark.parametrize(["format", "expected"], [
    ("GTiff", "image/tiff; application=geotiff"),
    ("NetCDF", "application/x-netcdf"),
    ("PNG", "image/png"),
    ("CovJSON", "application/json"),
])
def test_save_result_gtiff_mimetype(api, format, expected):
    pg = {
        "l": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "s": {"process_id": "save_result", "arguments": {"data": {"from_node": "l"}, "format": format}, "result": True}
    }
    res = api.check_result(pg)
    assert res.headers["Content-type"] == expected


def test_execute_load_collection_sar_backscatter_defaults(api100):
    api100.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FAPAR_CLOUDCOVER"}
        },
        "sar_backscatter": {
            "process_id": "sar_backscatter",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
            },
            "result": True
        },
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params.sar_backscatter == SarBackscatterArgs(
        coefficient="gamma0-terrain", elevation_model=None, mask=False, contributing_area=False,
        local_incidence_angle=False, ellipsoid_incidence_angle=False, noise_removal=True, options={}
    )


def test_execute_load_collection_sar_backscatter_none_values(api100):
    api100.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FAPAR_CLOUDCOVER"}
        },
        "sar_backscatter": {
            "process_id": "sar_backscatter",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "coefficient": None,
                "elevation_model": None,
            },
            "result": True
        },
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params.sar_backscatter == SarBackscatterArgs(
        coefficient=None, elevation_model=None, mask=False, contributing_area=False,
        local_incidence_angle=False, ellipsoid_incidence_angle=False, noise_removal=True, options={}
    )


def test_execute_load_collection_sar_backscatter(api100):
    api100.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FAPAR_CLOUDCOVER"}
        },
        "sar_backscatter": {
            "process_id": "sar_backscatter",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "coefficient": "gamma0-ellipsoid",
                "mask": True,
                "contributing_area": True,
                "ellipsoid_incidence_angle": True,
                "noise_removal": False,
                "options": {"tile_size": 1024}
            },
            "result": True
        },
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params.sar_backscatter == SarBackscatterArgs(
        coefficient='gamma0-ellipsoid', elevation_model=None, mask=True, contributing_area=True,
        local_incidence_angle=False, ellipsoid_incidence_angle=True, noise_removal=False, options={"tile_size": 1024}
    )


def test_execute_load_collection_sar_backscatter_compatibility(api100):
    # assert that we can differentiate between collections that are sar_backscatter compatible and those that are not
    api100.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FAPAR_CLOUDCOVER"},
            "result": True
        }
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params.sar_backscatter is None


def test_execute_load_collection_custom_properties(api100):
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
            "process_id": "load_collection", "arguments": {"id": "S2_FAPAR_CLOUDCOVER", "properties": properties}},
        "lc2": {
            "process_id": "load_collection", "arguments": {"id": "S2_FAPAR_CLOUDCOVER", "properties": asc_props}},
        "merge": {
            "process_id": "merge_cubes",
            "arguments": {"cube1": {"from_node": "lc"}, "cube2": {"from_node": "lc2"}},
            "result": True
        }
    }

    api100.check_result(pg)
    params = dummy_backend.all_load_collection_calls("S2_FAPAR_CLOUDCOVER")
    print(params)
    assert len(params) == 2
    assert params[0].properties == properties
    assert params[1].properties == asc_props


def test_execute_load_collection_custom_cloud_mask(api100):
    # assert that we can differentiate between collections that are sar_backscatter compatible and those that are not
    api100.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FAPAR_CLOUDCOVER"},
            "result": False
        },
        "mask": {
            "process_id": "mask_scl_dilation",
            "arguments": {"data": {"from_node": "loadcollection1"}},
            "result": False
        },
        'filterbands1': {
            'process_id': 'filter_bands',
            'arguments': {
                'data': {'from_node': 'mask'},
                'bands': ["SCL"],
            },
            'result': True
        },
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params.custom_mask == {"method": "mask_scl_dilation"}
    assert params.bands == None


def test_execute_load_collection_custom_l1c_cloud_mask(api100):
    api100.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FAPAR_CLOUDCOVER"},
            "result": False
        },
        "mask": {
            "process_id": "mask_l1c",
            "arguments": {"data": {"from_node": "loadcollection1"}},
            "result": True
        }
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params.custom_mask == {"method": "mask_l1c"}
    assert params.bands is None


def test_execute_load_collection_resolution_merge(api100):
    api100.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR"}
        },
        "resolution_merge": {
            "process_id": "resolution_merge",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "method": "improphe",
                "high_resolution_bands": ["B02", "B03"],
                "low_resolution_bands": ["B05", "B06"],
                "options": {"kernel_size": 8}
            },
            "result": True
        },
    })
    params = dummy_backend.last_load_collection_call("S2_FOOBAR")
    resolution_merge_mock = dummy_backend.get_collection("S2_FOOBAR").resolution_merge
    args, kwargs = resolution_merge_mock.call_args
    assert args[0] == ResolutionMergeArgs(
        method="improphe",
        high_resolution_bands=["B02", "B03"], low_resolution_bands=["B05", "B06"],
        options={'kernel_size': 8},
    )


def test_execute_custom_process_by_process_graph_minimal(api100):
    process_id = generate_unique_test_process_id()
    # Register a custom process with minimal process graph
    process_spec = {
        "id": process_id,
        "process_graph": {
            "increment": {"process_id": "add", "arguments": {"x": {"from_parameter": "x"}, "y": 1}, "result": True}
        }
    }
    custom_process_from_process_graph(process_spec=process_spec)
    # Apply process
    res = api100.check_result({
        "do_math": {
            "process_id": process_id,
            "arguments": {"x": 2},
            "result": True
        },
    }).json
    assert res == 3


def test_execute_custom_process_by_process_graph(api100):
    process_id = generate_unique_test_process_id()

    # Register a custom process with process graph
    process_spec = api100.load_json("add_and_multiply.json")
    process_spec["id"] = process_id
    custom_process_from_process_graph(process_spec=process_spec)
    # Apply process
    res = api100.check_result({
        "do_math": {
            "process_id": process_id,
            "arguments": {"data": 2},
            "result": True
        },
    }).json
    assert res == 25


def test_execute_custom_process_by_process_graph_json(api100, tmp_path):
    process_id = generate_unique_test_process_id()

    process_spec = api100.load_json("add_and_multiply.json")
    process_spec["id"] = process_id
    path = tmp_path / f"{process_id}.json"
    with path.open("w") as f:
        json.dump(process_spec, f)

    # Register a custom process with process graph
    custom_process_from_process_graph(path)

    # Apply process
    res = api100.check_result({
        "do_math": {
            "process_id": process_id,
            "arguments": {"data": 2},
            "result": True
        },
    }).json
    assert res == 25


def test_execute_custom_process_by_process_graph_namespaced(api100):
    process_id = generate_unique_test_process_id()

    # Register a custom process with process graph
    process_spec = api100.load_json("add_and_multiply.json")
    process_spec["id"] = process_id
    custom_process_from_process_graph(process_spec=process_spec, namespace="madmath")
    # Apply process
    res = api100.check_result({
        "do_math": {
            "process_id": process_id,
            "namespace": "madmath",
            "arguments": {"data": 3},
            "result": True
        },
    }).json
    assert res == 30


def test_normalized_difference(api100):
    res = api100.check_result({
        "do_math": {
            "process_id": "normalized_difference",
            "arguments": {"x": 3, "y": 5},
            "result": True
        },
    }).json
    assert res == -0.25


def test_ard_normalized_radar_backscatter(api100):
    api100.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR"}
        },
        "ardnormalizedradarbackscatter1": {
            "process_id": "ard_normalized_radar_backscatter",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "elevation_model": "MAPZEN",
                "ellipsoid_incidence_angle": True,
                "noise_removal": True,
                "contributing_area": True
            },
            "result": True
        }
    })

    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.sar_backscatter.call_count == 1
    args, kwargs = dummy.sar_backscatter.call_args
    assert args == (SarBackscatterArgs(
        coefficient="gamma0-terrain", elevation_model="MAPZEN", mask=True, contributing_area=True,
        local_incidence_angle=True, ellipsoid_incidence_angle=True, noise_removal=True, options={}),)
    assert kwargs == {}


def test_ard_normalized_radar_backscatter_without_optional_arguments(api100):
    api100.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR"}
        },
        "ardnormalizedradarbackscatter1": {
            "process_id": "ard_normalized_radar_backscatter",
            "arguments": {"data": {"from_node": "loadcollection1"}},
            "result": True
        }
    })

    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.sar_backscatter.call_count == 1
    args, kwargs = dummy.sar_backscatter.call_args
    assert args == (SarBackscatterArgs(
        coefficient="gamma0-terrain", elevation_model=None, mask=True, contributing_area=False,
        local_incidence_angle=True, ellipsoid_incidence_angle=False, noise_removal=True, options={}),)
    assert kwargs == {}


@pytest.mark.parametrize(["date", "value", "unit", "expected"], [
    # Examples from date_shift.json
    ("2020-02-01T17:22:45Z", 6, "month", "2020-08-01T17:22:45Z"),
    # TODO timezone support https://github.com/Open-EO/openeo-python-driver/issues/75
    # ("2021-03-31T00:00:00+02:00", -7, "day",  "2021-03-24T00:00:00+02:00"),
    ("2020-02-29T17:22:45Z", 1, "year", "2021-02-28T17:22:45Z"),
    ("2020-01-31", 1, "month", "2020-02-29"),
    ("2016-12-31T23:59:59Z", 1, "second", "2017-01-01T00:00:00Z"),
    # TODO millisecond support https://github.com/Open-EO/openeo-python-driver/issues/75
    # ("2018-12-31T17:22:45Z", 1150, "millisecond", "2018-12-31T17:22:46.150Z"),
    ("2018-01-01", 25, "hour", "2018-01-02"),
    ("2018-01-01", -1, "hour", "2017-12-31"),
    # Additional tests
    ("2011-12-13", 3, "day", "2011-12-16"),
    ("2011-12-13T14:15:16Z", 3, "day", "2011-12-16T14:15:16Z"),
    ("2011-12-13", -3, "day", "2011-12-10"),
    ("2011-12-13T14:15:16Z", -3, "day", "2011-12-10T14:15:16Z"),
    ("2018-01-01T01:02:03Z", 25, "hour", "2018-01-02T02:02:03Z"),
    ("2018-01-01T01:02:03Z", -2, "hour", "2017-12-31T23:02:03Z"),
    ("2018-01-01", 24 * 60 + 1, "minute", "2018-01-02"),
    ("2018-01-01", -1, "minute", "2017-12-31"),
    ("2018-01-01T00:01:02Z", 24 * 60 + 1, "minute", "2018-01-02T00:02:02Z"),
    ("2018-01-01T00:01:02Z", -2, "minute", "2017-12-31T23:59:02Z"),
])
def test_date_shift(api100, date, value, unit, expected):
    pg = {
        "dateshift1": {
            "process_id": "date_shift",
            "arguments": {"date": date, "value": value, "unit": unit},
            "result": True,
        }
    }
    res = api100.result(pg).assert_status_code(200).json
    assert res == expected


@pytest.mark.parametrize(["buf", "unit", "repr_geom", "expected_type", "bounds"], [
    # Template to calculate expected bounds of single geometries:
    #       import geopandas as gpd
    #       shape = Point(5.1, 51.22)
    #       series = gpd.GeoSeries([shape], crs=4326)
    #       buffered = series.to_crs(32631).buffer(100).to_crs(4326)
    #       print(buffered.iloc[0].bounds)
    (
            100, "meter", {"type": "Point", "coordinates": (5.1, 51.22)},
            "Polygon", (5.098569060859835, 51.219101388461105, 5.101430937032029, 51.22089861137662),
    ),
    (
            1000, "meter", {"type": "Point", "coordinates": (5.1, 51.22)},
            "Polygon", (5.085690513968438, 51.21101387730941, 5.114309275218151, 51.22898610646433),
    ),
    (
            100, "kilometer", {"type": "Point", "coordinates": (5.1, 51.22)},
            "Polygon", (3.668246824769951, 50.321307522288016, 6.529646150453797, 52.11853021385508),
    ),
    (
            100, "meter", {"type": "MultiPoint", "coordinates": ((5.1, 51.22), (5.11, 51.23))},
            "MultiPolygon", (5.098569060859835, 51.219101388461105, 5.111431237949069, 51.23089860406231),
    ),
    (
            750, "meter", {"type": "MultiPoint", "coordinates": ((5.1, 51.22), (5.11, 51.23))},
            "Polygon", (5.089267905161828, 51.213260409503235, 5.120734232861271, 51.23673952650676),
    ),
    (
            100, "meter", {"type": "LineString", "coordinates": ((5.1, 51.22), (5.11, 51.23))},
            "Polygon", (5.09856905615526, 51.21910138452717, 5.111431257271536, 51.23089861521429),
    ),
    (
            100, "meter", load_json("geojson/Polygon01.json"),
            "Polygon", (5.0985703229820665, 51.19910120996896, 5.141430817509717, 51.23089799863948),
    ),
    (
            -500, "meter", load_json("geojson/Polygon01.json"),
            "Polygon", (5.1084522374601145, 51.206647698484225, 5.129458164090344, 51.222687097228174),
    ),
    (
            1, "kilometer", load_json("geojson/Polygon01.json"),
            "Polygon", (5.087874843800581, 51.19113803453279, 5.154275007454256, 51.23894705603116),
    ),
    (
            100, "meter", load_json("geojson/MultiPolygon01.json"),
            "MultiPolygon", (5.098787598453318, 51.19911380673043, 5.141427521031006, 51.2408707459342),
    ),
    (
            500, "meter", load_json("geojson/MultiPolygon01.json"),
            "Polygon", (5.092851581512912, 51.19550604834508, 5.147157170251627, 51.244493919231424,)
    ),
    (
            1000, "meter", load_json("geojson/FeatureCollection01.json"),
            "FeatureCollection", (4.43568898, 51.09100882, 4.53429533, 51.20899105),
    ),
    (
            1000, "meter", str(get_path("geojson/FeatureCollection01.json")),
            "FeatureCollection", (4.43568898, 51.09100882, 4.53429533, 51.20899105),
    ),
])
def test_vector_buffer(api100, buf, unit, repr_geom, expected_type, bounds):
    pg = {
        "vectorbuffer1": {
            "process_id": "vector_buffer",
            "arguments": {"geometry": repr_geom, "distance": buf, "unit": unit},
            "result": True,
        }
    }
    res = api100.result(pg).assert_status_code(200).json
    assert res["type"] == expected_type
    if res["type"] == "FeatureCollection":
        res_gs = gpd.GeoSeries([shapely.geometry.shape(feat["geometry"]) for feat in res["features"]])
    else:
        res_gs = gpd.GeoSeries([shapely.geometry.shape(res)])
    assert res_gs.total_bounds == pytest.approx(bounds, 0.001)


@pytest.mark.parametrize(["distance", "unit", "expected"], [
    (-10, "meter", [5.5972, 50.8468, 5.6046, 50.8524]),
    (+10, "meter", [5.5970, 50.8467, 5.6049, 50.8526]),
    (1, "kilometer", [5.5860, 50.8395, 5.6163, 50.8596]),
])
def test_vector_buffer_non_epsg4326(api100, distance, unit, expected):
    geometry = load_json("geojson/FeatureCollection03.json")
    pg = {
        "vectorbuffer1": {
            "process_id": "vector_buffer",
            "arguments": {"geometry": geometry, "distance": distance, "unit": unit},
            "result": True,
        }
    }
    res = api100.result(pg).assert_status_code(200).json
    assert res["type"] == "FeatureCollection"
    res_gs = gpd.GeoSeries([shapely.geometry.shape(feat["geometry"]) for feat in res["features"]])
    assert res_gs.total_bounds == pytest.approx(expected, abs=0.0001)


@pytest.mark.parametrize(["distance", "unit", "expected"], [
    (-10, "meter", [5.0144, 51.1738, 5.0173, 51.1769]),
    (+10, "meter", [5.0141, 51.1736, 5.0176, 51.1771]),
    (1, "kilometer", [4.9999, 51.1647, 5.0318, 51.1860]),
])
def test_vector_buffer_ogc_crs84(api100, distance, unit, expected):
    geometry = load_json("geojson/FeatureCollection04.json")
    pg = {
        "vectorbuffer1": {
            "process_id": "vector_buffer",
            "arguments": {"geometry": geometry, "distance": distance, "unit": unit},
            "result": True,
        }
    }
    res = api100.result(pg).assert_status_code(200).json
    assert res["type"] == "FeatureCollection"
    res_gs = gpd.GeoSeries([shapely.geometry.shape(feat["geometry"]) for feat in res["features"]])
    assert res_gs.total_bounds == pytest.approx(expected, abs=0.0001)


def test_load_result(api100):
    api100.check_result("load_result.json")
    params = dummy_backend.last_load_collection_call("99a605a0-1a10-4ba9-abc1-6898544e25fc")

    assert params["temporal_extent"] == ('2019-09-22', '2019-09-22')


def test_chunk_polygon(api100):
    api100.check_result("chunk_polygon.json")
    params = dummy_backend.last_load_collection_call("S2_FOOBAR")
    assert params["spatial_extent"] == {'west': 1.0, 'south': 5.0, 'east': 12.0, 'north': 16.0, 'crs': 'EPSG:4326'}


def test_fit_class_random_forest(api100):
    res = api100.check_result("fit_class_random_forest.json")

    geom1 = {
        "type": "Polygon",
        "coordinates": [[[3.0, 5.0], [4.0, 5.0], [4.0, 6.0], [3.0, 6.0], [3.0, 5.0]]],
    }
    geom2 = {
        "type": "Polygon",
        "coordinates": [[[8.0, 1.0], [9.0, 1.0], [9.0, 2.0], [8.0, 2.0], [8.0, 1.0]]],
    }
    assert res.json == DictSubSet(
        {
            "type": "DummyMlModel",
            "creation_data": {
                "process_id": "fit_class_random_forest",
                "data": DictSubSet(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            DictSubSet(
                                {
                                    "type": "Feature",
                                    "id": "0",
                                    "geometry": geom1,
                                    "properties": {
                                        "agg~B02": 2.345,
                                        "agg~B03": None,
                                        "agg~B04": 2.0,
                                        "agg~B08": 3.0,
                                        "target": 0,
                                    },
                                }
                            ),
                            DictSubSet(
                                {
                                    "type": "Feature",
                                    "id": "1",
                                    "geometry": geom2,
                                    "properties": {
                                        "agg~B02": 4.0,
                                        "agg~B03": 5.0,
                                        "agg~B04": 6.0,
                                        "agg~B08": 7.0,
                                        "target": 1,
                                    },
                                }
                            ),
                        ],
                    }
                ),
                "target": DictSubSet(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            DictSubSet(
                                {
                                    "type": "Feature",
                                    "geometry": geom1,
                                    "properties": {"target": 0},
                                }
                            ),
                            DictSubSet(
                                {
                                    "type": "Feature",
                                    "geometry": geom2,
                                    "properties": {"target": 1},
                                }
                            ),
                        ],
                    }
                ),
                "max_variables": None,
                "num_trees": 200,
                "seed": None,
            },
        }
    )


def test_if_merge_cubes(api100):
    api100.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FOOBAR",
                "temporal_extent": ["2021-09-01", "2021-09-10"],
                "spatial_extent": {"west": 3, "south": 51, "east": 3.1, "north": 51.1},
                "bands": ["B04"],
            }},
        "eq1": {"process_id": "eq", "arguments": {"x": 4, "y": 3}},
        "if1": {
            "process_id": "if",
            "arguments": {
                "value": {"from_node": "eq1"},
                "accept": {"from_node": "loadcollection1"}, "reject": {"from_node": "loadcollection1"},
            }},
        "mergecubes1": {
            "process_id": "merge_cubes",
            "arguments": {"cube1": {"from_node": "loadcollection1"}, "cube2": {"from_node": "if1"}},
            "result": True
        }
    })


@pytest.mark.parametrize(["geojson", "expected"], [
    (
            {"type": "Polygon", "coordinates": [[(1, 1), (3, 1), (2, 3), (1, 1)]]},
            [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[1, 1], [3, 1], [2, 3], [1, 1]]]},
                    "properties": {},
                },
            ],
    ),
    (
            {"type": "MultiPolygon", "coordinates": [[[(1, 1), (3, 1), (2, 3), (1, 1)]]]},
            [
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiPolygon", "coordinates": [[[[1, 1], [3, 1], [2, 3], [1, 1]]]]},
                    "properties": {},
                },
            ],
    ),
    (
            {
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": [[[(1, 1), (3, 1), (2, 3), (1, 1)]]]},
                "properties": {"id": "12_3"},
            },
            [
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiPolygon", "coordinates": [[[[1, 1], [3, 1], [2, 3], [1, 1]]]]},
                    "properties": {"id": "12_3"},
                },
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
                ]},
            [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[1, 1], [3, 1], [2, 3], [1, 1]]]},
                    "properties": {"id": 1},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiPolygon", "coordinates": [[[[1, 1], [3, 1], [2, 3], [1, 1]]]]},
                    "properties": {"id": 2},
                },
            ],
    ),
])
def test_to_vector_cube(api100, geojson, expected):
    res = api100.check_result({
        "vc": {
            "process_id": "to_vector_cube",
            "arguments": {"data": geojson},
            "result": True,
        }
    })
    assert res.json == DictSubSet({
        "type": "FeatureCollection",
        "features": expected,
    })
