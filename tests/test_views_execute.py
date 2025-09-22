import logging
import shutil
import dataclasses
import json
import math
import re
import sys
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional
from unittest import mock
from zipfile import ZipFile
import http.client

import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry
from pyproj import CRS

from openeo_driver.datacube import DriverDataCube, DriverVectorCube
from openeo_driver.datastructs import ResolutionMergeArgs, SarBackscatterArgs
from openeo_driver.dry_run import ProcessType
from openeo_driver.dummy import dummy_backend
from openeo_driver.dummy.dummy_backend import DummyVisitor
from openeo_driver.errors import (
    ProcessGraphInvalidException,
    ProcessGraphMissingException,
)
from openeo_driver.processes import ProcessArgs, ProcessRegistry
from openeo_driver.ProcessGraphDeserializer import (
    custom_process_from_process_graph,
    collect,
    ENV_DRY_RUN_TRACER,
)
from openeo_driver.testing import (
    TEST_USER,
    TEST_USER_BEARER_TOKEN,
    ApiTester,
    ApproxGeoJSONByBounds,
    DictSubSet,
    RegexMatcher,
    ephemeral_fileserver,
    generate_unique_test_process_id,
    preprocess_check_and_replace,
    preprocess_regex_check_and_replace,
)
from openeo_driver.users import User
from openeo_driver.util.geometry import (
    as_geojson_feature,
    as_geojson_feature_collection, BoundingBox,
)
from openeo_driver.util.ioformats import IOFORMATS
from openeo_driver.util.logging import FlaskRequestCorrelationIdLogging
from openeo_driver.utils import EvalEnv

from .data import TEST_DATA_ROOT, get_path, load_json
from .test_dry_run import CRS_UTM


@pytest.fixture(
    params=[
        "1.0",
        # "1.1",
        "1.2",
    ]
)
def api_version(request):
    return request.param


@pytest.fixture
def api(api_version, client, backend_implementation) -> ApiTester:
    dummy_backend.reset(backend_implementation)

    if api_version.startswith("1."):
        # For now, use "pg/1.0" for all 1.x API requests (no need to differentiate yet)
        data_root = TEST_DATA_ROOT / "pg" / "1.0"
    else:
        raise ValueError(api_version)
    return ApiTester(api_version=api_version, client=client, data_root=data_root)


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


def test_execute_filter_temporal_extent_double_null(api):
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

    resp.assert_error(
        400,
        "ProcessParameterInvalid",
        message="The value passed for parameter 'extent' in process 'filter_temporal' is invalid:"
        " both start and end are null",
    )


def test_execute_filter_temporal_extent_equal_start_and_end(api, caplog):
    resp = api.result(
        {
            "loadcollection1": {"process_id": "load_collection", "arguments": {"id": "S2_FAPAR_CLOUDCOVER"}},
            "filtertemporal1": {
                "process_id": "filter_temporal",
                "arguments": {"data": {"from_node": "loadcollection1"}, "extent": ["2021-12-07", "2021-12-07"]},
                "result": True,
            },
        }
    )

    # TODO: This should trigger a TemporalExtentEmpty error instead of success with warning
    resp.assert_status_code(200)
    expected_warning = "filter_temporal extent with same start and end (['2021-12-07', '2021-12-07']) is invalid/deprecated and will trigger a TemporalExtentEmpty error in the future."
    assert expected_warning in caplog.text


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


def test_execute_apply_unary(api):
    api.check_result("apply_unary.json")
    assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").apply.call_count == 1


def test_execute_apply_unary_parent_scope(api):
    api.check_result(
        "apply_unary.json",
        preprocess=preprocess_check_and_replace('"from_parameter": "x"', '"from_parameter": "data"')
    )


#
@pytest.mark.skip('parameter checking of callback graphs now happens somewhere else')
def test_execute_apply_unary_invalid_from_parameter(api):
    resp = api.result(
        "apply_unary.json",
        preprocess=preprocess_check_and_replace('"from_parameter": "x"', '"from_parameter": "1nv8l16"')
    )
    resp.assert_error(400, "ProcessParameterRequired")


def test_execute_apply_run_udf_100(api):
    api.check_result("apply_run_udf.json")
    assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").apply.call_count == 1


def test_reduce_temporal_run_udf(api):
    api.check_result("reduce_temporal_run_udf.json")
    if api.api_version_compare.at_least("1.0.0"):
        loadCall = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
        assert loadCall.process_types == set([ProcessType.GLOBAL_TIME])
        assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").reduce_dimension.call_count == 1
    else:
        assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").apply_tiles_spatiotemporal.call_count == 1


def test_reduce_temporal_run_udf_invalid_dimension(api):
    resp = api.result(
        "reduce_temporal_run_udf.json",
        preprocess=preprocess_check_and_replace('"dimension": "t"', '"dimension": "tempo"')
    )
    resp.assert_error(
        400,
        "ProcessParameterInvalid",
        message="The value passed for parameter 'dimension' in process 'reduce_dimension' is invalid: Must be one of ['x', 'y', 't'] but got 'tempo'.",
    )


def test_reduce_bands_run_udf(api):
    api.check_result("reduce_bands_run_udf.json")
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
        400,
        "ProcessParameterInvalid",
        message="The value passed for parameter 'dimension' in process 'reduce_dimension' is invalid: Must be one of ['x', 'y', 't', 'bands'] but got 'layers'.",
    )


def test_apply_dimension_temporal_run_udf(api):
    api.check_result("apply_dimension_temporal_run_udf.json")
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    assert dummy.apply_dimension.call_count == 1
    callback = dummy.apply_dimension.call_args.kwargs["process"]
    # check if callback is valid
    DummyVisitor().accept_process_graph(callback)
    dummy.rename_dimension.assert_called_with("t", "new_time_dimension")
    load_parameters = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert load_parameters.process_types == set([ProcessType.GLOBAL_TIME])


def test_apply_dimension_temporal_run_udf_invalid_temporal_dimension(api):
    resp = api.result(
        "apply_dimension_temporal_run_udf.json",
        preprocess=preprocess_check_and_replace('"dimension": "t"', '"dimension": "letemps"')
    )
    resp.assert_error(
        400,
        "ProcessParameterInvalid",
        message="The value passed for parameter 'dimension' in process 'apply_dimension' is invalid: Must be one of ['x', 'y', 't'] but got 'letemps'.",
    )


def test_apply_neighborhood(api):
    api.check_result(
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


def test_reduce_max_invalid_dimension(api):
    res = api.result("reduce_max.json", preprocess=preprocess_check_and_replace("PLACEHOLDER", "orbit"))
    res.assert_error(
        400,
        "ProcessParameterInvalid",
        message="The value passed for parameter 'dimension' in process 'reduce_dimension' is invalid: Must be one of ['x', 'y', 't', 'bands'] but got 'orbit'.",
    )


def test_execute_merge_cubes(api):
    api.check_result("merge_cubes.json")
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    assert dummy.merge_cubes.call_count == 1
    args, kwargs = dummy.merge_cubes.call_args
    assert args[1:] == ('or',)


def test_execute_resample_spatial_defaults(api):
    api.check_result(
        {
            "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
            "resample": {
                "process_id": "resample_spatial",
                "arguments": {"data": {"from_node": "lc"}},
                "result": True,
            },
        }
    )
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.resample_spatial.call_count == 1
    assert dummy.resample_spatial.call_args.kwargs == {
        "resolution": 0,
        "projection": None,
        "method": "near",
        "align": "upper-left",
    }


def test_execute_resample_spatial_custom(api):
    api.check_result(
        {
            "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
            "resample": {
                "process_id": "resample_spatial",
                "arguments": {
                    "data": {"from_node": "lc"},
                    "resolution": [11, 123],
                    "projection": 3857,
                    "method": "cubic",
                    "align": "lower-right",
                },
                "result": True,
            },
        }
    )
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.resample_spatial.call_count == 1
    assert dummy.resample_spatial.call_args.kwargs == {
        "resolution": [11, 123],
        "projection": 3857,
        "method": "cubic",
        "align": "lower-right",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resolution": [1, 2, 3, 4, 5]},
        {"method": "glossy"},
        {"align": "justified"},
    ],
)
def test_execute_resample_spatial_invalid(api, kwargs):
    res = api.result(
        {
            "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
            "resample": {
                "process_id": "resample_spatial",
                "arguments": {"data": {"from_node": "lc"}, **kwargs},
                "result": True,
            },
        }
    )
    res.assert_error(status_code=400, error_code="ProcessParameterInvalid")


def test_execute_resample_cube_spatial_defaults(api):
    api.check_result(
        {
            "lc1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
            "lc2": {"process_id": "load_collection", "arguments": {"id": "SENTINEL1_GRD"}},
            "resample": {
                "process_id": "resample_cube_spatial",
                "arguments": {"data": {"from_node": "lc1"}, "target": {"from_node": "lc2"}},
                "result": True,
            },
        }
    )
    cube1 = dummy_backend.get_collection("S2_FOOBAR")
    cube2 = dummy_backend.get_collection("SENTINEL1_GRD")
    assert cube1.resample_cube_spatial.call_count == 1
    assert cube1.resample_cube_spatial.call_args.kwargs == {"target": cube2, "method": "near"}


def test_execute_resample_cube_spatial_custom(api):
    api.check_result(
        {
            "lc1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
            "lc2": {"process_id": "load_collection", "arguments": {"id": "SENTINEL1_GRD"}},
            "resample": {
                "process_id": "resample_cube_spatial",
                "arguments": {"data": {"from_node": "lc1"}, "target": {"from_node": "lc2"}, "method": "lanczos"},
                "result": True,
            },
        }
    )
    cube1 = dummy_backend.get_collection("S2_FOOBAR")
    cube2 = dummy_backend.get_collection("SENTINEL1_GRD")
    assert cube1.resample_cube_spatial.call_count == 1
    assert cube1.resample_cube_spatial.call_args.kwargs == {"target": cube2, "method": "lanczos"}


def test_execute_resample_cube_spatial_invalid(api):
    res = api.result(
        {
            "lc1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
            "lc2": {"process_id": "load_collection", "arguments": {"id": "SENTINEL1_GRD"}},
            "resample": {
                "process_id": "resample_cube_spatial",
                "arguments": {"data": {"from_node": "lc1"}, "target": {"from_node": "lc2"}, "method": "du chef"},
                "result": True,
            },
        }
    )
    res.assert_error(
        status_code=400,
        error_code="ProcessParameterInvalid",
        message=re.compile(r"Invalid enum value 'du chef'\. Expected one of.*cubic.*near"),
    )


def test_execute_resample_and_merge_cubes(api):
    api.check_result("resample_and_merge_cubes.json")
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    last_load_collection_call = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert last_load_collection_call.target_crs == CRS_UTM
    assert last_load_collection_call.target_resolution == (10, 10)
    assert dummy.merge_cubes.call_count == 1
    assert dummy.resample_cube_spatial.call_count == 1
    assert dummy.resample_cube_spatial.call_args.kwargs["method"] == "cubic"
    args, kwargs = dummy.merge_cubes.call_args
    assert args[1:] == ('or',)


def test_execute_merge_cubes_and_reduce(api):
    api.check_result("merge_cubes_and_reduce.json")
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


def test_reduce_bands_invalid_dimension(api):
    res = api.result("reduce_bands.json",
                     preprocess=preprocess_check_and_replace('"dimension": "bands"', '"dimension": "layor"'))
    res.assert_error(
        400,
        "ProcessParameterInvalid",
        message="The value passed for parameter 'dimension' in process 'reduce_dimension' is invalid: Must be one of ['x', 'y', 't', 'bands'] but got 'layor'.",
    )


def test_execute_mask(api):
    api.check_result("mask.json")
    assert dummy_backend.get_collection("ESA_WORLDCOVER_10M_2020_V1").mask.call_count == 1

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

    params = dummy_backend.last_load_collection_call('ESA_WORLDCOVER_10M_2020_V1')
    assert params["spatial_extent"] == expected_spatial_extent


def test_execute_diy_mask(api):
    api.check_result("scl_mask_custom.json")
    # assert dummy_backend.get_collection("TERRASCOPE_S2_FAPAR_V2").mask.call_count == 1  # Optimized away now

    load_collections = dummy_backend.all_load_collection_calls("TERRASCOPE_S2_FAPAR_V2")
    assert len(load_collections) == 3
    assert load_collections[0].pixel_buffer == [100.5,100.5]
    assert load_collections[0].bands == ['SCENECLASSIFICATION_20M']
    assert load_collections[1].pixel_buffer == [100.5, 100.5]
    assert load_collections[1].bands == ['SCENECLASSIFICATION_20M']
    assert load_collections[2].bands == ['FAPAR_10M']



def test_execute_mask_optimized_loading(api):
    api.check_result("mask.json",
                     preprocess=preprocess_check_and_replace('"10"', 'null')
                     )
    # assert dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER").mask.call_count == 1  # Optimized away now

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

    params = dummy_backend.last_load_collection_call('ESA_WORLDCOVER_10M_2020_V1')
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
                    "result": "true"
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
                  "result": "true"
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
            "data": {
              "from_node": "loadcollection1"
            },
            "elevation_model": "COPERNICUS_30"
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
          "result": "true"
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
def test_mask_polygon_types(api, mask, expected):
    pg = {
        "lc1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "mask1": {"process_id": "mask_polygon", "arguments": {
            "data": {"from_node": "lc1"},
            "mask": mask
        }, "result": True}
    }
    api.check_result(pg)
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.mask_polygon.call_count == 1
    args, kwargs = dummy.mask_polygon.call_args
    assert isinstance(kwargs['mask'], expected)


def test_mask_polygon_vector_cube(api):
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
    api.check_result(pg)
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.mask_polygon.call_count == 1
    args, kwargs = dummy.mask_polygon.call_args
    assert isinstance(kwargs['mask'], shapely.geometry.MultiPolygon)


def test_data_mask_optimized(api):
    pg = {
        "load_collection1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "load_collection2": {"process_id": "load_collection", "arguments": {"id": "S2_FAPAR_CLOUDCOVER"}},
        "filter1": {
            "process_id": "filter_bands",
            "arguments": {
                "data": {"from_node": "load_collection1"},
                "bands": ["B02"]
            }
        },
        "mask1": {
            "process_id": "mask",
            "arguments": {"data": {"from_node": "filter1"}, "mask": {"from_node": "load_collection2"}},
            "result": True
        }
    }
    api.check_result(pg)
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    # Even with filter_bands, the load_collection optimization should work
    # mask does not need to be called when it is already applied in load_collection
    assert dummy.mask.call_count == 0

def test_data_mask_use_data_twice(api):
    pg = {
        "load_collection1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "load_collection2": {"process_id": "load_collection", "arguments": {"id": "S2_FAPAR_CLOUDCOVER"}},
        "filter1": {
            "process_id": "filter_bands",
            "arguments": {
                "data": {"from_node": "load_collection1"},
                "bands": ["B02"]
            }
        },
        "resample1": {
            "process_id": "resample_cube_spatial",
            "arguments": {
                "data": {
                    "from_node": "load_collection2"
                },
                "target": {
                    "from_node": "filter1"
                }
            }
        },
        "mask1": {
            "process_id": "mask",
            "arguments": {"data": {"from_node": "filter1"}, "mask": {"from_node": "resample1"}},
            "result": True
        }
    }
    api.check_result(pg)
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    # Not handling overlaps between mask and data nodes.
    # A load_collection under the data node could be used twice and would not be pre-masked correctly.
    assert dummy.mask.call_count == 1

def test_data_mask_unoptimized(api):
    pg = {
        "load_collection1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "load_collection2": {"process_id": "load_collection", "arguments": {"id": "S2_FAPAR_CLOUDCOVER"}},
        "apply1": {
            "process_id": "apply_kernel",
            "arguments": {
                "data": {"from_node": "load_collection1"},
                "kernel": [
                    [+0, -1, +0],
                    [-1, +4, -1],
                    [+0, -1, +0]
                ]
            },
        },
        "mask1": {
            "process_id": "mask",
            "arguments": {"data": {"from_node": "apply1"}, "mask": {"from_node": "load_collection2"}},
            "result": True
        }
    }
    api.check_result(pg)
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.mask.call_count == 1


def test_aggregate_temporal_period(api):
    api.check_result("aggregate_temporal_period_max.json")


def test_aggregate_temporal_max(api):
    api.check_result("aggregate_temporal_max.json")


def test_aggregate_temporal_max_invalid_temporal_dimension(api):
    resp = api.result(
        "aggregate_temporal_max.json",
        preprocess=preprocess_check_and_replace('"dimension": "t"', '"dimension": "detijd"')
    )
    resp.assert_error(
        400,
        "ProcessParameterInvalid",
        message="The value passed for parameter 'dimension' in process 'aggregate_temporal' is invalid: Must be one of ['x', 'y', 't'] but got 'detijd'.",
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
    params = dummy_backend.last_load_collection_call("ESA_WORLDCOVER_10M_2020_V1")
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


def test_execute_aggregate_spatial_spatial_cube(api):
    resp = api.check_result("aggregate_spatial_spatial_cube.json")
    assert resp.json == [[2.345, None], [2.0, 3.0]]


@pytest.mark.parametrize(["geometries", "expected"], [
    ("some text", "Invalid type: <class 'str'> ('some text')"),
    (1234, "Invalid type: <class 'int'> (1234)"),
    (["a", "list"], "Invalid type: <class 'list'> (['a', 'list'])")
])
def test_aggregate_spatial_invalid_geometry(api, geometries, expected):
    pg = api.load_json("aggregate_spatial.json")
    assert pg["aggregate_spatial"]["arguments"]["geometries"]
    pg["aggregate_spatial"]["arguments"]["geometries"] = geometries
    _ = api.result(pg).assert_error(400, "ProcessParameterInvalid", expected)


@pytest.mark.parametrize(["feature_collection_test_path"], [
    ["geojson/FeatureCollection02.json"],
    ["geojson/FeatureCollection05.json"]
])
def test_aggregate_spatial_vector_cube_basic(api, feature_collection_test_path):
    path = get_path(feature_collection_test_path)
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR", "bands": ["B02","B03","B04"]}},
        "lf": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": [str(path)], "format": "GeoJSON", "options": {"columns_for_cube": []}},
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
        },
        "sr": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "ag"}, "format": "GeoJSON", "options": {"flatten_prefix": "agg"}},
            "result": True
        },
    }
    res = api.check_result(pg)

    params = dummy_backend.last_load_collection_call("S2_FOOBAR")
    assert params["spatial_extent"] == {"west": 277438.26352113695, "south": 110530.15880237566, "east":  722056.3830076031, "north": 442397.8228986237, "crs": "EPSG:32631"}
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
        api, info, preprocess_pg, aggregate_data, p1_properties, p2_properties
):
    path = get_path("geojson/FeatureCollection02.json")
    pg = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR", "bands": ["B02","B03","B04"]}},
        "lf": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": [str(path)], "format": "GeoJSON", "options": {"columns_for_cube": []}},
        },
        "ag": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": aggregate_data},
                "geometries": {"from_node": "lf"},
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
        "sr": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "ag"}, "format": "GeoJSON", "options": {"flatten_prefix": "agg"}},
            "result": True
        },
    }
    pg.update(preprocess_pg)
    res = api.check_result(pg)

    params = dummy_backend.last_load_collection_call("S2_FOOBAR")
    assert isinstance(params["aggregate_spatial_geometries"], DriverVectorCube)

    assert BoundingBox.from_dict(params["spatial_extent"]).as_wsen_tuple() == params.aggregate_spatial_geometries.reproject(CRS.from_epsg(32631)).get_bounding_box()

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


def test_create_wmts_100(api):
    api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    process_graph = api.load_json("filter_temporal.json")
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
    resp = api.post('/services', json=post_data).assert_status_code(201)
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
    assert resp.json == {"2015-07-06T00:00:00Z": [[2.345], [2.0]], "2015-08-22T00:00:00Z": [[None], [3.0]]}
    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == {"west": 5, "south": 51, "east": 6, "north": 52, "crs": 'EPSG:4326'}
    assert params["temporal_extent"] == ('2017-11-21', '2017-12-21')
    assert isinstance(params["aggregate_spatial_geometries"], DriverVectorCube)
    assert len(params["aggregate_spatial_geometries"].get_cube()) == 2
    assert len(params["aggregate_spatial_geometries"].get_geometries()) == 2


def test_read_vector_no_load_collection_spatial_extent(api):
    geometry_filename = str(get_path("geojson/GeometryCollection01.json"))
    preprocess1 = preprocess_check_and_replace("PLACEHOLDER", geometry_filename)
    preprocess2 = preprocess_regex_check_and_replace(r'"spatial_extent"\s*:\s*\{.*?\},', replacement='')
    process_graph = api.load_json(
        "read_vector.json", preprocess=lambda s: preprocess2(preprocess1(s))
    )
    resp = api.check_result(process_graph)
    assert b'NaN' not in resp.data
    assert resp.json == {"2015-07-06T00:00:00Z": [[2.345], [2.0]], "2015-08-22T00:00:00Z": [[None], [3.0]]}
    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == {"west": 5.05, "south": 51.21, "east": 5.15, "north": 51.3, "crs": 'EPSG:4326'}
    assert params["temporal_extent"] == ('2017-11-21', '2017-12-21')
    assert len(params["aggregate_spatial_geometries"].get_cube()) == 2
    assert len(params["aggregate_spatial_geometries"].get_geometries()) == 2


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
def test_run_udf_on_vector_read_vector(api, udf_code):
    udf_code = textwrap.dedent(udf_code)
    process_graph = {
        "get_vector_data": {
            "process_id": "read_vector",
            "arguments": {"filename": str(get_path("geojson/FeatureCollection01.json"))},
        },
        "udf": {
            "process_id": "run_udf",
            "arguments": {
                "data": {"from_node": "get_vector_data"},
                "udf": udf_code,
                "runtime": "Python",
            },
            "result": True,
        },
    }
    resp = api.check_result(process_graph)
    assert resp.json["features"] == [
        {
            "geometry": {
                "coordinates": [[[4.47, 51.1], [4.52, 51.1], [4.52, 51.15], [4.47, 51.15], [4.47, 51.1]]],
                "type": "Polygon",
            },
            "properties": {},
            "type": "Feature",
        },
        {
            "geometry": {
                "coordinates": [[[4.45, 51.17], [4.5, 51.17], [4.5, 51.2], [4.45, 51.2], [4.45, 51.17]]],
                "type": "Polygon",
            },
            "properties": {},
            "type": "Feature",
        },
    ]


@pytest.mark.parametrize(
    "udf_code",
    [
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
    ],
)
def test_run_udf_on_vector_get_geometries(api, udf_code):
    udf_code = textwrap.dedent(udf_code)
    process_graph = {
        "get_vector_data": {
            "process_id": "get_geometries",
            "arguments": {"filename": str(get_path("geojson/FeatureCollection01.json"))},
        },
        "udf": {
            "process_id": "run_udf",
            "arguments": {
                "data": {"from_node": "get_vector_data"},
                "udf": udf_code,
                "runtime": "Python",
            },
            "result": True,
        },
    }
    resp = api.check_result(process_graph)
    assert resp.json["features"] == [
        {
            "geometry": {
                "coordinates": [[[4.47, 51.1], [4.52, 51.1], [4.52, 51.15], [4.47, 51.15], [4.47, 51.1]]],
                "type": "Polygon",
            },
            "properties": {},
            "type": "Feature",
        },
        {
            "geometry": {
                "coordinates": [[[4.45, 51.17], [4.5, 51.17], [4.5, 51.2], [4.45, 51.2], [4.45, 51.17]]],
                "type": "Polygon",
            },
            "properties": {},
            "type": "Feature",
        },
    ]


@pytest.mark.parametrize(
    "udf_code",
    [
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
    ],
)
def test_run_udf_on_vector_load_uploaded_files(api, udf_code):
    """https://github.com/Open-EO/openeo-python-driver/issues/197"""
    udf_code = textwrap.dedent(udf_code)
    process_graph = {
        "get_vector_data": {
            "process_id": "load_uploaded_files",
            "arguments": {"paths": [str(get_path("geojson/FeatureCollection01.json"))], "format": "GeoJSON"},
        },
        "udf": {
            "process_id": "run_udf",
            "arguments": {
                "data": {"from_node": "get_vector_data"},
                "udf": udf_code,
                "runtime": "Python",
            },
            "result": True,
        },
    }
    resp = api.check_result(process_graph)
    assert resp.json == [None, None]


@pytest.mark.parametrize(
    "udf_code",
    [
        """
        from openeo_udf.api.datacube import DataCube  # Old style openeo_udf API
        def fct_buffer(udf_data: UdfData):
            gdf = udf_data.feature_collection_list[0].data
            gdf = gdf.buffer(1)
            udf_data.feature_collection_list = [FeatureCollection(id="", data=gdf)]
            return udf_data
        """,
        """
        from openeo.udf import UdfData, FeatureCollection
        def fct_buffer(udf_data: UdfData):
            gdf = udf_data.feature_collection_list[0].data
            gdf = gdf.buffer(1)
            udf_data.feature_collection_list = [FeatureCollection(id="_", data=gdf)]
            return udf_data
        """,
    ],
)
def test_run_udf_on_vector_inline_geojson(api, udf_code):
    udf_code = textwrap.dedent(udf_code)
    geometry = load_json(get_path("geojson/FeatureCollection02.json"))
    process_graph = {
        "udf": {
            "process_id": "run_udf",
            "arguments": {
                "data": geometry,
                "udf": udf_code,
                "runtime": "Python",
            },
            "result": True,
        },
    }
    resp = api.check_result(process_graph)
    geometries = [f["geometry"] for f in resp.json["features"]]
    assert geometries == [
        ApproxGeoJSONByBounds(0, 0, 4, 4, types=["Polygon"], abs=0.01),
        ApproxGeoJSONByBounds(2, 1, 6, 5, types=["Polygon"], abs=0.01),
    ]


@pytest.mark.parametrize(
    "udf_code",
    [
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
def test_run_udf_on_json(api, udf_code):
    udf_code = textwrap.dedent(udf_code)
    process_graph = api.load_json(
        "run_udf_on_timeseries.json",
        preprocess=lambda s: s.replace('"PLACEHOLDER_UDF"', repr(udf_code))
    )
    resp = api.check_result(process_graph)
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
def test_run_udf_on_list(api, udf_code):
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
    resp = api.check_result(process_graph)
    assert resp.json == [1, 4, 9, 25, 64]


@pytest.mark.parametrize(
    "udf_code",
    [
        """
        from openeo.udf import UdfData, StructuredData
        from geopandas import GeoDataFrame
        from shapely.geometry import Point
        def transform(data: UdfData) -> UdfData:
            data.set_feature_collection_list([FeatureCollection("t", GeoDataFrame([{"geometry": Point(0.0, 0.1)}]))])
            return data
    """,
    ],
)
def test_run_udf_on_aggregate_spatial(api, udf_code):
    udf_code = textwrap.dedent(udf_code)
    process_graph = {
        "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR", "bands": ["B02", "B03", "B04"]}},
        "ag": {
            "process_id": "aggregate_spatial",
            "arguments": {
                "data": {"from_node": "lc"},
                "geometries": {"type": "Point", "coordinates": [5.5, 51.5]},
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
        "udf": {
            "process_id": "run_udf",
            "arguments": {"data": {"from_node": "ag"}, "udf": udf_code, "runtime": "Python"},
        },
        "sr": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "udf"}, "format": "GeoJSON"},
            "result": True,
        },
    }
    resp = api.check_result(process_graph)
    assert resp.json["type"] == "FeatureCollection"
    assert len(resp.json["features"]) == 1
    assert resp.json["features"][0]["geometry"]["coordinates"] == [0.0, 0.1]


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
def test_run_udf_on_list_runtimes(api, runtime, version, failure):
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
    resp = api.result(process_graph)
    if failure:
        error_code, message = failure
        resp.assert_error(400, error_code=error_code, message=message)
    else:
        assert resp.assert_status_code(200).json == [1, 4, 9, 25, 64]



def test_process_reference_as_argument(api):
    process_graph = api.load_json(
        "process_reference_as_argument.json"
    )
    resp = api.check_result(process_graph)


def test_load_collection_without_spatial_extent_incorporates_read_vector_extent(api):
    process_graph = api.load_json(
        "read_vector_spatial_extent.json",
        preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("geojson/GeometryCollection01.json")))
    )
    resp = api.check_result(process_graph)
    assert b"NaN" not in resp.data
    assert resp.json == {"2015-07-06T00:00:00Z": [[2.345], [2.0]], "2015-08-22T00:00:00Z": [[None], [3.0]]}
    params = dummy_backend.last_load_collection_call("PROBAV_L3_S10_TOC_NDVI_333M_V2")
    assert params["spatial_extent"] == {"west": 5.05, "south": 51.21, "east": 5.15, "north": 51.3, "crs": "EPSG:4326"}


def test_read_vector_from_feature_collection(api):
    process_graph = api.load_json(
        "read_vector_feature_collection.json",
        preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("geojson/FeatureCollection01.json")))
    )
    resp = api.check_result(process_graph)
    assert b"NaN" not in resp.data
    assert resp.json == {"2015-07-06T00:00:00Z": [[2.345], [2.0]], "2015-08-22T00:00:00Z": [[None], [3.0]]}
    params = dummy_backend.last_load_collection_call("PROBAV_L3_S10_TOC_NDVI_333M_V2")
    assert params["spatial_extent"] == {"west": 5, "south": 51, "east": 6, "north": 52, "crs": "EPSG:4326"}


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

    @pytest.fixture(scope="class")
    def test_data_file_server(self) -> str:
        """Ephemeral file server of the test data"""
        with ephemeral_fileserver(path=TEST_DATA_ROOT) as fileserver_root:
            yield fileserver_root

    def test_geojson_url(self, api, test_data_file_server):
        """Load vector cube from GeoJSON URL"""
        url = f"{test_data_file_server}/geojson/FeatureCollection02.json"
        pg = {"lf": {
                "process_id": "load_uploaded_files",
                "arguments": {"paths": [url], "format": "GeoJSON"},
                "result": True,
            }
        }
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
    def test_vector_url(self, api, path, format, test_data_file_server):
        """Load vector cube from URL"""
        url = f"{test_data_file_server}/{path}"

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

    def test_shapefile_url(self, api, tmp_path, test_data_file_server):
        """Load vector cube from shapefile (zip) URL"""
        (tmp_path / "geom.shp.zip").write_bytes(self._zip_content(get_path("shapefile").glob("mol.*")))
        with ephemeral_fileserver(path=tmp_path) as fileserver_root:
            url = f"{fileserver_root}/geom.shp.zip"
            pg = {
                "lf": {
                    "process_id": "load_uploaded_files",
                    "arguments": {"paths": [url], "format": "ESRI Shapefile"},
                    "result": True,
                }
            }
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

    @pytest.mark.parametrize(
        ["geojson", "expected"],
        [
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
                    ],
                },
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
        ],
    )
    def test_to_vector_cube(self, api, geojson, expected):
        res = api.check_result(
            {
                "vc": {
                    "process_id": "to_vector_cube",
                    "arguments": {"data": geojson},
                    "result": True,
                }
            }
        )
        assert res.json == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": expected,
            }
        )

    @pytest.mark.parametrize(
        ["geojson", "expected"],
        [
            (
                {"type": "Point", "coordinates": (1, 2)},
                [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [1, 2]},
                        "properties": {},
                    },
                ],
            ),
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
                    ],
                },
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
        ],
    )
    def test_load_geojson(self, api, geojson, expected):
        # TODO: cover `properties` parameter
        res = api.check_result(
            {"vc": {"process_id": "load_geojson", "arguments": {"data": geojson}, "result": True}}
        )
        assert res.json == DictSubSet({"type": "FeatureCollection", "features": expected})

    @pytest.mark.parametrize(
        ["geometry", "expected"],
        [
            (
                {"type": "Point", "coordinates": (1, 2)},
                [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [1, 2]},
                        "properties": {},
                    },
                ],
            ),
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
                    ],
                },
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
        ],
    )
    def test_load_url_geojson(self, api, geometry, expected, tmp_path):
        (tmp_path / "geometry.json").write_text(json.dumps(geometry))
        with ephemeral_fileserver(tmp_path) as fileserver_root:
            url = f"{fileserver_root}/geometry.json"
            res = api.check_result(
                {
                    "load": {
                        "process_id": "load_url",
                        "arguments": {"url": url, "format": "GeoJSON"},
                        "result": True,
                    }
                }
            )
        assert res.json == DictSubSet({"type": "FeatureCollection", "features": expected})


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


def test_load_disk_data_and_reduce_dimension(api):
    """https://github.com/Open-EO/openeo-geopyspark-driver/issues/500"""
    pg = {
        "loaddiskdata": {
            "process_id": "load_disk_data",
            "arguments": {
                "format": "GTiff",
                "glob_pattern": "/data/MTDA/CGS_S2/CGS_S2_FAPAR/2019/04/24/*/*/10M/*_FAPAR_10M_V102.tif",
                "options": {"date_regex": ".*_(\\d{4})(\\d{2})(\\d{2})T.*"},
            },
        },
        "reducedimension": {
            "process_id": "reduce_dimension",
            "arguments": {
                "data": {"from_node": "loaddiskdata"},
                "dimension": "t",
                "reducer": {
                    "process_graph": {
                        "sum": {"process_id": "sum", "arguments": {"data": {"from_parameter": "data"}}, "result": True}
                    }
                },
            },
            "result": True,
        },
    }
    api.check_result(pg)


def test_mask_with_vector_file(api):
    process_graph = api.load_json(
        "mask_with_vector_file.json",
        preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("geojson/MultiPolygon02.json")))
    )
    api.check_result(process_graph)


def test_aggregate_feature_collection(api):
    api.check_result("aggregate_feature_collection.json")
    params = dummy_backend.last_load_collection_call('ESA_WORLDCOVER_10M_2020_V1')
    assert params["spatial_extent"] == {"west": 5, "south": 51, "east": 6, "north": 52, "crs": 'EPSG:4326'}


def test_aggregate_feature_collection_no_load_collection_spatial_extent(api):
    preprocess = preprocess_regex_check_and_replace(r'"spatial_extent"\s*:\s*\{.*?\},', replacement='')
    api.check_result("aggregate_feature_collection.json", preprocess=preprocess)
    params = dummy_backend.last_load_collection_call('ESA_WORLDCOVER_10M_2020_V1')
    assert params["spatial_extent"] == {
        "west": 5.076, "south": 51.21, "east": 5.166, "north": 51.26, "crs": 'EPSG:4326'
    }


@pytest.mark.parametrize("auth", [True, False])
def test_post_result_process_100(api, client, auth):
    if auth:
        api.set_auth_bearer_token()
    response = api.post(
        path='/result',
        json=api.get_process_graph_dict(api.load_json("basic.json")),
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


def test_fuzzy_mask_parent_scope(api):
    api.check_result(
        "fuzzy_mask.json",
        preprocess=preprocess_check_and_replace('"from_parameter": "x"', '"from_parameter": "data"')
    )


def test_fuzzy_mask_add_dim(api):
    api.check_result("fuzzy_mask_add_dim.json")


def test_rename_labels(api):
    api.check_result("rename_labels.json")


@pytest.mark.parametrize("namespace", ["user", None, "_undefined"])
def test_user_defined_process_bbox_mol_basic(api, namespace, udp_registry):
    api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    bbox_mol_spec = api.load_json("udp/bbox_mol.json")
    udp_registry.save(user_id=TEST_USER, process_id="bbox_mol", spec=bbox_mol_spec)
    pg = api.load_json("udp_bbox_mol_basic.json")
    if namespace != "_undefined":
        pg["bboxmol1"]["namespace"] = namespace
    elif "namespace" in pg["bboxmol1"]:
        del pg["bboxmol1"]["namespace"]
    api.check_result(pg)
    params = dummy_backend.last_load_collection_call('S2_FOOBAR')
    assert params["spatial_extent"] == {"west": 5.05, "south": 51.2, "east": 5.1, "north": 51.23, "crs": 'EPSG:4326'}


@pytest.mark.parametrize("namespace", ["backend", "foobar"])
def test_user_defined_process_bbox_mol_basic_other_namespace(api, udp_registry, namespace):
    api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    bbox_mol_spec = api.load_json("udp/bbox_mol.json")
    udp_registry.save(user_id=TEST_USER, process_id="bbox_mol", spec=bbox_mol_spec)
    pg = api.load_json("udp_bbox_mol_basic.json")
    pg["bboxmol1"]["namespace"] = namespace
    api.result(pg).assert_error(status_code=400, error_code="ProcessUnsupported")


@pytest.mark.parametrize(["udp_args", "expected_start_date", "expected_end_date"], [
    ({}, "2019-01-01", None),
    ({"start_date": "2019-12-12"}, "2019-12-12", None),
    ({"end_date": "2019-12-12"}, "2019-01-01", "2019-12-12"),
    ({"start_date": "2019-08-08", "end_date": "2019-12-12"}, "2019-08-08", "2019-12-12"),
])
def test_user_defined_process_date_window(
        api, udp_registry, udp_args, expected_start_date, expected_end_date
):
    api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api.load_json("udp/date_window.json")
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

    api.check_result(pg)
    params = dummy_backend.last_load_collection_call('S2_FOOBAR')
    assert params["temporal_extent"] == (expected_start_date, expected_end_date)


def test_user_defined_process_required_parameter(api, udp_registry):
    api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api.load_json("udp/date_window.json")
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

    response = api.result(pg)
    response.assert_error(400, "ProcessParameterRequired", message="parameter 'data' is required")


@pytest.mark.parametrize("set_parameter", [False, True])
def test_udp_udf_reduce_dimension(api, udp_registry, set_parameter):
    # TODO: eliminate this test?
    api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api.load_json("udp/udf_reduce_dimension.json")
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

    response = api.result(pg).assert_status_code(200)
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.reduce_dimension.call_count == 1
    dummy.reduce_dimension.assert_called_with(reducer=mock.ANY, dimension="bands", context=None, env=mock.ANY)
    args, kwargs = dummy.reduce_dimension.call_args
    assert "runudf1" in kwargs["reducer"]
    env: EvalEnv = kwargs["env"]
    expected_param = "test_the_udfparam" if set_parameter else "udfparam_default"
    assert env.collect_parameters()["udfparam"] == expected_param


class TestParameterHandingUdpWithUdf:
    """
    Tests for parameter handling with UDP doing a reduce_dimension/apply/apply_dimension/... and a UDF
    """

    def _build_udp(
        self,
        parent: str,
        *,
        parent_context: Optional[dict] = None,
        udf_context: Optional[dict] = None,
    ) -> dict:
        """Build UDP as dict"""
        run_udf_args = {"data": {"from_parameter": "data"}, "udf": "print('hello world')", "runtime": "Python"}
        if udf_context:
            run_udf_args["context"] = udf_context

        child = {"process_graph": {"runudf1": {"process_id": "run_udf", "arguments": run_udf_args, "result": True}}}

        parent_args = {
            "data": {"from_parameter": "data"},
        }
        if parent in ["reduce_dimension", "apply_dimension"]:
            parent_args["dimension"] = "bands"
        elif parent in ["apply_neighborhood"]:
            parent_args["size"] = [
                {"dimension": "x", "value": 128, "unit": "px"},
                {"dimension": "y", "value": 128, "unit": "px"},
            ]
            parent_args["overlap"] = [
                {"dimension": "x", "value": 16, "unit": "px"},
                {"dimension": "y", "value": 16, "unit": "px"},
            ]
        if parent in ["reduce_dimension"]:
            parent_args["reducer"] = child
        elif parent in ["apply", "apply_dimension", "apply_neighborhood"]:
            parent_args["process"] = child
        else:
            raise ValueError(parent)
        if parent_context:
            parent_args["context"] = parent_context

        udp = {
            "id": "parameterized_udf",
            "parameters": [
                {"name": "data", "schema": {"type": "object", "subtype": "raster-cube"}},
                {
                    "name": "udp_param",
                    "schema": {"type": "string"},
                    "optional": True,
                    "default": "udp_param_default",
                },
            ],
            "returns": {"schema": {"type": "object", "subtype": "raster-cube"}},
            "process_graph": {
                "parent1": {
                    "process_id": parent,
                    "arguments": parent_args,
                    "result": True,
                }
            },
        }
        return udp

    def _build_process_graph(self, udp_param: Optional[str] = None) -> dict:
        udp_args = {"data": {"from_node": "loadcollection1"}}
        if udp_param:
            udp_args["udp_param"] = udp_param
        pg = {
            "loadcollection1": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
            "parameterizedudf1": {
                "process_id": "parameterized_udf",
                "namespace": "user",
                "arguments": udp_args,
                "result": True,
            },
        }
        return pg

    @dataclasses.dataclass(frozen=True)
    class _UseCase:
        parent_context: Optional[dict]
        udf_context: Optional[dict]
        set_udp_parameter: bool
        expected_context: Optional[dict]
        expected_udp_param: Optional[str]

    _use_cases = [
        _UseCase(
            parent_context=None,
            udf_context={"udf_param": {"from_parameter": "udp_param"}},
            set_udp_parameter=False,
            expected_context=None,
            expected_udp_param="udp_param_default",
        ),
        _UseCase(
            parent_context=None,
            udf_context={"udf_param": {"from_parameter": "udp_param"}},
            set_udp_parameter=True,
            expected_context=None,
            expected_udp_param="udp_param_123",
        ),
        _UseCase(
            parent_context={"udf_param": {"from_parameter": "udp_param"}},
            udf_context={"from_parameter": "context"},
            set_udp_parameter=False,
            expected_context={"udf_param": "udp_param_default"},
            expected_udp_param="udp_param_default",
        ),
        _UseCase(
            parent_context={"udf_param": {"from_parameter": "udp_param"}},
            udf_context={"from_parameter": "context"},
            set_udp_parameter=True,
            expected_context={"udf_param": "udp_param_123"},
            expected_udp_param="udp_param_123",
        ),
    ]

    @pytest.mark.parametrize("use_case", _use_cases)
    def test_reduce_dimension(self, api, udp_registry, use_case: _UseCase):
        api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)

        # Build + register UDP, build process graph and execute.
        udp = self._build_udp(
            parent="reduce_dimension", parent_context=use_case.parent_context, udf_context=use_case.udf_context
        )
        udp_registry.save(user_id=TEST_USER, process_id="parameterized_udf", spec=udp)
        pg = self._build_process_graph(udp_param="udp_param_123" if use_case.set_udp_parameter else None)
        _ = api.result(pg).assert_status_code(200)

        parent_mock: mock.Mock = dummy_backend.get_collection("S2_FOOBAR").reduce_dimension
        assert parent_mock.mock_calls == [
            mock.call(reducer=mock.ANY, dimension="bands", context=use_case.expected_context, env=mock.ANY)
        ]
        parent_env: EvalEnv = parent_mock.call_args.kwargs["env"]
        assert parent_env.collect_parameters()["udp_param"] == use_case.expected_udp_param

    @pytest.mark.parametrize("use_case", _use_cases)
    def test_apply(self, api, udp_registry, use_case: _UseCase):
        api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)

        # Build + register UDP, build process graph and execute.
        udp = self._build_udp(parent="apply", parent_context=use_case.parent_context, udf_context=use_case.udf_context)
        udp_registry.save(user_id=TEST_USER, process_id="parameterized_udf", spec=udp)
        pg = self._build_process_graph(udp_param="udp_param_123" if use_case.set_udp_parameter else None)
        _ = api.result(pg).assert_status_code(200)

        parent_mock: mock.Mock = dummy_backend.get_collection("S2_FOOBAR").apply
        assert parent_mock.mock_calls == [mock.call(process=mock.ANY, context=use_case.expected_context, env=mock.ANY)]
        parent_env: EvalEnv = parent_mock.call_args.kwargs["env"]
        assert parent_env.collect_parameters()["udp_param"] == use_case.expected_udp_param

    @pytest.mark.parametrize("use_case", _use_cases)
    def test_apply_dimension(self, api, udp_registry, use_case: _UseCase):
        api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)

        # Build + register UDP, build process graph and execute.
        udp = self._build_udp(
            parent="apply_dimension", parent_context=use_case.parent_context, udf_context=use_case.udf_context
        )
        udp_registry.save(user_id=TEST_USER, process_id="parameterized_udf", spec=udp)
        pg = self._build_process_graph(udp_param="udp_param_123" if use_case.set_udp_parameter else None)
        _ = api.result(pg).assert_status_code(200)

        parent_mock: mock.Mock = dummy_backend.get_collection("S2_FOOBAR").apply_dimension
        assert parent_mock.mock_calls == [
            mock.call(
                process=mock.ANY,
                dimension="bands",
                target_dimension=None,
                context=use_case.expected_context,
                env=mock.ANY,
            )
        ]
        parent_env: EvalEnv = parent_mock.call_args.kwargs["env"]
        assert parent_env.collect_parameters()["udp_param"] == use_case.expected_udp_param

    @pytest.mark.parametrize("use_case", _use_cases)
    def test_apply_neighborhood(self, api, udp_registry, use_case: _UseCase):
        api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)

        # Build + register UDP, build process graph and execute.
        udp = self._build_udp(
            parent="apply_neighborhood", parent_context=use_case.parent_context, udf_context=use_case.udf_context
        )
        udp_registry.save(user_id=TEST_USER, process_id="parameterized_udf", spec=udp)
        pg = self._build_process_graph(udp_param="udp_param_123" if use_case.set_udp_parameter else None)
        _ = api.result(pg).assert_status_code(200)

        parent_mock: mock.Mock = dummy_backend.get_collection("S2_FOOBAR").apply_neighborhood
        assert parent_mock.mock_calls == [
            mock.call(
                process=mock.ANY,
                size=mock.ANY,
                overlap=mock.ANY,
                context=use_case.expected_context,
                env=mock.ANY,
            )
        ]
        parent_env: EvalEnv = parent_mock.call_args.kwargs["env"]
        assert parent_env.collect_parameters()["udp_param"] == use_case.expected_udp_param


@pytest.mark.parametrize("set_parameter", [False, True])
def test_udp_apply_neighborhood(api, udp_registry, set_parameter):
    api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api.load_json("udp/udf_apply_neighborhood.json")
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
    expected_param = "test_the_udfparam" if set_parameter else "udfparam_default"

    response = api.result(pg).assert_status_code(200)
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.apply_neighborhood.call_count == 1
    dummy.apply_neighborhood.assert_called_with(
        process=mock.ANY, size=mock.ANY, overlap=mock.ANY, env=mock.ANY, context=None
    )
    args, kwargs = dummy.apply_neighborhood.call_args
    assert "runudf1" in kwargs["process"]
    env: EvalEnv = kwargs["env"]
    assert env.collect_parameters()["udfparam"] == expected_param


def test_user_defined_process_udp_vs_pdp_priority(api, udp_registry):
    """
    See https://github.com/Open-EO/openeo-python-driver/issues/353
    This test was effectively asserting that the backend correctly follows API.
    Unfortunately, the prescribed behaviour of allowing to override predefined processes is somewhat questionable.
    On top of that, it lead to a major operational issue.
    When the above issue is resolved, this test should be adjusted to check the desired behaviour.
    """
    api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    # First without a defined "ndvi" UDP
    api.check_result("udp_ndvi.json")
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.ndvi.call_count == 1
    dummy.ndvi.assert_called_with(nir="nir", red="red", target_band=None)
    assert dummy.reduce_dimension.call_count == 0

    # Overload ndvi with UDP.
    udp_registry.save(user_id=TEST_USER, process_id="ndvi", spec=api.load_json("udp/myndvi.json"))
    api.check_result("udp_ndvi.json")
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.ndvi.call_count == 2
    assert dummy.reduce_dimension.call_count == 0
    #dummy.reduce_dimension.assert_called_with(reducer=mock.ANY, dimension="bands", context=None, env=mock.ANY)
    #args, kwargs = dummy.reduce_dimension.call_args
    #assert "red" in kwargs["reducer"]
    #assert "nir" in kwargs["reducer"]


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


@pytest.mark.parametrize(
    ["namespace", "url_mocks", "expected_error"],
    [
        pytest.param(
            "https://share.test/u42/bbox_mol.json",
            {"https://share.test/u42/bbox_mol.json": "udp/bbox_mol.json"},
            None,
            id="basic",
        ),
        pytest.param(
            "https://share.test/u42/bbox_mol",
            {"https://share.test/u42/bbox_mol": "udp/bbox_mol.json"},
            None,
            id="simple-no-extension",
        ),
        pytest.param(
            "https://share.test/u42/bbox_mol.json",
            {
                "https://share.test/u42/bbox_mol.json": (302, "https://shr976.test/45435"),
                "https://shr976.test/45435": "udp/bbox_mol.json",
            },
            None,
            id="redirect",
        ),
        pytest.param(
            "https://share.test/u42/",
            {
                "https://share.test/u42/": {
                    "processes": [
                        {"id": "foo", "process_graph": {}},
                        load_json("pg/1.0/udp/bbox_mol.json"),
                    ],
                    "links": [],
                }
            },
            None,
            id="process-listing",
        ),
        pytest.param(
            "https://share.test/u42/bbox_mol.json",
            {"https://share.test/u42/bbox_mol.json": 404},
            (
                400,
                "ProcessNamespaceInvalid",
                "Process 'bbox_mol' specified with invalid namespace 'https://share.test/u42/bbox_mol.json': HTTPError('404 Client Error: Not Found for url: https://share.test/u42/bbox_mol.json')",
            ),
            id="error-404",
        ),
        pytest.param(
            "https://share.test/u42/bbox_mol.json",
            {"https://share.test/u42/bbox_mol.json": "[1,2,3]"},
            (
                400,
                "ProcessNamespaceInvalid",
                "Process 'bbox_mol' specified with invalid namespace 'https://share.test/u42/bbox_mol.json': ValueError(\"Process definition should be a JSON object, but got <class 'list'>.\")",
            ),
            id="error-no-dict",
        ),
        pytest.param(
            "https://share.test/u42/bbox_mol.json",
            {"https://share.test/u42/bbox_mol.json": {"foo": "bar"}},
            (
                400,
                "ProcessNotFound",
                "No valid process definition for 'bbox_mol' found at 'https://share.test/u42/bbox_mol.json'.",
            ),
            id="error-no-id",
        ),
        pytest.param(
            "https://share.test/u42/bbox_mol.json",
            {"https://share.test/u42/bbox_mol.json": '{"foo": invalid json'},
            (
                400,
                "ProcessNamespaceInvalid",
                "Process 'bbox_mol' specified with invalid namespace 'https://share.test/u42/bbox_mol.json': JSONDecodeError('Expecting value: line 1 column 9 (char 8)')",
            ),
            id="error-invalid-json",
        ),
        pytest.param(
            "https://share.test/u42/bbox_mol.json",
            {
                "https://share.test/u42/bbox_mol.json": load_json(
                    "pg/1.0/udp/bbox_mol.json", preprocess=lambda t: t.replace("bbox_mol", "BBox_Mol")
                )
            },
            (
                400,
                "ProcessIdMismatch",
                "Mismatch between expected process 'bbox_mol' and process 'BBox_Mol' defined at 'https://share.test/u42/bbox_mol.json'.",
            ),
            id="error-id-mismatch-capitalization",
        ),
        pytest.param(
            "https://share.test/u42/bbox_mol.json",
            {
                "https://share.test/u42/bbox_mol.json": load_json(
                    "pg/1.0/udp/bbox_mol.json", preprocess=lambda t: t.replace("bbox_mol", "BoundingBox-Mol")
                )
            },
            (
                400,
                "ProcessIdMismatch",
                "Mismatch between expected process 'bbox_mol' and process 'BoundingBox-Mol' defined at 'https://share.test/u42/bbox_mol.json'.",
            ),
            id="error-id-mismatch-different-id",
        ),
        pytest.param(
            "https://share.test/u42/",
            {
                "https://share.test/u42/": {
                    "processes": [
                        {"id": "foo", "process_graph": {}},
                        {"id": "bar", "process_graph": {}},
                    ],
                    "links": [],
                }
            },
            (400, "ProcessNotFound", "Process 'bbox_mol' not found in process listing at 'https://share.test/u42/'."),
            id="process-listing-missing",
        ),
    ],
)
def test_evaluate_process_from_url(api, requests_mock, namespace, url_mocks, expected_error):
    for url, value in url_mocks.items():
        if isinstance(value, str):
            if value.endswith(".json"):
                bbox_mol_spec = api.load_json(value)
                requests_mock.get(url, json=bbox_mol_spec)
            else:
                requests_mock.get(url, text=value)
        elif isinstance(value, dict):
            requests_mock.get(url, json=value)
        elif value in [404, 500]:
            requests_mock.get(url, status_code=value, reason=http.client.responses.get(value))
        elif isinstance(value, tuple) and value[0] in [302]:
            status_code, target = value
            requests_mock.get(url, status_code=status_code, headers={"Location": target})
        else:
            raise ValueError(value)

    # Evaluate process graph (with URL namespace)
    pg = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR"},
        },
        "bboxmol1": {
            "process_id": "bbox_mol",
            "namespace": namespace,
            "arguments": {"data": {"from_node": "loadcollection1"}},
            "result": True,
        },
    }
    res = api.result(pg)
    if expected_error:
        status_code, error_code, message = expected_error
        res.assert_error(status_code=status_code, error_code=error_code, message=message)
    else:
        res.assert_status_code(200)
        params = dummy_backend.last_load_collection_call('S2_FOOBAR')
        assert params["spatial_extent"] == {"west": 5.05, "south": 51.2, "east": 5.1, "north": 51.23, "crs": 'EPSG:4326'}


def test_execute_no_cube_1_plus_2(api):
    # Calculator as a service!
    res = api.result({
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
    ({"s": {"process_id": "array_apply", "arguments": {"data": [3, -25, 8], "process":
        { "process_graph":{"s": {"process_id": "add", "arguments": {"x": {"from_parameter": "x"}, "y": 5}, "result": True}}
        }},
                "result": True}}, [8, -20, 13]),
])
def test_execute_no_cube_just_math(api, process_graph, expected):
    assert api.result(process_graph).assert_status_code(200).json == pytest.approx(expected,0.0001)


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
def test_execute_no_cube_logic(api, process_graph, expected):
    assert api.result(process_graph).assert_status_code(200).json == expected


@pytest.mark.parametrize(
    ["process_id", "arguments", "expected"],
    [
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
        # TODO: `text_merge` is deprecated (in favor of `text_concat`)
        ("text_merge", {"data": ["foo", "bar"]}, "foobar"),
        ("text_merge", {"data": ["foo", "bar"], "separator": "--"}, "foo--bar"),
        ("text_merge", {"data": [1, 2, 3], "separator": "/"}, "1/2/3"),
        ("text_merge", {"data": [1, "b"], "separator": 0}, "10b"),
        ("text_concat", {"data": ["foo", "bar"]}, "foobar"),
        ("text_concat", {"data": ["foo", "bar"], "separator": "--"}, "foo--bar"),
        ("text_concat", {"data": [1, 2, 3], "separator": "/"}, "1/2/3"),
        ("text_concat", {"data": [1, "b"], "separator": 0}, "10b"),
    ],
)
def test_text_processes(api, process_id, arguments, expected):
    if process_id == "text_merge" and api.api_version_compare >= "1.2":
        pytest.skip("text_merge is dropped since API version 1.2")
    # TODO: null propagation (`text_begins(data=null,...) -> null`) can not be tested at the moment
    pg = {"t": {"process_id": process_id, "arguments": arguments, "result":True}}
    assert api.result(pg).assert_status_code(200).json == expected


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
def test_execute_no_cube_just_arrays(api, process_graph, expected):
    assert api.result(process_graph).assert_status_code(200).json == expected


def test_execute_no_cube_dynamic_args(api):
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
    api.check_result(pg)
    apply_kernel_mock = dummy_backend.get_collection("S2_FOOBAR").apply_kernel
    args, kwargs = apply_kernel_mock.call_args
    assert kwargs["factor"] == 7.75


@pytest.mark.parametrize(["border", "expected"], [(0, 0)])
def test_execute_apply_kernel_border(api, border, expected):
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
    api.check_result(pg)
    apply_kernel_mock = dummy_backend.get_collection("S2_FOOBAR").apply_kernel
    args, kwargs = apply_kernel_mock.call_args
    assert kwargs["border"] == expected


# TODO: test using dynamic arguments in bbox_filter (not possible yet: see EP-3509)


def test_execute_EP3509_process_order(api):
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
            "data": {"from_node": "filterbands1"}, "kernel": [[1,1],[1,1]]
        }, "result": True}
    }
    api.check_result(pg)
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


def test_reduce_add_reduce_dim(api):
    """Test reduce_dimension -> add_dimension -> reduce_dimension"""
    content = api.check_result("reduce_add_reduce_dimension.json")
    dummy = dummy_backend.get_collection("S2_FOOBAR")

    assert dummy.reduce_dimension.call_count == 1

    dims = content.json["cube:dimensions"]
    names = [k for k in dims]
    assert names == ["x", "y", "t"]


def test_reduce_drop_dimension(api):
    content = api.check_result({
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


def test_dimension_labels(api):
    res = api.check_result(
        {
            "lc": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
            "dl": {
                "process_id": "dimension_labels",
                "arguments": {"data": {"from_node": "lc"}, "dimension": "bands"},
                "result": True,
            },
        }
    )
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.dimension_labels.call_count == 1
    assert res.json == ["B02", "B03", "B04", "B08"]


@pytest.mark.parametrize(["arguments", "expected"], [
    ({"data": {"from_node": "l"}, "name": "foo", "label": "bar"}, "other"),
    ({"data": {"from_node": "l"}, "name": "foo", "label": "bar", "type": "bands"}, "bands"),
])
def test_add_dimension_type_argument(api, arguments, expected):
    api.check_result({
        "l": {"process_id": "load_collection", "arguments": {"id": "S2_FOOBAR"}},
        "s": {"process_id": "add_dimension", "arguments": arguments, "result": True}
    })
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    dummy.add_dimension.assert_called_with(name="foo", label="bar", type=expected)


def test_add_dimension_duplicate(api):
    res = api.result({
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


def test_execute_load_collection_sar_backscatter_defaults(api):
    api.check_result({
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


def test_execute_load_collection_sar_backscatter_none_values(api):
    api.check_result({
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


def test_execute_load_collection_sar_backscatter(api):
    api.check_result({
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


def test_execute_load_collection_sar_backscatter_compatibility(api):
    # assert that we can differentiate between collections that are sar_backscatter compatible and those that are not
    api.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FAPAR_CLOUDCOVER"},
            "result": True
        }
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params.sar_backscatter is None


def test_execute_load_collection_custom_properties(api):
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

    api.check_result(pg)
    params = dummy_backend.all_load_collection_calls("S2_FAPAR_CLOUDCOVER")
    assert len(params) == 2
    assert params[0].properties == properties
    assert params[1].properties == asc_props


def test_execute_load_collection_custom_cloud_mask(api):
    # assert that we can differentiate between collections that are sar_backscatter compatible and those that are not
    api.check_result({
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


def test_execute_load_collection_custom_l1c_cloud_mask(api):
    api.check_result({
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


def test_execute_load_collection_resolution_merge(api):
    api.check_result({
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


def test_execute_custom_process_by_process_graph_minimal(api):
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
    res = api.check_result({
        "do_math": {
            "process_id": process_id,
            "arguments": {"x": 2},
            "result": True
        },
    }).json
    assert res == 3


def test_execute_custom_process_by_process_graph(api):
    process_id = generate_unique_test_process_id()

    # Register a custom process with process graph
    process_spec = api.load_json("add_and_multiply.json")
    process_spec["id"] = process_id
    custom_process_from_process_graph(process_spec=process_spec)
    # Apply process
    res = api.check_result({
        "do_math": {
            "process_id": process_id,
            "arguments": {"data": 2},
            "result": True
        },
    }).json
    assert res == 25


def test_execute_custom_process_by_process_graph_json(api, tmp_path):
    process_id = generate_unique_test_process_id()

    process_spec = api.load_json("add_and_multiply.json")
    process_spec["id"] = process_id
    path = tmp_path / f"{process_id}.json"
    with path.open("w") as f:
        json.dump(process_spec, f)

    # Register a custom process with process graph
    custom_process_from_process_graph(path)

    # Apply process
    res = api.check_result({
        "do_math": {
            "process_id": process_id,
            "arguments": {"data": 2},
            "result": True
        },
    }).json
    assert res == 25


def test_execute_custom_process_by_process_graph_namespaced(api):
    process_id = generate_unique_test_process_id()

    # Register a custom process with process graph
    process_spec = api.load_json("add_and_multiply.json")
    process_spec["id"] = process_id
    custom_process_from_process_graph(process_spec=process_spec, namespace="madmath")
    # Apply process
    res = api.check_result({
        "do_math": {
            "process_id": process_id,
            "namespace": "madmath",
            "arguments": {"data": 3},
            "result": True
        },
    }).json
    assert res == 30


@pytest.mark.parametrize("hidden", [False, True])
def test_execute_custom_process_by_process_graph_hidden(api, hidden):
    process_id = generate_unique_test_process_id()
    # Register a custom process with minimal process graph
    process_spec = {
        "id": process_id,
        "process_graph": {
            "increment": {"process_id": "add", "arguments": {"x": {"from_parameter": "x"}, "y": 1}, "result": True}
        },
    }
    custom_process_from_process_graph(process_spec=process_spec, hidden=hidden)
    # Apply process
    res = api.check_result(
        {
            "do_math": {"process_id": process_id, "arguments": {"x": 2}, "result": True},
        }
    ).json
    assert res == 3

    process_ids = set(p["id"] for p in api.get("/processes").assert_status_code(200).json["processes"])
    assert (process_id not in process_ids) == hidden


def test_normalized_difference(api):
    res = api.check_result({
        "do_math": {
            "process_id": "normalized_difference",
            "arguments": {"x": 3, "y": 5},
            "result": True
        },
    }).json
    assert res == -0.25


def test_ard_normalized_radar_backscatter(api):
    api.check_result({
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


def test_ard_normalized_radar_backscatter_without_optional_arguments(api):
    api.check_result({
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
def test_date_shift(api, date, value, unit, expected):
    pg = {
        "dateshift1": {
            "process_id": "date_shift",
            "arguments": {"date": date, "value": value, "unit": unit},
            "result": True,
        }
    }
    res = api.result(pg).assert_status_code(200).json
    assert res == expected


@pytest.mark.parametrize(
    ["buf", "unit", "repr_geom", "expected_type", "bounds"],
    [
        # Template to calculate expected bounds of single geometries:
        #       import geopandas as gpd
        #       shape = Point(5.1, 51.22)
        #       series = gpd.GeoSeries([shape], crs=4326)
        #       buffered = series.to_crs(32631).buffer(100).to_crs(4326)
        #       print(buffered.iloc[0].bounds)
        (
            100,
            "meter",
            {"type": "Point", "coordinates": (5.1, 51.22)},
            "Polygon",
            (5.098569060859835, 51.219101388461105, 5.101430937032029, 51.22089861137662),
        ),
        (
            1000,
            "meter",
            {"type": "Point", "coordinates": (5.1, 51.22)},
            "Polygon",
            (5.085690513968438, 51.21101387730941, 5.114309275218151, 51.22898610646433),
        ),
        (
            100,
            "kilometer",
            {"type": "Point", "coordinates": (5.1, 51.22)},
            "Polygon",
            (3.668246824769951, 50.321307522288016, 6.529646150453797, 52.11853021385508),
        ),
        (
            100,
            "meter",
            {"type": "MultiPoint", "coordinates": ((5.1, 51.22), (5.11, 51.23))},
            "MultiPolygon",
            (5.098569060859835, 51.219101388461105, 5.111431237949069, 51.23089860406231),
        ),
        (
            750,
            "meter",
            {"type": "MultiPoint", "coordinates": ((5.1, 51.22), (5.11, 51.23))},
            "Polygon",
            (5.089267905161828, 51.213260409503235, 5.120734232861271, 51.23673952650676),
        ),
        (
            100,
            "meter",
            {"type": "LineString", "coordinates": ((5.1, 51.22), (5.11, 51.23))},
            "Polygon",
            (5.09856905615526, 51.21910138452717, 5.111431257271536, 51.23089861521429),
        ),
        (
            100,
            "meter",
            load_json("geojson/Polygon01.json"),
            "Polygon",
            (5.0985703229820665, 51.19910120996896, 5.141430817509717, 51.23089799863948),
        ),
        (
            -500,
            "meter",
            load_json("geojson/Polygon01.json"),
            "Polygon",
            (5.1084522374601145, 51.206647698484225, 5.129458164090344, 51.222687097228174),
        ),
        (
            1,
            "kilometer",
            load_json("geojson/Polygon01.json"),
            "Polygon",
            (5.087874843800581, 51.19113803453279, 5.154275007454256, 51.23894705603116),
        ),
        (
            100,
            "meter",
            load_json("geojson/MultiPolygon01.json"),
            "MultiPolygon",
            (5.098787598453318, 51.19911380673043, 5.141427521031006, 51.2408707459342),
        ),
        (
            500,
            "meter",
            load_json("geojson/MultiPolygon01.json"),
            "Polygon",
            (5.092851581512912, 51.19550604834508, 5.147157170251627, 51.244493919231424),
        ),
        (
            1000,
            "meter",
            load_json("geojson/Feature01.json"),
            "Polygon",
            (0.9910, 0.9910, 3.009, 3.009),
        ),
        (
            1000,
            "meter",
            load_json("geojson/FeatureCollection01.json"),
            "FeatureCollection",
            (4.43568898, 51.09100882, 4.53429533, 51.20899105),
        ),
        (
            1000,
            "meter",
            str(get_path("geojson/FeatureCollection01.json")),
            "FeatureCollection",
            (4.43568898, 51.09100882, 4.53429533, 51.20899105),
        ),
    ],
)
def test_vector_buffer(api, buf, unit, repr_geom, expected_type, bounds):
    pg = {
        "vectorbuffer1": {
            "process_id": "vector_buffer",
            "arguments": {"geometry": repr_geom, "distance": buf, "unit": unit},
            "result": True,
        }
    }
    res = api.result(pg).assert_status_code(200).json
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
def test_vector_buffer_non_epsg4326(api, distance, unit, expected):
    geometry = load_json("geojson/FeatureCollection03.json")
    pg = {
        "vectorbuffer1": {
            "process_id": "vector_buffer",
            "arguments": {"geometry": geometry, "distance": distance, "unit": unit},
            "result": True,
        }
    }
    res = api.result(pg).assert_status_code(200).json
    assert res["type"] == "FeatureCollection"
    res_gs = gpd.GeoSeries([shapely.geometry.shape(feat["geometry"]) for feat in res["features"]])
    assert res_gs.total_bounds == pytest.approx(expected, abs=0.0001)


@pytest.mark.parametrize(["distance", "unit", "expected"], [
    (-10, "meter", [5.0144, 51.1738, 5.0173, 51.1769]),
    (+10, "meter", [5.0141, 51.1736, 5.0176, 51.1771]),
    (1, "kilometer", [4.9999, 51.1647, 5.0318, 51.1860]),
])
def test_vector_buffer_ogc_crs84(api, distance, unit, expected):
    geometry = load_json("geojson/FeatureCollection04.json")
    pg = {
        "vectorbuffer1": {
            "process_id": "vector_buffer",
            "arguments": {"geometry": geometry, "distance": distance, "unit": unit},
            "result": True,
        }
    }
    res = api.result(pg).assert_status_code(200).json
    assert res["type"] == "FeatureCollection"
    res_gs = gpd.GeoSeries([shapely.geometry.shape(feat["geometry"]) for feat in res["features"]])
    assert res_gs.total_bounds == pytest.approx(expected, abs=0.0001)


def test_load_result(api):
    if api.api_version_compare >= "1.2":
        pytest.skip("load_result is dropped since API version 1.2")
    api.check_result("load_result.json")
    params = dummy_backend.last_load_collection_call("99a605a0-1a10-4ba9-abc1-6898544e25fc")

    assert params["temporal_extent"] == ('2019-09-22', '2019-09-22')


def test_chunk_polygon(api):
    if api.api_version_compare >= "1.2":
        pytest.skip("chunk_polygon is dropped since API version 1.2")

    api.check_result("chunk_polygon.json")
    params = dummy_backend.last_load_collection_call("S2_FOOBAR")
    assert params["spatial_extent"] == {"west": 1.0, "south": 5.0, "east": 12.0, "north": 16.0, "crs": "EPSG:4326"}


def test_apply_polygon_legacy(api, caplog):
    caplog.set_level(logging.WARNING)
    api.check_result("apply_polygon.json", preprocess=lambda s: s.replace("geometries", "polygons"))
    params = dummy_backend.last_load_collection_call("S2_FOOBAR")
    assert params["spatial_extent"] == {"west": 1.0, "south": 5.0, "east": 12.0, "north": 16.0, "crs": "EPSG:4326"}
    assert "In process 'apply_polygon': parameter 'polygons' is deprecated, use 'geometries' instead." in caplog.text


def test_apply_polygon(api, caplog):
    caplog.set_level(logging.WARNING)
    api.check_result("apply_polygon.json")
    params = dummy_backend.last_load_collection_call("S2_FOOBAR")
    assert params["spatial_extent"] == {"west": 1.0, "south": 5.0, "east": 12.0, "north": 16.0, "crs": "EPSG:4326"}
    # TODO due to #288 we can not simply assert absence of any warnings/errors
    # assert caplog.text == ""
    assert "deprecated" not in caplog.text


def test_apply_polygon_no_geometries(api):
    res = api.result("apply_polygon.json", preprocess=lambda s: s.replace("geometries", "heometriez"))
    res.assert_error(400, "ProcessParameterRequired", "Process 'apply_polygon' parameter 'geometries' is required")


@pytest.mark.parametrize("geometries_argument", ["polygons", "geometries"])
def test_apply_polygon_with_vector_cube(api, tmp_path, geometries_argument):
    shutil.copy(get_path("geojson/FeatureCollection01.json"), tmp_path / "geometry.json")
    with ephemeral_fileserver(tmp_path) as fileserver_root:
        url = f"{fileserver_root}/geometry.json"

        pg = {
            "load_raster": {
                "process_id": "load_collection",
                "arguments": {"id": "S2_FOOBAR"},
            },
            "load_vector": {
                "process_id": "load_url",
                "arguments": {"url": str(url), "format": "GeoJSON"},
            },
            "apply_polygon": {
                "process_id": "apply_polygon",
                "arguments": {
                    "data": {"from_node": "load_raster"},
                    geometries_argument: {"from_node": "load_vector"},
                    "process": {
                        "process_graph": {
                            "constant": {
                                "process_id": "constant",
                                "arguments": {"x": {"from_parameter": "x"}},
                                "result": True,
                            }
                        }
                    },
                },
                "result": True,
            },
        }
        _ = api.check_result(pg)
        dummy = dummy_backend.get_collection("S2_FOOBAR")
        assert dummy.apply_polygon.call_count == 1
        polygons = dummy.apply_polygon.call_args.kwargs["polygons"]
        # TODO #288 instead of MultPolygon, this should actually be a vector cube, feature collection or something equivalent
        assert isinstance(polygons, shapely.geometry.MultiPolygon)
        assert polygons.bounds == (4.45, 51.1, 4.52, 51.2)


def test_fit_class_random_forest(api):
    res = api.check_result("fit_class_random_forest.json")

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
                                        "B02": 2.345,
                                        "B03": None,
                                        "B04": 2.0,
                                        "B08": 3.0,
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
                                        "B02": 4.0,
                                        "B03": 5.0,
                                        "B04": 6.0,
                                        "B08": 7.0,
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


def test_fit_class_catboost(api):
    res = api.check_result("fit_class_catboost.json")

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
                "process_id": "fit_class_catboost",
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
                                        "B02": 2.345,
                                        "B03": None,
                                        "B04": 2.0,
                                        "B08": 3.0,
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
                                        "B02": 4.0,
                                        "B03": 5.0,
                                        "B04": 6.0,
                                        "B08": 7.0,
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
                "iterations": 10,
                "depth": 16,
                "seed": 8,
                "border_count": 254,
            },
        }
    )


def test_if_merge_cubes(api):
    api.check_result({
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FOOBAR",
                "temporal_extent": ["2021-09-01", "2021-09-10"],
                "spatial_extent": {"west": 3, "south": 51, "east": 3.1, "north": 51.1},
                "bands": ["B04"],
            }},
        "eq1": {"process_id": "eq", "arguments": {"x": 4, "y": 3}},
        "errornode": {"process_id": "doesntExist", "arguments": {}},
        "if1": {
            "process_id": "if",
            "arguments": {
                "value": {"from_node": "eq1"},
                "accept": {"from_node": "errornode"}, "reject": {"from_node": "loadcollection1"},
            }},
        "mergecubes1": {
            "process_id": "merge_cubes",
            "arguments": {"cube1": {"from_node": "loadcollection1"}, "cube2": {"from_node": "if1"}},
            "result": True
        }
    })




def test_vector_buffer_returns_error_on_empty_result_geometry(api):
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "id": "52",
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[5.17540782, 51.2721762], [5.17541833, 51.27226422], [5.17332007, 51.27209092],
                                 [5.17331899, 51.27206998], [5.17331584, 51.27200861], [5.17540782, 51.2721762]]]
            }
        }, {
            "id": "53",
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[5.21184352, 51.16855893], [5.21189095, 51.16861253], [5.21162038, 51.16887065],
                                 [5.21141722, 51.16910133], [5.21117999, 51.16902886], [5.21103991, 51.16899166],
                                 [5.21143081, 51.16848752], [5.21155978, 51.16848455], [5.21184352, 51.16855893]]]
            }
        }]
    }

    resp = api.result({
        'vectorbuffer1': {
            'process_id': 'vector_buffer',
            'arguments': {
                'geometry': geojson,
                'distance': -10,
                'unit': 'meter'
            },
            'result': True
        }
    })

    resp.assert_error(400, "ProcessParameterInvalid",
                      message="The value passed for parameter 'geometry' in process 'vector_buffer' is invalid:"
                              " Buffering with distance -10 meter resulted in empty geometries at position(s) [0]")


@pytest.mark.parametrize(
    ["request_costs", "job_options", "expected_costs_header"],
    [
        # Default backend_implementation.request_costs
        (None, None, None),
        # request_costs override
        (
            lambda user, request_id, success, job_options: 1234 + isinstance(user, User),
            None,
            "1235",
        ),
        # Extra job options handling
        (
            lambda user, request_id, success, job_options: 1234 * job_options.get("extra", 0),
            {"extra": 2},
            "2468",
        ),
    ],
)
@pytest.mark.parametrize("success", [False, True])
def test_synchronous_processing_request_costs(
    api, backend_implementation, request_costs, job_options, success, expected_costs_header
):
    if request_costs is None:
        request_costs = backend_implementation.request_costs

    if success:
        side_effect = backend_implementation.catalog.load_collection
    else:
        side_effect = Exception("nope")

    with mock.patch.object(
        backend_implementation.catalog, "load_collection", side_effect=side_effect
    ) as load_collection, mock.patch.object(
        FlaskRequestCorrelationIdLogging, "_build_request_id", return_value="r-abc123"
    ), mock.patch.object(
        backend_implementation, "request_costs", side_effect=request_costs, autospec=request_costs
    ) as get_request_costs:
        api.ensure_auth_header()
        pg = {"lc": {"process_id": "load_collection", "arguments": {"id": "S2_FAPAR_CLOUDCOVER"}, "result": True}}
        post_data = {
            "process": {"process_graph": pg},
            "job_options": job_options,
        }
        resp = api.post(path="/result", json=post_data)
        if success:
            resp.assert_status_code(200)
            if expected_costs_header:
                assert resp.headers["OpenEO-Costs-experimental"] == expected_costs_header
        else:
            resp.assert_status_code(500)

    assert load_collection.call_count == 1

    env = load_collection.call_args[1]["env"]
    assert env["correlation_id"] == "r-abc123"

    get_request_costs.assert_called_with(
        user=User(TEST_USER, internal_auth_data={"authentication_method": "Basic"}),
        job_options=job_options,
        success=success,
        request_id="r-abc123",
    )


def test_load_stac_default_temporal_extent(api, backend_implementation):
    with mock.patch.object(backend_implementation, "load_stac") as load_stac:
        api.ensure_auth_header()
        pg = {"loadstac1": {"process_id": "load_stac", "arguments": {"url": "https://stac.test"}, "result": True}}
        post_data = {
            "process": {"process_graph": pg},
        }
        api.post(path="/result", json=post_data)

    assert load_stac.call_count == 1
    _, kwargs = load_stac.call_args

    assert kwargs["load_params"]["temporal_extent"] == ["1970-01-01", "2070-01-01"]


class TestVectorCubeRunUDF:
    """
    Tests about running UDF based manipulations on vector cubes

    References:
    - https://github.com/Open-EO/openeo-python-driver/issues/197
    - https://github.com/Open-EO/openeo-python-driver/pull/200
    - https://github.com/Open-EO/openeo-geopyspark-driver/issues/437
    """

    def _build_run_udf_callback(self, udf_code: str) -> dict:
        udf_code = textwrap.dedent(udf_code)
        return {
            "process_graph": {
                "runudf1": {
                    "process_id": "run_udf",
                    "arguments": {
                        "data": {"from_parameter": "data"},
                        "udf": udf_code,
                        "runtime": "Python",
                    },
                    "result": True,
                }
            },
        }

    @pytest.mark.parametrize(
        "dimension",
        [
            "properties",
            "geometry",
        ],
    )
    def test_apply_dimension_run_udf_change_geometry(self, api, dimension):
        """VectorCube + apply_dimension + UDF (changing geometry)"""
        process_graph = {
            "load": {
                "process_id": "load_geojson",
                "arguments": {
                    "data": load_json("geojson/FeatureCollection02.json"),
                    "properties": ["pop"],
                },
            },
            "apply_dimension": {
                "process_id": "apply_dimension",
                "arguments": {
                    "data": {"from_node": "load"},
                    "dimension": dimension,
                    "process": self._build_run_udf_callback(
                        """
                        from openeo.udf import UdfData, FeatureCollection
                        def process_vector_cube(udf_data: UdfData) -> UdfData:
                            [feature_collection] = udf_data.get_feature_collection_list()
                            gdf = feature_collection.data
                            gdf["geometry"] = gdf["geometry"].buffer(distance=1, resolution=2)
                            udf_data.set_feature_collection_list([
                                FeatureCollection(id="_", data=gdf),
                            ])
                        """
                    ),
                },
                "result": True,
            },
        }
        resp = api.check_result(process_graph)
        assert resp.json == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": ApproxGeoJSONByBounds(0, 0, 4, 4, types=["Polygon"], abs=0.1),
                        "properties": {"id": "first", "pop": 1234},
                    },
                    {
                        "type": "Feature",
                        "geometry": ApproxGeoJSONByBounds(2, 1, 6, 5, types=["Polygon"], abs=0.1),
                        "properties": {"id": "second", "pop": 5678},
                    },
                ],
            }
        )

    @pytest.mark.parametrize(
        "dimension",
        [
            # TODO: this "dimension="properties" use case does not strictly follow the openEO API spec
            #       `apply_dimension` only allows changing the cardinality of the provided dimension ("properties"),
            #       not any other dimension ("geometries" here).
            "properties",
            "geometry",
        ],
    )
    def test_apply_dimension_run_udf_filter_on_geometries(self, api, dimension):
        """
        Test to use `apply_dimension(dimension="...", process=UDF)` to filter out certain
        entries from geometries dimension based on geometry (e.g. intersection with another geometry)
        """
        process_graph = {
            "load": {
                "process_id": "load_geojson",
                "arguments": {
                    "data": load_json("geojson/FeatureCollection10.json"),
                    "properties": ["pop"],
                },
            },
            "apply_dimension": {
                "process_id": "apply_dimension",
                "arguments": {
                    "data": {"from_node": "load"},
                    "dimension": dimension,
                    "process": self._build_run_udf_callback(
                        """
                        from openeo.udf import UdfData, FeatureCollection
                        import shapely.geometry
                        def process_vector_cube(udf_data: UdfData) -> UdfData:
                            [feature_collection] = udf_data.get_feature_collection_list()
                            gdf = feature_collection.data
                            to_intersect = shapely.geometry.box(4, 3, 8, 4)
                            gdf = gdf[gdf["geometry"].intersects(to_intersect)]
                            udf_data.set_feature_collection_list([
                                FeatureCollection(id="_", data=gdf),
                            ])
                        """
                    ),
                },
                "result": True,
            },
        }
        resp = api.check_result(process_graph)
        assert resp.json == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": ApproxGeoJSONByBounds(3, 2, 5, 4, types=["Polygon"], abs=0.1),
                        "properties": {"id": "second", "pop": 456},
                    },
                    {
                        "type": "Feature",
                        "geometry": ApproxGeoJSONByBounds(6, 2, 12, 6, types=["Polygon"], abs=0.1),
                        "properties": {"id": "third", "pop": 789},
                    },
                ],
            }
        )

    @pytest.mark.parametrize(
        "dimension",
        [
            # TODO: this "dimension="properties" use case does not strictly follow the openEO API spec
            #       `apply_dimension` only allows changing the cardinality of the provided dimension ("properties"),
            #       not any other dimension ("geometries" here).
            "properties",
            "geometry",
        ],
    )
    def test_apply_dimension_run_udf_filter_on_properties(self, api, dimension):
        """
        Test to use `apply_dimension(dimension="...", process=UDF)` to filter out certain
        entries from geometries dimension, based on feature properties

        Note in case of dimension="properties":
            strictly speaking, this approach draws outside the lines of the openEO API spec
            as apply_dimension only allows changing the cardinality of the provided dimension ("properties" in this case),
            not any other dimension (like "geometries" in this case).
        """
        process_graph = {
            "load": {
                "process_id": "load_geojson",
                "arguments": {
                    "data": load_json("geojson/FeatureCollection10.json"),
                    "properties": ["pop"],
                },
            },
            "apply_dimension": {
                "process_id": "apply_dimension",
                "arguments": {
                    "data": {"from_node": "load"},
                    "dimension": dimension,
                    "process": self._build_run_udf_callback(
                        """
                        from openeo.udf import UdfData, FeatureCollection
                        def process_vector_cube(udf_data: UdfData) -> UdfData:
                            [feature_collection] = udf_data.get_feature_collection_list()
                            gdf = feature_collection.data
                            gdf = gdf[gdf["pop"] > 500]
                            udf_data.set_feature_collection_list([
                                FeatureCollection(id="_", data=gdf),
                            ])
                        """
                    ),
                },
                "result": True,
            },
        }
        resp = api.check_result(process_graph)
        assert resp.json == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": ApproxGeoJSONByBounds(6.0, 2.0, 12.0, 6.0, types=["Polygon"], abs=0.1),
                        "properties": {"id": "third", "pop": 789},
                    },
                    {
                        "type": "Feature",
                        "geometry": ApproxGeoJSONByBounds(-2.0, 7.0, 5.0, 14.0, types=["Polygon"], abs=0.1),
                        "properties": {"id": "fourth", "pop": 101112},
                    },
                ],
            }
        )

    @pytest.mark.parametrize(
        "dimension",
        [
            "properties",
            # TODO: this "dimension="geometry" use case does not strictly follow the openEO API spec
            #       `apply_dimension` only allows changing the cardinality of the provided dimension ("geometry"),
            #       not any other dimension ("properties" here).
            "geometry",
        ],
    )
    def test_apply_dimension_run_udf_add_properties(self, api, dimension):
        """
        Test to use `apply_dimension(dimension="...", process=UDF)` to add properties
        """
        process_graph = {
            "load": {
                "process_id": "load_geojson",
                "arguments": {
                    "data": load_json("geojson/FeatureCollection02.json"),
                    "properties": ["pop"],
                },
            },
            "apply_dimension": {
                "process_id": "apply_dimension",
                "arguments": {
                    "data": {"from_node": "load"},
                    "dimension": dimension,
                    "process": self._build_run_udf_callback(
                        """
                        from openeo.udf import UdfData, FeatureCollection
                        def process_vector_cube(udf_data: UdfData) -> UdfData:
                            [feature_collection] = udf_data.get_feature_collection_list()
                            gdf = feature_collection.data
                            gdf["poppop"] = gdf["pop"] ** 2
                            udf_data.set_feature_collection_list([
                                FeatureCollection(id="_", data=gdf),
                            ])
                        """
                    ),
                },
                "result": True,
            },
        }
        resp = api.check_result(process_graph)
        assert resp.json == DictSubSet(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": ApproxGeoJSONByBounds(1.0, 1.0, 3.0, 3.0, types=["Polygon"], abs=0.1),
                        "properties": {"id": "first", "pop": 1234, "poppop": 1234 * 1234},
                    },
                    {
                        "type": "Feature",
                        "geometry": ApproxGeoJSONByBounds(3.0, 2.0, 5.0, 4.0, types=["Polygon"], abs=0.1),
                        "properties": {"id": "second", "pop": 5678, "poppop": 5678 * 5678},
                    },
                ],
            }
        )


def test_request_id_in_response_header(api):
    result = api.check_result({
        'collection': {
            'process_id': 'load_collection',
            'arguments': {'id': 'S2_FAPAR_CLOUDCOVER'},
            'result': True
        }
    })

    request_id = result.headers["Request-Id"]
    assert request_id.startswith("r-"), request_id


@pytest.mark.parametrize(
    ["arguments", "expected"],
    [
        (
            {},
            {
                "erosion_kernel_size": 0,
                "kernel1_size": 17,
                "kernel2_size": 201,
                "mask1_values": [2, 4, 5, 6, 7],
                "mask2_values": [3, 8, 9, 10, 11],
            },
        ),
        (
            {"kernel1_size": 1717, "mask1_values": [666, 777]},
            {
                "erosion_kernel_size": 0,
                "kernel1_size": 1717,
                "kernel2_size": 201,
                "mask1_values": [666, 777],
                "mask2_values": [3, 8, 9, 10, 11],
            },
        ),
    ],
)
def test_to_scl_dilation_mask_defaults(api, arguments, expected):
    api.check_result(
        {
            "loadcollection1": {
                "process_id": "load_collection",
                "arguments": {"id": "SENTINEL2_L2A_SENTINELHUB", "bands": ["SCL"]},
            },
            "to_scl_dilation_mask": {
                "process_id": "to_scl_dilation_mask",
                "arguments": {**{"data": {"from_node": "loadcollection1"}}, **arguments},
                "result": True,
            },
        }
    )

    dummy = dummy_backend.get_collection("SENTINEL2_L2A_SENTINELHUB")
    assert dummy.to_scl_dilation_mask.call_count == 1
    args, kwargs = dummy.to_scl_dilation_mask.call_args
    assert args == ()
    assert kwargs == expected


def test_synchronous_processing_response_header_openeo_identifier(api):
    with mock.patch.object(FlaskRequestCorrelationIdLogging, "_build_request_id", return_value="r-abc123"):
        res = api.result({"add1": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}})
    assert res.assert_status_code(200).json == 8
    assert res.headers["OpenEO-Identifier"] == "r-abc123"


@pytest.fixture
def custom_process_registry(backend_implementation) -> ProcessRegistry:
    process_registry = ProcessRegistry()
    process_registry.add_hidden(collect)
    with mock.patch.object(backend_implementation.processing, "get_process_registry", return_value=process_registry):
        yield process_registry


@pytest.mark.parametrize(
    ["post_data_base", "expected_job_options"],
    [
        ({}, None),
        ({"job_options": {"speed": "slow"}}, {"speed": "slow"}),
        ({"_x_speed": "slow"}, {"_x_speed": "slow"}),
    ],
)
def test_synchronous_processing_job_options(api, custom_process_registry, post_data_base, expected_job_options):
    """Test job options handling in synchronous processing in EvalEnv"""
    actual_job_options = []
    def i_spy_with_my_little_eye(args: ProcessArgs, env: EvalEnv):
        nonlocal actual_job_options
        if not env.get(ENV_DRY_RUN_TRACER):
            actual_job_options.append(env.get("job_options"))
        return args.get("x")

    custom_process_registry.add_function(i_spy_with_my_little_eye, spec={"id": "i_spy_with_my_little_eye"})

    pg = {"ispy": {"process_id": "i_spy_with_my_little_eye", "arguments": {"x": 123}, "result": True}}
    post_data = {
        **post_data_base,
        **{"process": {"process_graph": pg}},
    }
    api.ensure_auth_header()
    res = api.post(path="/result", json=post_data)
    assert res.assert_status_code(200).json == 123
    assert actual_job_options == [expected_job_options]


@pytest.mark.parametrize(
    ["default_job_options", "given_job_options", "expected_job_options"],
    [
        (None, None, None),
        ({}, {}, {}),
        ({"cpu": "yellow"}, {}, {"cpu": "yellow"}),
        ({}, {"cpu": "yellow"}, {"cpu": "yellow"}),
        ({"cpu": "yellow"}, {"cpu": "blue"}, {"cpu": "blue"}),
        (
            {"memory": "2GB", "cpu": "yellow"},
            {"memory": "4GB", "queue": "fast"},
            {"cpu": "yellow", "memory": "4GB", "queue": "fast"},
        ),
    ],
)
def test_synchronous_processing_job_options_and_defaults_from_remote_process_definition(
    api, custom_process_registry, requests_mock, default_job_options, given_job_options, expected_job_options
):
    process_definition = {
        "id": "add3",
        "process_graph": {
            "add": {"process_id": "add", "arguments": {"x": {"from_parameter": "x"}, "y": 3}, "result": True}
        },
        "parameters": [
            {"name": "x", "schema": {"type": "number"}},
        ],
        "returns": {"schema": {"type": "number"}},
    }
    if default_job_options is not None:
        process_definition["default_synchronous_options"] = default_job_options
    requests_mock.get("https://share.test/add3.json", json=process_definition)

    actual_job_options = []

    def i_spy_with_my_little_eye(args: ProcessArgs, env: EvalEnv):
        nonlocal actual_job_options
        if not env.get(ENV_DRY_RUN_TRACER):
            actual_job_options.append(env.get("job_options"))
        return args.get("x")

    custom_process_registry.add_function(i_spy_with_my_little_eye, spec={"id": "i_spy_with_my_little_eye"})
    custom_process_registry.add_process(name="add", function=lambda args, env: args.get("x") + args.get("y"))

    pg = {
        "ispy": {"process_id": "i_spy_with_my_little_eye", "arguments": {"x": 123}},
        "add3": {
            "process_id": "add3",
            "namespace": "https://share.test/add3.json",
            "arguments": {"x": {"from_node": "ispy"}},
            "result": True,
        },
    }
    post_data = {
        "process": {"process_graph": pg},
    }
    if given_job_options is not None:
        post_data["job_options"] = given_job_options

    api.ensure_auth_header()
    res = api.post(path="/result", json=post_data)
    assert res.assert_status_code(200).json == 126
    assert actual_job_options == [expected_job_options]


def test_load_collection_property_from_parameter(api, udp_registry):
    # You can't execute a top-level process graph with parameters; instead, you execute a top-level process graph
    # that calls a UDP with parameters.

    api.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    udp_spec = api.load_json("udp/load_collection_property_from_parameter.json")
    udp_registry.save(user_id=TEST_USER, process_id="load_collection_property_from_parameter", spec=udp_spec)
    pg = api.load_json("udp_load_collection_property_from_parameter.json")
    api.check_result(pg)

    params = dummy_backend.last_load_collection_call("SENTINEL1_GRD")

    assert "sat:orbit_state" in params.properties
