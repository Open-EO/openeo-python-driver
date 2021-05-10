import json
import os
import re
from unittest import mock

import numpy as np
import pytest
import shapely.geometry

from openeo_driver.ProcessGraphDeserializer import custom_process_from_process_graph
from openeo_driver.backend import get_backend_implementation
from openeo_driver.datastructs import SarBackscatterArgs, ResolutionMergeArgs
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.dry_run import ProcessType
from openeo_driver.dummy import dummy_backend
from openeo_driver.errors import ProcessGraphMissingException
from openeo_driver.testing import ApiTester, preprocess_check_and_replace, TEST_USER, TEST_USER_BEARER_TOKEN, \
    preprocess_regex_check_and_replace, generate_unique_test_process_id
from openeo_driver.utils import EvalEnv
from openeo_driver.views import app
from .data import get_path, TEST_DATA_ROOT

os.environ["DRIVER_IMPLEMENTATION_PACKAGE"] = "openeo_driver.dummy.dummy_backend"

user_defined_process_registry = get_backend_implementation().user_defined_processes


@pytest.fixture(params=["0.4.0", "1.0.0"])
def api_version(request):
    return request.param


@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture
def api(api_version, client) -> ApiTester:
    dummy_backend.reset()
    data_root = TEST_DATA_ROOT / "pg" / (".".join(api_version.split(".")[:2]))
    return ApiTester(api_version=api_version, client=client, data_root=data_root)


@pytest.fixture
def api040(client) -> ApiTester:
    dummy_backend.reset()
    data_root = TEST_DATA_ROOT / "pg" / "0.4"
    return ApiTester(api_version="0.4.0", client=client, data_root=data_root)


@pytest.fixture
def api100(client) -> ApiTester:
    dummy_backend.reset()
    data_root = TEST_DATA_ROOT / "pg" / "1.0"
    return ApiTester(api_version="1.0.0", client=client, data_root=data_root)


def test_udf_runtimes(api):
    runtimes = api.get('/udf_runtimes').assert_status_code(200).json
    assert "Python" in runtimes
    assert "type" in runtimes["Python"]
    assert "default" in runtimes["Python"]


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


def test_load_collection_spatial_extent_geojson(api):
    api.check_result({
        'collection': {
            'process_id': 'load_collection',
            'arguments': {
                'id': 'S2_FAPAR_CLOUDCOVER',
                'spatial_extent': {
                    "type": "Polygon",
                    "coordinates": [[[7.0, 46.1], [7.3, 46.0], [7.6, 46.3], [7.2, 46.6], [7.0, 46.1]]]},
                'temporal_extent': ['2018-01-01', '2018-12-31']
            },
            'result': True
        }
    })
    params = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
    assert params["temporal_extent"] == ('2018-01-01', '2018-12-31')
    assert params["spatial_extent"] == {"west": 7.0, "south": 46.0, "east": 7.6, "north": 46.6, "crs": "EPSG:4326"}


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


def test_execute_apply_run_udf(api040):
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
    assert dummy.apply_tiles_spatiotemporal.call_count == 1
    assert dummy.apply_dimension.call_count == 1
    if api.api_version_compare.at_least("1.0.0"):
        dummy.rename_dimension.assert_called_with('t', 'new_time_dimension')
        load_parameters = dummy_backend.last_load_collection_call("S2_FAPAR_CLOUDCOVER")
        assert load_parameters.process_types == set([ProcessType.GLOBAL_TIME])


def test_apply_dimension_temporal_run_udf_legacy_client(api):
    api.check_result(
        "apply_dimension_temporal_run_udf.json",
        preprocess=preprocess_check_and_replace('"dimension": "t"', '"dimension": "temporal"')
    )
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    assert dummy.apply_tiles_spatiotemporal.call_count == 1
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

    expected = {
        "west": 7.02,
        "south": 51.2,
        "east": 7.65,
        "north": 51.7,
        "crs": 'EPSG:4326',
    }
    expected_geometry = shapely.geometry.shape({
        "type": "Polygon",
        "coordinates": [[[7.02, 51.7], [7.65, 51.7], [7.65, 51.2], [7.04, 51.3], [7.02, 51.7]]]
    })

    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == expected
    assert params["aggregate_spatial_geometries"] == expected_geometry

    params = dummy_backend.last_load_collection_call('S2_FAPAR_CLOUDCOVER')
    assert params["spatial_extent"] == expected


def test_execute_mask_polygon(api):
    api.check_result("mask_polygon.json")
    dummy = dummy_backend.get_collection("S2_FAPAR_CLOUDCOVER")
    assert dummy.mask_polygon.call_count == 1
    args, kwargs = dummy.mask_polygon.call_args
    assert isinstance(kwargs['mask'], shapely.geometry.Polygon)


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


def test_execute_aggregate_spatial(api):
    resp = api.check_result("aggregate_spatial.json")
    assert resp.json == {
        "2015-07-06T00:00:00": [2.345],
        "2015-08-22T00:00:00": [None]
    }
    params = dummy_backend.last_load_collection_call('S2_FAPAR_CLOUDCOVER')
    assert params["spatial_extent"] == {"west": 7.02, "south": 51.29, "east": 7.65, "north": 51.75, "crs": 'EPSG:4326'}
    assert params["aggregate_spatial_geometries"] == shapely.geometry.shape({
        "type": "Polygon",
        "coordinates": [[[7.02, 51.75], [7.65, 51.74], [7.65, 51.29], [7.04, 51.31], [7.02, 51.75]]]
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
    geometry_filename = str(get_path("GeometryCollection.geojson"))
    process_graph = api.load_json(
        "read_vector.json",
        preprocess=preprocess_check_and_replace("PLACEHOLDER", geometry_filename)
    )
    resp = api.check_result(process_graph)
    assert b'NaN' not in resp.data
    assert resp.json == {"2015-07-06T00:00:00": [2.345], "2015-08-22T00:00:00": [None]}
    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == {"west": 5, "south": 51, "east": 6, "north": 52, "crs": 'EPSG:4326'}
    assert params["temporal_extent"] == ('2017-11-21', '2017-12-21')
    assert params["aggregate_spatial_geometries"] == DelayedVector(geometry_filename)


def test_read_vector_no_load_collection_spatial_extent(api):
    geometry_filename = str(get_path("GeometryCollection.geojson"))
    preprocess1 = preprocess_check_and_replace("PLACEHOLDER", geometry_filename)
    preprocess2 = preprocess_regex_check_and_replace(r'"spatial_extent"\s*:\s*\{.*?\},', replacement='')
    process_graph = api.load_json(
        "read_vector.json", preprocess=lambda s: preprocess2(preprocess1(s))
    )
    resp = api.check_result(process_graph)
    assert b'NaN' not in resp.data
    assert resp.json == {"2015-07-06T00:00:00": [2.345], "2015-08-22T00:00:00": [None]}
    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == {"west": 5.05, "south": 51.21, "east": 5.15, "north": 51.3, "crs": 'EPSG:4326'}
    assert params["temporal_extent"] == ('2017-11-21', '2017-12-21')
    assert params["aggregate_spatial_geometries"] == DelayedVector(geometry_filename)


def test_run_udf_on_vector(api100):
    process_graph = api100.load_json(
        "run_udf_on_vector.json",
        preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("GeometryCollection.geojson")))
    )
    resp = api100.check_result(process_graph)
    print(resp.json)
    assert len(resp.json) == 2
    assert resp.json[0]['type'] == 'Polygon'


def test_run_udf_on_json(api100):
    process_graph = api100.load_json(
        "run_udf_on_timeseries.json"
    )
    resp = api100.check_result(process_graph)
    assert resp.json == {'len': 2, 'keys': ['2015-07-06T00:00:00', '2015-08-22T00:00:00'], 'values': [[2.345], [None]]}


def test_process_reference_as_argument(api100):
    process_graph = api100.load_json(
        "process_reference_as_argument.json"
    )
    resp = api100.check_result(process_graph)
    print(resp.json)


def test_load_collection_without_spatial_extent_incorporates_read_vector_extent(api):
    process_graph = api.load_json(
        "read_vector_spatial_extent.json",
        preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("GeometryCollection.geojson")))
    )
    resp = api.check_result(process_graph)
    assert b'NaN' not in resp.data
    assert resp.json == {
        "2015-07-06T00:00:00": [2.345],
        "2015-08-22T00:00:00": [None]
    }
    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == {"west": 5.05, "south": 51.21, "east": 5.15, "north": 51.3, "crs": 'EPSG:4326'}


def test_read_vector_from_feature_collection(api):
    process_graph = api.load_json(
        "read_vector_feature_collection.json",
        preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("FeatureCollection.geojson")))
    )
    resp = api.check_result(process_graph)
    assert b'NaN' not in resp.data
    assert resp.json == {
        "2015-07-06T00:00:00": [2.345],
        "2015-08-22T00:00:00": [None]
    }
    params = dummy_backend.last_load_collection_call('PROBAV_L3_S10_TOC_NDVI_333M_V2')
    assert params["spatial_extent"] == {"west": 5, "south": 51, "east": 6, "north": 52, "crs": 'EPSG:4326'}


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
        preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("mask_polygons_3.43_51.00_3.46_51.02.json")))
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


def test_missing_process_graph(api):
    api.set_auth_bearer_token()
    response = api.post(path='/result', json={"foo": "bar"})
    response.assert_error(status_code=ProcessGraphMissingException.status_code, error_code='ProcessGraphMissing')


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
def test_user_defined_process_bbox_mol_basic(api100, namespace):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    bbox_mol_spec = api100.load_json("udp/bbox_mol.json")
    user_defined_process_registry.save(user_id=TEST_USER, process_id="bbox_mol", spec=bbox_mol_spec)
    pg = api100.load_json("udp_bbox_mol_basic.json")
    if namespace != "_undefined":
        pg["bboxmol1"]["namespace"] = namespace
    elif "namespace" in pg["bboxmol1"]:
        del pg["bboxmol1"]["namespace"]
    api100.check_result(pg)
    params = dummy_backend.last_load_collection_call('S2_FOOBAR')
    assert params["spatial_extent"] == {"west": 5.05, "south": 51.2, "east": 5.1, "north": 51.23, "crs": 'EPSG:4326'}


@pytest.mark.parametrize("namespace", ["backend", "foobar"])
def test_user_defined_process_bbox_mol_basic_other_namespace(api100, namespace):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    bbox_mol_spec = api100.load_json("udp/bbox_mol.json")
    user_defined_process_registry.save(user_id=TEST_USER, process_id="bbox_mol", spec=bbox_mol_spec)
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
        api100, udp_args, expected_start_date, expected_end_date
):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api100.load_json("udp/date_window.json")
    user_defined_process_registry.save(user_id=TEST_USER, process_id="date_window", spec=spec)

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


def test_user_defined_process_required_parameter(api100):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api100.load_json("udp/date_window.json")
    user_defined_process_registry.save(user_id=TEST_USER, process_id="date_window", spec=spec)

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
def test_udp_udf_reduce_dimension(api100, set_parameter):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api100.load_json("udp/udf_reduce_dimension.json")
    user_defined_process_registry.save(user_id=TEST_USER, process_id="udf_reduce_dimension", spec=spec)

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
    dummy.reduce_dimension.assert_called_with(reducer=mock.ANY, dimension="bands", env=mock.ANY)
    args, kwargs = dummy.reduce_dimension.call_args
    assert "runudf1" in kwargs["reducer"]
    env: EvalEnv = kwargs["env"]
    expected_param = "test_the_udfparam" if set_parameter else "udfparam_default"
    assert env.collect_parameters()["udfparam"] == expected_param


@pytest.mark.parametrize("set_parameter", [False, True])
def test_udp_apply_neighborhood(api100, set_parameter):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    spec = api100.load_json("udp/udf_apply_neighborhood.json")
    user_defined_process_registry.save(user_id=TEST_USER, process_id="udf_apply_neighborhood", spec=spec)

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


def test_user_defined_process_udp_vs_pdp_priority(api100):
    api100.set_auth_bearer_token(TEST_USER_BEARER_TOKEN)
    # First without a defined "ndvi" UDP
    api100.check_result("udp_ndvi.json")
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.ndvi.call_count == 1
    dummy.ndvi.assert_called_with(nir=None, red=None, target_band=None)
    assert dummy.reduce_dimension.call_count == 0

    # Overload ndvi with UDP.
    user_defined_process_registry.save(user_id=TEST_USER, process_id="ndvi", spec=api100.load_json("udp/myndvi.json"))
    api100.check_result("udp_ndvi.json")
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.ndvi.call_count == 1
    assert dummy.reduce_dimension.call_count == 1
    dummy.reduce_dimension.assert_called_with(reducer=mock.ANY, dimension="bands", env=mock.ANY)
    args, kwargs = dummy.reduce_dimension.call_args
    assert "red" in kwargs["reducer"]
    assert "nir" in kwargs["reducer"]


def test_execute_03_style_filter_bbox(api):
    res = api.result({"filterbbox1": {
        "process_id": "filter_bbox",
        "arguments": {
            "data": 123,
            "west": 4.6511, "east": 4.6806, "north": 51.20859, "south": 51.18997, "crs": "epsg:4326"
        },
        "result": True
    }})
    res.assert_error(
        status_code=400, error_code="ProcessParameterRequired",
        message="Process 'filter_bbox' parameter 'extent' is required"
    )


def test_execute_03_style_filter_temporal(api):
    res = api.result({"filtertemporal1": {
        "process_id": "filter_temporal",
        "arguments": {
            "data": 123,
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


@pytest.mark.parametrize(["url", "namespace"], [
    ("https://oeo.net/user/123/procs/bbox_mol.json", "https://oeo.net/user/123/procs"),
    ("https://oeo.net/user/123/procs/bbox_mol.json", "https://oeo.net/user/123/procs/"),
    ("https://oeo.net/user/123/procs/bbox_mol.json", "https://oeo.net/user/123/procs/bbox_mol.json"),
    ("https://oeo.net/user/123/procs/foo.json", "https://oeo.net/user/123/procs/foo.json"),
    ("http://oeo.net/user/123/procs/bbox_mol.json", "http://oeo.net/user/123/procs"),
])
def test_evaluate_process_from_url(api100, requests_mock, url, namespace):
    # Setup up "online" definition of `bbox_mol` process
    bbox_mol_spec = api100.load_json("udp/bbox_mol.json")
    url_mock = requests_mock.get(url, json=bbox_mol_spec)

    # Evaluate process graph (with URL namespace)
    pg = api100.load_json("udp_bbox_mol_basic.json")
    pg["bboxmol1"]["namespace"] = namespace
    api100.check_result(pg)

    params = dummy_backend.last_load_collection_call('S2_FOOBAR')
    assert params["spatial_extent"] == {"west": 5.05, "south": 51.2, "east": 5.1, "north": 51.23, "crs": 'EPSG:4326'}
    assert url_mock.called


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
    ({"arc1": {"process_id": "array_contains", "arguments": {"data": [2, 8, 5, 3], "value": 5}, "result": True}}, True),
    ({"arf1": {"process_id": "array_find", "arguments": {"data": [2, 8, 5, 3], "value": 5}, "result": True}}, 2),
    ({"srt1": {"process_id": "sort", "arguments": {"data": [2, 8, 5, 3]}, "result": True}}, [2, 3, 5, 8]),
    # any/all implementation is bit weird at the moment https://github.com/Open-EO/openeo-processes-python/issues/16
    # ({"any1": {"process_id": "any", "arguments": {"data": [False, True, False]}, "result": True}}, True),
    # ({"all1": {"process_id": "all", "arguments": {"data": [False, True, False]}, "result": True}}, False),
])
def test_execute_no_cube_just_math(api100, process_graph, expected):
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
    api100.check_result("reduce_add_reduce_dimension.json")
    dummy = dummy_backend.get_collection("S2_FOOBAR")
    assert dummy.reduce_dimension.call_count == 2
    assert dummy.add_dimension.call_count == 1


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
