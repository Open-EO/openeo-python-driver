import json
import os
from unittest import TestCase

import pytest
from flask import Response
from flask.testing import FlaskClient

import dummy_impl
from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo_driver.views import app
from .data import load_json, get_path

os.environ["DRIVER_IMPLEMENTATION_PACKAGE"] = "dummy_impl"


class Test(TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        dummy_impl.collections = {}

    def _post_process_graph(self, process_graph: dict, url='preview'):
        # TODO #33 "preview" is an old 0.3-style endpoint. The 1.0-style endpoint for synchronous execution is "/result"
        if not url.startswith('/openeo/'):
            url = '/openeo/0.4.2/' + url.lstrip("/")
        resp = self.client.post(
            url, content_type='application/json',
            data=json.dumps({'process_graph': process_graph}))
        return resp

    def test_udf_runtimes(self):
        runtimes = self.client.get('/openeo/0.4.0/udf_runtimes').json
        print(runtimes)
        self.assertIn("Python", runtimes)

    def test_load_collection(self):
        resp = self._post_process_graph({
            'collection': {
                'process_id': 'load_collection',
                'arguments': {'id': 'S2_FAPAR_CLOUDCOVER'},
                'result': True
            }
        })
        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_filter_temporal(self):
        resp = self._post_process_graph({
            'filter_temp': {
                'process_id': 'filter_temporal',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'extent': ['2018-01-01', '2018-12-31']
                },
                'result': True
            },
            'collection': {
                'process_id': 'load_collection',
                'arguments': {
                    'id': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        })
        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_apply_kernel(self):
        kernel_list = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        resp = self._post_process_graph(load_json("pg/0.4/apply_kernel.json"))
        assert resp.status_code == 200
        assert resp.content_length > 0
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_kernel.call_count == 1

        np_kernel = dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_kernel.call_args[0][0]
        self.assertListEqual(np_kernel.tolist(), kernel_list)
        self.assertEqual(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_kernel.call_args[0][1], 3)

    def test_load_collection_filter(self):
        resp = self._post_process_graph({
            'collection': {
                'process_id': 'load_collection',
                'arguments': {
                    'id': 'S2_FAPAR_CLOUDCOVER',
                    'spatial_extent': {
                        'west': 5.027, 'east': 5.0438, 'north': 51.2213,
                        'south': 51.1974, 'crs': 'EPSG:4326'
                    },
                    'temporal_extent': ['2018-01-01', '2018-12-31']
                },
                'result': True
            }
        })
        assert resp.status_code == 200
        assert resp.content_length > 0
        # assert resp.headers['Content-Type'] == "application/octet-stream"

        assert dummy_impl.collections['S2_FAPAR_CLOUDCOVER'].download.call_count == 1
        assert dummy_impl.collections['S2_FAPAR_CLOUDCOVER'].viewingParameters == {
            'version': '0.4.2', 'from': '2018-01-01', 'to': '2018-12-31',
            'left': 5.027, 'right': 5.0438, 'top': 51.2213, 'bottom': 51.1974, 'srs': 'EPSG:4326'}

    def test_execute_apply_unary(self):
        resp = self._post_process_graph(load_json("pg/0.4/apply_unary.json"))
        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_apply_run_udf(self):
        resp = self._post_process_graph(load_json("pg/0.4/apply_run_udf.json"))
        assert resp.status_code == 200
        assert resp.content_length > 0
        print(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"])
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_tiles.call_count == 1

    def test_execute_reduce_temporal_run_udf(self):
        resp = self._post_process_graph(load_json("pg/0.4/reduce_temporal_run_udf.json"))
        assert resp.status_code == 200
        assert resp.content_length > 0
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_tiles_spatiotemporal.call_count == 1

    def test_execute_reduce_bands_run_udf(self):
        request = load_json("udf.json")
        resp = self.client.post('/openeo/0.4.0/result', content_type='application/json', json=request)

        assert resp.status_code == 200
        assert resp.content_length > 0
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_tiles.call_count == 1

    def test_execute_apply_dimension_temporal_run_udf(self):
        resp = self._post_process_graph(load_json("pg/0.4/apply_dimension_temporal_run_udf.json"))
        assert resp.status_code == 200
        assert resp.content_length > 0
        self.assertEqual(1, dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_tiles_spatiotemporal.call_count)
        self.assertEqual(1, dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_dimension.call_count)

    def test_execute_reduce_max(self):
        resp = self._post_process_graph(load_json("pg/0.4/reduce_max.json"))
        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_merge_cubes(self):
        resp = self._post_process_graph(load_json("pg/0.4/merge_cubes.json"))
        assert resp.status_code == 200
        assert resp.content_length > 0
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].merge.call_count == 1
        args, kwargs = dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].merge.call_args
        assert args[1:] == ('or', )

    def test_execute_reduce_bands(self):
        resp = self._post_process_graph(load_json("pg/0.4/reduce_bands.json"))
        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_mask(self):
        resp = self._post_process_graph(load_json("pg/0.4/mask.json"))
        self.assertEqual(200, resp.status_code, msg=resp.get_data(as_text=True))
        assert resp.content_length > 0
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].mask.call_count == 1

        def check_params(viewing_parameters):
            self.assertAlmostEqual(viewing_parameters['left'], 7.022705078125007, delta=0.0001)
            self.assertAlmostEqual(viewing_parameters['bottom'], 51.29289899553571, delta=0.0001)
            self.assertAlmostEqual(viewing_parameters['right'], 7.659912109375007, delta=0.0001)
            self.assertAlmostEqual(viewing_parameters['top'], 51.75432477678571, delta=0.0001)
            self.assertEquals(viewing_parameters['srs'], 'EPSG:4326')
        check_params(dummy_impl.collections['PROBAV_L3_S10_TOC_NDVI_333M_V2'].viewingParameters)
        check_params(dummy_impl.collections['S2_FAPAR_CLOUDCOVER'].viewingParameters)


    def test_execute_mask_polygon(self):
        resp = self._post_process_graph(load_json("pg/0.4/mask_polygon.json"))
        assert resp.status_code == 200
        assert resp.content_length > 0
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].mask.call_count == 1
        import shapely.geometry
        self.assertIsInstance(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].mask.call_args[1]['polygon'],
                              shapely.geometry.Polygon)

    def test_preview_aggregate_temporal_max(self):
        resp = self._post_process_graph(load_json("pg/0.4/aggregate_temporal_max.json"))
        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_zonal_statistics(self):
        process_graph = load_json("pg/0.4/zonal_statistics.json")
        resp = self._post_process_graph(process_graph)
        self.assertEqual(200, resp.status_code, msg=resp.get_data(as_text=True))
        assert json.loads(resp.get_data(as_text=True)) == {
            "2015-07-06T00:00:00": [2.9829132080078127],
            "2015-08-22T00:00:00": [None]
        }
        assert dummy_impl.collections['S2_FAPAR_CLOUDCOVER'].viewingParameters['srs'] == 'EPSG:4326'

    def test_create_wmts(self):
        process_graph = load_json("pg/0.4/filter_temporal.json")
        resp = self.client.post('/openeo/0.4.0/services', content_type='application/json', json={
            "custom_param": 45,
            "process_graph": process_graph,
            "type": 'WMTS',
            "title": "My Service",
            "description": "Service description"
        })

        assert resp.status_code == 201
        assert resp.headers['OpenEO-Identifier'] == 'c63d6c27-c4c2-4160-b7bd-9e32f582daec'
        assert resp.headers['Location'].endswith("/services/c63d6c27-c4c2-4160-b7bd-9e32f582daec/service/wmts")

        tiled_viewing_service = dummy_impl.collections["S2"].tiled_viewing_service
        assert tiled_viewing_service.call_count == 1
        ProcessGraphVisitor.dereference_from_node_arguments(process_graph)
        tiled_viewing_service.assert_called_with(
            custom_param=45, description='Service description', process_graph=process_graph, title='My Service',
            type='WMTS'
        )

    def test_read_vector(self):
        process_graph = load_json(
            "pg/0.4/read_vector.json",
            preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("GeometryCollection.geojson")))
        )
        resp = self._post_process_graph(process_graph)
        body = resp.get_data(as_text=True)

        assert resp.status_code == 200
        assert 'NaN' not in body
        assert json.loads(body) == {
            "2015-07-06T00:00:00": [2.9829132080078127],
            "2015-08-22T00:00:00": [None]
        }

    def test_load_collection_without_spatial_extent_incorporates_read_vector_extent(self):
        process_graph = load_json(
            "pg/0.4/read_vector_spatial_extent.json",
            preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("GeometryCollection.geojson")))
        )

        resp = self._post_process_graph(process_graph)
        body = resp.get_data(as_text=True)

        assert resp.status_code == 200
        assert 'NaN' not in body
        assert json.loads(body) == {
            "2015-07-06T00:00:00": [2.9829132080078127],
            "2015-08-22T00:00:00": [None]
        }

        viewing_parameters = dummy_impl.collections['PROBAV_L3_S10_TOC_NDVI_333M_V2'].viewingParameters
        self.assertAlmostEqual(viewing_parameters['left'], 5.07616, delta=0.01)
        self.assertAlmostEqual(viewing_parameters['bottom'], 51.2122, delta=0.01)
        self.assertAlmostEqual(viewing_parameters['right'], 5.16685, delta=0.01)
        self.assertAlmostEqual(viewing_parameters['top'], 51.2689, delta=0.01)
        self.assertEquals(viewing_parameters['srs'], 'EPSG:4326')

    def test_read_vector_from_FeatureCollection(self):
        process_graph = load_json(
            "pg/0.4/read_vector_feature_collection.json",
            preprocess=lambda s: s.replace("PLACEHOLDER", str(get_path("FeatureCollection.geojson")))
        )

        resp = self._post_process_graph(process_graph)
        body = resp.get_data(as_text=True)

        assert resp.status_code == 200, body
        assert 'NaN' not in body
        assert json.loads(body) == {
            "2015-07-06T00:00:00": [2.9829132080078127],
            "2015-08-22T00:00:00": [None]
        }

    def test_no_nested_JSONResult(self):
        process_graph = load_json("pg/0.4/no_nested_json_result.json")
        resp = self.client.post('/openeo/0.4.0/result', content_type='application/json', data=json.dumps(process_graph))

        self.assertEqual(200, resp.status_code, msg=resp.get_data(as_text=True))

    def test_point_with_bbox(self):
        process_graph = {
            "loadcollection1": {
                'process_id': 'load_collection',
                'arguments': {'id': 'S2_FAPAR_CLOUDCOVER'},
            },
            "filterbbox": {
                "process_id": "filter_bbox",
                "arguments": {
                    "data": {"from_node": "loadcollection1"},
                    "extent": {"west": 3, "east": 6, "south": 50, "north": 51, "crs": "EPSG:4326"}
                },
                "result": True
            }
        }
        resp = self._post_process_graph(process_graph, url="timeseries/point?x=1&y=2")
        assert resp.status_code == 200
        assert resp.json == {"viewingParameters": {
            "left": 3, "right": 6, "bottom": 50, "top": 51, "srs": "EPSG:4326", "version": "0.4.2"
        }}

    def test_load_disk_data(self):
        process_graph = load_json("pg/0.4/load_disk_data.json")
        with self._post_process_graph(process_graph) as resp:
            self.assertEqual(200, resp.status_code)

    def test_mask_with_vector_file(self):
        process_graph = load_json(
            "pg/0.4/mask_with_vector_file.json",
            preprocess=lambda s:s.replace("PLACEHOLDER", str(get_path("mask_polygons_3.43_51.00_3.46_51.02.json")))
        )
        with self._post_process_graph(process_graph) as resp:
            self.assertEqual(200, resp.status_code, msg=resp.get_data(as_text=True))

    def test_aggregate_feature_collection(self):
        process_graph = load_json("pg/0.4/aggregate_feature_collection.json")
        with self._post_process_graph(process_graph) as resp:
            self.assertEqual(200, resp.status_code, msg=resp.get_data(as_text=True))


@pytest.fixture(params=["0.4.0", "1.0.0"])
def api_version(request):
    return request.param


@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()


class TestApi:
    """Helper container class for compact writing of api version aware `views` tests"""

    def __init__(self, api_version: str, client: FlaskClient, impl=dummy_impl):
        self.api_version = api_version
        self.client = client
        self.impl = impl

    def load_json(self, filename) -> dict:
        """Load test process graph from json file"""
        version = ".".join(self.api_version.split(".")[:2])
        return load_json("pg/{v}/{f}".format(v=version, f=filename))

    def result(self, process_graph: dict) -> Response:
        """post process graph dict to api and get result"""
        response = self.client.post(
            path="/openeo/{v}/result".format(v=self.api_version),
            content_type='application/json',
            json={'process_graph': process_graph},
        )
        # TODO Make basic asserts optional
        assert response.status_code == 200
        assert response.content_length > 0

        return response

    def result_from_file(self, filename: str) -> Response:
        return self.result(self.load_json(filename))

    @property
    def collections(self) ->dict:
        return self.impl.collections


@pytest.fixture
def api(api_version, client) -> TestApi:
    dummy_impl.collections = {}
    return TestApi(api_version=api_version, client=client, impl=dummy_impl)



def test_execute_simple_download(api):
    resp = api.result_from_file("basic.json")
    assert api.collections["S2_FAPAR_CLOUDCOVER"].download.call_count == 1
