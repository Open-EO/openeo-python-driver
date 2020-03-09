import json
import os
from multiprocessing import Pool
from unittest import TestCase, skip

import dummy_impl
from openeo_driver.views import app

os.environ["DRIVER_IMPLEMENTATION_PACKAGE"] = "dummy_impl"

client = app.test_client()


class Test(TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self._auth_header = {"Authorization": "Bearer basic.dGVzdA=="}
        dummy_impl.collections = {}

    def test_health(self):
        resp = self.client.get('/openeo/health')

        assert resp.status_code == 200
        assert "OK" in resp.get_data(as_text=True)

    def test_output_formats(self):
        resp = self.client.get('/openeo/output_formats')
        assert resp.status_code == 200
        assert resp.json == {"GTiff": {"title": "GeoTiff", "gis_data_types": ["raster"]}, }

    def test_file_formats(self):
        resp = self.client.get('/openeo/file_formats')
        assert resp.status_code == 200
        assert resp.json == {
            "input": {"GeoJSON": {"gis_data_type": ["vector"]}},
            "output": {
                "GTiff": {"title": "GeoTiff", "gis_data_types": ["raster"]},
            }
        }

    def test_collections(self):
        resp = self.client.get('/openeo/collections')
        assert resp.status_code == 200
        collections = resp.json
        assert 'DUMMY_S2_FAPAR_CLOUDCOVER' in [c['id'] for c in collections['collections']]

    def test_collections_detail(self):
        resp = self.client.get('/openeo/collections/DUMMY_S2_FAPAR_CLOUDCOVER')
        assert resp.status_code == 200
        collection = resp.json
        assert collection['id'] == 'DUMMY_S2_FAPAR_CLOUDCOVER'

    def test_data_detail_error(self):
        resp = self.client.get('/openeo/collections/S2_FAPAR_CLOUDCOVER')
        assert resp.status_code == 404
        error = resp.json
        assert error["code"] == "CollectionNotFound"
        assert error["message"] == "Collection 'S2_FAPAR_CLOUDCOVER' does not exist."

    def test_processes(self):
        resp = self.client.get('/openeo/processes')
        assert resp.status_code == 200
        processes = {spec['id']: spec for spec in resp.json['processes']}

        assert 'max' in processes.keys()

        histogram_spec = processes['histogram']
        assert 'data' in histogram_spec['parameters'].keys()

    def test_process_details(self):
        resp = self.client.get('/openeo/processes/max')
        assert resp.status_code == 200

    @classmethod
    def _post_download(cls, index):
        download_expected_graph = {'process_id': 'filter_bbox', 'args': {'imagery': {'process_id': 'filter_daterange',
                                                                                     'args': {'imagery': {
                                                                                         'collection_id': 'S2_FAPAR_SCENECLASSIFICATION_V102_PYRAMID'},
                                                                                         'from': '2018-08-06T00:00:00Z',
                                                                                         'to': '2018-08-06T00:00:00Z'}},
                                                                         'left': 5.027, 'right': 5.0438, 'top': 51.2213,
                                                                         'bottom': 51.1974, 'srs': 'EPSG:4326'}}

        post_request = json.dumps({"process_graph": download_expected_graph})
        resp = client.post('/openeo/execute', content_type='application/json', data=post_request)
        assert resp.status_code == 200
        assert resp.content_length > 0
        return index

    @skip
    def test_execute_download_parallel(self):
        """
        Tests downloading in parallel, see EP-2743
        Spark related issues are only exposed/tested when not using the dummy backend
        :return:
        """
        Test._post_download(1)

        with Pool(2) as pool:
            result = pool.map(Test._post_download, range(1, 3))

        print(result)

    def test_create_job(self):
        resp = self.client.post('/openeo/jobs', content_type='application/json', headers=self._auth_header, data=json.dumps({
            'process_graph': {},
            'output': {}
        }))

        assert resp.status_code == 201
        assert resp.content_length == 0
        assert resp.headers['Location'].endswith('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc')
        assert resp.headers['OpenEO-Identifier'] == '07024ee9-7847-4b8a-b260-6c879a2b3cdc'

    def test_queue_job(self):
        resp = self.client.post('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self._auth_header)

        self.assertEqual(202, resp.status_code)

    def test_get_job_info(self):
        resp = self.client.get('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc', headers=self._auth_header)

        self.assertEqual(200, resp.status_code)

        info = resp.get_json()

        self.assertEqual(info['job_id'], '07024ee9-7847-4b8a-b260-6c879a2b3cdc')
        self.assertEqual(info['status'], 'running')

    def test_cancel_job(self):
        resp = self.client.delete('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self._auth_header)

        self.assertEqual(204, resp.status_code)

    def test_api_propagates_http_status_codes(self):
        resp = self.client.get('/openeo/jobs/unknown_job_id/results/some_file', headers=self._auth_header)

        assert resp.status_code == 404

    def test_service_types_v040(self):
        resp = self.client.get('/openeo/0.4.0/service_types')
        service_types = resp.json
        assert list(service_types.keys()) == ["WMTS"]
        wmts = service_types["WMTS"]
        assert wmts["parameters"]["version"]["default"] == "1.0.0"
        assert wmts["variables"] == []
        assert wmts["attributes"] == []

    def test_service_types_v100(self):
        resp = self.client.get('/openeo/1.0.0/service_types')
        service_types = resp.json
        assert list(service_types.keys()) == ["WMTS"]
        wmts = service_types["WMTS"]
        assert wmts["configuration"]["version"]["default"] == "1.0.0"
        assert wmts["process_parameters"] == []
        assert wmts["links"] == []

    def test_create_unsupported_service_type_returns_BadRequest(self):
        resp = self.client.post('/openeo/services', content_type='application/json', json={
            "process_graph": {'product_id': 'S2'},
            "type": '???',
        })

        self.assertEqual(400, resp.status_code)

    def test_unsupported_services_methods_return_MethodNotAllowed(self):
        resp = self.client.put('/openeo/services', content_type='application/json', json={
            "process_graph": {'product_id': 'S2'},
            "type": 'WMTS',
        })

        self.assertEqual(405, resp.status_code)

    def test_uncaught_exceptions_return_InternalServerError(self):
        resp = self.client.post('/openeo/services', content_type='application/json', json={
            "process_graph": {'product_id': 'S2'}
        })

        self.assertEqual(500, resp.status_code)

    def test_get_batch_job_logs(self):
        resp = self.client.get('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/logs', headers=self._auth_header)

        self.assertEqual(200, resp.status_code)

        log_entries = resp.get_json()['logs']
        self.assertEqual([], log_entries)

    def test_list_results_for_unknown_job_returns_NotFound(self):
        resp = self.client.get('/openeo/jobs/unknown_job_id/results', headers=self._auth_header)

        self.assertEqual(404, resp.status_code)
