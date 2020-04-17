import json
import os
from multiprocessing import Pool
from pathlib import Path
import tempfile
from unittest import TestCase, skip, mock
import flask
from openeo_driver.dummy import dummy_backend
from openeo.capabilities import ComparableVersion
from openeo_driver.users import HttpAuthHandler
from openeo_driver.views import app, EndpointRegistry

os.environ["DRIVER_IMPLEMENTATION_PACKAGE"] = "openeo_driver.dummy.dummy_backend"
app.config["OPENEO_TITLE"] = "OpenEO Test API"

client = app.test_client()


class Test(TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SERVER_NAME'] = 'oeo.net'
        self.client = app.test_client()
        self._auth_header = {
            "Authorization": "Bearer " + HttpAuthHandler().build_basic_access_token(user_id=dummy_backend.TEST_USER)
        }
        dummy_backend.collections = {}

    def test_capabilities(self):
        resp = self.client.get('/openeo/1.0.0/')
        capabilities = resp.json
        assert capabilities["api_version"] == "1.0.0"
        assert capabilities["stac_version"] == "0.9.0"
        assert capabilities["title"] == "OpenEO Test API"
        assert capabilities["id"] == "openeotestapi1.0.0"

    def test_capabilities_endpoints(self):
        capabilities = self.client.get('/openeo/1.0.0/').json
        endpoints = {e["path"]: sorted(e["methods"]) for e in capabilities["endpoints"]}
        assert endpoints["/collections"] == ["GET"]
        assert endpoints["/collections/{collection_id}"] == ["GET"]
        assert endpoints["/result"] == ["POST"]
        assert endpoints["/jobs"] == ["GET", "POST"]
        assert endpoints["/processes"] == ["GET"]
        assert endpoints["/processes/{process_id}"] == ["GET"]
        assert endpoints["/udf_runtimes"] == ["GET"]
        assert endpoints["/file_formats"] == ["GET"]
        assert endpoints["/service_types"] == ["GET"]
        assert endpoints["/services"] == ["GET", "POST"]
        assert endpoints["/services/{service_id}"] == ["DELETE", "GET", "PATCH"]
        assert endpoints["/subscription"] == ["GET"]
        assert endpoints["/jobs/{job_id}"] == ["DELETE", "GET", "PATCH"]
        assert endpoints["/jobs/{job_id}/results"] == ["DELETE", "GET", "POST"]
        assert endpoints["/credentials/basic"] == ["GET"]
        assert endpoints["/credentials/oidc"] == ["GET"]
        assert endpoints["/me"] == ["GET"]

    def test_capabilities_endpoints_issue_28_v040(self):
        """https://github.com/Open-EO/openeo-python-driver/issues/28"""
        capabilities = self.client.get('/openeo/0.4.0/').json
        endpoints = {e["path"]: e["methods"] for e in capabilities["endpoints"]}
        assert endpoints["/output_formats"] == ["GET"]
        assert "/file_formats" not in endpoints

    def test_capabilities_endpoints_issue_28_v100(self):
        """https://github.com/Open-EO/openeo-python-driver/issues/28"""
        capabilities = self.client.get('/openeo/1.0.0/').json
        endpoints = {e["path"]: e["methods"] for e in capabilities["endpoints"]}
        assert endpoints["/file_formats"] == ["GET"]
        assert "/output_formats" not in endpoints

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

    def test_create_job_040(self):
        resp = self.client.post('/openeo/0.4.0/jobs', headers=self._auth_header, json={
            'title': 'foo job',
            'process_graph': {"foo": {"process_id": "foo", "arguments": {}}},
        })
        assert resp.status_code == 201
        assert resp.content_length == 0
        assert resp.headers['Location'] == 'http://oeo.net/openeo/0.4.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc'
        assert resp.headers['OpenEO-Identifier'] == '07024ee9-7847-4b8a-b260-6c879a2b3cdc'

    def test_create_job_100(self):
        resp = self.client.post('/openeo/1.0.0/jobs', headers=self._auth_header, json={
            'process': {
                'process_graph': {"foo": {"process_id": "foo", "arguments": {}}},
                'summary': 'my foo job',
            },
        })
        assert resp.status_code == 201
        assert resp.content_length == 0
        assert resp.headers['Location'] == 'http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc'
        assert resp.headers['OpenEO-Identifier'] == '07024ee9-7847-4b8a-b260-6c879a2b3cdc'

    def test_start_job(self):
        resp = self.client.post('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self._auth_header)
        assert resp.status_code == 202
        assert resp.content_length == 0

    def test_start_job_invalid(self):
        resp = self.client.post('/openeo/jobs/deadbeef-f00/results', headers=self._auth_header)
        assert resp.status_code == 404
        error = resp.json
        assert error["code"] == "JobNotFound"
        assert error["message"] == "The job 'deadbeef-f00' does not exist."

    def test_get_job_info_040(self):
        resp = self.client.get('/openeo/0.4.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc', headers=self._auth_header)
        assert resp.status_code == 200
        assert resp.json == {
            'id': '07024ee9-7847-4b8a-b260-6c879a2b3cdc',
            'status': 'running',
            'submitted': "2017-01-01T09:32:12Z",
            'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}},
        }

    def test_get_job_info_100(self):
        resp = self.client.get('/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc', headers=self._auth_header)
        assert resp.status_code == 200
        assert resp.json == {
            'id': '07024ee9-7847-4b8a-b260-6c879a2b3cdc',
            'status': 'running',
            'created': "2017-01-01T09:32:12Z",
            'process': {'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}}},
        }

    def test_get_job_info_invalid(self):
        resp = self.client.get('/openeo/1.0.0/jobs/deadbeef-f00', headers=self._auth_header)
        assert resp.status_code == 404
        error = resp.json
        assert error["code"] == "JobNotFound"
        assert error["message"] == "The job 'deadbeef-f00' does not exist."

    def test_list_user_jobs_040(self):
        resp = self.client.get('/openeo/0.4.0/jobs', headers=self._auth_header)
        assert resp.status_code == 200
        assert resp.json == {
            "jobs": [
                {
                    'id': '07024ee9-7847-4b8a-b260-6c879a2b3cdc',
                    'status': 'running',
                    'submitted': "2017-01-01T09:32:12Z",
                }
            ],
            "links": []
        }

    def test_list_user_jobs_100(self):
        resp = self.client.get('/openeo/1.0.0/jobs', headers=self._auth_header)
        assert resp.status_code == 200
        assert resp.json == {
            "jobs": [
                {
                    'id': '07024ee9-7847-4b8a-b260-6c879a2b3cdc',
                    'status': 'running',
                    'created': "2017-01-01T09:32:12Z",
                }
            ],
            "links": []
        }


    def test_get_job_results_040(self):
        resp = self.client.get('/openeo/0.4.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results',
                               headers=self._auth_header)
        assert resp.status_code == 200
        assert resp.json == {
            "links": [
                {
                    "href": "http://oeo.net/openeo/0.4.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/output.tiff"
                }
            ]
        }

    def test_get_job_results_100(self):
        resp = self.client.get('/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results',
                               headers=self._auth_header)
        assert resp.status_code == 200
        assert resp.json == {
            "assets": {
                "output.tiff": {
                    "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/output.tiff"
                }
            }
        }

    def test_get_job_results_invalid_job(self):
        resp = self.client.get('/openeo/jobs/deadbeef-f00/results', headers=self._auth_header)
        assert resp.status_code == 404
        assert resp.json["code"] == "JobNotFound"

    def test_download_result_invalid_job(self):
        resp = self.client.get('/openeo/jobs/deadbeef-f00/results/some_file', headers=self._auth_header)
        assert resp.status_code == 404
        assert resp.json["code"] == "JobNotFound"

    def test_download_result(self):
        # TODO: use fixture for tmp_dir?
        with tempfile.TemporaryDirectory() as d:
            output_root = Path(d)
            with mock.patch.object(dummy_backend.DummyBatchJobs, '_output_root', return_value=output_root):
                output = output_root / "07024ee9-7847-4b8a-b260-6c879a2b3cdc" / "out" / "output.tiff"
                output.parent.mkdir(parents=True)
                with output.open("wb") as f:
                    f.write(b"tiffdata")
                resp = self.client.get("/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/output.tiff", headers=self._auth_header)
        assert resp.status_code == 200
        assert resp.data == b"tiffdata"

    def test_get_batch_job_logs(self):
        resp = self.client.get('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/logs', headers=self._auth_header)
        assert resp.status_code == 200
        assert resp.json == {
            "logs": [
                {"id": "1", "level": "info", "message": "hello world", "path": []}
            ],
            "links": []
        }

    def test_cancel_job(self):
        resp = self.client.delete('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self._auth_header)
        assert resp.status_code == 204

    def test_cancel_job_invalid(self):
        resp = self.client.delete('/openeo/jobs/deadbeef-f00/results', headers=self._auth_header)
        assert resp.status_code == 404
        assert resp.json["code"] == "JobNotFound"

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

    def test_list_services_040(self):
        metadata = self.client.get('/openeo/0.4.0/services').json
        assert metadata == {
            "services": [{
                'id': 'wmts-foo',
                'type': 'WMTS',
                'enabled': True,
                'url': 'https://oeo.net/wmts/foo',
                'submitted': '2020-04-09T15:05:08Z',
                'title': 'Test service',
            }],
            "links": []
        }

    def test_list_services_100(self):
        metadata = self.client.get('/openeo/1.0.0/services').json
        assert metadata == {
            "services": [{
                'id': 'wmts-foo',
                'type': 'WMTS',
                'enabled': True,
                'url': 'https://oeo.net/wmts/foo',
                'title': 'Test service',
                'created': '2020-04-09T15:05:08Z',
            }],
            "links": []
        }

    def test_get_service_metadata_040(self):
        metadata = self.client.get('/openeo/0.4.0/services/wmts-foo').json
        assert metadata == {
            "id": "wmts-foo",
            "process_graph": {"foo": {"process_id": "foo", "arguments": {}}},
            "url": "https://oeo.net/wmts/foo",
            "type": "WMTS",
            "enabled": True,
            "parameters": {},
            "attributes": {},
            "title": "Test service",
            'submitted': '2020-04-09T15:05:08Z',
        }

    def test_get_service_metadata_100(self):
        metadata = self.client.get('/openeo/1.0.0/services/wmts-foo').json
        assert metadata == {
            "id": "wmts-foo",
            "process": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
            "url": "https://oeo.net/wmts/foo",
            "type": "WMTS",
            "enabled": True,
            "attributes": {},
            "title": "Test service",
            'created': '2020-04-09T15:05:08Z',
        }

    def test_get_service_metadata_wrong_id(self):
        res = self.client.get('/openeo/1.0.0/services/wmts-invalid')
        assert res.status_code == 404
        assert res.json['code'] == 'ServiceNotFound'



def test_endpoint_registry():
    app = flask.Flask(__name__)
    bp = flask.Blueprint("test", __name__)
    endpoint = EndpointRegistry()

    @bp.route("/hello")
    def hello():
        return "not an endpoint"

    @endpoint
    @bp.route("/foo")
    def foo():
        return "simple endpoint"

    @endpoint(hidden=True)
    @bp.route("/secret")
    def secret():
        return "secret endpoint"

    @endpoint(version=ComparableVersion("1.0.0").accept_lower)
    @bp.route("/old")
    def old():
        return "old endpoint"

    app.register_blueprint(bp, url_prefix="/bar")

    result = endpoint.get_path_metadata(bp)

    # Check metadata
    assert len(result) == 3
    paths, methods, metadatas = zip(*sorted(result))
    assert paths == ('/foo', '/old', '/secret')
    assert methods == ({"GET"},) * 3
    assert metadatas[0].hidden is False
    assert metadatas[0].for_version is None
    assert metadatas[1].for_version("0.4.0") is True
    assert metadatas[1].for_version("1.0.0") is False
    assert metadatas[2].hidden is True

    # Check that view functions still work
    client = app.test_client()
    assert b"not an endpoint" == client.get("/bar/hello").data
    assert b"simple endpoint" == client.get("/bar/foo").data
    assert b"secret endpoint" == client.get("/bar/secret").data
    assert b"old endpoint" == client.get("/bar/old").data


def test_endpoint_registry_multiple_methods():
    bp = flask.Blueprint("test", __name__)
    endpoint = EndpointRegistry()

    @endpoint
    @bp.route("/foo", methods=["GET"])
    def foo_get():
        return "get"

    @endpoint
    @bp.route("/foo", methods=["POST"])
    def foo_post():
        return "post"

    result = endpoint.get_path_metadata(bp)

    # Check metadata
    assert len(result) == 2
    paths, methods, metadatas = zip(*sorted(result))
    assert paths == ('/foo', '/foo')
    assert methods == ({"GET"}, {"POST"})
