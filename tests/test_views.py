import logging
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase, mock

import flask
import pytest

from openeo.capabilities import ComparableVersion
from openeo_driver.ProcessGraphDeserializer import custom_process_from_process_graph
from openeo_driver.backend import BatchJobMetadata, UserDefinedProcessMetadata
from openeo_driver.dummy import dummy_backend
from openeo_driver.testing import ApiTester
from openeo_driver.testing import TEST_USER, ApiResponse, TEST_USER_AUTH_HEADER, generate_unique_test_process_id
from openeo_driver.views import app, EndpointRegistry, _normalize_collection_metadata, \
    backend_implementation
from .data import TEST_DATA_ROOT
from .test_users import _build_basic_http_auth_header

os.environ["DRIVER_IMPLEMENTATION_PACKAGE"] = "openeo_driver.dummy.dummy_backend"
app.config["OPENEO_TITLE"] = "OpenEO Test API"


@pytest.fixture(params=["0.4.0", "1.0.0"])
def api_version(request):
    return request.param


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SERVER_NAME'] = 'oeo.net'
    return app.test_client()


@pytest.fixture
def api(api_version, client) -> ApiTester:
    return ApiTester(api_version=api_version, client=client, data_root=TEST_DATA_ROOT)


@pytest.fixture
def api040(client) -> ApiTester:
    return ApiTester(api_version="0.4.0", client=client, data_root=TEST_DATA_ROOT)


@pytest.fixture
def api100(client) -> ApiTester:
    return ApiTester(api_version="1.0.0", client=client, data_root=TEST_DATA_ROOT)


class TestGeneral:
    """
    General tests (capabilities, collections, processes)
    """

    def test_well_known_openeo(self, client):
        resp = client.get('/.well-known/openeo')
        assert resp.status_code == 200
        versions = resp.json["versions"]
        # in .well-known/openeo there should only be one item per api_version (no aliases)
        by_api_version = {d["api_version"]: d for d in versions}
        assert len(versions) == len(by_api_version)
        assert by_api_version == {
            "0.4.2": {'api_version': '0.4.2', 'production': True, 'url': 'http://oeo.net/openeo/0.4/'},
            "1.0.0": {'api_version': '1.0.0', 'production': True, 'url': 'http://oeo.net/openeo/1.0/'},
        }

    def test_versioned_well_known_openeo(self, api):
        api.get('/.well-known/openeo').assert_error(404, "NotFound")

    def test_capabilities_040(self, api040):
        capabilities = api040.get('/').assert_status_code(200).json
        assert capabilities["api_version"] == "0.4.0"
        assert capabilities["version"] == "0.4.0"
        assert capabilities["stac_version"] == "0.9.0"
        assert capabilities["title"] == "OpenEO Test API"
        assert capabilities["id"] == "openeotestapi-0.4.0"
        assert capabilities["production"] is True

    def test_capabilities_100(self, api100):
        capabilities = api100.get('/').assert_status_code(200).json
        assert capabilities["api_version"] == "1.0.0"
        assert capabilities["stac_version"] == "0.9.0"
        assert capabilities["title"] == "OpenEO Test API"
        assert capabilities["id"] == "openeotestapi-1.0.0"
        assert capabilities["production"] is True

        def get_link(rel):
            links = [link for link in capabilities["links"] if link["rel"] == rel]
            assert len(links) == 1
            return links[0]

        assert get_link("version-history")["href"] == "http://oeo.net/.well-known/openeo"
        assert get_link("data")["href"] == "http://oeo.net/openeo/1.0.0/collections"
        assert get_link("conformance")["href"] == "http://oeo.net/openeo/1.0.0/conformance"

    def test_capabilities_version_alias(self, client):
        resp = ApiResponse(client.get('/openeo/0.4/')).assert_status_code(200).json
        assert resp["api_version"] == "0.4.2"
        assert resp["production"] == True

    def test_capabilities_invalid_api_version(self, client):
        resp = ApiResponse(client.get('/openeo/0.0.0/'))
        resp.assert_error(501, 'UnsupportedApiVersion')
        assert "Unsupported version component in URL: '0.0.0'" in resp.json['message']

    def test_capabilities_endpoints(self, api100):
        capabilities = api100.get("/").assert_status_code(200).json
        endpoints = {e["path"]: sorted(e["methods"]) for e in capabilities["endpoints"]}
        assert endpoints["/collections"] == ["GET"]
        assert endpoints["/collections/{collection_id}"] == ["GET"]
        assert endpoints["/result"] == ["POST"]
        assert endpoints["/jobs"] == ["GET", "POST"]
        assert endpoints["/processes"] == ["GET"]
        assert endpoints["/udf_runtimes"] == ["GET"]
        assert endpoints["/file_formats"] == ["GET"]
        assert endpoints["/service_types"] == ["GET"]
        assert endpoints["/services"] == ["GET", "POST"]
        assert endpoints["/services/{service_id}"] == ["DELETE", "GET"]
        # assert endpoints["/subscription"] == ["GET"]
        assert endpoints["/jobs/{job_id}"] == ["GET"]
        assert endpoints["/jobs/{job_id}/results"] == ["DELETE", "GET", "POST"]
        assert endpoints["/credentials/basic"] == ["GET"]
        assert endpoints["/credentials/oidc"] == ["GET"]
        assert endpoints["/me"] == ["GET"]

    def test_capabilities_endpoints_issue_28_v040(self, api040):
        """https://github.com/Open-EO/openeo-python-driver/issues/28"""
        capabilities = api040.get("/").assert_status_code(200).json
        endpoints = {e["path"]: e["methods"] for e in capabilities["endpoints"]}
        assert endpoints["/output_formats"] == ["GET"]
        assert "/file_formats" not in endpoints

    def test_capabilities_endpoints_issue_28_v100(self, api100):
        """https://github.com/Open-EO/openeo-python-driver/issues/28"""
        capabilities = api100.get("/").assert_status_code(200).json
        endpoints = {e["path"]: e["methods"] for e in capabilities["endpoints"]}
        assert endpoints["/file_formats"] == ["GET"]
        assert "/output_formats" not in endpoints

    def test_conformance(self, api100):
        res = api100.get('/conformance').assert_status_code(200).json
        assert "conformsTo" in res

    @pytest.mark.parametrize("path", ["/", "/collections", "/processes", "/jobs", ])
    def test_cors_headers_options(self, api, path):
        # Mimic browser's OPTION request
        api.client.environ_base["HTTP_ORIGIN"] = "https://editor.openeo.org"
        api.client.environ_base["HTTP_ACCESS_CONTROL_REQUEST_METHOD"] = "GET"
        api.client.environ_base["HTTP_ACCESS_CONTROL_REQUEST_HEADERS"] = "Authorization, Content-Type"
        resp = api.client.options(api.url(path))

        assert resp.status_code == 204
        assert len(resp.data) == 0
        # General CORS headers
        assert resp.access_control_allow_origin == "*"
        assert resp.access_control_allow_credentials is False
        assert {"Location", "OpenEO-Identifier", "OpenEO-Costs", "Link"}.issubset(resp.access_control_expose_headers)

        # OPTIONS specific CORS headers
        assert {"Authorization", "Content-Type"}.issubset(resp.access_control_allow_headers)
        assert {"GET", "POST", "PATCH"}.issubset(resp.access_control_allow_methods)
        assert resp.content_type == "application/json"

    @pytest.mark.parametrize("path", ["/", "/collections", "/processes"])
    def test_cors_headers_get(self, api, path):
        response = api.get(path)
        resp = response.assert_status_code(200).response
        # General CORS headers
        assert resp.access_control_allow_origin == "*"
        assert resp.access_control_allow_credentials is False
        assert {"Location", "OpenEO-Identifier", "OpenEO-Costs", "Link"}.issubset(resp.access_control_expose_headers)

    def test_cors_headers_post(self, api):
        pg = {"process_graph": {"add": {"process_id": "add", "arguments": {"x": 1, "y": 2}, "result": True}}}
        response = api.post(
            "/jobs",
            headers=TEST_USER_AUTH_HEADER,
            json={"process": pg} if api.api_version_compare.at_least("1.0.0") else pg
        )
        resp = response.assert_status_code(201).response
        # General CORS headers
        assert resp.access_control_allow_origin == "*"
        assert resp.access_control_allow_credentials is False
        assert {"Location", "OpenEO-Identifier", "OpenEO-Costs", "Link"}.issubset(resp.access_control_expose_headers)

    def test_cors_headers_put(self, api100):
        response = api100.put(
            "/process_graphs/myndvi",
            headers=TEST_USER_AUTH_HEADER,
            json={
                "id": "oneplustwo",
                "process_graph": {"a": {"process_id": "add", "arguments": {"x": 1, "y": 2}, "result": True}}
            }
        )
        resp = response.assert_status_code(200).response
        # General CORS headers
        assert resp.access_control_allow_origin == "*"
        assert resp.access_control_allow_credentials is False
        assert {"Location", "OpenEO-Identifier", "OpenEO-Costs", "Link"}.issubset(resp.access_control_expose_headers)

    def test_health(self, api):
        resp = api.get('/health').assert_status_code(200).json
        assert resp == {"health": "OK"}

    def test_credentials_oidc_040(self, api040):
        resp = api040.get('/credentials/oidc').assert_status_code(303)
        assert resp.headers["Location"] == "https://oidc.oeo.net/.well-known/openid-configuration"

    def test_credentials_oidc_100(self, api100):
        resp = api100.get('/credentials/oidc').assert_status_code(200).json
        assert resp == {'providers': [
            {'id': 'testprovider', 'issuer': 'https://oidc.oeo.net', 'scopes': ['openid'], 'title': 'Test'},
            {
                'id': 'eoidc', 'issuer': 'https://eo.id', 'scopes': ['openid'], 'title': 'e-OIDC',
                'default_client': {'id': 'badcafef00d'},
                'default_clients': [{
                    'id': 'badcafef00d',
                    'grant_types': ['urn:ietf:params:oauth:grant-type:device_code+pkce', 'refresh_token']}],
            },
            {
                'id': 'local', 'issuer': 'http://localhost:9090/auth/realms/master', 'scopes': ['openid'],
                'title': 'Local Keycloak'}
        ]}

    def test_output_formats(self, api040):
        resp = api040.get('/output_formats').assert_status_code(200).json
        assert resp == {"GTiff": {"title": "GeoTiff", "gis_data_types": ["raster"], "parameters": {}}, }

    def test_file_formats(self, api100):
        response = api100.get('/file_formats')
        resp = response.assert_status_code(200).json
        assert resp == {
            "input": {"GeoJSON": {"gis_data_types": ["vector"], "parameters": {}}},
            "output": {
                "GTiff": {"title": "GeoTiff", "gis_data_types": ["raster"], "parameters": {}},
            }
        }

    @pytest.mark.parametrize("endpoint", [
        "/processes",
        "/processes/backend"
    ])
    def test_processes(self, api, endpoint):
        resp = api.get(endpoint).assert_status_code(200).json
        processes = resp["processes"]
        process_ids = set(p['id'] for p in processes)
        assert {"load_collection", "min", "max", "sin", "merge_cubes", "mask"}.issubset(process_ids)
        expected_keys = {"id", "description", "parameters", "returns"}
        for process in processes:
            assert all(k in process for k in expected_keys)

    def test_processes_non_standard_histogram(self, api):
        resp = api.get('/processes').assert_status_code(200).json
        histogram_spec, = [p for p in resp["processes"] if p['id'] == "histogram"]
        assert "into bins" in histogram_spec["description"]
        if api.api_version_compare.at_least("1.0.0"):
            assert histogram_spec["parameters"] == [
                {
                    'name': 'data',
                    'description': 'An array of numbers',
                    'optional': False,
                    'schema': {'type': 'array', 'items': {'type': ['number', 'null']}, }
                }
            ]
        else:
            assert histogram_spec["parameters"] == {
                'data': {
                    'description': 'An array of numbers',
                    'required': True,
                    'schema': {'type': 'array', 'items': {'type': ['number', 'null']}, }
                }
            }
        assert histogram_spec["returns"] == {
            'description': 'A sequence of (bin, count) pairs',
            'schema': {'type': 'object'}
        }

    def test_processes_040_vs_100(self, api040, api100):
        pids040 = {p['id'] for p in api040.get("/processes").assert_status_code(200).json["processes"]}
        pids100 = {p['id'] for p in api100.get("/processes").assert_status_code(200).json["processes"]}
        expected_only_040 = {'reduce', 'aggregate_polygon'}
        expected_only_100 = {'reduce_dimension', 'aggregate_spatial', 'mask_polygon', 'add'}
        for pid in expected_only_040:
            assert pid in pids040
            assert pid not in pids100
        for pid in expected_only_100:
            assert pid not in pids040
            assert pid in pids100

    def test_custom_process_listing(self, api100):
        process_id = generate_unique_test_process_id()

        # Register a custom process with process graph
        process_spec = api100.load_json("pg/1.0/add_and_multiply.json")
        process_spec["id"] = process_id
        custom_process_from_process_graph(process_spec=process_spec)

        processes = api100.get("/processes").assert_status_code(200).json["processes"]
        processes_by_id = {p["id"]: p for p in processes}
        assert process_id in processes_by_id
        assert processes_by_id[process_id] == process_spec

    def test_processes_from_namespace(self, api100):
        process_id = generate_unique_test_process_id()

        # Register a custom process with process graph
        process_spec = api100.load_json("pg/1.0/add_and_multiply.json")
        process_spec["id"] = process_id
        custom_process_from_process_graph(process_spec=process_spec, namespace="foobar")
        process_spec_short = process_spec.copy()
        process_spec_short.pop("process_graph")

        processes = api100.get("/processes/foobar").assert_status_code(200).json["processes"]
        processes_by_id = {p["id"]: p for p in processes}
        assert process_id in processes_by_id
        assert processes_by_id[process_id] == process_spec_short

        processes = api100.get("/processes/foobar?full=yes").assert_status_code(200).json["processes"]
        processes_by_id = {p["id"]: p for p in processes}
        assert process_id in processes_by_id
        assert processes_by_id[process_id] == process_spec

        processes = api100.get("/processes").assert_status_code(200).json["processes"]
        assert "increment" not in set(p['id'] for p in processes)


class TestCollections:

    def test_normalize_collection_metadata_no_id(self, caplog):
        with pytest.raises(KeyError):
            _normalize_collection_metadata({"foo": "bar"}, api_version=ComparableVersion("1.0.0"))
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("should have 'id' field" in m for m in errors)

    def test_normalize_collection_metadata_minimal_040(self, caplog):
        assert _normalize_collection_metadata({"id": "foobar"}, api_version=ComparableVersion("0.4.2")) == {
            'id': 'foobar',
            'stac_version': '0.6.2',
            'stac_extensions': ['datacube'],
            'description': 'foobar',
            'extent': {'spatial': [0, 0, 0, 0], 'temporal': [None, None]},
            'license': 'proprietary',
            'links': [],
        }
        warnings = set(r.getMessage() for r in caplog.records if r.levelno == logging.WARN)
        assert warnings == {"Collection 'foobar' metadata does not have field 'extent'."}

    def test_normalize_collection_metadata_minimal_full_040(self, caplog):
        assert _normalize_collection_metadata({"id": "foobar"}, api_version=ComparableVersion("0.4.2"), full=True) == {
            'id': 'foobar',
            'stac_version': '0.6.2',
            'stac_extensions': ['datacube'],
            'description': 'foobar',
            'extent': {'spatial': [0, 0, 0, 0], 'temporal': [None, None]},
            'license': 'proprietary',
            'properties': {},
            'other_properties': {},
            'links': [],
        }
        warnings = set(r.getMessage() for r in caplog.records if r.levelno == logging.WARN)
        assert warnings == {
            "Collection 'foobar' metadata does not have field 'extent'.",
            "Collection 'foobar' metadata does not have field 'other_properties'.",
            "Collection 'foobar' metadata does not have field 'properties'.",
        }

    def test_normalize_collection_metadata_minimal_100(self, caplog):
        assert _normalize_collection_metadata({"id": "foobar"}, api_version=ComparableVersion("1.0.0")) == {
            'id': 'foobar',
            'stac_version': '0.9.0',
            'stac_extensions': ['datacube'],
            'description': 'foobar',
            'extent': {'spatial': {'bbox': [[0, 0, 0, 0]]}, 'temporal': {'interval': [[None, None]]}},
            'license': 'proprietary',
            'links': [],
        }
        warnings = set(r.getMessage() for r in caplog.records if r.levelno == logging.WARN)
        assert warnings == {"Collection 'foobar' metadata does not have field 'extent'."}

    def test_normalize_collection_metadata_minimal_full_100(self, caplog):
        assert _normalize_collection_metadata({"id": "foobar"}, api_version=ComparableVersion("1.0.0"), full=True) == {
            'id': 'foobar',
            'stac_version': '0.9.0',
            'stac_extensions': ['datacube'],
            'description': 'foobar',
            'extent': {'spatial': {'bbox': [[0, 0, 0, 0]]}, 'temporal': {'interval': [[None, None]]}},
            'license': 'proprietary',
            'cube:dimensions': {},
            'summaries': {},
            'links': [],
        }
        warnings = set(r.getMessage() for r in caplog.records if r.levelno == logging.WARN)
        assert warnings == {
            "Collection 'foobar' metadata does not have field 'cube:dimensions'.",
            "Collection 'foobar' metadata does not have field 'extent'.",
            "Collection 'foobar' metadata does not have field 'summaries'."
        }

    def test_normalize_collection_metadata_cube_dimensions_extent_full_100(self, caplog):
        metadata = {
            "id": "foobar",
            "extent": {
                "spatial": {"bbox": [[-180, -56, 180, 83]]},
                "temporal": {"interval": [["2015-07-06", None]]}
            },
            "cube:dimensions": {
                "x": {"type": "spatial", "axis": "x"},
                "y": {"type": "spatial", "axis": "y"},
                "t": {"type": "temporal"},
            },
        }
        assert _normalize_collection_metadata(metadata, api_version=ComparableVersion("1.0.0"), full=True) == {
            'id': 'foobar',
            'stac_version': '0.9.0',
            'stac_extensions': ['datacube'],
            'description': 'foobar',
            'extent': {
                'spatial': {'bbox': [[-180, -56, 180, 83]]},
                'temporal': {'interval': [["2015-07-06T00:00:00Z", None]]}
            },
            'license': 'proprietary',
            "cube:dimensions": {
                "x": {"type": "spatial", "axis": "x", "extent": [-180, 180]},
                "y": {"type": "spatial", "axis": "y", "extent": [-56, 83]},
                "t": {"type": "temporal", "extent": ["2015-07-06T00:00:00Z", None]},
            }, 'summaries': {},
            'links': [],
        }

    def test_normalize_collection_metadata_dimensions_and_bands_040(self, caplog):
        metadata = {
            "id": "foobar",
            "cube:dimensions": {
                "x": {"type": "spatial"},
                "b": {"type": "bands", "values": ["B02", "B03"]}
            },
            "summaries": {
                "eo:bands": [{"name": "B02"}, {"name": "B03"}]
            }
        }
        res = _normalize_collection_metadata(metadata, api_version=ComparableVersion("0.4.0"), full=True)
        assert res["properties"]["cube:dimensions"] == {
            "x": {"type": "spatial"},
            "b": {"type": "bands", "values": ["B02", "B03"]}
        }
        assert res["properties"]["eo:bands"] == [{"name": "B02"}, {"name": "B03"}]

    def test_normalize_collection_metadata_dimensions_and_bands_100(self, caplog):
        metadata = {
            "id": "foobar",
            "properties": {
                "cube:dimensions": {
                    "x": {"type": "spatial"},
                    "b": {"type": "bands", "values": ["B02", "B03"]}
                },
                "eo:bands": [{"name": "B02"}, {"name": "B03"}]
            }
        }
        res = _normalize_collection_metadata(metadata, api_version=ComparableVersion("1.0.0"), full=True)
        assert res["cube:dimensions"] == {
            "x": {"type": "spatial"},
            "b": {"type": "bands", "values": ["B02", "B03"]}
        }
        assert res["summaries"]["eo:bands"] == [{"name": "B02"}, {"name": "B03"}]

    def test_normalize_collection_metadata_datetime(self, caplog):
        metadata = {
            "id": "foobar",
            "extent": {
                "temporal": {
                    "interval": [["2009-08-07", "2009-10-11"], ["2011-12-13 14:15:16", None]],
                }
            }
        }
        res = _normalize_collection_metadata(metadata, api_version=ComparableVersion("1.0.0"), full=True)
        assert res["extent"]["temporal"]["interval"] == [
            ['2009-08-07T00:00:00Z', '2009-10-11T00:00:00Z'],
            ['2011-12-13T14:15:16Z', None],
        ]

    def test_collections(self, api):
        resp = api.get('/collections').assert_status_code(200).json
        assert "links" in resp
        assert "collections" in resp
        assert 'S2_FAPAR_CLOUDCOVER' in [c['id'] for c in resp['collections']]
        assert 'S2_FOOBAR' in [c['id'] for c in resp['collections']]
        for collection in resp['collections']:
            assert 'id' in collection
            assert 'stac_version' in collection
            assert 'description' in collection
            assert 'license' in collection
            assert 'extent' in collection
            assert 'links' in collection

    def test_strip_private_fields(self, api):
        assert '_private' in dummy_backend.DummyCatalog().get_collection_metadata("S2_FOOBAR")
        # All metadata
        collections = api.get('/collections').assert_status_code(200).json["collections"]
        metadata, = (c for c in collections if c["id"] == "S2_FOOBAR")
        assert '_private' not in metadata
        # Single collection metadata
        metadata = api.get('/collections/S2_FOOBAR').assert_status_code(200).json
        assert '_private' not in metadata

    def test_collections_detail_invalid_collection(self, api):
        error = api.get('/collections/FOOBOO').assert_error(404, "CollectionNotFound").json
        assert error["message"] == "Collection 'FOOBOO' does not exist."

    def test_collections_detail(self, api):
        collection = api.get('/collections/S2_FOOBAR').assert_status_code(200).json
        assert collection['id'] == 'S2_FOOBAR'
        assert collection['description'] == 'S2_FOOBAR'
        assert collection['license'] == 'free'
        cube_dimensions = {
            'x': {'extent': [2.5, 6.2],
                  'reference_system': 'AUTO:42001',
                  'step': 10,
                  'type': 'spatial'},
            'y': {'extent': [49.5, 51.5],
                  'reference_system': 'AUTO:42001',
                  'step': 10,
                  'type': 'spatial'},
            "t": {"type": "temporal", "extent": ["2019-01-01", None]},
            "bands": {"type": "bands", "values": ["B02", "B03", "B04", "B08"]}
        }
        eo_bands = [
            {"name": "B02", "common_name": "blue"}, {"name": "B03", "common_name": "green"},
            {"name": "B04", "common_name": "red"}, {"name": "B08", "common_name": "nir"},
        ]
        if api.api_version_compare.at_least("1.0.0"):
            assert collection['stac_version'] == '0.9.0'
            assert collection['cube:dimensions'] == cube_dimensions
            assert collection['summaries']['eo:bands'] == eo_bands
            assert collection['extent']['spatial'] == {'bbox': [[2.5, 49.5, 6.2, 51.5]]}
            assert collection['extent']['temporal'] == {'interval': [['2019-01-01T00:00:00Z', None]]}
        else:
            assert collection['stac_version'] == '0.6.2'
            assert collection['properties']['cube:dimensions'] == cube_dimensions
            assert collection['properties']["eo:bands"] == eo_bands
            assert collection['extent'] == {
                'spatial': [2.5, 49.5, 6.2, 51.5],
                'temporal': ['2019-01-01T00:00:00Z', None]
            }


class TestBatchJobs:
    AUTH_HEADER = TEST_USER_AUTH_HEADER

    @staticmethod
    @contextmanager
    def _fresh_job_registry(next_job_id):
        """Set up a fresh job registry and predefine next job id"""
        with mock.patch.object(dummy_backend.DummyBatchJobs, 'generate_job_id', return_value=next_job_id):
            dummy_backend.DummyBatchJobs._job_registry = {
                (TEST_USER, '07024ee9-7847-4b8a-b260-6c879a2b3cdc'): BatchJobMetadata(
                    id='07024ee9-7847-4b8a-b260-6c879a2b3cdc',
                    status='running',
                    process={'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}}},
                    created=datetime(2017, 1, 1, 9, 32, 12),
                ),
                (TEST_USER, '53c71345-09b4-46b4-b6b0-03fd6fe1f199'): BatchJobMetadata(
                    id='53c71345-09b4-46b4-b6b0-03fd6fe1f199',
                    status='finished',
                    process={'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}}},
                    created=datetime(2020, 6, 11, 11, 51, 29),
                    started=datetime(2020, 6, 11, 11, 55, 9),
                    finished=datetime(2020, 6, 11, 11, 55, 15),
                    memory_time_megabyte=timedelta(seconds=18704944),
                    cpu_time=timedelta(seconds=1621),
                    geometry={
                        "type": "Polygon",
                        "coordinates": [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]]
                    },
                    bbox=[-180, -90, 180, 90],
                    start_datetime=datetime(1981, 4, 24, 3, 0, 0),
                    end_datetime=datetime(1981, 4, 24, 3, 0, 0),
                    instruments=['MSI'],
                    epsg=4326
                )
            }
            yield

    def test_create_job_040(self, api040):
        with self._fresh_job_registry(next_job_id="job-220"):
            resp = api040.post('/jobs', headers=self.AUTH_HEADER, json={
                'title': 'foo job',
                'process_graph': {"foo": {"process_id": "foo", "arguments": {}}},
            }).assert_status_code(201)
        assert resp.headers['Location'] == 'http://oeo.net/openeo/0.4.0/jobs/job-220'
        assert resp.headers['OpenEO-Identifier'] == 'job-220'
        job_info = dummy_backend.DummyBatchJobs._job_registry[TEST_USER, 'job-220']
        assert job_info.id == "job-220"
        assert job_info.process == {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}}
        assert job_info.status == "created"
        assert job_info.created == dummy_backend.DEFAULT_DATETIME
        assert job_info.job_options is None

    def test_create_job_with_options_040(self, api040):
        with self._fresh_job_registry(next_job_id="job-230"):
            resp = api040.post('/jobs', headers=self.AUTH_HEADER, json={
                'title': 'foo job',
                'process_graph': {"foo": {"process_id": "foo", "arguments": {}}},
                'job_options': {"driver-memory": "3g", "executor-memory": "5g"},
            }).assert_status_code(201)
        assert resp.headers['Location'] == 'http://oeo.net/openeo/0.4.0/jobs/job-230'
        assert resp.headers['OpenEO-Identifier'] == 'job-230'
        job_info = dummy_backend.DummyBatchJobs._job_registry[TEST_USER, 'job-230']
        assert job_info.job_options == {"driver-memory": "3g", "executor-memory": "5g"}

    def test_create_job_100(self, api100):
        with self._fresh_job_registry(next_job_id="job-245"):
            resp = api100.post('/jobs', headers=self.AUTH_HEADER, json={
                'process': {
                    'process_graph': {"foo": {"process_id": "foo", "arguments": {}}},
                    'summary': 'my foo job',
                },
                "title": "Foo job", "description": "Run the `foo` process!"
            }).assert_status_code(201)
        assert resp.headers['Location'] == 'http://oeo.net/openeo/1.0.0/jobs/job-245'
        assert resp.headers['OpenEO-Identifier'] == 'job-245'
        job_info = dummy_backend.DummyBatchJobs._job_registry[TEST_USER, 'job-245']
        assert job_info.id == "job-245"
        assert job_info.process == {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}}
        assert job_info.status == "created"
        assert job_info.created == dummy_backend.DEFAULT_DATETIME
        assert job_info.job_options is None
        assert job_info.title == "Foo job"
        assert job_info.description == "Run the `foo` process!"

    def test_create_job_100_with_options(self, api100):
        with self._fresh_job_registry(next_job_id="job-256"):
            resp = api100.post('/jobs', headers=self.AUTH_HEADER, json={
                'process': {
                    'process_graph': {"foo": {"process_id": "foo", "arguments": {}}},
                    'summary': 'my foo job',
                },
                'job_options': {"driver-memory": "3g", "executor-memory": "5g"},
            }).assert_status_code(201)
        assert resp.headers['Location'] == 'http://oeo.net/openeo/1.0.0/jobs/job-256'
        assert resp.headers['OpenEO-Identifier'] == 'job-256'
        job_info = dummy_backend.DummyBatchJobs._job_registry[TEST_USER, 'job-256']
        assert job_info.job_options == {"driver-memory": "3g", "executor-memory": "5g"}

    def test_start_job(self, api):
        with self._fresh_job_registry(next_job_id="job-267"):
            api.post('/jobs', headers=self.AUTH_HEADER, json=api.get_process_graph_dict(
                {"foo": {"process_id": "foo", "arguments": {}}},
            )).assert_status_code(201)
            assert dummy_backend.DummyBatchJobs._job_registry[TEST_USER, 'job-267'].status == "created"
            api.post('/jobs/job-267/results', headers=self.AUTH_HEADER, json={}).assert_status_code(202)
            assert dummy_backend.DummyBatchJobs._job_registry[TEST_USER, 'job-267'].status == "running"

    def test_start_job_invalid(self, api):
        resp = api.post('/jobs/deadbeef-f00/results', headers=self.AUTH_HEADER)
        resp.assert_error(404, "JobNotFound")
        assert resp.json["message"] == "The batch job 'deadbeef-f00' does not exist."

    def test_get_job_info_040(self, api040):
        resp = api040.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc', headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            'id': '07024ee9-7847-4b8a-b260-6c879a2b3cdc',
            'status': 'running',
            'submitted': "2017-01-01T09:32:12Z",
            'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}},
        }

    def test_get_job_info_metrics_100(self, api100):
        resp = api100.get('/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199', headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            'id': '53c71345-09b4-46b4-b6b0-03fd6fe1f199',
            'status': 'finished',
            'created': "2020-06-11T11:51:29Z",
            'process': {'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}}},
            'duration_seconds': 6,
            'duration_human_readable': "0:00:06",
            'memory_time_megabyte_seconds': 18704944,
            'memory_time_human_readable': "18704944 MB-seconds",
            'cpu_time_seconds': 1621,
            'cpu_time_human_readable': "1621 cpu-seconds"
        }

    def test_get_job_info_100(self, api100):
        resp = api100.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc', headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            'id': '07024ee9-7847-4b8a-b260-6c879a2b3cdc',
            'status': 'running',
            'created': "2017-01-01T09:32:12Z",
            'process': {'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}}},
        }

    def test_get_job_info_invalid(self, api):
        resp = api.get('/jobs/deadbeef-f00', headers=self.AUTH_HEADER).assert_error(404, "JobNotFound")
        assert resp.json["message"] == "The batch job 'deadbeef-f00' does not exist."

    def test_list_user_jobs_040(self, api040):
        with self._fresh_job_registry(next_job_id="job-318"):
            resp = api040.get('/jobs', headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            "jobs": [
                {
                    'id': '07024ee9-7847-4b8a-b260-6c879a2b3cdc',
                    'status': 'running',
                    'submitted': "2017-01-01T09:32:12Z",
                },
                {
                    'id': '53c71345-09b4-46b4-b6b0-03fd6fe1f199',
                    'status': 'finished',
                    'submitted': "2020-06-11T11:51:29Z"
                }
            ],
            "links": []
        }

    def test_list_user_jobs_100(self, api100):
        with self._fresh_job_registry(next_job_id="job-332"):
            resp = api100.get('/jobs', headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            "jobs": [
                {
                    'id': '07024ee9-7847-4b8a-b260-6c879a2b3cdc',
                    'status': 'running',
                    'created': "2017-01-01T09:32:12Z",
                },
                {
                    'id': '53c71345-09b4-46b4-b6b0-03fd6fe1f199',
                    'status': 'finished',
                    'created': "2020-06-11T11:51:29Z"
                }
            ],
            "links": []
        }

    def test_get_job_results_unfinished(self, api):
        with self._fresh_job_registry(next_job_id="job-345"):
            resp = api.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self.AUTH_HEADER)
        resp.assert_error(400, "JobNotFinished")

    def test_get_job_results_040(self, api040):
        with self._fresh_job_registry(next_job_id="job-349"):
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="finished")
            resp = api040.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            "links": [
                {
                    "href": "http://oeo.net/openeo/0.4.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/output.tiff"
                }
            ]
        }

    def test_get_job_results_100(self, api100):
        with self._fresh_job_registry(next_job_id="job-362"):
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="finished")
            resp = api100.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self.AUTH_HEADER)
            assert resp.assert_status_code(200).json == {
                'assets': {
                    'output.tiff': {
                        'roles': ['data'],
                        'title': 'output.tiff',
                        'href': 'http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/output.tiff',
                        'type': 'image/tiff; application=geotiff',
                        'eo:bands': [{
                            'name': "NDVI",
                            'center_wavelength': 1.23
                        }],
                        'file:nodata':[123]
                    }
                },
                'geometry': None,
                'id': '07024ee9-7847-4b8a-b260-6c879a2b3cdc',
                'links': [
                    {
                        "rel": "self",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results",
                        "type": "application/json"
                    },
                    {
                        "rel": "card4l-document",
                        "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                        "type": "application/pdf"
                    }
                ],
                'properties': {
                    'created': '2017-01-01T09:32:12Z',
                    'datetime': None,
                    'card4l:processing_chain': {'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}}},
                    'card4l:specification': 'SR',
                    'card4l:specification_version': '5.0',
                    'processing:facility': 'VITO - SPARK',
                    'processing:software': 'openeo-geotrellis-0.0.1'
                },
                'stac_extensions': ['processing',
                                    'card4l-eo',
                                    'https://stac-extensions.github.io/file/v1.0.0/schema.json',
                                    'eo'],
                 'stac_version': '0.9.0',
                 'type': 'Feature'
            }

            resp = api100.get('/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results', headers=self.AUTH_HEADER)

            assert resp.assert_status_code(200).json == {
                'assets': {
                    'output.tiff': {
                        'roles': ['data'],
                        'title': 'output.tiff',
                        'href': 'http://oeo.net/openeo/1.0.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/output.tiff',
                        'type': 'image/tiff; application=geotiff',
                        'eo:bands': [{
                            'name': "NDVI",
                            'center_wavelength': 1.23
                        }],
                        'file:nodata': [123]
                    }
                },
                'geometry': {
                    "type": "Polygon",
                    "coordinates": [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]]
                },
                'bbox': [-180, -90, 180, 90],
                'id': '53c71345-09b4-46b4-b6b0-03fd6fe1f199',
                'links': [
                    {
                        "rel": "self",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results",
                        "type": "application/json"
                    },
                    {
                        "rel": "card4l-document",
                        "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                        "type": "application/pdf"
                    }
                ],
                'properties': {
                    'created': '2020-06-11T11:51:29Z',
                    'datetime': '1981-04-24T03:00:00Z',
                    'instruments': ['MSI'],
                    'proj:epsg': 4326,
                    'card4l:processing_chain': {'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}}},
                    'card4l:specification': 'SR',
                    'card4l:specification_version': '5.0',
                    'processing:facility': 'VITO - SPARK',
                    'processing:software': 'openeo-geotrellis-0.0.1'},
                'stac_extensions': ['processing',
                                    'card4l-eo',
                                    'https://stac-extensions.github.io/file/v1.0.0/schema.json',
                                    'eo',
                                    'projection'],
                'stac_version': '0.9.0',
                'type': 'Feature'
            }

    @mock.patch.dict(app.config, {'SIGNED_URL': 'TRUE', 'SIGNED_URL_SECRET': '123&@#'})
    def test_get_job_results_signed_040(self, api040):
        with self._fresh_job_registry(next_job_id='job-370'):
            dummy_backend.DummyBatchJobs._update_status(
                job_id='07024ee9-7847-4b8a-b260-6c879a2b3cdc', user_id=TEST_USER, status='finished')
            resp = api040.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            'links': [
                {
                    'href': 'http://oeo.net/openeo/0.4.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/TXIuVGVzdA%3D%3D/50afb0cad129e61d415278c4ffcd8a83/output.tiff'
                }
            ]
        }

    @mock.patch.dict(app.config, {'SIGNED_URL': 'TRUE', 'SIGNED_URL_SECRET': '123&@#'})
    def test_get_job_results_signed_100(self, api100):
        with self._fresh_job_registry(next_job_id='job-372'):
            dummy_backend.DummyBatchJobs._update_status(
                job_id='07024ee9-7847-4b8a-b260-6c879a2b3cdc', user_id=TEST_USER, status='finished')
            resp = api100.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self.AUTH_HEADER)
            assert resp.assert_status_code(200).json == {
                'assets': {
                    'output.tiff': {
                        'roles': ['data'],
                        'title': 'output.tiff',
                        'href': 'http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/TXIuVGVzdA%3D%3D/50afb0cad129e61d415278c4ffcd8a83/output.tiff',
                        'type': 'image/tiff; application=geotiff',
                        'eo:bands': [{
                            'name': 'NDVI',
                            'center_wavelength': 1.23
                        }],
                        'file:nodata': [123]
                    }
                },
                'geometry': None,
                'id': '07024ee9-7847-4b8a-b260-6c879a2b3cdc',
                'links': [
                    {
                        'rel': 'self',
                        'href': 'http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results',
                        'type': 'application/json'
                    },
                    {
                        'rel': 'card4l-document',
                        'href': 'http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf',
                        'type': 'application/pdf'
                    }
                ],
                'properties': {
                    'created': '2017-01-01T09:32:12Z',
                    'datetime': None,
                    'card4l:processing_chain': {'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}}},
                    'card4l:specification': 'SR',
                    'card4l:specification_version': '5.0',
                    'processing:facility': 'VITO - SPARK',
                    'processing:software': 'openeo-geotrellis-0.0.1'
                },
                'stac_extensions': ['processing',
                                    'card4l-eo',
                                    'https://stac-extensions.github.io/file/v1.0.0/schema.json',
                                    'eo'],
                'stac_version': '0.9.0',
                'type': 'Feature'
            }

    @mock.patch('time.time', mock.MagicMock(return_value=1234))
    @mock.patch.dict(app.config, {'SIGNED_URL': 'TRUE', 'SIGNED_URL_SECRET': '123&@#', 'SIGNED_URL_EXPIRATION': '1000'})
    def test_get_job_results_signed_with_expiration_040(self, api040):
        with self._fresh_job_registry(next_job_id='job-371'):
            dummy_backend.DummyBatchJobs._update_status(
                job_id='07024ee9-7847-4b8a-b260-6c879a2b3cdc', user_id=TEST_USER, status='finished')
            resp = api040.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            'links': [
                {
                    'href': 'http://oeo.net/openeo/0.4.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/TXIuVGVzdA%3D%3D/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234'
                }
            ]
        }

    @mock.patch('time.time', mock.MagicMock(return_value=1234))
    @mock.patch.dict(app.config, {'SIGNED_URL': 'TRUE', 'SIGNED_URL_SECRET': '123&@#', 'SIGNED_URL_EXPIRATION': '1000'})
    def test_get_job_results_signed_with_expiration_100(self, api100):
        with self._fresh_job_registry(next_job_id='job-373'):
            dummy_backend.DummyBatchJobs._update_status(
                job_id='07024ee9-7847-4b8a-b260-6c879a2b3cdc', user_id=TEST_USER, status='finished')
            resp = api100.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self.AUTH_HEADER)
            assert resp.assert_status_code(200).json == {
                'assets': {
                    'output.tiff': {
                        'roles': ['data'],
                        'title': 'output.tiff',
                        'href': 'http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/TXIuVGVzdA%3D%3D/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234',
                        'type': 'image/tiff; application=geotiff',
                        'eo:bands': [{
                            'name': 'NDVI',
                            'center_wavelength': 1.23
                        }],
                        'file:nodata': [123]
                    }
                },
                'geometry': None,
                'id': '07024ee9-7847-4b8a-b260-6c879a2b3cdc',
                'links': [
                    {
                        'rel': 'self',
                        'href': 'http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results',
                        'type': 'application/json'
                    },
                    {
                        'rel': 'card4l-document',
                        'href': 'http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf',
                        'type': 'application/pdf'
                    }
                ],
                'properties': {
                    'created': '2017-01-01T09:32:12Z',
                    'datetime': None,
                    'card4l:processing_chain': {'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}}},
                    'card4l:specification': 'SR',
                    'card4l:specification_version': '5.0',
                    'processing:facility': 'VITO - SPARK',
                    'processing:software': 'openeo-geotrellis-0.0.1'
                },
                'stac_extensions': ['processing',
                                    'card4l-eo',
                                    'https://stac-extensions.github.io/file/v1.0.0/schema.json',
                                    'eo'],
                'stac_version': '0.9.0',
                'type': 'Feature'
            }

    def test_get_job_results_invalid_job(self, api):
        api.get('/jobs/deadbeef-f00/results', headers=self.AUTH_HEADER).assert_error(404, "JobNotFound")

    def test_download_result_invalid_job(self, api):
        api.get('/jobs/deadbeef-f00/results/some_file', headers=self.AUTH_HEADER).assert_error(404, "JobNotFound")

    def test_download_result(self, api, tmp_path):
        output_root = Path(tmp_path)
        with mock.patch.object(dummy_backend.DummyBatchJobs, '_output_root', return_value=output_root):
            output = output_root / "07024ee9-7847-4b8a-b260-6c879a2b3cdc" / "output.tiff"
            output.parent.mkdir(parents=True)
            with output.open("wb") as f:
                f.write(b"tiffdata")
            resp = api.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/output.tiff", headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).data == b"tiffdata"
        assert resp.headers["Content-Type"] == "image/tiff; application=geotiff"

    @mock.patch.dict(app.config, {'SIGNED_URL': 'TRUE', 'SIGNED_URL_SECRET': '123&@#'})
    def test_download_result_signed(self, api, tmp_path):
        output_root = Path(tmp_path)
        with mock.patch.object(dummy_backend.DummyBatchJobs, '_output_root', return_value=output_root):
            output = output_root / '07024ee9-7847-4b8a-b260-6c879a2b3cdc' / 'output.tiff'
            output.parent.mkdir(parents=True)
            with output.open('wb') as f:
                f.write(b'tiffdata')
            resp = api.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/TXIuVGVzdA%3D%3D/50afb0cad129e61d415278c4ffcd8a83/output.tiff')
        assert resp.assert_status_code(200).data == b'tiffdata'
        assert resp.headers['Content-Type'] == 'image/tiff; application=geotiff'

    @mock.patch.dict(app.config, {'SIGNED_URL': 'TRUE', 'SIGNED_URL_SECRET': '123&@#'})
    def test_download_result_signed_invalid(self, api):
        resp = api.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/TXIuVGVzdA%3D%3D/test123/output.tiff')
        assert resp.assert_error(403, 'CredentialsInvalid')

    @mock.patch('time.time', mock.MagicMock(return_value=1234))
    @mock.patch.dict(app.config, {'SIGNED_URL': 'TRUE', 'SIGNED_URL_SECRET': '123&@#', 'SIGNED_URL_EXPIRATION': '1000'})
    def test_download_result_signed_with_expiration(self, api, tmp_path):
        output_root = Path(tmp_path)
        with mock.patch.object(dummy_backend.DummyBatchJobs, '_output_root', return_value=output_root):
            output = output_root / '07024ee9-7847-4b8a-b260-6c879a2b3cdc' / 'output.tiff'
            output.parent.mkdir(parents=True)
            with output.open('wb') as f:
                f.write(b'tiffdata')
            resp = api.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/TXIuVGVzdA%3D%3D/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234')
        assert resp.assert_status_code(200).data == b'tiffdata'
        assert resp.headers['Content-Type'] == 'image/tiff; application=geotiff'

    @mock.patch('time.time', mock.MagicMock(return_value=3456))
    @mock.patch.dict(app.config, {'SIGNED_URL': 'TRUE', 'SIGNED_URL_SECRET': '123&@#', 'SIGNED_URL_EXPIRATION': '1000'})
    def test_download_result_signed_with_expiration_invalid(self, api, tmp_path):
        resp = api.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/TXIuVGVzdA%3D%3D/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234')
        assert resp.assert_error(410, 'ResultLinkExpired')

    def test_get_batch_job_logs(self, api):
        resp = api.get('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/logs', headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            "logs": [
                {"id": "1", "level": "info", "message": "hello world"}
            ],
            "links": []
        }

    def test_cancel_job(self, api):
        with self._fresh_job_registry(next_job_id="job-403"):
            resp = api.delete('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results', headers=self.AUTH_HEADER)
        assert resp.status_code == 204

    def test_cancel_job_invalid(self, api):
        with self._fresh_job_registry(next_job_id="job-403"):
            resp = api.delete('/jobs/deadbeef-f00/results', headers=self.AUTH_HEADER)
        resp.assert_error(404, "JobNotFound")

    def test_delete_job(self, api):
        with self._fresh_job_registry(next_job_id="job-403"):
            resp = api.delete('/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc', headers=self.AUTH_HEADER)
        assert resp.status_code == 204


class TestSecondaryServices:
    AUTH_HEADER = TEST_USER_AUTH_HEADER

    def test_service_types_v100(self, api):
        resp = api.get('/service_types').assert_status_code(200)
        service_types = resp.json
        assert list(service_types.keys()) == ["WMTS"]
        wmts = service_types["WMTS"]
        assert wmts["configuration"]["version"]["default"] == "1.0.0"
        assert wmts["process_parameters"] == []
        assert wmts["links"] == []

    def test_create_unsupported_service_type_returns_400_BadRequest(self, api):
        resp = api.post('/services', json={
            "process": {"process_graph": {'product_id': 'S2'}},
            "type": '???',
        }, headers=self.AUTH_HEADER)
        resp.assert_status_code(400)

    def test_unsupported_services_methods_return_405_MethodNotAllowed(self, api):
        res = api.put('/services', json={
            "process": {"process_graph": {'product_id': 'S2'}},
            "type": 'WMTS',
        })
        res.assert_status_code(405)

    def test_list_services_040(self, api040):
        metadata = api040.get('/services', headers=self.AUTH_HEADER).json
        assert metadata == {
            "services": [{
                'id': 'wmts-foo',
                'type': 'WMTS',
                'enabled': True,
                'url': 'https://oeo.net/wmts/foo',
                'submitted': '2020-04-09T15:05:08Z',
                'title': 'Test service',
                'parameters': {'version': '0.5.8'},
            }],
            "links": []
        }

    def test_list_services_100(self, api100):
        metadata = api100.get('/services', headers=self.AUTH_HEADER).json
        assert metadata == {
            "services": [{
                'id': 'wmts-foo',
                'type': 'WMTS',
                'enabled': True,
                'url': 'https://oeo.net/wmts/foo',
                'title': 'Test service',
                'created': '2020-04-09T15:05:08Z',
                'configuration': {'version': '0.5.8'},
            }],
            "links": []
        }

    def test_get_service_metadata_040(self, api040):
        metadata = api040.get('/services/wmts-foo', headers=self.AUTH_HEADER).json
        assert metadata == {
            "id": "wmts-foo",
            "process_graph": {"foo": {"process_id": "foo", "arguments": {}}},
            "url": "https://oeo.net/wmts/foo",
            "type": "WMTS",
            "enabled": True,
            "parameters": {"version": "0.5.8"},
            "attributes": {},
            "title": "Test service",
            'submitted': '2020-04-09T15:05:08Z',
        }

    def test_get_service_metadata_100(self, api100):
        metadata = api100.get('/services/wmts-foo', headers=self.AUTH_HEADER).json
        assert metadata == {
            "id": "wmts-foo",
            "process": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
            "url": "https://oeo.net/wmts/foo",
            "type": "WMTS",
            "enabled": True,
            "configuration": {"version": "0.5.8"},
            "attributes": {},
            "title": "Test service",
            'created': '2020-04-09T15:05:08Z',
        }

    def test_get_service_metadata_wrong_id(self, api):
        api.get('/services/wmts-invalid', headers=self.AUTH_HEADER).assert_error(404, 'ServiceNotFound')

    def test_services_requires_authentication(self, api):
        api.get('/services').assert_error(401, 'AuthenticationRequired')

    def test_get_service_requires_authentication(self, api):
        api.get('/services/wmts-foo').assert_error(401, 'AuthenticationRequired')

    def test_patch_service_requires_authentication(self, api):
        api.patch('/services/wmts-foo').assert_error(401, 'AuthenticationRequired')

    def test_delete_service_requires_authentication(self, api):
        api.delete('/services/wmts-foo').assert_error(401, 'AuthenticationRequired')

    def test_service_logs_100(self, api):
        logs = api.get('/services/wmts-foo/logs', headers=self.AUTH_HEADER).json
        assert logs == {
            "logs": [
                {"id": 3, "level": "info", "message": "Loaded data."},
            ],
            "links": []
        }


def test_credentials_basic_no_headers(api):
    api.get("/credentials/basic").assert_error(401, 'AuthenticationRequired')


def test_credentials_basic_wrong_password(api):
    headers = {"Authorization": _build_basic_http_auth_header(username="john", password="password123")}
    api.get("/credentials/basic", headers=headers).assert_error(403, 'CredentialsInvalid')


def test_credentials_basic(api):
    headers = {"Authorization": _build_basic_http_auth_header(username="john", password="john123")}
    response = api.get("/credentials/basic", headers=headers).assert_status_code(200).json
    expected = {"access_token"}
    if api.api_version_compare.below("1.0.0"):
        expected.add("user_id")
    assert set(response.keys()) == expected


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


@pytest.fixture
def udp_store() -> dummy_backend.DummyUserDefinedProcesses:
    udps = backend_implementation.user_defined_processes
    assert isinstance(udps, dummy_backend.DummyUserDefinedProcesses)
    udps.reset({
        ('Mr.Test', 'udp1'): UserDefinedProcessMetadata(
            id='udp1',
            process_graph={'process1': {}},
            public=True,
        ),
        ('Mr.Test', 'udp2'): UserDefinedProcessMetadata(
            id='udp2',
            process_graph={'process1': {}},
            public=False,
        )
    })
    return udps


class TestUserDefinedProcesses:
    def test_add_udp(self, api100, udp_store):
        api100.put('/process_graphs/evi', headers=TEST_USER_AUTH_HEADER, json={
            'id': 'evi',
            'parameters': [
                {'name': 'red'}
            ],
            'process_graph': {
                'sub': {}
            },
            'public': True
        }).assert_status_code(200)

        new_udp = udp_store._processes['Mr.Test', 'evi']
        assert new_udp.id == 'evi'
        assert new_udp.parameters == [{'name': 'red'}]
        assert new_udp.process_graph == {'sub': {}}
        assert new_udp.public

    def test_update_udp(self, api100, udp_store):
        api100.put('/process_graphs/udp1', headers=TEST_USER_AUTH_HEADER, json={
            'id': 'udp1',
            'parameters': [
                {'name': 'blue'}
            ],
            'process_graph': {
                'add': {}
            },
            'public': True
        }).assert_status_code(200)

        modified_udp = udp_store._processes['Mr.Test', 'udp1']
        assert modified_udp.id == 'udp1'
        assert modified_udp.process_graph == {'add': {}}
        assert modified_udp.parameters == [{'name': 'blue'}]
        assert modified_udp.public

    def test_list_udps(self, api100, udp_store):
        resp = api100.get('/process_graphs', headers=TEST_USER_AUTH_HEADER).assert_status_code(200)

        udps = resp.json['processes']
        udp1 = next(udp for udp in udps if udp['id'] == 'udp1')

        assert 'process_graph' not in udp1
        assert udp1['public']

    def test_get_udp(self, api100, udp_store):
        resp = api100.get('/process_graphs/udp1', headers=TEST_USER_AUTH_HEADER).assert_status_code(200)

        udp = resp.json
        assert udp['id'] == 'udp1'
        assert udp['public']

    def test_get_unknown_udp(self, api100, udp_store):
        api100.get('/process_graphs/unknown', headers=TEST_USER_AUTH_HEADER).assert_status_code(404)

    def test_delete_udp(self, api100, udp_store):
        assert ('Mr.Test', 'udp2') in udp_store._processes

        api100.delete('/process_graphs/udp2', headers=TEST_USER_AUTH_HEADER).assert_status_code(204)

        assert ('Mr.Test', 'udp1') in udp_store._processes
        assert ('Mr.Test', 'udp2') not in udp_store._processes

    def test_delete_unknown_udp(self, api100, udp_store):
        api100.delete('/process_graphs/unknown', headers=TEST_USER_AUTH_HEADER).assert_status_code(404)

    def test_public_udp(self, api100, udp_store):
        api100.put('/process_graphs/evi', headers=TEST_USER_AUTH_HEADER, json={
            'id': 'evi',
            'parameters': [{'name': 'red'}],
            'process_graph': {'sub': {}},
            'public': True
        }).assert_status_code(200)
        api100.put('/process_graphs/secret', headers=TEST_USER_AUTH_HEADER, json={
            'id': 'secret',
            'parameters': [{'name': 'red'}],
            'process_graph': {'sub': {}},
            'public': False
        }).assert_status_code(200)

        r = api100.get("/processes/u:Mr.Test").assert_status_code(200)
        assert r.json == {
            "processes": [
                {"id": "udp1", "public": True},
                {"id": "evi", "parameters": [{"name": "red"}], "public": True}
            ],
            "links": [],
        }


def test_debug_echo_get(api):
    res = api.get("/_debug/echo?xev=lol", headers={"foo": "bar"}).assert_status_code(200).json
    assert res["method"] == "GET"
    assert res["args"] == {"xev": "lol"}
    assert res["data"] == "b''"
    assert res["headers"]["Foo"] == "bar"


def test_debug_echo_post(api):
    res = api.post("/_debug/echo", json={"foo": "bar"}).assert_status_code(200).json
    assert res["method"] == "POST"
    assert res["data"] == 'b\'{"foo": "bar"}\''
