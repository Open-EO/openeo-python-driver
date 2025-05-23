import json
import logging
import re
import urllib.parse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest import mock

import boto3
import dirty_equals
import flask
import geopandas as gpd
import moto
import pystac.validation.stac_validator
import pytest
import re_assert
import time_machine
import werkzeug.exceptions
from openeo.utils.version import ComparableVersion

from openeo_driver.asset_urls import AssetUrl
from openeo_driver.backend import (
    BatchJobMetadata,
    BatchJobResultMetadata,
    BatchJobs,
    OpenEoBackendImplementation,
    Processing,
    UserDefinedProcessMetadata,
    not_implemented,
)
from openeo_driver.config import OpenEoBackendConfig
from openeo_driver.datacube import DriverVectorCube
from openeo_driver.dummy import dummy_backend, dummy_config
from openeo_driver.dummy.dummy_backend import DummyBackendImplementation, DummyProcessing
from openeo_driver.errors import OpenEOApiException
from openeo_driver.ProcessGraphDeserializer import custom_process_from_process_graph
from openeo_driver.processes import ProcessesListing
from openeo_driver.integrations.s3.client import S3ClientBuilder
from openeo_driver.save_result import VectorCubeResult
from openeo_driver.testing import (
    TEST_USER,
    TEST_USER_AUTH_HEADER,
    ApiResponse,
    ApiTester,
    DictSubSet,
    ListSubSet,
    RegexMatcher,
    build_basic_http_auth_header,
    generate_unique_test_process_id,
)
from openeo_driver.urlsigning import UrlSigner
from openeo_driver.users import User
from openeo_driver.users.auth import AccessTokenException, HttpAuthHandler
from openeo_driver.users.oidc import OidcProvider
from openeo_driver.util.logging import (
    LOGGING_CONTEXT_FLASK,
    FlaskRequestCorrelationIdLogging,
)
from openeo_driver.views import (
    STREAM_CHUNK_SIZE_DEFAULT,
    EndpointRegistry,
    _normalize_collection_metadata,
    build_app,
)

from .conftest import TEST_APP_CONFIG, enhanced_logging
from .data import TEST_DATA_ROOT, get_path
from .test_dry_run import CRS_UTM

EXPECTED_PROCESSING_EXPRESSION = {
    "expression": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
    "format": "openeo",
}


EXPECTED_PROVIDERS = [
    {
        "name": dummy_config.config.capabilities_title,
        "description": dummy_config.config.capabilities_description,
        "processing:expression": EXPECTED_PROCESSING_EXPRESSION,
        "processing:facility": dummy_config.config.processing_facility,
        "processing:software": {
            dummy_config.config.processing_software: dummy_config.config.capabilities_backend_version
        },
        "roles": ["processor"],
    }
]


@pytest.fixture(
    params=[
        "1.0",
        "1.1",
        "1.2",
    ]
)
def api_version(request) -> str:
    return request.param


@pytest.fixture
def api(api_version, client) -> ApiTester:
    return ApiTester(api_version=api_version, client=client, data_root=TEST_DATA_ROOT)


@pytest.fixture
def api100(client) -> ApiTester:
    return ApiTester(api_version="1.0.0", client=client, data_root=TEST_DATA_ROOT)


@pytest.fixture
def api110(client) -> ApiTester:
    return ApiTester(api_version="1.1.0", client=client, data_root=TEST_DATA_ROOT)


@pytest.fixture
def api120(client) -> ApiTester:
    return ApiTester(api_version="1.2", client=client, data_root=TEST_DATA_ROOT)


class DummyDirectS3Assets(AssetUrl):
    S3_ENDPOINT = "https://s3.oeo.test"

    def build_url(self, asset_metadata: dict, asset_name: str, job_id: str, user_id: str) -> str:
        href = asset_metadata.get("href", "")
        if href.startswith("s3://"):
            return href.replace("s3://", f"{self.S3_ENDPOINT}/")
        return super().build_url(asset_metadata=asset_metadata, asset_name=asset_name, job_id=job_id, user_id=user_id)


def api_from_backend_implementation(
    backend_implementation: OpenEoBackendImplementation, api_version="1.0.0", data_root=TEST_DATA_ROOT
) -> ApiTester:
    app: flask.Flask = build_app(backend_implementation)
    app.config.from_mapping(TEST_APP_CONFIG)
    client = app.test_client()
    api = ApiTester(api_version=api_version, client=client, data_root=data_root)
    return api


TEST_AWS_REGION_NAME = "eu-central-1"


@pytest.fixture(scope="function")
def aws_credentials(monkeypatch):
    """Mocked AWS Credentials and related environment variables for moto/boto3."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", TEST_AWS_REGION_NAME)
    monkeypatch.setenv("AWS_REGION", TEST_AWS_REGION_NAME)
    monkeypatch.setenv("SWIFT_BUCKET", "openeo-test-bucket")


@pytest.fixture(scope="function")
def mock_s3_resource(aws_credentials, monkeypatch):
    with moto.mock_aws():
        # Make sure clients with custom endpoints are within Moto scope as by default they are not
        def _get_client(*args, **kwargs):
            return boto3.client("s3")

        monkeypatch.setattr(S3ClientBuilder, "_s3_client", _get_client)
        yield boto3.resource("s3", region_name=TEST_AWS_REGION_NAME)


def create_s3_bucket(s3_resource, bucket_name):
    bucket = s3_resource.Bucket(bucket_name)
    bucket.create(CreateBucketConfiguration={"LocationConstraint": TEST_AWS_REGION_NAME})
    return bucket


class TestGeneral:
    """
    General tests (capabilities, collections, processes)
    """

    def test_well_known_openeo(self, client):
        resp = client.get("/.well-known/openeo")
        assert resp.status_code == 200
        versions = resp.json["versions"]
        # in .well-known/openeo there should only be one item per api_version (no aliases)
        by_api_version = {d["api_version"]: d for d in versions}
        assert len(versions) == len(by_api_version)
        assert by_api_version == {
            "1.0.0": {"api_version": "1.0.0", "production": True, "url": "http://oeo.net/openeo/1.0/"},
            "1.1.0": {"api_version": "1.1.0", "production": True, "url": "http://oeo.net/openeo/1.1/"},
            "1.2.0": {"api_version": "1.2.0", "production": True, "url": "http://oeo.net/openeo/1.2/"},
        }
        assert resp.headers["Cache-Control"] == "max-age=900, public"

    def test_versioned_well_known_openeo(self, api):
        api.get("/.well-known/openeo").assert_error(404, "NotFound")

    @pytest.mark.parametrize(
        ["headers", "expected"],
        [
            ({}, "http://oeo.net/"),
            ({"X-Forwarded-Proto": "https"}, "https://oeo.net/"),
        ],
    )
    def test_https_proxy_handling(self, client, headers, expected):
        resp = client.get("/.well-known/openeo", headers=headers)
        for url in [v["url"] for v in resp.json["versions"]]:
            assert url.startswith(expected)

    @pytest.mark.parametrize(
        ["url", "expected_version"],
        [
            ("/openeo/1.0/", "1.0.0"),
            ("/openeo/1.0.0/", "1.0.0"),
            ("/openeo/1.1/", "1.1.0"),
            ("/openeo/1.1.0/", "1.1.0"),
            ("/openeo/1.2/", "1.2.0"),
            ("/openeo/1/", "1.2.0"),
            ("/openeo/", "1.2.0"),
        ],
    )
    def test_versioned_urls(self, client, url, expected_version):
        resp = client.get(url)
        assert resp.status_code == 200
        capabilities = resp.json
        assert capabilities["title"] == "Dummy openEO Backend"
        assert capabilities["api_version"] == expected_version

    def test_capabilities_100(self, api100):
        capabilities = api100.get("/").assert_status_code(200).json
        assert capabilities["api_version"] == "1.0.0"
        assert capabilities["stac_version"] == "0.9.0"
        assert capabilities["title"] == "Dummy openEO Backend"
        assert (
            capabilities["description"]
            == "Dummy openEO backend provided by [openeo-python-driver](https://github.com/Open-EO/openeo-python-driver)."
        )
        assert capabilities["backend_version"] == "1.2.3-foo"
        assert capabilities["id"] == "dummyopeneobackend-1.2.3-foo"
        assert capabilities["production"] is True

        assert capabilities["conformsTo"] == dirty_equals.Contains("https://api.openeo.org/1.2.0")

        def get_link(rel):
            links = [link for link in capabilities["links"] if link["rel"] == rel]
            assert len(links) == 1
            return links[0]

        assert get_link("version-history")["href"] == "http://oeo.net/.well-known/openeo"
        assert get_link("data")["href"] == "http://oeo.net/openeo/1.0.0/collections"
        assert get_link("conformance")["href"] == "http://oeo.net/openeo/1.0.0/conformance"

    def test_capabilities_invalid_api_version(self, client):
        resp = ApiResponse(client.get("/openeo/0.0.0/"))
        resp.assert_error(501, "UnsupportedApiVersion")
        assert "Unsupported version component in URL: '0.0.0'" in resp.json["message"]

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
        assert endpoints["/processing_parameters"] == ["GET"]
        assert endpoints["/service_types"] == ["GET"]
        assert endpoints["/services"] == ["GET", "POST"]
        assert endpoints["/services/{service_id}"] == ["DELETE", "GET"]
        # assert endpoints["/subscription"] == ["GET"]
        assert endpoints["/jobs/{job_id}"] == ["DELETE", "GET"]
        assert endpoints["/jobs/{job_id}/results"] == ["DELETE", "GET", "POST"]
        assert endpoints["/credentials/basic"] == ["GET"]
        assert endpoints["/credentials/oidc"] == ["GET"]
        assert endpoints["/me"] == ["GET"]
        assert endpoints["/validation"] == ["POST"]

    def test_capabilities_caching(self, api):
        headers = api.get("/").assert_status_code(200).headers
        assert headers["Cache-Control"] == "max-age=900, public"
        assert "Expires" in headers

    def test_capabilities_endpoints_hiding(self):
        class MyProcessing(Processing):
            @not_implemented
            def validate(self, *args, **kwargs): ...

        backend_implementation = DummyBackendImplementation(processing=MyProcessing())
        api = api_from_backend_implementation(backend_implementation)
        capabilities = api.get("/").assert_status_code(200).json
        endpoints = {e["path"]: sorted(e["methods"]) for e in capabilities["endpoints"]}
        assert "/validation" not in endpoints

    def test_capabilities_endpoints_issue_28_v100(self, api100):
        """https://github.com/Open-EO/openeo-python-driver/issues/28"""
        capabilities = api100.get("/").assert_status_code(200).json
        endpoints = {e["path"]: e["methods"] for e in capabilities["endpoints"]}
        assert endpoints["/file_formats"] == ["GET"]
        assert "/output_formats" not in endpoints

    @pytest.mark.parametrize("enable_basic_auth", [False, True])
    def test_capabilities_basic_auth_disabled(self, enable_basic_auth):
        config = OpenEoBackendConfig(enable_basic_auth=enable_basic_auth)
        backend_implementation = DummyBackendImplementation(config=config)
        api100 = api_from_backend_implementation(backend_implementation)
        capabilities = api100.get("/").assert_status_code(200).json
        endpoints = {e["path"] for e in capabilities["endpoints"]}
        if enable_basic_auth:
            assert "/credentials/basic" in endpoints
            api100.get("/credentials/basic").assert_error(401, "AuthenticationRequired")
        else:
            assert "/credentials/basic" not in endpoints
            api100.get("/credentials/basic").assert_error(404, "NotFound")

    @pytest.mark.parametrize("enable_oidc_auth", [False, True])
    def test_capabilities_oidc_auth_default(self, enable_oidc_auth):
        config = OpenEoBackendConfig(enable_oidc_auth=enable_oidc_auth)
        backend_implementation = DummyBackendImplementation(config=config)
        api100 = api_from_backend_implementation(backend_implementation)
        capabilities = api100.get("/").assert_status_code(200).json
        endpoints = {e["path"] for e in capabilities["endpoints"]}
        if enable_oidc_auth:
            assert "/credentials/oidc" in endpoints
            api100.get("/credentials/oidc").assert_status_code(200)
        else:
            assert "/credentials/oidc" not in endpoints
            api100.get("/credentials/oidc").assert_error(404, "NotFound")

    def test_capabilities_processing_software(self, api100):
        capabilities = api100.get("/").assert_status_code(200).json
        assert capabilities["processing:software"] == {
            "openeo": RegexMatcher(r"0.\d+\.\d+"),
            "openeo_driver": RegexMatcher(r"0.\d+\.\d+"),
        }
        assert any("stac-extensions.github.io/processing" in e for e in capabilities.get("stac_extensions", []))

    def test_conformance(self, api100):
        res = api100.get("/conformance").assert_status_code(200).json
        assert res == {
            "conformsTo": dirty_equals.Contains(
                "https://api.openeo.org/1.2.0",
                "https://api.openeo.org/extensions/remote-process-definition/0.1.0",
            ),
        }

    @pytest.mark.parametrize(
        "path",
        [
            "/",
            "/collections",
            "/processes",
            "/jobs",
        ],
    )
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
            json={"process": pg} if api.api_version_compare.at_least("1.0.0") else pg,
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
                "process_graph": {"a": {"process_id": "add", "arguments": {"x": 1, "y": 2}, "result": True}},
            },
        )
        resp = response.assert_status_code(200).response
        # General CORS headers
        assert resp.access_control_allow_origin == "*"
        assert resp.access_control_allow_credentials is False
        assert {"Location", "OpenEO-Identifier", "OpenEO-Costs", "Link"}.issubset(resp.access_control_expose_headers)

    def test_error_handling_generic(self, api, caplog):
        caplog.set_level(logging.WARNING)
        resp = api.get("/_debug/error")
        assert (resp.status_code, resp.json) == (
            500,
            {
                "code": "Internal",
                "message": "Server error: Exception('Computer says no.')",
                "id": re_assert.Matches("r-[0-9a-f]{32}"),
            },
        )
        assert caplog.record_tuples == [
            (
                "openeo_driver.views.error",
                logging.ERROR,
                "Exception('Computer says no.')",
            ),
        ]

    def test_error_handling_not_implemented_error(self, api, caplog):
        caplog.set_level(logging.WARNING)
        resp = api.get("/_debug/error/basic/NotImplementedError")
        assert (resp.status_code, resp.json) == (
            500,
            {
                "code": "Internal",
                "message": "Server error: NotImplementedError()",
                "id": re_assert.Matches("r-[0-9a-f]{32}"),
            },
        )
        assert caplog.record_tuples == [
            (
                "openeo_driver.views.error",
                logging.ERROR,
                "NotImplementedError()",
            )
        ]

    def test_error_handling_error_summary(self, api, caplog):
        caplog.set_level(logging.WARNING)
        resp = api.get("/_debug/error/basic/ErrorSummary")
        assert (resp.status_code, resp.json) == (
            500,
            {
                "code": "Internal",
                "message": "Server error: No negatives please.",
                "id": re_assert.Matches("r-[0-9a-f]{32}"),
            },
        )
        assert caplog.record_tuples == [
            (
                "openeo_driver.views.error",
                logging.ERROR,
                "No negatives please.",
            )
        ]

    @pytest.mark.parametrize(
        ["url", "error_status", "error_code", "error_message"],
        [
            ("/_debug/error/api", 500, "Internal", "Computer says no."),
            ("/_debug/error/api/404/CollectionNotFound", 404, "CollectionNotFound", "Computer says no."),
            ("/_debug/error/http", 500, "Internal", "500 Internal Server Error: Computer says no."),
            ("/_debug/error/http/404", 404, "NotFound", "404 Not Found: Computer says no."),
            ("/_debug/error/http/501", 501, "Internal", "501 Not Implemented: Computer says no."),
            ("/invalid/url", 404, "NotFound", f"404 Not Found: {werkzeug.exceptions.NotFound.description}"),
        ],
    )
    def test_error_handling_api_error(self, api, caplog, url, error_status, error_code, error_message):
        caplog.set_level(logging.WARNING)
        resp = api.get(url)
        assert resp.status_code == error_status
        assert resp.json == {"code": error_code, "message": error_message, "id": re_assert.Matches("r-[0-9a-f]{32}")}
        assert caplog.record_tuples == [
            (
                "openeo_driver.views.error",
                logging.ERROR,
                re_assert.Matches(
                    re.escape(
                        f"OpenEOApiException(status_code={error_status}, code={error_code!r}, message={error_message!r}, id='"
                    )
                    + "r-[0-9a-f]{32}"
                    + re.escape("')")
                ),
            ),
        ]

    def test_health_basic(self, api):
        resp = api.get("/health").assert_status_code(200).json
        assert resp == {"health": "OK"}

    def test_health_dict(self, api, backend_implementation):
        with mock.patch.object(backend_implementation, "health_check") as health_check:
            health_check.return_value = {"status": "OK", "color": "green"}
            resp = api.get("/health").assert_status_code(200).json
            assert resp == {"status": "OK", "color": "green"}

    def test_health_flask_response(self, api, flask_app, backend_implementation):
        with mock.patch.object(backend_implementation, "health_check") as health_check:
            health_check.return_value = flask_app.make_response(
                ('{"code": "meh"}', 500, {"Content-type": "application/json"})
            )
            api.get("/health").assert_error(500, "meh")

    def test_health_dynamic(self, api, backend_implementation):
        def health(options: Optional[dict] = None):
            return {
                "status": "OK",
                "color": (options or {}).get("color", "green"),
            }

        with mock.patch.object(backend_implementation, "health_check", new=health):
            resp = api.get("/health").assert_status_code(200).json
            assert resp == {"status": "OK", "color": "green"}
            resp = api.get("/health?color=blue").assert_status_code(200).json
            assert resp == {"status": "OK", "color": "blue"}
            resp = api.get("/health?shape=square&color=red").assert_status_code(200).json
            assert resp == {"status": "OK", "color": "red"}

    def test_credentials_oidc_100(self, api100):
        resp = api100.get("/credentials/oidc").assert_status_code(200).json
        assert resp == {
            "providers": ListSubSet(
                [
                    {"id": "testprovider", "issuer": "https://oidc.test", "scopes": ["openid"], "title": "Test"},
                    DictSubSet(
                        {
                            "id": "eoidc",
                            "issuer": "https://eoidc.test",
                            "scopes": ["openid"],
                            "default_clients": [
                                {
                                    "id": "badcafef00d",
                                    "grant_types": [
                                        "urn:ietf:params:oauth:grant-type:device_code+pkce",
                                        "refresh_token",
                                    ],
                                }
                            ],
                        }
                    ),
                    DictSubSet({"id": "local"}),
                ]
            )
        }

    def test_file_formats(self, api100):
        response = api100.get("/file_formats")
        resp = response.assert_status_code(200).json
        assert resp == {
            "input": DictSubSet(
                {
                    "GeoJSON": {"gis_data_types": ["vector"], "parameters": {}},
                }
            ),
            "output": DictSubSet(
                {
                    "GTiff": {"title": "GeoTiff", "gis_data_types": ["raster"], "parameters": {}},
                }
            ),
        }
        assert response.headers["Cache-Control"] == "max-age=900, public"

    def test_processing_parameters(self, api100):
        response = api100.get('/processing_parameters')
        resp = response.assert_status_code(200).json
        assert resp == []
        assert response.headers["Cache-Control"] == "max-age=900, public"


    @pytest.mark.parametrize(
        ["user_id", "expect_success"],
        [
            ("Mark", False),
            ("John", True),
        ],
    )
    def test_user_access_validation(self, api, user_id, expect_success):
        headers = {"Authorization": "Bearer basic//" + HttpAuthHandler.build_basic_access_token(user_id=user_id)}
        response = api.get("/jobs", headers=headers)
        if expect_success:
            response.assert_status_code(200)
        else:
            response.assert_error(403, "PermissionsInsufficient", message="No access for Mark.")

    def test_after_request(self):
        backend_implementation = DummyBackendImplementation()
        backend_implementation.after_request = mock.Mock()
        api = api_from_backend_implementation(backend_implementation)

        with mock.patch.object(FlaskRequestCorrelationIdLogging, "_build_request_id", return_value="r-abc123"):
            api.get("/health")

        backend_implementation.after_request.assert_called_with("r-abc123")

    @pytest.mark.parametrize(
        ["backend_config_overrides", "expected413"],
        [
            ({}, False),
            ({"flask_settings": {"MAX_CONTENT_LENGTH": 100}}, True),
        ],
    )
    def test_max_content_length(self, api, expected413):
        pg = {
            "process": {
                "process_graph": {"add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}},
            },
            "too": "muuuuuuuch" * 100,
        }
        response = api.post("/jobs", headers=TEST_USER_AUTH_HEADER, json=pg)
        if expected413:
            response.assert_error(status_code=413, error_code="Internal", message="Request Entity Too Large")
        else:
            response.assert_status_code(201)


class TestProcessRegistry:
    @pytest.mark.parametrize(
        "endpoint",
        [
            "/processes",
            "/processes/backend",
        ],
    )
    def test_processes_basic(self, api_version, api, endpoint):
        resp = api.get(endpoint).assert_status_code(200).json
        processes = resp["processes"]
        process_ids = set(p["id"] for p in processes)
        assert {"load_collection", "min", "max", "sin", "merge_cubes", "mask"}.issubset(process_ids)
        expected_keys = {"id", "description", "parameters", "returns"}
        for process in processes:
            assert all(k in process for k in expected_keys)

        expected_version = "2.0.0-rc.1" if ComparableVersion(api_version) >= "1.2" else "1.2.0"
        assert resp["version"] == expected_version

    @pytest.mark.parametrize(
        "backend_config_overrides",
        [
            {"processes_exclusion_list": {"1.0.0": ["merge_cubes"]}},
        ],
    )
    def test_processes_exclusion(self, api, backend_config_overrides):
        resp = api.get("/processes").assert_status_code(200).json
        ids = [c["id"] for c in resp["processes"]]
        if ComparableVersion(api.api_version) == "1.0.0":
            assert "merge_cubes" not in ids
        else:
            assert "merge_cubes" in ids

    def test_process_details(self, api100):
        spec = api100.get("/processes/backend/add").assert_status_code(200).json
        assert spec["id"] == "add"
        assert spec["summary"].lower() == "addition of two numbers"
        assert "x + y" in spec["description"]
        assert set(p["name"] for p in spec["parameters"]) == {"x", "y"}
        assert "computed sum" in spec["returns"]["description"]
        assert "process_graph" in spec

    def test_processes_non_standard_atmospheric_correction(self, api):
        if api.api_version_compare.below("1.0.0"):
            pytest.skip()
        resp = api.get("/processes").assert_status_code(200).json
        (spec,) = [p for p in resp["processes"] if p["id"] == "atmospheric_correction"]
        assert spec["summary"] == "Apply atmospheric correction"
        assert spec["categories"] == ["cubes", "optical"]
        assert spec["experimental"] is True
        assert spec["links"][0]["rel"] == "about"
        assert "DigitalElevationModelInvalid" in spec["exceptions"]

    def test_custom_processes(self, api100):
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
        assert "increment" not in set(p["id"] for p in processes)

    @pytest.mark.parametrize(
        ["dummy_processing", "expected"],
        [
            (
                DummyProcessing(use_dummy_process_registry=["add"]),
                {
                    "processes": dirty_equals.IsList(length=1),
                    "flavor": "salt and pepper",
                    "links": [{"rel": "docs", "href": "http://processing.test/dummy"}],
                    "version": "dummy-v2",
                },
            ),
            (
                # Default: just use standard ProcessRegistries
                None,
                {
                    "processes": dirty_equals.IsList(length=...),
                    "links": [],
                    "version": "2.0.0-rc.1",
                },
            ),
        ],
    )
    def test_custom_process_listing_response(self, api120, dummy_processing, expected):
        resp = api120.get("/processes").assert_status_code(200).json
        assert resp == expected


@pytest.fixture
def oidc_provider(requests_mock):
    oidc_issuer = "https://eoidc.test"
    user_db = {
        # Access token to introspection data mapping
        "j0hn": {"sub": "john", "active": True, "scope": "openid", "iss": oidc_issuer, "typ": "Bearer"},
        "4l1c3": {"sub": "Alice", "active": True, "scope": "openid", "iss": oidc_issuer},
        "4l1c3.old": {"sub": "Alice", "active": False, "scope": "openid"},
        "b0b": {"sub": "b0b1b08571101437a2", "active": True, "scope": "openid"},
        "c6r01": {"sub": "Carol", "active": True, "scope": "openid"},
        "cust0m.cl13nt.j0": {
            "active": True,
            "sub": "s3rv1c36cc-0fj0hn",
            "username": "service-account-0fj0hn",
            "scope": "openid",
        },
        "cust0m.cl13nt.4l": {
            "active": True,
            "sub": "s3rv1c36cc-0f6l1c3",
            "preferred_username": "service-account-0f6l1c3",
            "scope": "openid",
        },
    }
    oidc_conf = f"{oidc_issuer}/.well-known/openid-configuration"
    oidc_userinfo_url = f"{oidc_issuer}/userinfo"
    oidc_introspection_url = f"{oidc_issuer}/token/inspect"
    requests_mock.get(
        oidc_conf,
        json={
            "userinfo_endpoint": oidc_userinfo_url,
            "introspection_endpoint": oidc_introspection_url,
        },
    )

    def userinfo(request, context):
        """Fake OIDC /userinfo endpoint handler"""
        _, _, token = request.headers["Authorization"].partition("Bearer ")
        if token in user_db:
            return {"sub": user_db[token]["sub"]}
        else:
            context.status_code = 401
            return {"code": "InvalidToken"}

    def introspection(request, context):
        """Fake OIDC token introspection endpoint"""
        try:
            data = urllib.parse.parse_qs(request.body)
            token = data["token"][0]
            return user_db[token]
        except Exception:
            context.status_code = 401
            return {"code": "InvalidToken"}

    requests_mock.get(oidc_userinfo_url, json=userinfo)
    requests_mock.post(oidc_introspection_url, json=introspection)


class TestUser:
    def test_no_auth(self, api):
        api.get("/me").assert_error(401, "AuthenticationRequired")

    def test_basic_auth(self, api):
        response = api.get("/me", headers=TEST_USER_AUTH_HEADER).assert_status_code(200).json
        assert response == {"name": TEST_USER, "user_id": TEST_USER}

    def test_oidc_basic(self, api, oidc_provider):
        response = api.get("/me", headers={"Authorization": "Bearer oidc/eoidc/j0hn"}).assert_status_code(200).json
        assert response == DictSubSet({"user_id": "john"})

    def test_oidc_invalid_access_token(self, api, oidc_provider):
        api.get("/me", headers={"Authorization": "Bearer oidc/eoidc/invalid"}).assert_error(403, "TokenInvalid")

    def test_oidc_invalid_provider(self, api, oidc_provider):
        api.get("/me", headers={"Authorization": "Bearer oidc/invalid/j0hn"}).assert_error(403, "TokenInvalid")

    def test_default_plan(self, api, oidc_provider):
        response = api.get("/me", headers={"Authorization": "Bearer oidc/eoidc/4l1c3"}).assert_status_code(200).json
        assert response == DictSubSet({"user_id": "Alice", "default_plan": "alice-plan"})

    def test_roles(self, api, oidc_provider):
        response = api.get("/me", headers={"Authorization": "Bearer oidc/eoidc/c6r01"}).assert_status_code(200).json
        assert response == DictSubSet({"user_id": "Carol", "roles": ["admin", "devops"]})

    def _get_api_with_advanced_token_handling(
        self,
        oidc_token_introspection: bool = True,
        oidc_client_user_map: Optional[dict] = None,
        api_version: str = "1.0.0",
    ):
        """
        Special API tester setup with token introspection stuff that is not standard yet
        or in the default fixtures.
        """
        provider = OidcProvider(
            id="eoidc",
            issuer="https://eoidc.test",
            scopes=["openid"],
            title="e-OIDC",
            service_account=("service-account-123", "s3rv1c3-6cc0unt-123"),
        )
        backend_config = OpenEoBackendConfig(
            id="_get_api_with_client_mapping",
            oidc_providers=[provider],
            oidc_token_introspection=oidc_token_introspection,
            oidc_user_map=oidc_client_user_map or {},
        )
        backend_implementation = DummyBackendImplementation(config=backend_config)
        flask_app = build_app(
            backend_implementation=backend_implementation,
            # error_handling=False,
        )
        flask_app.config.from_mapping(TEST_APP_CONFIG)

        auth_handler = flask_app.extensions["auth_handler"]

        @flask_app.route(f"/openeo/<version>/dump-user-internal-auth-data", methods=["GET"])
        @auth_handler.requires_bearer_auth
        def dump_user_internal_auth_data(user: User):
            return flask.jsonify(user.internal_auth_data)

        client = flask_app.test_client()
        return ApiTester(api_version=api_version, client=client, data_root=TEST_DATA_ROOT)

    @pytest.mark.parametrize(
        ["oidc_token_introspection", "oidc_client_user_map", "access_token", "expected"],
        [
            (False, {}, "cust0m.cl13nt.j0", "s3rv1c36cc-0fj0hn"),
            (False, {("eoidc", "s3rv1c36cc-0fj0hn"): {"user_id": "john"}}, "cust0m.cl13nt.j0", "s3rv1c36cc-0fj0hn"),
            (
                True,
                {},
                "cust0m.cl13nt.j0",
                AccessTokenException("Client credentials access token without user mapping: sub='s3rv1c36cc-0fj0hn'."),
            ),
            (True, {("eoidc", "s3rv1c36cc-0fj0hn"): {"user_id": "john"}}, "cust0m.cl13nt.j0", "s3rv1c36cc-0fj0hn"),
            (
                True,
                {("eoidc", "s3rv1c36cc-0fj0hn"): {"user_id": "john"}},
                "cust0m.cl13nt.4l",
                AccessTokenException("Client credentials access token without user mapping: sub='s3rv1c36cc-0f6l1c3'."),
            ),
            (True, {("eoidc", "s3rv1c36cc-0f6l1c3"): {"user_id": "Alice"}}, "cust0m.cl13nt.4l", "s3rv1c36cc-0f6l1c3"),
        ],
    )
    def test_oidc_client_to_user_mapping(
        self, oidc_provider, oidc_token_introspection, oidc_client_user_map, access_token, expected
    ):
        api = self._get_api_with_advanced_token_handling(
            oidc_token_introspection=oidc_token_introspection, oidc_client_user_map=oidc_client_user_map
        )
        headers = {"Authorization": f"Bearer oidc/eoidc/{access_token}"}
        response = api.get("/me", headers=headers)
        if isinstance(expected, str):
            assert response.assert_status_code(200).json == DictSubSet({"user_id": expected})
        elif isinstance(expected, OpenEOApiException):
            assert response.assert_error(expected.status_code, expected.code, message=expected.message)
        else:
            raise ValueError(expected)

    @pytest.mark.parametrize(
        ["oidc_token_introspection", "access_token", "expected"],
        [
            (
                False,
                "j0hn",
                DictSubSet(
                    {
                        "access_token": "j0hn",
                        "authentication_method": "OIDC",
                        "oidc_issuer": "https://eoidc.test",
                        "oidc_provider_id": "eoidc",
                        "oidc_token_sub": "john",
                    }
                ),
            ),
            (
                True,
                "j0hn",
                DictSubSet(
                    {
                        "access_token": "j0hn",
                        "authentication_method": "OIDC",
                        "oidc_access_token_introspection": {
                            "active": True,
                            "iss": "https://eoidc.test",
                            "scope": "openid",
                            "sub": "john",
                            "typ": "Bearer",
                        },
                        "oidc_issuer": "https://eoidc.test",
                        "oidc_provider_id": "eoidc",
                        "oidc_token_sub": "john",
                    }
                ),
            ),
        ],
    )
    def test_internal_auth_data(self, oidc_provider, oidc_token_introspection, access_token, expected):
        api = self._get_api_with_advanced_token_handling(oidc_token_introspection=oidc_token_introspection)
        headers = {"Authorization": f"Bearer oidc/eoidc/{access_token}"}
        response = api.get("/dump-user-internal-auth-data", headers=headers)
        data = response.assert_status_code(200).json
        assert data == expected

    @pytest.mark.parametrize(
        ["access_token", "expected"],
        [
            (
                "4l1c3",
                DictSubSet(
                    {
                        "oidc_access_token_introspection": {
                            "sub": "Alice",
                            "active": True,
                            "scope": "openid",
                            "iss": "https://eoidc.test",
                        }
                    }
                ),
            ),
            ("4l1c3.old", AccessTokenException("Access token not active.")),
        ],
    )
    def test_token_introspection_active(self, oidc_provider, access_token, expected):
        api = self._get_api_with_advanced_token_handling(oidc_token_introspection=True)
        headers = {"Authorization": f"Bearer oidc/eoidc/{access_token}"}
        response = api.get("/dump-user-internal-auth-data", headers=headers)
        if isinstance(expected, OpenEOApiException):
            response.assert_error(status_code=expected.status_code, error_code=expected.code, message=expected.message)
        else:
            assert response.assert_status_code(200).json == expected


class TestLogging:
    def test_user_id_logging(self, api, oidc_provider):
        with enhanced_logging(json=True, context=LOGGING_CONTEXT_FLASK) as logs:
            # Make request that requires auth (so that we have a user) and fails (so that we have a log to look at)
            resp = api.post("/result", json={"broken": "yezz"}, headers={"Authorization": "Bearer oidc/eoidc/b0b"})
            resp.assert_error(400, "ProcessGraphMissing")

        logs = [l for l in logs.getvalue().strip().split("\n")]
        logs = [json.loads(l) for l in logs]
        error = next(l for l in logs if "ProcessGraphMissing" in l["message"])

        assert error["user_id"] == "b0b1b08571101437a2"
        assert error["req_id"] == "123-456"


class TestCollections:
    def test_normalize_collection_metadata_no_id(self, caplog):
        with pytest.raises(KeyError):
            _normalize_collection_metadata({"foo": "bar"}, api_version=ComparableVersion("1.0.0"))
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("should have 'id' field" in m for m in errors)

    def test_normalize_collection_metadata_minimal_100(self, caplog):
        assert _normalize_collection_metadata({"id": "foobar"}, api_version=ComparableVersion("1.0.0")) == {
            "id": "foobar",
            "stac_version": "0.9.0",
            "stac_extensions": [
                "https://stac-extensions.github.io/datacube/v2.2.0/schema.json",
                "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
            ],
            "description": "foobar",
            "extent": {"spatial": {"bbox": [[0, 0, 0, 0]]}, "temporal": {"interval": [[None, None]]}},
            "license": "proprietary",
            "links": [],
        }
        warnings = set(r.getMessage() for r in caplog.records if r.levelno == logging.WARN)
        assert warnings == {"Collection 'foobar' metadata does not have field 'extent'."}

    def test_normalize_collection_metadata_minimal_full_100(self, caplog):
        assert _normalize_collection_metadata({"id": "foobar"}, api_version=ComparableVersion("1.0.0"), full=True) == {
            "id": "foobar",
            "stac_version": "0.9.0",
            "stac_extensions": [
                "https://stac-extensions.github.io/datacube/v2.2.0/schema.json",
                "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
            ],
            "description": "foobar",
            "extent": {"spatial": {"bbox": [[0, 0, 0, 0]]}, "temporal": {"interval": [[None, None]]}},
            "license": "proprietary",
            "cube:dimensions": {},
            "summaries": {},
            "links": [],
        }
        warnings = set(r.getMessage() for r in caplog.records if r.levelno == logging.WARN)
        assert warnings == {
            "Collection 'foobar' metadata does not have field 'cube:dimensions'.",
            "Collection 'foobar' metadata does not have field 'extent'.",
            "Collection 'foobar' metadata does not have field 'summaries'.",
        }

    def test_normalize_collection_metadata_cube_dimensions_extent_full_100(self, caplog):
        metadata = {
            "id": "foobar",
            "extent": {"spatial": {"bbox": [[-180, -56, 180, 83]]}, "temporal": {"interval": [["2015-07-06", None]]}},
            "cube:dimensions": {
                "x": {"type": "spatial", "axis": "x"},
                "y": {"type": "spatial", "axis": "y"},
                "t": {"type": "temporal"},
            },
        }
        assert _normalize_collection_metadata(metadata, api_version=ComparableVersion("1.0.0"), full=True) == {
            "id": "foobar",
            "stac_version": "0.9.0",
            "stac_extensions": [
                "https://stac-extensions.github.io/datacube/v2.2.0/schema.json",
                "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
            ],
            "description": "foobar",
            "extent": {
                "spatial": {"bbox": [[-180, -56, 180, 83]]},
                "temporal": {"interval": [["2015-07-06T00:00:00Z", None]]},
            },
            "license": "proprietary",
            "cube:dimensions": {
                "x": {"type": "spatial", "axis": "x", "extent": [-180, 180]},
                "y": {"type": "spatial", "axis": "y", "extent": [-56, 83]},
                "t": {"type": "temporal", "extent": ["2015-07-06T00:00:00Z", None]},
            },
            "summaries": {},
            "links": [],
        }

    def test_normalize_collection_metadata_dimensions_and_bands_100(self, caplog):
        metadata = {
            "id": "foobar",
            "properties": {
                "cube:dimensions": {"x": {"type": "spatial"}, "b": {"type": "bands", "values": ["B02", "B03"]}},
                "eo:bands": [{"name": "B02"}, {"name": "B03"}],
            },
        }
        res = _normalize_collection_metadata(metadata, api_version=ComparableVersion("1.0.0"), full=True)
        assert res["cube:dimensions"] == {"x": {"type": "spatial"}, "b": {"type": "bands", "values": ["B02", "B03"]}}
        assert res["summaries"]["eo:bands"] == [{"name": "B02"}, {"name": "B03"}]

    def test_normalize_collection_metadata_bands_stac110(self, caplog):
        """Test normalization of legacy "eo:bands" metadata to common "bands" metadata"""
        metadata = {
            "id": "foobar",
            "summaries": {
                "eo:bands": [
                    {"name": "B02", "common_name": "blue", "center_wavelength": 0.47},
                    {"name": "B03", "common_name": "green", "center_wavelength": 0.56},
                ],
            },
        }
        res = _normalize_collection_metadata(metadata, api_version=ComparableVersion("1.0.0"), full=True)
        assert res == dirty_equals.IsPartialDict(
            id="foobar",
            stac_version="0.9.0",  # TODO #363
            stac_extensions=dirty_equals.Contains(
                "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
            ),
            summaries={
                "eo:bands": [
                    {"name": "B02", "common_name": "blue", "center_wavelength": 0.47},
                    {"name": "B03", "common_name": "green", "center_wavelength": 0.56},
                ],
                "bands": [
                    {"name": "B02", "eo:common_name": "blue", "eo:center_wavelength": 0.47},
                    {"name": "B03", "eo:common_name": "green", "eo:center_wavelength": 0.56},
                ],
            },
        )

    def test_normalize_collection_metadata_datetime(self, caplog):
        metadata = {
            "id": "foobar",
            "extent": {
                "temporal": {
                    "interval": [["2009-08-07", "2009-10-11"], ["2011-12-13 14:15:16", None]],
                }
            },
        }
        res = _normalize_collection_metadata(metadata, api_version=ComparableVersion("1.0.0"), full=True)
        assert res["extent"]["temporal"]["interval"] == [
            ["2009-08-07T00:00:00Z", "2009-10-11T00:00:00Z"],
            ["2011-12-13T14:15:16Z", None],
        ]

    def test_collections(self, api):
        resp = api.get("/collections").assert_status_code(200).json
        assert "links" in resp
        assert "collections" in resp
        assert "S2_FAPAR_CLOUDCOVER" in [c["id"] for c in resp["collections"]]
        assert "S2_FOOBAR" in [c["id"] for c in resp["collections"]]
        for collection in resp["collections"]:
            assert "id" in collection
            assert "stac_version" in collection
            assert "description" in collection
            assert "license" in collection
            assert "extent" in collection
            assert "links" in collection

    @pytest.mark.parametrize("backend_config_overrides", [{"collection_exclusion_list": {"1.0.0": ["S2_FOOBAR"]}}])
    def test_collections_exclusion(self, api, backend_config_overrides):
        resp = api.get("/collections").assert_status_code(200).json
        ids = [c["id"] for c in resp["collections"]]
        if ComparableVersion(api.api_version) == "1.0.0":
            assert "S2_FOOBAR" not in ids
        else:
            assert "S2_FOOBAR" in ids

    def test_collections_caching(self, api):
        resp = api.get("/collections").assert_status_code(200)
        assert resp.headers["Cache-Control"] == "max-age=900, public"

    def test_strip_private_fields(self, api):
        assert "_private" in dummy_backend.DummyCatalog().get_collection_metadata("S2_FOOBAR")
        # All metadata
        collections = api.get("/collections").assert_status_code(200).json["collections"]
        (metadata,) = (c for c in collections if c["id"] == "S2_FOOBAR")
        assert "_private" not in metadata
        # Single collection metadata
        metadata = api.get("/collections/S2_FOOBAR").assert_status_code(200).json
        assert "_private" not in metadata

    def test_collection_full_metadata_invalid_collection(self, api):
        error = api.get("/collections/FOOBOO").assert_error(404, "CollectionNotFound").json
        assert error["message"] == "Collection 'FOOBOO' does not exist."

    def test_collection_full_metadata(self, api, api_version):
        collection = api.get("/collections/S2_FOOBAR").assert_status_code(200).json
        assert collection["id"] == "S2_FOOBAR"
        assert collection["description"] == "S2_FOOBAR"
        assert collection["license"] == "free"
        cube_dimensions = {
            "x": {"extent": [2.5, 6.2], "reference_system": CRS_UTM, "step": 10, "type": "spatial"},
            "y": {"extent": [49.5, 51.5], "reference_system": CRS_UTM, "step": 10, "type": "spatial"},
            "t": {"type": "temporal", "extent": ["2019-01-01", None]},
            "bands": {"type": "bands", "values": ["B02", "B03", "B04", "B08"]},
        }
        eo_bands = [
            {"name": "B02", "common_name": "blue"},
            {"name": "B03", "common_name": "green"},
            {"name": "B04", "common_name": "red"},
            {"name": "B08", "common_name": "nir"},
        ]
        if api.api_version_compare.at_least("1.0.0"):
            assert collection["stac_version"] == "0.9.0"
            assert collection["cube:dimensions"] == cube_dimensions
            assert collection["summaries"]["eo:bands"] == eo_bands
            assert collection["extent"]["spatial"] == {"bbox": [[2.5, 49.5, 6.2, 51.5]]}
            assert collection["extent"]["temporal"] == {"interval": [["2019-01-01T00:00:00Z", None]]}
        else:
            assert collection["stac_version"] == "0.6.2"
            assert collection["properties"]["cube:dimensions"] == cube_dimensions
            assert collection["properties"]["eo:bands"] == eo_bands
            assert collection["extent"] == {
                "spatial": [2.5, 49.5, 6.2, 51.5],
                "temporal": ["2019-01-01T00:00:00Z", None],
            }

        assert collection["links"] == [
            {"rel": "root", "href": f"http://oeo.net/openeo/{api_version}/collections"},
            {"rel": "parent", "href": f"http://oeo.net/openeo/{api_version}/collections"},
            {"rel": "self", "href": f"http://oeo.net/openeo/{api_version}/collections/S2_FOOBAR"},
        ]

    @pytest.mark.parametrize(
        ["orig", "expected"],
        [
            (
                [],
                lambda api_version: [
                    {"rel": "root", "href": f"http://oeo.net/openeo/{api_version}/collections"},
                    {"rel": "parent", "href": f"http://oeo.net/openeo/{api_version}/collections"},
                    {"rel": "self", "href": f"http://oeo.net/openeo/{api_version}/collections/S2_WITH_LINKS"},
                ],
            ),
            (
                [
                    {"rel": "root", "href": "https://s2.test/foobar"},
                    {"rel": "about", "href": "https://s2.test/about"},
                ],
                lambda api_version: [
                    {"rel": "root", "href": "https://s2.test/foobar"},
                    {"rel": "about", "href": "https://s2.test/about"},
                    {"rel": "parent", "href": f"http://oeo.net/openeo/{api_version}/collections"},
                    {"rel": "self", "href": f"http://oeo.net/openeo/{api_version}/collections/S2_WITH_LINKS"},
                ],
            ),
            (
                [
                    {"rel": "root", "href": "https://s2.test/foobar"},
                    {"rel": "parent", "href": "https://s2.test/about"},
                    {"rel": "self", "href": "https://s2.test/s2"},
                ],
                lambda api_version: [
                    {"rel": "root", "href": "https://s2.test/foobar"},
                    {"rel": "parent", "href": "https://s2.test/about"},
                    {"rel": "self", "href": "https://s2.test/s2"},
                ],
            ),
        ],
    )
    def test_collection_full_metadata_links(self, api, api_version, backend_implementation, orig, expected):
        extra_collections = {
            "S2_WITH_LINKS": {
                "id": "S2_WITH_LINKS",
                "links": orig,
            }
        }
        with backend_implementation.catalog.patch_collections(extra_collections):
            collection = api.get("/collections/S2_WITH_LINKS").assert_status_code(200).json
        assert collection["id"] == "S2_WITH_LINKS"
        assert collection["links"] == expected(api_version)

    def test_collection_full_metadata_caching(self, api):
        resp = api.get("/collections/S2_FOOBAR").assert_status_code(200)
        assert resp.headers["Cache-Control"] == "max-age=900, public"

    def test_collection_full_metadata_invalid_caching(self, api):
        resp = api.get("/collections/FOOBOO").assert_error(404, "CollectionNotFound")
        assert "Cache-Control" not in resp.headers

    @pytest.mark.parametrize(
        ["backend_config_overrides", "expected_present", "expected_absent"],
        [
            (
                {},
                {"S2_FOOBAR", "S2_FAPAR_CLOUDCOVER", "SENTINEL1_GRD"},
                set(),
            ),
            (
                {"collection_exclusion_list": {"1.2.0": []}},
                {"S2_FOOBAR", "S2_FAPAR_CLOUDCOVER", "SENTINEL1_GRD"},
                set(),
            ),
            (
                {"collection_exclusion_list": {"1.2.0": ["S2_FOOBAR"]}},
                {"S2_FAPAR_CLOUDCOVER", "SENTINEL1_GRD"},
                {"S2_FOOBAR"},
            ),
        ],
    )
    def test_collection_exclusion_list(self, api120, expected_present, expected_absent):
        resp = api120.get("/collections").assert_status_code(200)
        collection_ids = set(c["id"] for c in resp.json["collections"])
        assert expected_present.issubset(collection_ids)
        assert not expected_absent.intersection(collection_ids)


class TestBatchJobs:
    AUTH_HEADER = TEST_USER_AUTH_HEADER

    @staticmethod
    @contextmanager
    def _fresh_job_registry(next_job_id="job-1234", output_root: Optional[Path] = None, jobs: Optional[dict] = None):
        """Set up a fresh job registry and predefine next job id"""

        with ExitStack() as exit_stack:
            # Conditional setup of some mock contexts
            if next_job_id:
                exit_stack.enter_context(
                    mock.patch.object(dummy_backend.DummyBatchJobs, "generate_job_id", return_value=next_job_id)
                )
            if output_root:
                exit_stack.enter_context(
                    mock.patch.object(dummy_backend.DummyBatchJobs, "_output_root", return_value=output_root)
                )

            dummy_backend.DummyBatchJobs._job_registry = {
                (TEST_USER, "07024ee9-7847-4b8a-b260-6c879a2b3cdc"): BatchJobMetadata(
                    id="07024ee9-7847-4b8a-b260-6c879a2b3cdc",
                    status="running",
                    process={"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
                    created=datetime(2017, 1, 1, 9, 32, 12),
                    started=datetime(2017, 1, 1, 12, 0, 0),
                ),
                (TEST_USER, "53c71345-09b4-46b4-b6b0-03fd6fe1f199"): BatchJobMetadata(
                    id="53c71345-09b4-46b4-b6b0-03fd6fe1f199",
                    title="Your title here.",
                    description="Your description here.",
                    status="finished",
                    progress=100,
                    process={"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
                    created=datetime(2020, 6, 11, 11, 51, 29),
                    updated=datetime(2020, 6, 11, 11, 55, 15),
                    started=datetime(2020, 6, 11, 11, 55, 9),
                    finished=datetime(2020, 6, 11, 11, 55, 15),
                    memory_time_megabyte=timedelta(seconds=18704944),
                    cpu_time=timedelta(seconds=1621),
                    geometry={
                        "type": "Polygon",
                        "coordinates": [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]],
                    },
                    bbox=[-180, -90, 180, 90],
                    start_datetime=datetime(1981, 4, 24, 3, 0, 0),
                    end_datetime=datetime(1981, 4, 24, 3, 0, 0),
                    instruments=["MSI"],
                    epsg=4326,
                    plan="some_plan",
                    costs=1.23,
                    budget=4.56,
                    proj_shape=[300, 600],
                ),
                (TEST_USER, "j-2406047c20fc4966ab637d387502728f"): BatchJobMetadata(
                    id="j-2406047c20fc4966ab637d387502728f",
                    status="finished",
                    created=datetime(2024, 6, 4, 14, 20, 23),
                    bbox=[2.70964374625748, 51.00377983772219, 2.777548414187305, 51.10589339112414],
                    start_datetime=datetime(2020, 1, 1, 0, 0, 0),
                    end_datetime=datetime(2020, 3, 15, 0, 0, 0),
                ),
                (TEST_USER, "j-24083059866540a38cab32b028be0ab5"): BatchJobMetadata(
                    id="j-24083059866540a38cab32b028be0ab5",
                    status="finished",
                    created=datetime(2024, 8, 30, 12, 16, 34),
                    bbox=[34.456156, -0.910085, 34.796396, -0.345477],
                    start_datetime=None,
                    end_datetime=None,
                ),
                (TEST_USER, "j-24111211111111111111111111111111"): BatchJobMetadata(
                    id="j-24111211111111111111111111111111",
                    status="finished",
                    created=datetime(2024, 11, 12, 12, 16, 34),
                    bbox=[34.456156, -0.910085, 34.796396, -0.345477],
                    start_datetime=None,
                    end_datetime=None,
                ),
            }
            dummy_backend.DummyBatchJobs._job_result_registry = {}

            if jobs:
                for job_id, job_settings in jobs.items():
                    key = (job_settings.get("user", TEST_USER), job_id)
                    dummy_backend.DummyBatchJobs._job_registry[key] = BatchJobMetadata(
                        id=job_id,
                        status=job_settings.get("status", "running"),
                        process={"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
                        created=datetime(2017, 1, 1, 9, 32, 12),
                    )
            yield dummy_backend.DummyBatchJobs._job_registry

    @time_machine.travel("2024-01-02T13:14:15Z", tick=False)
    def test_create_job_100(self, api100):
        with self._fresh_job_registry(next_job_id="job-245"):
            resp = api100.post(
                "/jobs",
                headers=self.AUTH_HEADER,
                json={
                    "process": {
                        "process_graph": {"foo": {"process_id": "foo", "arguments": {}}},
                        "summary": "my foo job",
                    },
                    "title": "Foo job",
                    "description": "Run the `foo` process!",
                },
            ).assert_status_code(201)
        assert resp.headers["Location"] == "http://oeo.net/openeo/1.0.0/jobs/job-245"
        assert resp.headers["OpenEO-Identifier"] == "job-245"
        job_info = dummy_backend.DummyBatchJobs._job_registry[TEST_USER, "job-245"]
        assert job_info.id == "job-245"
        assert job_info.process == {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}}
        assert job_info.status == "created"
        assert job_info.created == datetime(2024, 1, 2, 13, 14, 15, tzinfo=timezone.utc)
        assert job_info.job_options is None
        assert job_info.title == "Foo job"
        assert job_info.description == "Run the `foo` process!"

    @pytest.mark.parametrize(
        ["post_data", "expected_job_options"],
        [
            (
                # Under  "job_options" field
                {
                    "process": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
                    "job_options": {
                        "driver-memory": "3g",
                        "executor-memory": "5g",
                    },
                },
                {"driver-memory": "3g", "executor-memory": "5g"},
            ),
            (
                # separate, top-level job options
                {
                    "process": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
                    "driver-memory": "3g",
                    "executor-memory": "5g",
                },
                {"driver-memory": "3g", "executor-memory": "5g"},
            ),
        ],
    )
    def test_create_job_100_with_job_options(self, api100, post_data, expected_job_options):
        with self._fresh_job_registry(next_job_id="job-256"):
            resp = api100.post(
                "/jobs",
                headers=self.AUTH_HEADER,
                json=post_data,
            ).assert_status_code(201)
        job_info = dummy_backend.DummyBatchJobs._job_registry[TEST_USER, "job-256"]
        assert job_info.job_options == expected_job_options

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
    def test_create_job_100_with_job_options_and_defaults_from_remote_process_definition(
        self, api100, requests_mock, default_job_options, given_job_options, expected_job_options
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
            process_definition["default_job_options"] = default_job_options
        requests_mock.get("https://share.test/add3.json", json=process_definition)

        pg = {
            "add12": {"process_id": "add", "arguments": {"x": 1, "y": 2}},
            "add3": {
                "process_id": "add3",
                "namespace": "https://share.test/add3.json",
                "arguments": {"x": {"from_node": "add12"}},
                "result": True,
            },
        }
        post_data = {
            "process": {"process_graph": pg},
        }
        if given_job_options is not None:
            post_data["job_options"] = given_job_options

        with self._fresh_job_registry(next_job_id="job-256"):
            resp = api100.post(
                "/jobs",
                headers=self.AUTH_HEADER,
                json=post_data,
            ).assert_status_code(201)

        job_info = dummy_backend.DummyBatchJobs._job_registry[TEST_USER, "job-256"]
        assert job_info.job_options == expected_job_options

    @time_machine.travel("2024-01-02T13:14:15Z")
    def test_create_job_get_metadata(self, api100):
        with self._fresh_job_registry(next_job_id="job-245"):
            resp = api100.post(
                "/jobs",
                headers=self.AUTH_HEADER,
                json={
                    "process": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
                    "title": "Foo job",
                },
            ).assert_status_code(201)
            assert resp.headers["OpenEO-Identifier"] == "job-245"
            job_info = dummy_backend.DummyBatchJobs._job_registry[TEST_USER, "job-245"]
            assert job_info.id == "job-245"
            assert job_info.status == "created"

            resp = api100.get("/jobs/job-245", headers=self.AUTH_HEADER)
            assert resp.assert_status_code(200).json == {
                "created": "2024-01-02T13:14:15Z",
                "id": "job-245",
                "process": {"process_graph": {"foo": {"arguments": {}, "process_id": "foo"}}},
                "progress": 0,
                "status": "created",
                "title": "Foo job",
            }

    def test_start_job(self, api):
        with self._fresh_job_registry(next_job_id="job-267") as registry:
            api.post(
                "/jobs",
                headers=self.AUTH_HEADER,
                json=api.get_process_graph_dict(
                    {"foo": {"process_id": "foo", "arguments": {}}},
                ),
            ).assert_status_code(201)
            assert registry[TEST_USER, "job-267"].status == "created"
            api.post("/jobs/job-267/results", headers=self.AUTH_HEADER, json={}).assert_status_code(202)
            assert registry[TEST_USER, "job-267"].status == "running"

    @pytest.mark.parametrize(
        ["orig_status", "start"],
        [
            ("created", True),
            ("queued", False),
            ("running", False),
            ("finished", False),
            ("error", False),
            ("canceled", True),
        ],
    )
    def test_start_job_existing_status(self, api, orig_status, start):
        """Only really start jobs that are started (or canceled)"""
        with self._fresh_job_registry() as registry:
            registry[TEST_USER, "job-267"] = BatchJobMetadata(
                id="job-267", status=orig_status, created=datetime(2017, 1, 1, 9, 32, 12)
            )
            # Try to start job
            with mock.patch.object(dummy_backend.DummyBatchJobs, "start_job") as start_job:
                api.post("/jobs/job-267/results", headers=self.AUTH_HEADER, json={}).assert_status_code(202)
                assert start_job.call_count == (1 if start else 0)

    def test_start_job_invalid(self, api):
        resp = api.post("/jobs/deadbeef-f00/results", headers=self.AUTH_HEADER)
        resp.assert_error(404, "JobNotFound")
        assert resp.json["message"] == "The batch job 'deadbeef-f00' does not exist."

    def test_get_job_info_metrics_100(self, api100):
        resp = api100.get("/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199", headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            "id": "53c71345-09b4-46b4-b6b0-03fd6fe1f199",
            "title": "Your title here.",
            "description": "Your description here.",
            "process": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
            "status": "finished",
            "progress": 100,
            "created": "2020-06-11T11:51:29Z",
            "updated": "2020-06-11T11:55:15Z",
            "plan": "some_plan",
            "costs": 1.23,
            "budget": 4.56,
            "usage": {
                "cpu": {"value": 1621, "unit": "cpu-seconds"},
                "duration": {"value": 6, "unit": "seconds"},
                "memory": {"value": 18704944, "unit": "mb-seconds"},
            },
        }

    def test_get_job_info_100(self, api100):
        with self._fresh_job_registry():
            resp = api100.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc", headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            "id": "07024ee9-7847-4b8a-b260-6c879a2b3cdc",
            "status": "running",
            "created": "2017-01-01T09:32:12Z",
            "process": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
        }

    def test_get_job_info_invalid(self, api):
        resp = api.get("/jobs/deadbeef-f00", headers=self.AUTH_HEADER).assert_error(404, "JobNotFound")
        assert resp.json["message"] == "The batch job 'deadbeef-f00' does not exist."

    @pytest.mark.parametrize("backend_config_overrides", [{"simple_job_progress_estimation": 600}])
    def test_get_job_info_simple_job_progress_estimation(self, api100, backend_config_overrides):
        with self._fresh_job_registry(), time_machine.travel(datetime(2017, 1, 1, 12, 5, 0)):
            resp = api100.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc", headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            "id": "07024ee9-7847-4b8a-b260-6c879a2b3cdc",
            "status": "running",
            "progress": pytest.approx(33.33, abs=0.1),
            "created": "2017-01-01T09:32:12Z",
            "process": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
        }

    def test_list_user_jobs_100(self, api100):
        with self._fresh_job_registry():
            resp = api100.get("/jobs", headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            "jobs": [
                {
                    "id": "07024ee9-7847-4b8a-b260-6c879a2b3cdc",
                    "status": "running",
                    "created": "2017-01-01T09:32:12Z",
                },
                {
                    "id": "53c71345-09b4-46b4-b6b0-03fd6fe1f199",
                    "title": "Your title here.",
                    "description": "Your description here.",
                    "status": "finished",
                    "progress": 100,
                    "created": "2020-06-11T11:51:29Z",
                    "updated": "2020-06-11T11:55:15Z",
                    "plan": "some_plan",
                    "costs": 1.23,
                    "budget": 4.56,
                },
                {
                    "id": "j-2406047c20fc4966ab637d387502728f",
                    "status": "finished",
                    "progress": 100,
                    "created": "2024-06-04T14:20:23Z",
                },
                {
                    "id": "j-24083059866540a38cab32b028be0ab5",
                    "status": "finished",
                    "progress": 100,
                    "created": "2024-08-30T12:16:34Z",
                },
                {
                    "id": "j-24111211111111111111111111111111",
                    "status": "finished",
                    "progress": 100,
                    "created": "2024-11-12T12:16:34Z",
                },
            ],
            "links": [],
        }

    def test_list_user_jobs_100_extra(self, api100):
        """
        `get_user_jobs` returns a richer `dict` instead of just `List[BatchJobMetadata]`
        """

        def get_user_jobs(
            self,
            user_id: str,
            limit: Optional[int] = None,
            request_parameters: Optional[dict] = None,
        ):
            return {
                "jobs": [BatchJobMetadata(id="id-123", status="running", created=datetime(2017, 1, 1, 9, 32, 12))],
                "links": [{"rel": "info", "href": "https://info.test"}],
                "federation:missing": ["b4"],
                "something else": "ignore me",
            }

        with mock.patch.object(dummy_backend.DummyBatchJobs, "get_user_jobs", new=get_user_jobs):
            resp = api100.get("/jobs", headers=self.AUTH_HEADER)

        assert resp.assert_status_code(200).json == {
            "jobs": [
                {
                    "id": "id-123",
                    "status": "running",
                    "created": "2017-01-01T09:32:12Z",
                },
            ],
            "links": [{"rel": "info", "href": "https://info.test"}],
            "federation:missing": ["b4"],
        }

    def test_list_user_jobs_paging(self, api100):
        # Setup some jobs in registry
        dummy_backend.DummyBatchJobs._job_registry = {
            (TEST_USER, f"j-{x}{x}"): BatchJobMetadata(
                id=f"j-{x}{x}",
                status="created",
                created=datetime(2024, 12, x),
            )
            for x in [1, 2, 3, 4, 5, 6, 7, 8]
        }

        # Initial job listing with explicit limit (first page)
        resp = api100.get("/jobs?limit=3", headers=self.AUTH_HEADER)
        data = resp.assert_status_code(200).json
        expected_next_href = dirty_equals.IsStr(
            regex=re.escape("http://oeo.net/openeo/1.0.0/jobs?limit=3&search_after=") + "[a-zA-Z0-9_/-]+(%3D)*"
        )
        assert data == {
            "jobs": [
                dirty_equals.IsPartialDict(id="j-88", created="2024-12-08T00:00:00Z"),
                dirty_equals.IsPartialDict(id="j-77", created="2024-12-07T00:00:00Z"),
                dirty_equals.IsPartialDict(id="j-66", created="2024-12-06T00:00:00Z"),
            ],
            "links": [
                {"rel": "next", "href": expected_next_href},
            ],
        }

        # Follow link to second page (which links to third page)
        next_url = data["links"][0]["href"]
        resp = api100.get(path=api100.extract_path(next_url), headers=self.AUTH_HEADER)
        data = resp.assert_status_code(200).json
        assert data == {
            "jobs": [
                dirty_equals.IsPartialDict(id="j-55", created="2024-12-05T00:00:00Z"),
                dirty_equals.IsPartialDict(id="j-44", created="2024-12-04T00:00:00Z"),
                dirty_equals.IsPartialDict(id="j-33", created="2024-12-03T00:00:00Z"),
            ],
            "links": [
                {"rel": "next", "href": expected_next_href},
            ],
        }

        # Follow link to third page (which has no next link)
        next_url = data["links"][0]["href"]
        resp = api100.get(path=api100.extract_path(next_url), headers=self.AUTH_HEADER)
        data = resp.assert_status_code(200).json
        assert data == {
            "jobs": [
                dirty_equals.IsPartialDict(id="j-22", created="2024-12-02T00:00:00Z"),
                dirty_equals.IsPartialDict(id="j-11", created="2024-12-01T00:00:00Z"),
            ],
            "links": [],
        }

    def test_get_job_results_unfinished(self, api):
        with self._fresh_job_registry(next_job_id="job-345"):
            resp = api.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results", headers=self.AUTH_HEADER)
        resp.assert_error(400, "JobNotFinished")

    def test_get_job_results_unfinished_with_partial_explicitly_false(self, api):
        with self._fresh_job_registry(next_job_id="job-345"):
            resp = api.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results?partial=false", headers=self.AUTH_HEADER)
        resp.assert_error(400, "JobNotFinished")

    @pytest.mark.parametrize(
        ["job_status", "expected_openeo_status"],
        [
            ("running", "running"),
            ("created", "running"),
            ("queued", "running"),
            ("error", "error"),
            ("canceled", "canceled"),
        ],
    )
    def test_get_job_results_unfinished_with_partial_true(self, api, job_status, expected_openeo_status):
        job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
        with self._fresh_job_registry(next_job_id="job-345"):
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status=job_status
            )
            resp: ApiResponse = api.get(f"/jobs/{job_id}/results?partial=true", headers=self.AUTH_HEADER)
        resp.assert_status_code(200)

        job_result = resp.json
        expected_canonical_url = f"http://oeo.net/openeo/{api.api_version}/jobs/{job_id}/results"
        assert job_result == DictSubSet(
            {
                "openeo:status": expected_openeo_status,
                "type": "Collection",
                "stac_version": "1.0.0",
                "id": job_id,
                "title": "Unfinished batch job {job_id}",
                "description": f"Results for batch job {job_id}",
                "license": "proprietary",
                "links": [
                    {
                        "rel": "canonical",
                        "href": expected_canonical_url,
                        "type": "application/json",
                    }
                ],
            }
        )

    def test_get_job_results_unfinished_with_partial_true_and_unsupported_jobstatus_raises_assertionerror(self, api):
        """Verify that we raise an error when the backend gives us a job status value that we don't support yet."""
        job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
        with self._fresh_job_registry(next_job_id="job-345"):
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="unsupported_status"
            )
            resp: ApiResponse = api.get(f"/jobs/{job_id}/results?partial=true", headers=self.AUTH_HEADER)

        resp.assert_status_code(500)
        assert resp.json["message"] == """Server error: AssertionError("unexpected job status: 'unsupported_status'")"""

    def test_get_job_results_100(self, api100):
        with self._fresh_job_registry(next_job_id="job-362"):
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="finished"
            )
            resp = api100.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results", headers=self.AUTH_HEADER)
            assert resp.assert_status_code(200).json == {
                "assets": {
                    "output.tiff": {
                        "roles": ["data"],
                        "title": "output.tiff",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/output.tiff",
                        "type": "image/tiff; application=geotiff",
                        "eo:bands": [{"name": "B02", "common_name": "blue", "center_wavelength": 0.665}],
                        "bands": [{"name": "B02", "eo:common_name": "blue", "eo:center_wavelength": 0.665}],
                    },
                    "output.nc": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/output.nc",
                        "roles": ["data"],
                        "title": "output.nc",
                        "type": "application/x-netcdf",
                    },
                    "randomforest.model": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/randomforest.model",
                        "roles": ["data"],
                        "title": "randomforest.model",
                        "type": "application/octet-stream",
                    },
                    "vectorcube.geojson": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/vectorcube.geojson",
                        "roles": ["data"],
                        "title": "vectorcube.geojson",
                        "type": "application/geo+json",
                    },
                },
                "geometry": None,
                "id": "07024ee9-7847-4b8a-b260-6c879a2b3cdc",
                "links": [
                    {
                        "rel": "self",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "canonical",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "card4l-document",
                        "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                        "type": "application/pdf",
                    },
                ],
                "properties": {
                    "created": "2017-01-01T09:32:12Z",
                    "datetime": None,
                    "card4l:processing_chain": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
                    "card4l:specification": "SR",
                    "card4l:specification_version": "5.0",
                    "processing:facility": "Dummy openEO API",
                    "processing:software": "openeo-python-driver",
                },
                "providers": EXPECTED_PROVIDERS,
                "stac_extensions": [
                    "https://stac-extensions.github.io/processing/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/card4l/v0.1.0/optical/schema.json",
                    "https://stac-extensions.github.io/file/v2.1.0/schema.json",
                    "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
                ],
                "stac_version": "0.9.0",
                "type": "Feature",
                "openeo:status": "finished",
            }

            resp = api100.get("/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results", headers=self.AUTH_HEADER)

            assert resp.assert_status_code(200).json == {
                "assets": {
                    "output.tiff": {
                        "roles": ["data"],
                        "title": "output.tiff",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/output.tiff",
                        "type": "image/tiff; application=geotiff",
                        "proj:epsg": 4326,
                        "proj:shape": [300, 600],
                        "eo:bands": [{"name": "B02", "common_name": "blue", "center_wavelength": 0.665}],
                        "bands": [{"name": "B02", "eo:common_name": "blue", "eo:center_wavelength": 0.665}],
                    },
                    "output.nc": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/output.nc",
                        "roles": ["data"],
                        "title": "output.nc",
                        "type": "application/x-netcdf",
                        "proj:epsg": 4326,
                        "proj:shape": [300, 600],
                    },
                    "randomforest.model": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/randomforest.model",
                        "roles": ["data"],
                        "title": "randomforest.model",
                        "type": "application/octet-stream",
                    },
                    "vectorcube.geojson": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/vectorcube.geojson",
                        "roles": ["data"],
                        "title": "vectorcube.geojson",
                        "type": "application/geo+json",
                        "proj:epsg": 4326,
                        "proj:shape": [300, 600],
                    },
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]],
                },
                "bbox": [-180, -90, 180, 90],
                "id": "53c71345-09b4-46b4-b6b0-03fd6fe1f199",
                "links": [
                    {
                        "rel": "self",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "canonical",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "card4l-document",
                        "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                        "type": "application/pdf",
                    },
                ],
                "properties": {
                    "created": "2020-06-11T11:51:29Z",
                    "updated": "2020-06-11T11:55:15Z",
                    "datetime": "1981-04-24T03:00:00Z",
                    "title": "Your title here.",
                    "description": "Your description here.",
                    "instruments": ["MSI"],
                    "proj:epsg": 4326,
                    "proj:shape": [300, 600],
                    "card4l:processing_chain": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
                    "card4l:specification": "SR",
                    "card4l:specification_version": "5.0",
                    "processing:facility": "Dummy openEO API",
                    "processing:software": "openeo-python-driver",
                },
                "providers": EXPECTED_PROVIDERS,
                "stac_extensions": [
                    "https://stac-extensions.github.io/processing/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/card4l/v0.1.0/optical/schema.json",
                    "https://stac-extensions.github.io/file/v2.1.0/schema.json",
                    "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
                ],
                "stac_version": "0.9.0",
                "type": "Feature",
                "openeo:status": "finished",
            }

    def test_get_job_results_110(self, api110):
        with self._fresh_job_registry(next_job_id="job-362"):
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="finished"
            )
            resp = api110.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results", headers=self.AUTH_HEADER)

            assert resp.assert_status_code(200).json == {
                "description": "Results for batch job 07024ee9-7847-4b8a-b260-6c879a2b3cdc",
                "extent": {"spatial": {"bbox": [[-180, -90, 180, 90]]}, "temporal": {"interval": [[None, None]]}},
                "license": "proprietary",
                "summaries": {
                    "ml-model:architecture": ["random-forest"],
                    "ml-model:learning_approach": ["supervised"],
                    "ml-model:prediction_type": ["classification"],
                },
                "assets": {
                    "output.tiff": {
                        "roles": ["data"],
                        "title": "output.tiff",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/output.tiff",
                        "type": "image/tiff; application=geotiff",
                        "eo:bands": [{"name": "B02", "common_name": "blue", "center_wavelength": 0.665}],
                        "bands": [{"name": "B02", "eo:common_name": "blue", "eo:center_wavelength": 0.665}],
                    },
                    "output.nc": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/output.nc",
                        "roles": ["data"],
                        "title": "output.nc",
                        "type": "application/x-netcdf",
                    },
                    "randomforest.model": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/randomforest.model",
                        "roles": ["data"],
                        "title": "randomforest.model",
                        "type": "application/octet-stream",
                    },
                    "vectorcube.geojson": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/vectorcube.geojson",
                        "roles": ["data"],
                        "title": "vectorcube.geojson",
                        "type": "application/geo+json",
                    },
                },
                "id": "07024ee9-7847-4b8a-b260-6c879a2b3cdc",
                "links": [
                    {
                        "rel": "self",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "canonical",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "card4l-document",
                        "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                        "type": "application/pdf",
                    },
                    {
                        "rel": "item",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/items/output.tiff",
                        "type": "application/geo+json",
                    },
                    {
                        "rel": "item",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/items/output.nc",
                        "type": "application/geo+json",
                    },
                    {
                        "rel": "item",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/items/ml_model_metadata.json",
                        "type": "application/json",
                    },
                ],
                "providers": EXPECTED_PROVIDERS,
                "stac_extensions": [
                    "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/file/v2.1.0/schema.json",
                    "https://stac-extensions.github.io/processing/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/ml-model/v1.0.0/schema.json",
                ],
                "stac_version": "1.0.0",
                "type": "Collection",
                "openeo:status": "finished",
            }

            resp = api110.get("/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results", headers=self.AUTH_HEADER)

            assert resp.assert_status_code(200).json == {
                "description": "Your description here.",
                "extent": {
                    "spatial": {"bbox": [[-180, -90, 180, 90]]},
                    "temporal": {"interval": [["1981-04-24T03:00:00Z", "1981-04-24T03:00:00Z"]]},
                },
                "license": "proprietary",
                "assets": {
                    "output.tiff": {
                        "roles": ["data"],
                        "title": "output.tiff",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/output.tiff",
                        "type": "image/tiff; application=geotiff",
                        "proj:epsg": 4326,
                        "proj:shape": [300, 600],
                        "eo:bands": [{"name": "B02", "common_name": "blue", "center_wavelength": 0.665}],
                        "bands": [{"name": "B02", "eo:common_name": "blue", "eo:center_wavelength": 0.665}],
                    },
                    "output.nc": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/output.nc",
                        "roles": ["data"],
                        "title": "output.nc",
                        "type": "application/x-netcdf",
                        "proj:epsg": 4326,
                        "proj:shape": [300, 600],
                    },
                    "randomforest.model": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/randomforest.model",
                        "roles": ["data"],
                        "title": "randomforest.model",
                        "type": "application/octet-stream",
                    },
                    "vectorcube.geojson": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/vectorcube.geojson",
                        "proj:epsg": 4326,
                        "proj:shape": [300, 600],
                        "roles": ["data"],
                        "title": "vectorcube.geojson",
                        "type": "application/geo+json",
                    },
                },
                "id": "53c71345-09b4-46b4-b6b0-03fd6fe1f199",
                "links": [
                    {
                        "rel": "self",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "canonical",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "card4l-document",
                        "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                        "type": "application/pdf",
                    },
                    {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/items/output.tiff",
                        "rel": "item",
                        "type": "application/geo+json",
                    },
                    {
                        "rel": "item",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/items/output.nc",
                        "type": "application/geo+json",
                    },
                    {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/items/ml_model_metadata.json",
                        "rel": "item",
                        "type": "application/json",
                    },
                ],
                "providers": EXPECTED_PROVIDERS,
                "stac_extensions": [
                    "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/file/v2.1.0/schema.json",
                    "https://stac-extensions.github.io/processing/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/ml-model/v1.0.0/schema.json",
                ],
                "summaries": {
                    "instruments": ["MSI"],
                    "ml-model:architecture": ["random-forest"],
                    "ml-model:learning_approach": ["supervised"],
                    "ml-model:prediction_type": ["classification"],
                },
                "title": "Your title here.",
                "stac_version": "1.0.0",
                "type": "Collection",
                "openeo:status": "finished",
            }

            resp = api110.get("/jobs/j-2406047c20fc4966ab637d387502728f/results", headers=self.AUTH_HEADER)

            assert resp.assert_status_code(200).json == DictSubSet(
                {
                    "stac_version": "1.0.0",
                    "type": "Collection",
                    "id": "j-2406047c20fc4966ab637d387502728f",
                    "description": "Results for batch job j-2406047c20fc4966ab637d387502728f",
                    "license": "proprietary",
                    "extent": {
                        "spatial": {
                            "bbox": [[2.70964374625748, 51.00377983772219, 2.777548414187305, 51.10589339112414]]
                        },
                        "temporal": {"interval": [["2020-01-01T00:00:00Z", "2020-03-15T00:00:00Z"]]},
                    },
                    "assets": {
                        "timeseries.csv": {
                            "href": "http://oeo.net/openeo/1.1.0/jobs/j-2406047c20fc4966ab637d387502728f/results/assets/timeseries.csv",
                            "roles": ["data"],
                            "title": "timeseries.csv",
                            "type": "text/csv",
                            "eo:bands": [
                                {"name": "S2-L2A-EVI_t0"},
                                {"name": "S2-L2A-EVI_t1"},
                                {"name": "S2-L2A-EVI_t2"},
                            ],
                            "bands": [{"name": "S2-L2A-EVI_t0"}, {"name": "S2-L2A-EVI_t1"}, {"name": "S2-L2A-EVI_t2"}],
                        },
                        "timeseries.parquet": {
                            "href": "http://oeo.net/openeo/1.1.0/jobs/j-2406047c20fc4966ab637d387502728f/results/assets/timeseries.parquet",
                            "roles": ["data"],
                            "title": "timeseries.parquet",
                            "type": "application/parquet; profile=geo",
                            "eo:bands": [
                                {"name": "S2-L2A-EVI_t0"},
                                {"name": "S2-L2A-EVI_t1"},
                                {"name": "S2-L2A-EVI_t2"},
                            ],
                            "bands": [{"name": "S2-L2A-EVI_t0"}, {"name": "S2-L2A-EVI_t1"}, {"name": "S2-L2A-EVI_t2"}],
                        },
                    },
                    "links": [
                        {
                            "href": "http://oeo.net/openeo/1.1.0/jobs/j-2406047c20fc4966ab637d387502728f/results",
                            "rel": "self",
                            "type": "application/json",
                        },
                        {
                            "href": "http://oeo.net/openeo/1.1.0/jobs/j-2406047c20fc4966ab637d387502728f/results",
                            "rel": "canonical",
                            "type": "application/json",
                        },
                        {
                            "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                            "rel": "card4l-document",
                            "type": "application/pdf",
                        },
                        {
                            "href": "http://oeo.net/openeo/1.1.0/jobs/j-2406047c20fc4966ab637d387502728f/results/items/timeseries.csv",
                            "rel": "item",
                            "type": "application/geo+json",
                        },
                        {
                            "href": "http://oeo.net/openeo/1.1.0/jobs/j-2406047c20fc4966ab637d387502728f/results/items/timeseries.parquet",
                            "rel": "item",
                            "type": "application/geo+json",
                        },
                    ],
                }
            )

    def test_get_job_results_public_href_asset_100(self, api, backend_implementation):
        import numpy as np

        results_data = {
            "output.tiff": {
                BatchJobs.ASSET_PUBLIC_HREF: "http://storage.test/r362/res.tiff?sgn=23432ldf348fl4r349",
                "type": "application/tiff",
                "nodata": np.nan,
            }
        }
        with self._fresh_job_registry(
            jobs={"07024ee9-7847-4b8a-b260-6c879a2b3cdc": {"status": "finished"}}
        ), mock.patch.object(backend_implementation.batch_jobs, "get_result_assets", return_value=results_data):
            resp = api.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results", headers=self.AUTH_HEADER)
        res = resp.assert_status_code(200).json
        assert res["assets"] == {
            "output.tiff": {
                "href": "http://storage.test/r362/res.tiff?sgn=23432ldf348fl4r349",
                "roles": ["data"],
                "title": "output.tiff",
                "type": "application/tiff",
            }
        }

    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#")}])
    def test_get_job_results_signed_100(self, api100, flask_app, backend_config_overrides):
        with self._fresh_job_registry():
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="finished"
            )
            resp = api100.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results", headers=self.AUTH_HEADER)
            assert resp.assert_status_code(200).json == {
                "assets": {
                    "output.tiff": {
                        "roles": ["data"],
                        "title": "output.tiff",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/50afb0cad129e61d415278c4ffcd8a83/output.tiff",
                        "type": "image/tiff; application=geotiff",
                        "eo:bands": [{"name": "B02", "common_name": "blue", "center_wavelength": 0.665}],
                        "bands": [{"name": "B02", "eo:common_name": "blue", "eo:center_wavelength": 0.665}],
                    },
                    "output.nc": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/e28f17365e698783574dd313de0d64cd/output.nc",
                        "roles": ["data"],
                        "title": "output.nc",
                        "type": "application/x-netcdf",
                    },
                    "randomforest.model": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/741cfd7379a9eda4bc1c8b0c5155bfe9/randomforest.model",
                        "roles": ["data"],
                        "title": "randomforest.model",
                        "type": "application/octet-stream",
                    },
                    "vectorcube.geojson": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/03aa954288d152431cae6949810c330f/vectorcube.geojson",
                        "roles": ["data"],
                        "title": "vectorcube.geojson",
                        "type": "application/geo+json",
                    },
                },
                "geometry": None,
                "id": "07024ee9-7847-4b8a-b260-6c879a2b3cdc",
                "links": [
                    {
                        "rel": "self",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "canonical",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/TXIuVGVzdA==/05cb8b78f20c68a5aa9eb05249928d24",
                        "type": "application/json",
                    },
                    {
                        "rel": "card4l-document",
                        "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                        "type": "application/pdf",
                    },
                ],
                "properties": {
                    "created": "2017-01-01T09:32:12Z",
                    "datetime": None,
                    "card4l:processing_chain": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
                    "card4l:specification": "SR",
                    "card4l:specification_version": "5.0",
                    "processing:facility": "Dummy openEO API",
                    "processing:software": "openeo-python-driver",
                },
                "providers": EXPECTED_PROVIDERS,
                "stac_extensions": [
                    "https://stac-extensions.github.io/processing/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/card4l/v0.1.0/optical/schema.json",
                    "https://stac-extensions.github.io/file/v2.1.0/schema.json",
                    "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
                ],
                "stac_version": "0.9.0",
                "type": "Feature",
                "openeo:status": "finished",
            }

    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#")}])
    def test_get_job_results_signed_100_unfinished_and_partial_false(self, api100, flask_app, backend_config_overrides):
        with self._fresh_job_registry():
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="running"
            )
            resp = api100.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results?partial=false", headers=self.AUTH_HEADER
            )
            resp.assert_error(400, "JobNotFinished")

    @pytest.mark.parametrize(
        ["backend_config_overrides", "job_status", "expected_openeo_status"],
        [
            [{"url_signer": UrlSigner(secret="123&@#")}, "running", "running"],
            [{"url_signer": UrlSigner(secret="123&@#")}, "error", "error"],
            [{"url_signer": UrlSigner(secret="123&@#")}, "canceled", "canceled"],
            [{"url_signer": UrlSigner(secret="123&@#")}, "queued", "running"],
            [{"url_signer": UrlSigner(secret="123&@#")}, "created", "running"],
        ],
    )
    def test_get_job_results_signed_100_unfinished_and_partial_true(
        self, api100, flask_app, backend_config_overrides, job_status, expected_openeo_status
    ):
        job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
        with self._fresh_job_registry():
            dummy_backend.DummyBatchJobs._update_status(job_id=job_id, user_id=TEST_USER, status=job_status)
            resp = api100.get(f"/jobs/{job_id}/results?partial=true", headers=self.AUTH_HEADER)

            resp.assert_status_code(200)
            expected_canonical_url = f"http://oeo.net/openeo/1.0.0/jobs/{job_id}/results/TXIuVGVzdA==/05cb8b78f20c68a5aa9eb05249928d24?partial=true"
            assert resp.json == DictSubSet(
                {
                    "openeo:status": expected_openeo_status,
                    "type": "Collection",
                    "stac_version": "1.0.0",
                    "id": job_id,
                    "title": "Unfinished batch job {job_id}",
                    "description": f"Results for batch job {job_id}",
                    "license": "proprietary",
                    "links": [
                        {
                            "rel": "canonical",
                            "href": expected_canonical_url,
                            "type": "application/json",
                        }
                    ],
                }
            )

    @pytest.mark.parametrize(
        "backend_config_overrides",
        [
            {"asset_url": DummyDirectS3Assets(), "url_signer": UrlSigner(secret="123&@#")},
            {"url_signer": UrlSigner(secret="123&@#")},
        ],
    )
    def test_get_job_results_signed_110(self, api110, flask_app, backend_config_overrides):
        with self._fresh_job_registry():
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="finished"
            )
            resp = api110.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results", headers=self.AUTH_HEADER)

            resp_data = resp.assert_status_code(200).json
            assert resp_data == {
                "description": "Results for batch job 07024ee9-7847-4b8a-b260-6c879a2b3cdc",
                "extent": {"spatial": {"bbox": [[-180, -90, 180, 90]]}, "temporal": {"interval": [[None, None]]}},
                "license": "proprietary",
                "summaries": {
                    "ml-model:architecture": ["random-forest"],
                    "ml-model:learning_approach": ["supervised"],
                    "ml-model:prediction_type": ["classification"],
                },
                "assets": {
                    "output.tiff": {
                        "roles": ["data"],
                        "title": "output.tiff",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/50afb0cad129e61d415278c4ffcd8a83/output.tiff",
                        "type": "image/tiff; application=geotiff",
                        "eo:bands": [{"name": "B02", "common_name": "blue", "center_wavelength": 0.665}],
                        "bands": [{"name": "B02", "eo:common_name": "blue", "eo:center_wavelength": 0.665}],
                    },
                    "output.nc": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/e28f17365e698783574dd313de0d64cd/output.nc",
                        "roles": ["data"],
                        "title": "output.nc",
                        "type": "application/x-netcdf",
                    },
                    "randomforest.model": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/741cfd7379a9eda4bc1c8b0c5155bfe9/randomforest.model",
                        "roles": ["data"],
                        "title": "randomforest.model",
                        "type": "application/octet-stream",
                    },
                    "vectorcube.geojson": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/03aa954288d152431cae6949810c330f/vectorcube.geojson",
                        "roles": ["data"],
                        "title": "vectorcube.geojson",
                        "type": "application/geo+json",
                    },
                },
                "id": "07024ee9-7847-4b8a-b260-6c879a2b3cdc",
                "links": [
                    {
                        "rel": "self",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "canonical",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/TXIuVGVzdA==/05cb8b78f20c68a5aa9eb05249928d24",
                        "type": "application/json",
                    },
                    {
                        "rel": "card4l-document",
                        "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                        "type": "application/pdf",
                    },
                    {
                        "rel": "item",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/items/TXIuVGVzdA==/50afb0cad129e61d415278c4ffcd8a83/output.tiff",
                        "type": "application/geo+json",
                    },
                    {
                        "rel": "item",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/items/TXIuVGVzdA==/e28f17365e698783574dd313de0d64cd/output.nc",
                        "type": "application/geo+json",
                    },
                    {
                        "rel": "item",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/items/TXIuVGVzdA==/272d7aa46727ee3f11a7211d5be953e4/ml_model_metadata.json",
                        "type": "application/json",
                    },
                ],
                "providers": EXPECTED_PROVIDERS,
                "stac_extensions": [
                    "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/file/v2.1.0/schema.json",
                    "https://stac-extensions.github.io/processing/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/ml-model/v1.0.0/schema.json",
                ],
                "stac_version": "1.0.0",
                "type": "Collection",
                "openeo:status": "finished",
            }

        pystac.validation.stac_validator.JsonSchemaSTACValidator().validate(
            stac_dict=resp_data,
            stac_object_type=pystac.STACObjectType.COLLECTION,
            stac_version=resp_data.get("stac_version", "1.0.0"),
            extensions=resp_data.get("stac_extensions", []),
        )

    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#")}])
    def test_get_job_results_signed_110_unfinished_and_partial_false(self, api110, flask_app, backend_config_overrides):
        with self._fresh_job_registry():
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="running"
            )
            resp = api110.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results?partial=false", headers=self.AUTH_HEADER
            )
            resp.assert_error(400, "JobNotFinished")

    @pytest.mark.parametrize(
        ["backend_config_overrides", "job_status", "expected_openeo_status"],
        [
            [{"url_signer": UrlSigner(secret="123&@#")}, "running", "running"],
            [{"url_signer": UrlSigner(secret="123&@#")}, "error", "error"],
            [{"url_signer": UrlSigner(secret="123&@#")}, "canceled", "canceled"],
            [{"url_signer": UrlSigner(secret="123&@#")}, "queued", "running"],
            [{"url_signer": UrlSigner(secret="123&@#")}, "created", "running"],
        ],
    )
    def test_get_job_results_signed_110_unfinished_and_partial_true(
        self, api110, flask_app, backend_config_overrides, job_status, expected_openeo_status
    ):
        job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
        with self._fresh_job_registry():
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status=job_status
            )
            resp: ApiResponse = api110.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results?partial=true", headers=self.AUTH_HEADER
            )

            resp.assert_status_code(200)
            expected_canonical_url = f"http://oeo.net/openeo/1.1.0/jobs/{job_id}/results/TXIuVGVzdA==/05cb8b78f20c68a5aa9eb05249928d24?partial=true"
            assert resp.json == DictSubSet(
                {
                    "openeo:status": expected_openeo_status,
                    "type": "Collection",
                    "stac_version": "1.0.0",
                    "id": job_id,
                    "title": "Unfinished batch job {job_id}",
                    "description": f"Results for batch job {job_id}",
                    "license": "proprietary",
                    "links": [
                        {
                            "rel": "canonical",
                            "href": expected_canonical_url,
                            "type": "application/json",
                        }
                    ],
                }
            )

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}])
    def test_get_job_results_signed_with_expiration_100(self, api100, flask_app, backend_config_overrides):
        with self._fresh_job_registry():
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="finished"
            )
            resp = api100.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results", headers=self.AUTH_HEADER)
            assert resp.assert_status_code(200).json == {
                "assets": {
                    "output.tiff": {
                        "roles": ["data"],
                        "title": "output.tiff",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234",
                        "type": "image/tiff; application=geotiff",
                        "eo:bands": [{"name": "B02", "common_name": "blue", "center_wavelength": 0.665}],
                        "bands": [{"name": "B02", "eo:common_name": "blue", "eo:center_wavelength": 0.665}],
                    },
                    "output.nc": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/3ed7b944c4f9cc88d2c3ef7534a27596/output.nc?expires=2234",
                        "roles": ["data"],
                        "title": "output.nc",
                        "type": "application/x-netcdf",
                    },
                    "randomforest.model": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/22b76413158c59acaccc74e74841a473/randomforest.model?expires=2234",
                        "roles": ["data"],
                        "title": "randomforest.model",
                        "type": "application/octet-stream",
                    },
                    "vectorcube.geojson": {
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/d6daa44b4320cd9cd3fc3c0fd37832f5/vectorcube.geojson?expires=2234",
                        "roles": ["data"],
                        "title": "vectorcube.geojson",
                        "type": "application/geo+json",
                    },
                },
                "geometry": None,
                "id": "07024ee9-7847-4b8a-b260-6c879a2b3cdc",
                "links": [
                    {
                        "rel": "self",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "canonical",
                        "href": "http://oeo.net/openeo/1.0.0/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/TXIuVGVzdA==/9fea29cd94195399cc4d902388a3c32c?expires=2234",
                        "type": "application/json",
                    },
                    {
                        "rel": "card4l-document",
                        "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                        "type": "application/pdf",
                    },
                ],
                "properties": {
                    "created": "2017-01-01T09:32:12Z",
                    "datetime": None,
                    "card4l:processing_chain": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
                    "card4l:specification": "SR",
                    "card4l:specification_version": "5.0",
                    "processing:facility": "Dummy openEO API",
                    "processing:software": "openeo-python-driver",
                },
                "providers": EXPECTED_PROVIDERS,
                "stac_extensions": [
                    "https://stac-extensions.github.io/processing/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/card4l/v0.1.0/optical/schema.json",
                    "https://stac-extensions.github.io/file/v2.1.0/schema.json",
                    "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
                ],
                "stac_version": "0.9.0",
                "type": "Feature",
                "openeo:status": "finished",
            }

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}])
    def test_get_job_results_signed_with_expiration_100_unfinished_and_partial_false(
        self, api100, flask_app, backend_config_overrides
    ):
        with self._fresh_job_registry():
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="running"
            )
            resp = api100.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results?partial=false", headers=self.AUTH_HEADER
            )
            resp.assert_error(400, "JobNotFinished")

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize(
        ["backend_config_overrides", "job_status", "expected_openeo_status"],
        [
            [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}, "running", "running"],
            [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}, "error", "error"],
            [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}, "canceled", "canceled"],
            [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}, "queued", "running"],
            [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}, "created", "running"],
        ],
    )
    def test_get_job_results_signed_with_expiration_100_unfinished_and_partial_true(
        self, api100, flask_app, backend_config_overrides, job_status, expected_openeo_status
    ):
        job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
        with self._fresh_job_registry():
            dummy_backend.DummyBatchJobs._update_status(job_id=job_id, user_id=TEST_USER, status=job_status)
            resp = api100.get(f"/jobs/{job_id}/results?partial=true", headers=self.AUTH_HEADER)
            resp.assert_status_code(200)

            expected_canonical_url = f"http://oeo.net/openeo/1.0.0/jobs/{job_id}/results/TXIuVGVzdA==/9fea29cd94195399cc4d902388a3c32c?expires=2234&partial=true"
            assert resp.json == DictSubSet(
                {
                    "openeo:status": expected_openeo_status,
                    "type": "Collection",
                    "stac_version": "1.0.0",
                    "id": job_id,
                    "title": "Unfinished batch job {job_id}",
                    "description": f"Results for batch job {job_id}",
                    "license": "proprietary",
                    "links": [
                        {
                            "rel": "canonical",
                            "href": expected_canonical_url,
                            "type": "application/json",
                        }
                    ],
                }
            )

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}])
    def test_get_job_results_signed_with_expiration_110(self, api110, flask_app, backend_config_overrides):
        with self._fresh_job_registry(next_job_id="job-373"):
            dummy_backend.DummyBatchJobs._update_status(
                job_id="07024ee9-7847-4b8a-b260-6c879a2b3cdc", user_id=TEST_USER, status="finished"
            )
            resp = api110.get("/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results", headers=self.AUTH_HEADER)
            assert resp.assert_status_code(200).json == {
                "type": "Collection",
                "stac_version": "1.0.0",
                "stac_extensions": [
                    "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/file/v2.1.0/schema.json",
                    "https://stac-extensions.github.io/processing/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/ml-model/v1.0.0/schema.json",
                ],
                "id": "53c71345-09b4-46b4-b6b0-03fd6fe1f199",
                "title": "Your title here.",
                "description": "Your description here.",
                "license": "proprietary",
                "extent": {
                    "spatial": {"bbox": [[-180, -90, 180, 90]]},
                    "temporal": {"interval": [["1981-04-24T03:00:00Z", "1981-04-24T03:00:00Z"]]},
                },
                "providers": EXPECTED_PROVIDERS,
                "links": [
                    {
                        "rel": "self",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results",
                        "type": "application/json",
                    },
                    {
                        "rel": "canonical",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/TXIuVGVzdA==/748b8c91160fbb6e137c91d7d33b0c4a?expires=2234",
                        "type": "application/json",
                    },
                    {
                        "rel": "card4l-document",
                        "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                        "type": "application/pdf",
                    },
                    {
                        "rel": "item",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/items/TXIuVGVzdA==/f5d336336d36e3e987ba6a34b87cde01/output.tiff?expires=2234",
                        "type": "application/geo+json",
                    },
                    {
                        "rel": "item",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/items/TXIuVGVzdA==/a5d63417ff2eac2c632af5abfce6c425/output.nc?expires=2234",
                        "type": "application/geo+json",
                    },
                    {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/items/TXIuVGVzdA==/30fac5af7fe96123c923e94c2732f9aa/ml_model_metadata.json?expires=2234",
                        "rel": "item",
                        "type": "application/json",
                    },
                ],
                "summaries": {
                    "instruments": ["MSI"],
                    "ml-model:architecture": ["random-forest"],
                    "ml-model:learning_approach": ["supervised"],
                    "ml-model:prediction_type": ["classification"],
                },
                "assets": {
                    "output.tiff": {
                        "title": "output.tiff",
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/TXIuVGVzdA==/f5d336336d36e3e987ba6a34b87cde01/output.tiff?expires=2234",
                        "type": "image/tiff; application=geotiff",
                        "proj:epsg": 4326,
                        "proj:shape": [300, 600],
                        "eo:bands": [{"name": "B02", "common_name": "blue", "center_wavelength": 0.665}],
                        "bands": [{"name": "B02", "eo:common_name": "blue", "eo:center_wavelength": 0.665}],
                        "roles": ["data"],
                    },
                    "output.nc": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/TXIuVGVzdA==/a5d63417ff2eac2c632af5abfce6c425/output.nc?expires=2234",
                        "roles": ["data"],
                        "title": "output.nc",
                        "type": "application/x-netcdf",
                        "proj:epsg": 4326,
                        "proj:shape": [300, 600],
                    },
                    "randomforest.model": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/TXIuVGVzdA==/18fd2346c52945c0caba7b13246f5a63/randomforest.model?expires=2234",
                        "roles": ["data"],
                        "title": "randomforest.model",
                        "type": "application/octet-stream",
                    },
                    "vectorcube.geojson": {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/TXIuVGVzdA==/1672093c942dbbb9c8d5f1041dcc6c06/vectorcube.geojson?expires=2234",
                        "roles": ["data"],
                        "title": "vectorcube.geojson",
                        "type": "application/geo+json",
                        "proj:epsg": 4326,
                        "proj:shape": [300, 600],
                    },
                },
                "openeo:status": "finished",
            }

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}])
    def test_get_job_results_signed_with_expiration_110_unfinished_and_partial_false(
        self, api110, flask_app, backend_config_overrides
    ):
        job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
        with self._fresh_job_registry(next_job_id="job-373"):
            dummy_backend.DummyBatchJobs._update_status(job_id=job_id, user_id=TEST_USER, status="running")
            resp = api110.get(f"/jobs/{job_id}/results?partial=false", headers=self.AUTH_HEADER)
            resp.assert_error(400, "JobNotFinished")

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize(
        ["backend_config_overrides", "job_status", "expected_openeo_status"],
        [
            [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}, "running", "running"],
            [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}, "error", "error"],
            [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}, "canceled", "canceled"],
            [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}, "queued", "running"],
            [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}, "created", "running"],
        ],
    )
    def test_get_job_results_signed_with_expiration_110_unfinished_and_partial_true(
        self, api110, flask_app, backend_config_overrides, job_status, expected_openeo_status
    ):
        job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
        with self._fresh_job_registry(next_job_id="job-373"):
            dummy_backend.DummyBatchJobs._update_status(job_id=job_id, user_id=TEST_USER, status=job_status)
            resp = api110.get(f"/jobs/{job_id}/results?partial=true", headers=self.AUTH_HEADER)
            resp.assert_status_code(200)

            expected_canonical_url = f"http://oeo.net/openeo/1.1.0/jobs/{job_id}/results/TXIuVGVzdA==/9fea29cd94195399cc4d902388a3c32c?expires=2234&partial=true"
            assert resp.json == DictSubSet(
                {
                    "openeo:status": expected_openeo_status,
                    "type": "Collection",
                    "stac_version": "1.0.0",
                    "id": job_id,
                    "title": "Unfinished batch job {job_id}",
                    "description": f"Results for batch job {job_id}",
                    "license": "proprietary",
                    "links": [
                        {
                            "rel": "canonical",
                            "href": expected_canonical_url,
                            "type": "application/json",
                        }
                    ],
                }
            )

    def test_get_job_results_custom_links(self, api100):
        with self._fresh_job_registry(next_job_id="job-362"):
            job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
            dummy_backend.DummyBatchJobs._update_status(job_id=job_id, user_id=TEST_USER, status="finished")
            dummy_backend.DummyBatchJobs.set_result_metadata(
                job_id=job_id,
                user_id=TEST_USER,
                metadata=BatchJobResultMetadata(
                    assets={},
                    links=[
                        {"rel": "canonical", "href": "https://other.test/j123/results"},
                        {"rel": "self", "href": "https://other.test/j123"},
                        {"rel": "card4l-document", "href": "https://c4ld.test"},
                        {"rel": "cu5t0m!", "href": "https://other.test/j123.readme"},
                    ],
                ),
            )
            resp = api100.get(f"/jobs/{job_id}/results", headers=self.AUTH_HEADER)
            assert resp.assert_status_code(200).json == DictSubSet(
                {
                    "id": "07024ee9-7847-4b8a-b260-6c879a2b3cdc",
                    "links": [
                        {"rel": "canonical", "href": "https://other.test/j123/results"},
                        {"rel": "self", "href": "https://other.test/j123"},
                        {"rel": "card4l-document", "href": "https://c4ld.test"},
                        {"rel": "cu5t0m!", "href": "https://other.test/j123.readme"},
                    ],
                }
            )

    def test_get_job_results_invalid_job(self, api):
        api.get("/jobs/deadbeef-f00/results", headers=self.AUTH_HEADER).assert_error(404, "JobNotFound")

    def test_download_result_invalid_job(self, api):
        api.get("/jobs/deadbeef-f00/results/assets/some_file", headers=self.AUTH_HEADER).assert_error(
            404, "JobNotFound"
        )

    def test_download_result(self, api, tmp_path):
        output_root = Path(tmp_path)
        jobs = {"07024ee9-7847-4b8a-b260-6c879a2b3cdc": {"status": "finished"}}
        with self._fresh_job_registry(output_root=output_root, jobs=jobs):
            output = output_root / "07024ee9-7847-4b8a-b260-6c879a2b3cdc" / "output.tiff"
            output.parent.mkdir(parents=True)
            with output.open("wb") as f:
                f.write(b"tiffdata")
            resp = api.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/output.tiff", headers=self.AUTH_HEADER
            )
        assert resp.assert_status_code(200).data == b"tiffdata"
        assert resp.headers["Content-Type"] == "image/tiff; application=geotiff"

    def test_download_result_nested_path(self, api110, tmp_path):
        output_root = Path(tmp_path)
        jobs = {"j-24111211111111111111111111111111": {"status": "finished"}}
        with self._fresh_job_registry(output_root=output_root, jobs=jobs):
            output = output_root / "j-24111211111111111111111111111111/subfolder/output.tiff"
            output.parent.mkdir(parents=True)
            with output.open("wb") as f:
                f.write(b"tiffdata")
            resp = api110.get(
                "/jobs/j-24111211111111111111111111111111/results/assets/subfolder/output.tiff",
                headers=self.AUTH_HEADER,
            )
        assert resp.assert_status_code(200).data == b"tiffdata"
        assert resp.headers["Content-Type"] == "image/tiff; application=geotiff"

    def test_download_result_vectorcube(self, api, tmp_path):
        output_root = Path(tmp_path)
        job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
        jobs = {job_id: {"status": "finished"}}
        with self._fresh_job_registry(output_root=output_root, jobs=jobs):
            output = output_root / job_id / "vectorcube.geojson"
            output.parent.mkdir(parents=True)
            path = str(get_path("geojson/FeatureCollection02.json"))
            vectorcube = DriverVectorCube(geometries=gpd.read_file(path))
            VectorCubeResult(cube=vectorcube, format="GeoJSON", options=None).write_assets(directory=output)
            url = f"/jobs/{job_id}/results/assets/vectorcube.geojson"
            resp = api.get(url, headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200)
        assert "features" in resp.json.keys()
        assert resp.headers["Content-Type"] == "application/geo+json"

    def test_download_result_with_s3_object_storage(self, api, mock_s3_resource, swift_url):
        job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
        s3_bucket_name = "openeo-test-bucket"
        output_root = f"s3://{s3_bucket_name}/some-data-dir"
        s3_key = f"some-data-dir/{job_id}/output.tiff"

        # Simulate that we have a large file so we would need to stream the download in chunks.
        # A size that is at least one byte larger than the STREAM_CHUNK_SIZE_DEFAULT should trigger streaming.
        # We should still receive the entire file contents in streaming mode.
        large_tiff_data = b"tiffdata" * STREAM_CHUNK_SIZE_DEFAULT

        jobs = {job_id: {"status": "finished"}}
        with self._fresh_job_registry(output_root=output_root, jobs=jobs):
            s3_bucket = create_s3_bucket(mock_s3_resource, s3_bucket_name)
            s3_bucket.put_object(Key=s3_key, Body=large_tiff_data)
            resp = api.get(f"/jobs/{job_id}/results/assets/output.tiff", headers=self.AUTH_HEADER)

        assert resp.assert_status_code(200).data == large_tiff_data
        assert resp.headers["Content-Type"] == "image/tiff; application=geotiff"

    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#")}])
    def test_download_result_signed(self, api, tmp_path, flask_app, backend_config_overrides):
        output_root = Path(tmp_path)
        jobs = {"07024ee9-7847-4b8a-b260-6c879a2b3cdc": {"status": "finished"}}
        with self._fresh_job_registry(output_root=output_root, jobs=jobs):
            output = output_root / "07024ee9-7847-4b8a-b260-6c879a2b3cdc" / "output.tiff"
            output.parent.mkdir(parents=True)
            with output.open("wb") as f:
                f.write(b"tiffdata")
            resp = api.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/50afb0cad129e61d415278c4ffcd8a83/output.tiff"
            )
        assert resp.assert_status_code(200).data == b"tiffdata"
        assert resp.headers["Content-Type"] == "image/tiff; application=geotiff"

    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#")}])
    def test_download_result_signed_invalid(self, api, flask_app, backend_config_overrides):
        jobs = {"07024ee9-7847-4b8a-b260-6c879a2b3cdc": {"status": "finished"}}
        with self._fresh_job_registry(jobs=jobs):
            resp = api.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/test123/output.tiff")
        assert resp.assert_error(403, "CredentialsInvalid")

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}])
    def test_download_result_signed_with_expiration(self, api, tmp_path, flask_app, backend_config_overrides):
        output_root = Path(tmp_path)
        jobs = {"07024ee9-7847-4b8a-b260-6c879a2b3cdc": {"status": "finished"}}
        with self._fresh_job_registry(output_root=output_root, jobs=jobs):
            output = output_root / "07024ee9-7847-4b8a-b260-6c879a2b3cdc" / "output.tiff"
            output.parent.mkdir(parents=True)
            with output.open("wb") as f:
                f.write(b"tiffdata")
            resp = api.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234"
            )
        assert resp.assert_status_code(200).data == b"tiffdata"
        assert resp.headers["Content-Type"] == "image/tiff; application=geotiff"

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}])
    def test_download_result_signed_with_expiration_supports_range_request(
        self, api, tmp_path, flask_app, backend_config_overrides, swift_url
    ):
        output_root = Path(tmp_path)
        output = output_root / "07024ee9-7847-4b8a-b260-6c879a2b3cdc" / "output.tiff"
        output.parent.mkdir(parents=True)
        with output.open("wb") as f:
            f.write(b"tiffdata")

        jobs = {"07024ee9-7847-4b8a-b260-6c879a2b3cdc": {"status": "finished"}}
        with self._fresh_job_registry(output_root=output_root, jobs=jobs):
            head_resp = api.head(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234"
            )
            assert head_resp.assert_status_code(200).data == b""
            assert head_resp.headers["Content-Type"] == "image/tiff; application=geotiff"
            assert head_resp.headers["Accept-Ranges"] == "bytes"
            assert head_resp.headers["Content-Length"] == "8"

            full_get_resp = api.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234"
            )
            assert full_get_resp.assert_status_code(200).data == b"tiffdata"
            assert full_get_resp.headers["Content-Length"] == "8"

            ranged_get_resp = api.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234",
                headers={"Range": "bytes=0-3"},
            )
            assert ranged_get_resp.assert_status_code(206).data == b"tiff"
            assert ranged_get_resp.headers["Content-Length"] == "4"

            out_of_range_get_resp = api.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234",
                headers={"Range": "bytes=8-10"},
            )
            out_of_range_get_resp.assert_status_code(416)

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}])
    def test_download_result_with_s3_object_storage_with_expiration_supports_range_request(
        self, api, mock_s3_resource, backend_config_overrides, swift_url
    ):
        job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
        s3_bucket_name = "openeo-test-bucket"
        output_root = f"s3://{s3_bucket_name}/some-data-dir"
        s3_key = f"some-data-dir/{job_id}/output.tiff"

        s3_bucket = create_s3_bucket(mock_s3_resource, s3_bucket_name)
        s3_bucket.put_object(Key=s3_key, Body=b"tiffdata")

        jobs = {job_id: {"status": "finished"}}
        with self._fresh_job_registry(output_root=output_root, jobs=jobs):
            head_resp = api.head(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234"
            )
            assert head_resp.assert_status_code(200).data == b""
            assert head_resp.headers["Content-Type"] == "image/tiff; application=geotiff"
            assert head_resp.headers["Accept-Ranges"] == "bytes"
            assert head_resp.headers["Content-Length"] == "8"

            full_get_resp = api.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234"
            )
            assert full_get_resp.assert_status_code(200).data == b"tiffdata"
            assert full_get_resp.headers["Content-Length"] == "8"

            ranged_get_resp = api.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234",
                headers={"Range": "bytes=0-3"},
            )
            assert ranged_get_resp.assert_status_code(206).data == b"tiff"
            assert ranged_get_resp.headers["Content-Length"] == "4"

            out_of_range_get_resp = api.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234",
                headers={"Range": "bytes=8-10"},
            )
            out_of_range_get_resp.assert_status_code(416)

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}])
    def test_download_result_with_s3_object_storage_with_expiration_NoSuchKey_error(
        self, api, mock_s3_resource, backend_config_overrides, caplog, swift_url
    ):
        job_id = "07024ee9-7847-4b8a-b260-6c879a2b3cdc"
        s3_bucket_name = "openeo-test-bucket"
        output_root = f"s3://{s3_bucket_name}/some-data-dir"
        s3_key = f"some-data-dir/{job_id}/output.tiff"

        create_s3_bucket(mock_s3_resource, s3_bucket_name)

        jobs = {job_id: {"status": "finished"}}
        with self._fresh_job_registry(output_root=output_root, jobs=jobs):
            full_get_resp = api.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA%3D%3D/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234"
            )

        assert full_get_resp.assert_status_code(404).json["message"] == "Not found: output.tiff"
        assert (
            "openeo_driver.views",
            logging.ERROR,
            f"Not found: s3://{s3_bucket_name}/{s3_key}",
        ) in caplog.record_tuples

    @mock.patch("time.time", mock.MagicMock(return_value=3456))
    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}])
    def test_download_result_signed_with_expiration_invalid(self, api, tmp_path, flask_app, backend_config_overrides):
        jobs = {"07024ee9-7847-4b8a-b260-6c879a2b3cdc": {"status": "finished"}}
        with self._fresh_job_registry(jobs=jobs):
            resp = api.get(
                "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results/assets/TXIuVGVzdA==/fd0ca65e29c6d223da05b2e73a875683/output.tiff?expires=2234"
            )
        assert resp.assert_error(410, "ResultLinkExpired")

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}])
    def test_get_job_result_item(self, flask_app, api110, backend_config_overrides):
        with self._fresh_job_registry():
            resp = api110.get(
                "/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/items/output.tiff", headers=self.AUTH_HEADER
            )

        resp_data = resp.assert_status_code(200).json

        assert resp_data == {
            "type": "Feature",
            "stac_version": "1.0.0",
            "stac_extensions": [
                "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
                "https://stac-extensions.github.io/file/v2.1.0/schema.json",
                "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
            ],
            "id": "output.tiff",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0.0, 50.0], [0.0, 55.0], [5.0, 55.0], [5.0, 50.0], [0.0, 50.0]]],
            },
            "bbox": [0.0, 50.0, 5.0, 55.0],
            "epsg": 4326,
            "properties": {"datetime": "1981-04-24T03:00:00Z", "proj:shape": [300, 600], "proj:epsg": 4326},
            "links": [
                {
                    "rel": "self",
                    "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/items/output.tiff",
                    "type": "application/geo+json",
                },
                {
                    "rel": "collection",
                    "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results",
                    "type": "application/json",
                },
            ],
            "assets": {
                "output.tiff": {
                    "title": "output.tiff",
                    "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/TXIuVGVzdA==/f5d336336d36e3e987ba6a34b87cde01/output.tiff?expires=2234",
                    "type": "image/tiff; application=geotiff",
                    "proj:epsg": 4326,
                    "proj:shape": [300, 600],
                    "eo:bands": [{"name": "B02", "common_name": "blue", "center_wavelength": 0.665}],
                    "bands": [{"name": "B02", "eo:common_name": "blue", "eo:center_wavelength": 0.665}],
                    "roles": ["data"],
                }
            },
            "collection": "53c71345-09b4-46b4-b6b0-03fd6fe1f199",
        }
        assert resp.headers["Content-Type"] == "application/geo+json"

        pystac.validation.stac_validator.JsonSchemaSTACValidator().validate(
            stac_dict=resp_data,
            stac_object_type=pystac.STACObjectType.ITEM,
            stac_version=resp_data.get("stac_version", "0.9.0"),
            extensions=resp_data.get("stac_extensions", []),
        )

    @pytest.mark.parametrize(
        ["vector_item_id", "vector_asset_media_type", "backend_config_overrides", "expected_asset_href"],
        [
            (
                "timeseries.csv",
                "text/csv",
                {"asset_url": DummyDirectS3Assets()},
                f"{DummyDirectS3Assets.S3_ENDPOINT}/OpenEO-data/batch_jobs/j-2406047c20fc4966ab637d387502728f/timeseries.csv",
            ),
            (
                "timeseries.csv",
                "text/csv",
                {},
                "http://oeo.net/openeo/1.1.0/jobs/j-2406047c20fc4966ab637d387502728f/results/assets/timeseries.csv",
            ),
            (
                "timeseries.parquet",
                "application/parquet; profile=geo",
                {"asset_url": DummyDirectS3Assets()},
                f"{DummyDirectS3Assets.S3_ENDPOINT}/OpenEO-data/batch_jobs/j-2406047c20fc4966ab637d387502728f/timeseries.parquet",
            ),
            (
                "timeseries.parquet",
                "application/parquet; profile=geo",
                {},
                "http://oeo.net/openeo/1.1.0/jobs/j-2406047c20fc4966ab637d387502728f/results/assets/timeseries.parquet",
            ),
        ],
    )
    def test_get_vector_cube_job_result_item(
        self, flask_app, api110, vector_item_id, vector_asset_media_type, backend_config_overrides, expected_asset_href
    ):
        vector_asset_filename = vector_item_id

        with self._fresh_job_registry():
            resp = api110.get(
                f"/jobs/j-2406047c20fc4966ab637d387502728f/results/items/{vector_item_id}", headers=self.AUTH_HEADER
            )

        resp_data = resp.assert_status_code(200).json

        assert resp_data == DictSubSet(
            {
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": vector_item_id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [2.70964374625748, 51.00377983772219],
                            [2.70964374625748, 51.10589339112414],
                            [2.777548414187305, 51.10589339112414],
                            [2.777548414187305, 51.00377983772219],
                            [2.70964374625748, 51.00377983772219],
                        ]
                    ],
                },
                "bbox": [2.70964374625748, 51.00377983772219, 2.777548414187305, 51.10589339112414],
                "properties": {
                    "datetime": None,
                    "start_datetime": "2020-01-01T00:00:00Z",
                    "end_datetime": "2020-03-15T00:00:00Z",
                },
                "collection": "j-2406047c20fc4966ab637d387502728f",
                "links": [
                    {
                        "href": f"http://oeo.net/openeo/1.1.0/jobs/j-2406047c20fc4966ab637d387502728f/results/items/{vector_item_id}",
                        "rel": "self",
                        "type": "application/geo+json",
                    },
                    {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/j-2406047c20fc4966ab637d387502728f/results",
                        "rel": "collection",
                        "type": "application/json",
                    },
                ],
                "assets": {
                    vector_asset_filename: {
                        "href": expected_asset_href,
                        "type": vector_asset_media_type,
                        "roles": ["data"],
                        "title": vector_asset_filename,
                        "eo:bands": [{"name": "S2-L2A-EVI_t0"}, {"name": "S2-L2A-EVI_t1"}, {"name": "S2-L2A-EVI_t2"}],
                        "bands": [{"name": "S2-L2A-EVI_t0"}, {"name": "S2-L2A-EVI_t1"}, {"name": "S2-L2A-EVI_t2"}],
                    }
                },
            }
        )

        assert resp.headers["Content-Type"] == "application/geo+json"

        pystac.validation.stac_validator.JsonSchemaSTACValidator().validate(
            stac_dict=resp_data,
            stac_object_type=pystac.STACObjectType.ITEM,
            stac_version=resp_data.get("stac_version", "0.9.0"),
            extensions=resp_data.get("stac_extensions", []),
        )

    def test_get_job_result_item_with_temporal_extent_on_asset(self, flask_app, api110):
        with self._fresh_job_registry():
            resp = api110.get(
                f"/jobs/j-24083059866540a38cab32b028be0ab5/results/items/timeseries.parquet", headers=self.AUTH_HEADER
            )

        resp_data = resp.assert_status_code(200).json

        assert resp_data == DictSubSet(
            {
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": "timeseries.parquet",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [34.456156, -0.910085],
                            [34.456156, -0.345477],
                            [34.796396, -0.345477],
                            [34.796396, -0.910085],
                            [34.456156, -0.910085],
                        ]
                    ],
                },
                "bbox": [34.456156, -0.910085, 34.796396, -0.345477],
                "properties": {
                    "datetime": None,
                    "start_datetime": "2016-10-30T00:00:00+00:00",
                    "end_datetime": "2018-05-03T00:00:00+00:00",
                },
                "collection": "j-24083059866540a38cab32b028be0ab5",
                "links": [
                    {
                        "href": f"http://oeo.net/openeo/1.1.0/jobs/j-24083059866540a38cab32b028be0ab5/results/items/timeseries.parquet",
                        "rel": "self",
                        "type": "application/geo+json",
                    },
                    {
                        "href": "http://oeo.net/openeo/1.1.0/jobs/j-24083059866540a38cab32b028be0ab5/results",
                        "rel": "collection",
                        "type": "application/json",
                    },
                ],
                "assets": {
                    "timeseries.parquet": {
                        "href": f"http://oeo.net/openeo/1.1.0/jobs/j-24083059866540a38cab32b028be0ab5/results/assets/timeseries.parquet",
                        "type": "application/parquet; profile=geo",
                        "roles": ["data"],
                        "title": "timeseries.parquet",
                        "eo:bands": [{"name": "S1-SIGMA0-VV"}, {"name": "S1-SIGMA0-VH"}, {"name": "S2-L2A-B01"}],
                        "bands": [{"name": "S1-SIGMA0-VV"}, {"name": "S1-SIGMA0-VH"}, {"name": "S2-L2A-B01"}],
                    }
                },
            }
        )

        assert resp.headers["Content-Type"] == "application/geo+json"

        pystac.validation.stac_validator.JsonSchemaSTACValidator().validate(
            stac_dict=resp_data,
            stac_object_type=pystac.STACObjectType.ITEM,
            stac_version=resp_data.get("stac_version", "0.9.0"),
            extensions=resp_data.get("stac_extensions", []),
        )

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    @pytest.mark.parametrize("backend_config_overrides", [{"url_signer": UrlSigner(secret="123&@#", expiration=1000)}])
    def test_download_ml_model_metadata_signed(self, flask_app, api110, backend_config_overrides):
        with self._fresh_job_registry():
            resp = api110.get(
                "/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/items/ml_model_metadata.json",
                headers=self.AUTH_HEADER,
            )
        assert "expires" in resp.assert_status_code(200).json["assets"]["model"]["href"]  # signed result

    @mock.patch("time.time", mock.MagicMock(return_value=1234))
    def test_download_ml_model_metadata(self, flask_app, api110, backend_config_overrides):
        with self._fresh_job_registry():
            resp = api110.get(
                "/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/items/ml_model_metadata.json",
                headers=self.AUTH_HEADER,
            )
        random_id = resp.assert_status_code(200).json["id"]
        assert "expires" not in resp.assert_status_code(200).json["assets"]["model"]["href"]  # not signed result
        assert resp.assert_status_code(200).json == {
            "id": random_id,
            "type": "Feature",
            "stac_version": "1.0.0",
            "stac_extensions": ["https://stac-extensions.github.io/ml-model/v1.0.0/schema.json"],
            "assets": {
                "model": {
                    "href": "http://oeo.net/openeo/1.1.0/jobs/53c71345-09b4-46b4-b6b0-03fd6fe1f199/results/assets/randomforest.model",
                    "roles": ["ml-model:checkpoint"],
                    "title": "org.apache.spark.mllib.tree.model.RandomForestModel",
                    "type": "application/octet-stream",
                }
            },
            "bbox": [-179.999, -89.999, 179.999, 89.999],
            "collection": "53c71345-09b4-46b4-b6b0-03fd6fe1f199",
            "geometry": {
                "coordinates": [
                    [
                        [-179.999, -89.999],
                        [179.999, -89.999],
                        [179.999, 89.999],
                        [-179.999, 89.999],
                        [-179.999, -89.999],
                    ]
                ],
                "type": "Polygon",
            },
            "links": [],
            "properties": {
                "datetime": None,
                "end_datetime": "9999-12-31T23:59:59Z",
                "ml-model:architecture": "random-forest",
                "ml-model:learning_approach": "supervised",
                "ml-model:prediction_type": "classification",
                "ml-model:training-os": "linux",
                "ml-model:training-processor-type": "cpu",
                "ml-model:type": "ml-model",
                "start_datetime": "1970-01-01T00:00:00Z",
            },
        }

    def test_get_batch_job_logs(self, api):
        with self._fresh_job_registry():
            resp = api.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/logs", headers=self.AUTH_HEADER)
        assert resp.assert_status_code(200).json == {
            "level": "debug",
            "logs": [
                {"id": "1", "level": "info", "message": "hello world"},
            ],
            "links": [],
        }

    def test_get_batch_job_logs_level(self, api):
        log_db = {
            "07024ee9-7847-4b8a-b260-6c879a2b3cdc": [
                {"id": "1", "level": "info", "message": "howdy world"},
                {"id": "2", "level": "error", "message": "oh no"},
            ]
        }
        with self._fresh_job_registry():
            with mock.patch.dict(dummy_backend.DummyBatchJobs._custom_job_logs, log_db):
                resp = api.get(
                    "/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/logs",
                    headers=self.AUTH_HEADER,
                    params={"level": "warning"},
                )
                assert resp.assert_status_code(200).json == {
                    "level": "warning",
                    "logs": [{"id": "2", "level": "error", "message": "oh no"}],
                    "links": [],
                }

    def test_get_batch_job_logs_failure(self, api):
        with self._fresh_job_registry():
            with mock.patch.dict(
                dummy_backend.DummyBatchJobs._custom_job_logs,
                {"07024ee9-7847-4b8a-b260-6c879a2b3cdc": [RuntimeError("nope")]},
            ):
                resp = api.get("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/logs", headers=self.AUTH_HEADER)
                assert resp.assert_status_code(200).json == {
                    "level": "debug",
                    "logs": [
                        {
                            "id": "",
                            "code": "Internal",
                            "level": "error",
                            "message": re_assert.Matches(
                                r"Log collection for job 07024ee9-7847-4b8a-b260-6c879a2b3cdc failed. \(req_id: r-[0-9a-f]{32}\) RuntimeError\('nope'\)"
                            ),
                        }
                    ],
                    "links": [],
                }

    def test_cancel_job(self, api):
        with self._fresh_job_registry():
            resp = api.delete("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results", headers=self.AUTH_HEADER)
        assert resp.status_code == 204

    def test_cancel_job_invalid(self, api):
        with self._fresh_job_registry():
            resp = api.delete("/jobs/deadbeef-f00/results", headers=self.AUTH_HEADER)
        resp.assert_error(404, "JobNotFound")

    def test_delete_job(self, api):
        with self._fresh_job_registry():
            resp = api.delete("/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc", headers=self.AUTH_HEADER)
        assert resp.status_code == 204


class TestSecondaryServices:
    AUTH_HEADER = TEST_USER_AUTH_HEADER

    def test_service_types_v100(self, api):
        resp = api.get("/service_types").assert_status_code(200)
        service_types = resp.json
        assert list(service_types.keys()) == ["WMTS"]
        wmts = service_types["WMTS"]
        assert wmts["configuration"]["version"]["default"] == "1.0.0"
        assert wmts["process_parameters"] == []
        assert wmts["links"] == []

    def test_create_unsupported_service_type_returns_400_BadRequest(self, api):
        resp = api.post(
            "/services",
            json={
                "process": {"process_graph": {"product_id": "S2"}},
                "type": "???",
            },
            headers=self.AUTH_HEADER,
        )
        resp.assert_status_code(400)

    def test_unsupported_services_methods_return_405_MethodNotAllowed(self, api):
        res = api.put(
            "/services",
            json={
                "process": {"process_graph": {"product_id": "S2"}},
                "type": "WMTS",
            },
        )
        res.assert_status_code(405)

    def test_list_services_100(self, api100):
        metadata = api100.get("/services", headers=self.AUTH_HEADER).json
        assert metadata == {
            "services": [
                {
                    "id": "wmts-foo",
                    "type": "WMTS",
                    "enabled": True,
                    "url": "https://oeo.net/wmts/foo",
                    "title": "Test service",
                    "created": "2020-04-09T15:05:08Z",
                    "configuration": {"version": "0.5.8"},
                }
            ],
            "links": [],
        }

    def test_get_service_metadata_100(self, api100):
        metadata = api100.get("/services/wmts-foo", headers=self.AUTH_HEADER).json
        assert metadata == {
            "id": "wmts-foo",
            "process": {"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
            "url": "https://oeo.net/wmts/foo",
            "type": "WMTS",
            "enabled": True,
            "configuration": {"version": "0.5.8"},
            "attributes": {},
            "title": "Test service",
            "created": "2020-04-09T15:05:08Z",
        }

    def test_get_service_metadata_wrong_id(self, api):
        api.get("/services/wmts-invalid", headers=self.AUTH_HEADER).assert_error(404, "ServiceNotFound")

    def test_services_requires_authentication(self, api):
        api.get("/services").assert_error(401, "AuthenticationRequired")

    def test_get_service_requires_authentication(self, api):
        api.get("/services/wmts-foo").assert_error(401, "AuthenticationRequired")

    def test_patch_service_requires_authentication(self, api):
        api.patch("/services/wmts-foo").assert_error(401, "AuthenticationRequired")

    def test_delete_service_requires_authentication(self, api):
        api.delete("/services/wmts-foo").assert_error(401, "AuthenticationRequired")

    def test_service_logs_100(self, api):
        logs = api.get("/services/wmts-foo/logs", headers=self.AUTH_HEADER).json
        assert logs == {
            "level": "debug",
            "logs": [
                {"id": 3, "level": "info", "message": "Loaded data."},
            ],
            "links": [],
        }


def test_credentials_basic_no_headers(api):
    api.get("/credentials/basic").assert_error(401, "AuthenticationRequired")


def test_credentials_basic_wrong_password(api):
    headers = {"Authorization": build_basic_http_auth_header(username="john", password="password123")}
    api.get("/credentials/basic", headers=headers).assert_error(403, "CredentialsInvalid")


def test_credentials_basic(api):
    headers = {"Authorization": build_basic_http_auth_header(username="Alice", password="alice123")}
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
    assert paths == ("/foo", "/old", "/secret")
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
    assert paths == ("/foo", "/foo")
    assert methods == ({"GET"}, {"POST"})


@pytest.fixture
def udp_store(backend_implementation) -> dummy_backend.DummyUserDefinedProcesses:
    udps = backend_implementation.user_defined_processes
    assert isinstance(udps, dummy_backend.DummyUserDefinedProcesses)
    udps.reset(
        {
            ("Mr.Test", "udp1"): UserDefinedProcessMetadata(
                id="udp1",
                process_graph={"add23": {"process_id": "add", "arguments": {"x": 2, "y": 3}, "result": True}},
            ),
            ("Mr.Test", "udp2"): UserDefinedProcessMetadata(
                id="udp2",
                process_graph={"add58": {"process_id": "add", "arguments": {"x": 5, "y": 8}, "result": True}},
                public=True,
            ),
        }
    )
    return udps


class TestUserDefinedProcesses:
    @pytest.mark.parametrize("body_id", [None, "evi", "meh"])
    def test_add_udp(self, api100, udp_store, body_id):
        spec = {"parameters": [{"name": "red"}], "process_graph": {"sub": {}}, "public": True}
        if body_id:
            spec["id"] = body_id
        api100.put("/process_graphs/evi", headers=TEST_USER_AUTH_HEADER, json=spec).assert_status_code(200)

        new_udp = udp_store._processes["Mr.Test", "evi"]
        assert new_udp.id == "evi"
        assert new_udp.parameters == [{"name": "red"}]
        assert new_udp.process_graph == {"sub": {}}
        assert new_udp.public

    def test_add_udp_no_pg(self, api100):
        spec = {"id": "evi", "parameters": [{"name": "red"}]}
        res = api100.put("/process_graphs/evi", headers=TEST_USER_AUTH_HEADER, json=spec)
        res.assert_error(400, "ProcessGraphMissing")

    def test_add_udp_invalid_id(self, api100):
        spec = {"id": "foob@r", "process_graph": {"sub": {}}}
        res = api100.put("/process_graphs/foob@r", headers=TEST_USER_AUTH_HEADER, json=spec)
        res.assert_error(400, "InvalidId")

    @pytest.mark.parametrize("body_id", [None, "udp1", "meh"])
    def test_update_udp(self, api100, udp_store, body_id):
        spec = {"parameters": [{"name": "blue"}], "process_graph": {"add": {}}, "public": True}
        if body_id:
            spec["id"] = body_id
        api100.put("/process_graphs/udp1", headers=TEST_USER_AUTH_HEADER, json=spec).assert_status_code(200)

        modified_udp = udp_store._processes["Mr.Test", "udp1"]
        assert modified_udp.id == "udp1"
        assert modified_udp.process_graph == {"add": {}}
        assert modified_udp.parameters == [{"name": "blue"}]
        assert modified_udp.public

    def test_list_udps(self, api100, udp_store):
        resp = api100.get("/process_graphs", headers=TEST_USER_AUTH_HEADER).assert_status_code(200)

        udps = resp.json["processes"]
        assert udps == [{"id": "udp1"}, {"id": "udp2"}]

    def test_get_udp(self, api100, udp_store):
        udp = api100.get("/process_graphs/udp1", headers=TEST_USER_AUTH_HEADER).assert_status_code(200).json
        assert udp == {
            "id": "udp1",
            "process_graph": {"add23": {"process_id": "add", "arguments": {"x": 2, "y": 3}, "result": True}},
        }

    def test_get_udp_public(self, api100, udp_store):
        udp = api100.get("/process_graphs/udp2", headers=TEST_USER_AUTH_HEADER).assert_status_code(200).json
        assert udp == {
            "id": "udp2",
            "process_graph": {"add58": {"process_id": "add", "arguments": {"x": 5, "y": 8}, "result": True}},
            "links": [
                {
                    "rel": "canonical",
                    "href": "http://oeo.net/openeo/1.0.0/processes/u:Mr.Test/udp2",
                    "title": "Public URL for user-defined process 'udp2'",
                }
            ],
            "public": True,
        }

    def test_get_unknown_udp(self, api100, udp_store):
        res = api100.get("/process_graphs/unknown", headers=TEST_USER_AUTH_HEADER)
        res.assert_error(404, "ProcessGraphNotFound")

    def test_get_invalid_id(self, api100, udp_store):
        res = api100.get("/process_graphs/foob@r", headers=TEST_USER_AUTH_HEADER)
        res.assert_error(400, "InvalidId")

    def test_delete_udp(self, api100, udp_store):
        assert ("Mr.Test", "udp2") in udp_store._processes

        api100.delete("/process_graphs/udp2", headers=TEST_USER_AUTH_HEADER).assert_status_code(204)

        assert ("Mr.Test", "udp1") in udp_store._processes
        assert ("Mr.Test", "udp2") not in udp_store._processes

    def test_delete_unknown_udp(self, api100, udp_store):
        res = api100.delete("/process_graphs/unknown", headers=TEST_USER_AUTH_HEADER)
        res.assert_error(404, "ProcessGraphNotFound")

    def test_delete_invalid_id(self, api100, udp_store):
        res = api100.delete("/process_graphs/foob@r", headers=TEST_USER_AUTH_HEADER)
        res.assert_error(400, "InvalidId")

    def test_public_udp(self, api100, udp_store):
        api100.put(
            "/process_graphs/evi",
            headers=TEST_USER_AUTH_HEADER,
            json={
                "parameters": [{"name": "red"}],
                "process_graph": {"sub": {}},
                "links": [{"rel": "about", "href": "https://wikipedia.test/evi"}],
                "public": True,
            },
        ).assert_status_code(200)
        api100.put(
            "/process_graphs/secret",
            headers=TEST_USER_AUTH_HEADER,
            json={"parameters": [{"name": "red"}], "process_graph": {"sub": {}}, "public": False},
        ).assert_status_code(200)

        r = api100.get("/processes/u:Mr.Test").assert_status_code(200)
        assert r.json == {
            "processes": [{"id": "udp2"}, {"id": "evi", "parameters": [{"name": "red"}]}],
            "links": [],
            "version": None,
        }

    def test_public_udp_link(self, api100, udp_store):
        api100.put(
            "/process_graphs/evi",
            headers=TEST_USER_AUTH_HEADER,
            json={
                "parameters": [{"name": "red"}],
                "process_graph": {"add35": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}},
                "links": [{"rel": "about", "href": "https://wikipedia.test/evi"}],
                "public": True,
            },
        ).assert_status_code(200)

        expected = {
            "id": "evi",
            "parameters": [{"name": "red"}],
            "process_graph": {"add35": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}},
            "public": True,
            "links": [
                {"rel": "about", "href": "https://wikipedia.test/evi"},
                {
                    "rel": "canonical",
                    "href": "http://oeo.net/openeo/1.0.0/processes/u:Mr.Test/evi",
                    "title": "Public URL for user-defined process 'evi'",
                },
            ],
        }

        api100.get("/process_graphs/evi").assert_error(401, "AuthenticationRequired")
        udp = api100.get("/process_graphs/evi", headers=TEST_USER_AUTH_HEADER).assert_status_code(200).json
        assert udp == expected

        udp = api100.get("/processes/u:Mr.Test/evi").assert_status_code(200).json
        assert udp == expected

    @pytest.mark.parametrize("public", [True, False])
    def test_udp_public_access(self, api100, udp_store, public):
        api100.put(
            "/process_graphs/evi",
            headers=TEST_USER_AUTH_HEADER,
            json={
                "process_graph": {"add35": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}},
                "public": public,
            },
        ).assert_status_code(200)

        resp = api100.get("/processes/u:Mr.Test/evi")
        if public:
            assert resp.assert_status_code(200).json == {
                "id": "evi",
                "process_graph": {"add35": {"arguments": {"x": 3, "y": 5}, "process_id": "add", "result": True}},
                "public": True,
                "links": [
                    {
                        "href": "http://oeo.net/openeo/1.0.0/processes/u:Mr.Test/evi",
                        "rel": "canonical",
                        "title": "Public URL for user-defined process 'evi'",
                    }
                ],
            }
        else:
            resp.assert_error(400, "ProcessUnsupported")

    @pytest.mark.parametrize("body_id", [None, "evi", "meh"])
    def test_add_and_get_udp(self, api100, udp_store, body_id):
        spec = {
            "parameters": [{"name": "red"}],
            "returns": {"schema": {"type": "number"}},
            "process_graph": {"sub": {}},
            "summary": "evify it",
            "description": "Calculate the EVI",
            "public": True,
        }
        if body_id:
            spec["id"] = body_id
        api100.put("/process_graphs/evi", headers=TEST_USER_AUTH_HEADER, json=spec).assert_status_code(200)

        udp = udp_store._processes["Mr.Test", "evi"]
        assert udp.id == "evi"
        assert udp.parameters == [{"name": "red"}]
        assert udp.returns == {"schema": {"type": "number"}}
        assert udp.process_graph == {"sub": {}}
        assert udp.summary == "evify it"
        assert udp.description == "Calculate the EVI"
        assert udp.public is True

        udp = api100.get("/process_graphs/evi", headers=TEST_USER_AUTH_HEADER).assert_status_code(200).json
        assert udp["id"] == "evi"
        assert udp["parameters"] == [{"name": "red"}]
        assert udp["returns"] == {"schema": {"type": "number"}}
        assert udp["process_graph"] == {"sub": {}}
        assert udp["summary"] == "evify it"
        assert udp["description"] == "Calculate the EVI"
        assert udp["public"] is True


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
