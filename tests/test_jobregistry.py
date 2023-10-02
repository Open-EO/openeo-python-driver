import logging
import urllib.parse
from typing import Union, Optional, List
from unittest import mock

import pytest
import requests
import requests_mock
import time_machine
from openeo.rest.auth.testing import OidcMock
from openeo_driver.errors import JobNotFoundException, InternalException

from openeo_driver.jobregistry import (
    DEPENDENCY_STATUS,
    JOB_STATUS,
    EjrError,
    EjrHttpError,
    ElasticJobRegistry,
    ElasticJobRegistryCredentials,
)
from openeo_driver.testing import (
    DictSubSet,
    RegexMatcher,
    ListSubSet,
    IgnoreOrder,
    caplog_with_custom_formatter,
)

DUMMY_PROCESS = {
    "summary": "calculate 3+5, please",
    "process_graph": {
        "add": {
            "process_id": "add",
            "arguments": {"x": 3, "y": 5},
            "result": True,
        },
    },
}


class TestElasticJobRegistryCredentials:
    def test_basic(self):
        creds = ElasticJobRegistryCredentials(
            oidc_issuer="https://oidc.test/", client_id="c123", client_secret="@#$"
        )
        assert creds.oidc_issuer == "https://oidc.test/"
        assert creds.client_id == "c123"
        assert creds.client_secret == "@#$"
        assert creds == ("https://oidc.test/", "c123", "@#$")

    def test_repr(self):
        creds = ElasticJobRegistryCredentials(
            oidc_issuer="https://oidc.test/", client_id="c123", client_secret="@#$"
        )
        expected = "ElasticJobRegistryCredentials(oidc_issuer='https://oidc.test/', client_id='c123', client_secret='***')"
        assert repr(creds) == expected
        assert str(creds) == expected

    def test_get_from_mapping(self):
        creds = ElasticJobRegistryCredentials.from_mapping(
            {"oidc_issuer": "https://oidc.test/", "client_id": "c456789", "client_secret": "s3cr3t"},
        )
        assert creds == ("https://oidc.test/", "c456789", "s3cr3t")

    def test_get_from_mapping_strictness(self):
        data = {"oidc_issuer": "https://oidc.test/", "client_id": "c456789"}
        with pytest.raises(
            EjrError, match="Failed building ElasticJobRegistryCredentials from mapping: missing {'client_secret'}"
        ):
            _ = ElasticJobRegistryCredentials.from_mapping(data)
        creds = ElasticJobRegistryCredentials.from_mapping(data, strict=False)
        assert creds is None

    def test_get_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENEO_EJR_OIDC_ISSUER", "https://id.example")
        monkeypatch.setenv("OPENEO_EJR_OIDC_CLIENT_ID", "c-9876")
        monkeypatch.setenv("OPENEO_EJR_OIDC_CLIENT_SECRET", "!@#$%%")
        creds = ElasticJobRegistryCredentials.from_env()
        assert creds == ("https://id.example", "c-9876", "!@#$%%")

    def test_get_from_env_strictness(self, monkeypatch):
        monkeypatch.setenv("OPENEO_EJR_OIDC_ISSUER", "https://id.example")
        monkeypatch.setenv("OPENEO_EJR_OIDC_CLIENT_ID", "c-9876")
        with pytest.raises(
            EjrError,
            match="Failed building ElasticJobRegistryCredentials from env: missing {'OPENEO_EJR_OIDC_CLIENT_SECRET'}",
        ):
            _ = ElasticJobRegistryCredentials.from_env()
        creds = ElasticJobRegistryCredentials.from_env(strict=False)
        assert creds is None


class TestElasticJobRegistry:
    EJR_API_URL = "https://ejr.test"

    OIDC_CLIENT_INFO = {
        "oidc_issuer": "https://oidc.test",
        "client_id": "ejrclient",
        "client_secret": "6j7$6c76T",
    }

    @pytest.fixture
    def oidc_mock(self, requests_mock) -> OidcMock:
        oidc_issuer = self.OIDC_CLIENT_INFO["oidc_issuer"]
        oidc_mock = OidcMock(
            requests_mock=requests_mock,
            oidc_issuer=oidc_issuer,
            expected_grant_type="client_credentials",
            expected_client_id=self.OIDC_CLIENT_INFO["client_id"],
            expected_fields={"client_secret": self.OIDC_CLIENT_INFO["client_secret"], "scope": "openid"},
        )
        return oidc_mock

    @pytest.fixture
    def ejr(self, oidc_mock) -> ElasticJobRegistry:
        """ElasticJobRegistry set up with authentication"""
        ejr = ElasticJobRegistry(api_url=self.EJR_API_URL, backend_id="unittests")
        credentials = ElasticJobRegistryCredentials(
            oidc_issuer=self.OIDC_CLIENT_INFO["oidc_issuer"],
            client_id=self.OIDC_CLIENT_INFO["client_id"],
            client_secret=self.OIDC_CLIENT_INFO["client_secret"],
        )
        ejr.setup_auth_oidc_client_credentials(credentials)
        return ejr

    def test_access_token_caching(self, requests_mock, oidc_mock, ejr):
        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=[])

        with time_machine.travel("2020-01-02 12:00:00+00"):
            result = ejr.list_user_jobs(user_id="john")
            assert result == []
            assert len(oidc_mock.get_request_history(url="/token")) == 1

        with time_machine.travel("2020-01-02 12:01:00+00"):
            result = ejr.list_user_jobs(user_id="john")
            assert result == []
            assert len(oidc_mock.get_request_history(url="/token")) == 1

        with time_machine.travel("2020-01-03 12:00:00+00"):
            result = ejr.list_user_jobs(user_id="john")
            assert result == []
            assert len(oidc_mock.get_request_history(url="/token")) == 2

    def _auth_is_valid(
        self,
        request: requests.Request,
        *,
        expected_access_token: Optional[str] = None,
        oidc_mock: Optional[OidcMock] = None,
    ) -> bool:
        if expected_access_token:
            access_token = expected_access_token
        elif oidc_mock:
            access_token = oidc_mock.state["access_token"]
        else:
            raise RuntimeError("`expected_access_token` or `oidc_mock` should be set")
        return request.headers["Authorization"] == f"Bearer {access_token}"

    def _handle_get_health(
        self,
        *,
        expected_access_token: Optional[str] = None,
        oidc_mock: Optional[OidcMock] = None,
    ):
        """Create a mocking handler for `GET /health` requests."""

        def get_health(request, context):
            if "Authorization" not in request.headers:
                status, state = "down", "missing"
            elif self._auth_is_valid(
                request=request,
                expected_access_token=expected_access_token,
                oidc_mock=oidc_mock,
            ):
                status, state = "up", "ok"
            else:
                status, state = "down", "expired"
            return {"info": {"auth": {"status": status, "state": state}}}

        return get_health

    def test_health_check(self, requests_mock, oidc_mock, ejr):
        requests_mock.get(
            f"{self.EJR_API_URL}/health",
            json=self._handle_get_health(oidc_mock=oidc_mock),
        )

        # Health check without auth
        response = ejr.health_check(use_auth=False)
        assert response == {"info": {"auth": {"status": "down", "state": "missing"}}}

        # With auth
        response = ejr.health_check(use_auth=True)
        assert response == {"info": {"auth": {"status": "up", "state": "ok"}}}

        # Try again with aut,
        # but invalidate access token at provider side (depends on caching of access token in EJR)
        oidc_mock.invalidate_access_token()
        response = ejr.health_check(use_auth=True)
        assert response == {"info": {"auth": {"status": "down", "state": "expired"}}}

    def test_session_headers(self, requests_mock, ejr):
        def get_health(request, context):
            assert request.headers["User-Agent"] == RegexMatcher(
                r"openeo_driver-\d+\.\d+.*/ElasticJobRegistry/unittests"
            )
            return {"info": {"auth": {"status": "down", "state": "missing"}}}

        requests_mock.get(f"{self.EJR_API_URL}/health", json=get_health)

        response = ejr.health_check(use_auth=False)
        assert response == {"info": {"auth": {"status": "down", "state": "missing"}}}

    def test_health_check_custom_requests_session(self):
        # Instead of using the requests_mock/oidc_mock fixtures like we do in other tests:
        # explicitly create a requests session to play with, using an adapter from requests_mock
        session = requests.Session()
        session_history: List[requests.Response] = []
        session.hooks["response"].append(
            lambda resp, *args, **kwargs: session_history.append(resp)
        )
        adapter = requests_mock.Adapter()
        session.mount("https://", adapter)

        oidc_issuer = "https://oidc.test"
        adapter.register_uri(
            "GET",
            f"{oidc_issuer}/.well-known/openid-configuration",
            json={
                "issuer": oidc_issuer,
                "scopes_supported": ["openid"],
                "token_endpoint": f"{oidc_issuer}/token",
            },
        )

        def post_token(request, context):
            # Very simple handler here (compared to OidcMock implementation)
            assert "grant_type=client_credentials" in request.text
            return {"access_token": "6cce5-t0k3n"}

        adapter.register_uri("POST", f"{oidc_issuer}/token", json=post_token)
        adapter.register_uri(
            "GET",
            "https://ejr.test/health",
            json=self._handle_get_health(expected_access_token="6cce5-t0k3n"),
        )

        # Setup up ElasticJobRegistry
        ejr = ElasticJobRegistry(
            api_url=self.EJR_API_URL, backend_id="unittests", session=session
        )
        credentials = ElasticJobRegistryCredentials(
            oidc_issuer=self.OIDC_CLIENT_INFO["oidc_issuer"],
            client_id=self.OIDC_CLIENT_INFO["client_id"],
            client_secret=self.OIDC_CLIENT_INFO["client_secret"],
        )
        ejr.setup_auth_oidc_client_credentials(credentials)

        # Check requests so far on custom session
        assert [
            f"{r.status_code} {r.request.method} {r.request.url}"
            for r in session_history
        ] == [
            "200 GET https://oidc.test/.well-known/openid-configuration",
        ]

        # Do a EJR /health request
        response = ejr.health_check(use_auth=True)
        assert response == {"info": {"auth": {"state": "ok", "status": "up"}}}

        # Check requests on custom session
        assert [
            f"{r.status_code} {r.request.method} {r.request.url}"
            for r in session_history
        ] == [
            "200 GET https://oidc.test/.well-known/openid-configuration",
            "200 POST https://oidc.test/token",
            "200 GET https://ejr.test/health",
        ]

    def test_create_job(self, requests_mock, oidc_mock, ejr):
        def post_jobs(request, context):
            """Handler of `POST /jobs`"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            # TODO: what to return? What does API return?  https://github.com/Open-EO/openeo-job-tracker-elastic-api/issues/3
            context.status_code = 201
            return request.json()

        requests_mock.post(f"{self.EJR_API_URL}/jobs", json=post_jobs)

        with time_machine.travel("2020-01-02 03:04:05+00", tick=False):
            result = ejr.create_job(process=DUMMY_PROCESS, user_id="john")
        assert result == DictSubSet(
            {
                "backend_id": "unittests",
                "job_id": RegexMatcher("j-[0-9a-f]+"),
                "user_id": "john",
                "process": DUMMY_PROCESS,
                "created": "2020-01-02T03:04:05Z",
                "updated": "2020-01-02T03:04:05Z",
                "status": "created",
                "job_options": None,
            }
        )

    @pytest.mark.parametrize("status_code", [204, 400, 500])
    def test_create_job_with_error(self, requests_mock, oidc_mock, ejr, status_code):
        def post_jobs(request, context):
            """Handler of `POST /jobs`"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            context.status_code = status_code
            return {"error": "meh"}

        requests_mock.post(f"{self.EJR_API_URL}/jobs", json=post_jobs)

        with pytest.raises(EjrError) as e:
            _ = ejr.create_job(process=DUMMY_PROCESS, user_id="john")

    def test_get_job(self, requests_mock, oidc_mock, ejr):
        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            assert request.json() == {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"backend_id": "unittests"}},
                            {"term": {"job_id": "job-123"}},
                        ]
                    }
                },
                "_source": IgnoreOrder(
                    ["job_id", "user_id", "created", "status", "updated", "*"]
                ),
            }
            return [{"job_id": "job-123", "user_id": "john", "status": "created"}]

        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=post_jobs_search)

        result = ejr.get_job(job_id="job-123")
        assert result == {"job_id": "job-123", "user_id": "john", "status": "created"}

    def test_get_job_not_found(self, requests_mock, oidc_mock, ejr):
        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            return []

        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=post_jobs_search)

        with pytest.raises(JobNotFoundException):
            _ = ejr.get_job(job_id="job-123")

    def test_delete_job(self, requests_mock, oidc_mock, ejr):
        delete_mock = requests_mock.delete(f"{self.EJR_API_URL}/jobs/job-123", status_code=200, content=b"")
        _ = ejr.delete_job(job_id="job-123")
        assert delete_mock.call_count == 1

    def test_delete_job_not_found(self, requests_mock, oidc_mock, ejr):
        delete_mock = requests_mock.delete(
            f"{self.EJR_API_URL}/jobs/job-123",
            status_code=404,
            json={"statusCode": 404, "message": "Could not find job with job-123", "error": "Not Found"},
        )
        with pytest.raises(JobNotFoundException):
            _ = ejr.delete_job(job_id="job-123")
        assert delete_mock.call_count == 1

    def test_get_job_multiple_results(self, requests_mock, oidc_mock, ejr):
        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            return [
                {"job_id": "job-123", "user_id": "alice"},
                {"job_id": "job-123", "user_id": "bob"},
            ]

        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=post_jobs_search)

        with pytest.raises(
            InternalException, match="Found 2 jobs for job_id='job-123'"
        ):
            _ = ejr.get_job(job_id="job-123")

    def test_list_user_jobs(self, requests_mock, oidc_mock, ejr):
        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            assert request.json() == {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"backend_id": "unittests"}},
                            {"term": {"user_id": "john"}},
                        ]
                    }
                },
                "_source": ListSubSet(["job_id", "user_id", "status"]),
            }
            # TODO: what to return? What does API return?  https://github.com/Open-EO/openeo-job-tracker-elastic-api/issues/3
            return [DUMMY_PROCESS]

        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=post_jobs_search)

        result = ejr.list_user_jobs(user_id="john")
        assert result == [DUMMY_PROCESS]

    @pytest.mark.parametrize(
        ["fields", "expected_fields"],
        [
            (None, ["job_id", "user_id", "created", "status", "updated"]),
            (
                ["created", "started"],
                ["job_id", "user_id", "created", "status", "updated", "started"],
            ),
        ],
    )
    def test_list_active_jobs(
        self, requests_mock, oidc_mock, ejr, fields, expected_fields
    ):
        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            assert request.json() == {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"backend_id": "unittests"}},
                            {"terms": {"status": ["created", "queued", "running"]}},
                            {"range": {"created": {"gte": "now-7d"}}},
                        ]
                    }
                },
                "_source": IgnoreOrder(expected_fields),
            }
            return [
                {"job_id": "job-123", "user_id": "alice"},
                {"job_id": "job-456", "user_id": "bob"},
            ]

        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=post_jobs_search)
        result = ejr.list_active_jobs(fields=fields)
        assert result == [
            {"job_id": "job-123", "user_id": "alice"},
            {"job_id": "job-456", "user_id": "bob"},
        ]

    def _handle_patch_jobs(
        self, oidc_mock: OidcMock, expected_data: Union[dict, DictSubSet]
    ):
        """Create a mocking handler for `PATCH /jobs` requests."""

        def patch_jobs(request: requests.Request, context):
            """Handler of `PATCH /jobs"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            data = request.json()
            assert data == expected_data
            # TODO: what to return? What does API return?  https://github.com/Open-EO/openeo-job-tracker-elastic-api/issues/3
            return data

        return patch_jobs

    def test_set_status(self, requests_mock, oidc_mock, ejr):
        handler = self._handle_patch_jobs(
            oidc_mock=oidc_mock,
            expected_data={
                "status": "running",
                "updated": "2022-12-14T12:34:56Z",
            },
        )
        requests_mock.patch(f"{self.EJR_API_URL}/jobs/job-123", json=handler)

        with time_machine.travel("2022-12-14T12:34:56Z"):
            result = ejr.set_status(job_id="job-123", status=JOB_STATUS.RUNNING)
        assert result["status"] == "running"

    def test_set_status_with_started(self, requests_mock, oidc_mock, ejr):
        handler = self._handle_patch_jobs(
            oidc_mock=oidc_mock,
            expected_data=DictSubSet(
                {
                    "status": "running",
                    "updated": "2022-12-14T10:00:00Z",
                    "started": "2022-12-14T10:00:00Z",
                }
            ),
        )
        requests_mock.patch(f"{self.EJR_API_URL}/jobs/job-123", json=handler)

        result = ejr.set_status(
            job_id="job-123",
            status=JOB_STATUS.RUNNING,
            updated="2022-12-14T10:00:00",
            started="2022-12-14T10:00:00",
        )
        assert result["status"] == "running"

    def test_set_status_with_finished(self, requests_mock, oidc_mock, ejr):
        handler = self._handle_patch_jobs(
            oidc_mock=oidc_mock,
            expected_data=DictSubSet(
                {
                    "status": "running",
                    "updated": "2022-12-14T12:34:56Z",
                    "finished": "2022-12-14T10:00:00Z",
                }
            ),
        )
        requests_mock.patch(f"{self.EJR_API_URL}/jobs/job-123", json=handler)
        with time_machine.travel("2022-12-14T12:34:56Z"):
            result = ejr.set_status(
                job_id="job-123",
                status=JOB_STATUS.RUNNING,
                finished="2022-12-14T10:00:00",
            )
        assert result["status"] == "running"

    def test_set_dependencies(self, requests_mock, oidc_mock, ejr):
        handler = self._handle_patch_jobs(
            oidc_mock=oidc_mock, expected_data={"dependencies": [{"foo": "bar"}]}
        )
        patch_mock = requests_mock.patch(
            f"{self.EJR_API_URL}/jobs/job-123", json=handler
        )

        ejr.set_dependencies(job_id="job-123", dependencies=[{"foo": "bar"}])
        assert patch_mock.call_count == 1

    def test_remove_dependencies(self, requests_mock, oidc_mock, ejr):
        handler = self._handle_patch_jobs(
            oidc_mock=oidc_mock,
            expected_data={"dependencies": None, "dependency_status": None},
        )
        patch_mock = requests_mock.patch(
            f"{self.EJR_API_URL}/jobs/job-123", json=handler
        )

        ejr.remove_dependencies(job_id="job-123")
        assert patch_mock.call_count == 1

    def test_set_dependency_status(self, requests_mock, oidc_mock, ejr):
        handler = self._handle_patch_jobs(
            oidc_mock=oidc_mock,
            expected_data={"dependency_status": "awaiting"},
        )
        patch_mock = requests_mock.patch(
            f"{self.EJR_API_URL}/jobs/job-123", json=handler
        )

        ejr.set_dependency_status(
                job_id="job-123", dependency_status=DEPENDENCY_STATUS.AWAITING
            )
        assert patch_mock.call_count == 1

    def test_set_proxy_user(self, requests_mock, oidc_mock, ejr):
        handler = self._handle_patch_jobs(
            oidc_mock=oidc_mock, expected_data={"proxy_user": "john"}
        )
        patch_mock = requests_mock.patch(
            f"{self.EJR_API_URL}/jobs/job-123", json=handler
        )

        ejr.set_proxy_user(job_id="job-123", proxy_user="john")
        assert patch_mock.call_count == 1

    def test_set_application_id(self, requests_mock, oidc_mock, ejr):
        handler = self._handle_patch_jobs(
            oidc_mock=oidc_mock, expected_data={"application_id": "app-456"}
        )
        patch_mock = requests_mock.patch(
            f"{self.EJR_API_URL}/jobs/job-123", json=handler
        )

        ejr.set_application_id(job_id="job-123", application_id="app-456")
        assert patch_mock.call_count == 1

    def test_set_usage(self, requests_mock, oidc_mock, ejr):
        handler = self._handle_patch_jobs(
            oidc_mock=oidc_mock, expected_data={"costs": 22, "usage": {
                "cpu": {"value": 3283, "unit": "cpu-seconds"},
                "memory": {"value": 8040202, "unit": "mb-seconds"},
                "sentinelhub": {"value": 108.33333656191826, "unit": "sentinelhub_processing_unit"}}}
        )
        patch_mock = requests_mock.patch(
            f"{self.EJR_API_URL}/jobs/job-123", json=handler
        )

        ejr.set_usage(job_id="job-123", costs=22, usage={
            "cpu": {"value": 3283, "unit": "cpu-seconds"},
            "memory": {"value": 8040202, "unit": "mb-seconds"},
            "sentinelhub": {"value": 108.33333656191826, "unit": "sentinelhub_processing_unit"}})
        assert patch_mock.call_count == 1

    def test_just_log_errors(self, caplog):
        with ElasticJobRegistry.just_log_errors("some math"):
            x = (2 + 3) / 0
        assert caplog.record_tuples == [
            (
                "openeo_driver.jobregistry.elastic",
                logging.WARN,
                "In context 'some math': caught ZeroDivisionError('division by zero')",
            )
        ]

    def test_job_id_logging(self, requests_mock, oidc_mock, ejr, caplog):
        """Check that job_id logging is passed through as logging extra in appropriate places"""
        caplog.set_level(logging.DEBUG)

        class Formatter:
            def format(self, record: logging.LogRecord):
                job_id = getattr(record, "job_id", None)
                return f"{record.name}:{job_id}:{record.message}"

        job_id = "j-123"

        def post_jobs(request, context):
            """Handler of `POST /jobs`"""
            context.status_code = 201
            return {"job_id": request.json()["job_id"]}

        def patch_job(request, context):
            return request.json()

        requests_mock.post(f"{self.EJR_API_URL}/jobs", json=post_jobs)
        requests_mock.patch(f"{self.EJR_API_URL}/jobs/{job_id}", json=patch_job)

        with caplog_with_custom_formatter(caplog=caplog, format=Formatter()):
            with time_machine.travel("2020-01-02 03:04:05+00", tick=False):
                result = ejr.create_job(
                    job_id=job_id, process=DUMMY_PROCESS, user_id="john"
                )
                assert result == {"job_id": job_id}
            with time_machine.travel("2020-01-02 03:04:10+00", tick=False):
                ejr.set_application_id(job_id=job_id, application_id="app-123")
            with time_machine.travel("2020-01-02 03:44:55+00", tick=False):
                ejr.set_status(job_id=job_id, status=JOB_STATUS.RUNNING)

        logs = caplog.text.strip().split("\n")

        for expected in [
            # Create
            "openeo_driver.jobregistry.elastic:j-123:EJR creating job_id='j-123' created='2020-01-02T03:04:05Z'",
            "openeo_driver.jobregistry.elastic:j-123:EJR Request `POST /jobs`: start 2020-01-02 03:04:05",
            "openeo_driver.jobregistry.elastic:j-123:Doing EJR request `POST https://ejr.test/jobs` headers.keys()=dict_keys(['Authorization'])",
            "openeo_driver.jobregistry.elastic:j-123:EJR response on `POST /jobs`: 201",
            "openeo_driver.jobregistry.elastic:j-123:EJR Request `POST /jobs`: end 2020-01-02 03:04:05, elapsed 0:00:00",
            # set_application_id
            "openeo_driver.jobregistry.elastic:j-123:EJR update job_id='j-123' data={'application_id': 'app-123'}",
            "openeo_driver.jobregistry.elastic:j-123:EJR Request `PATCH /jobs/j-123`: start 2020-01-02 03:04:10",
            "openeo_driver.jobregistry.elastic:j-123:EJR response on `PATCH /jobs/j-123`: 200",
            # set_status
            "openeo_driver.jobregistry.elastic:j-123:EJR update job_id='j-123' data={'status': 'running', 'updated': '2020-01-02T03:44:55Z'}",
            "openeo_driver.jobregistry.elastic:j-123:EJR response on `PATCH /jobs/j-123`: 200",
            "openeo_driver.jobregistry.elastic:j-123:EJR Request `PATCH /jobs/j-123`: end 2020-01-02 03:44:55, elapsed 0:00:00",
        ]:
            assert expected in logs
