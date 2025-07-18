import logging
import urllib.parse
from typing import Callable, List, Optional, Sequence, Union
from unittest import mock

import dirty_equals
import pytest
import requests
import requests_mock
import time_machine
from openeo.rest.auth.testing import OidcMock
from openeo.util import dict_no_none

from openeo_driver.backend import JobListing
from openeo_driver.constants import JOB_STATUS
from openeo_driver.errors import InternalException, JobNotFoundException
from openeo_driver.jobregistry import (
    DEPENDENCY_STATUS,
    PARTIAL_JOB_STATUS,
    EjrApiError,
    EjrApiResponseError,
    EjrError,
    ElasticJobRegistry,
    ejr_job_info_to_metadata,
    get_ejr_credentials_from_env,
)
from openeo_driver.testing import (
    DictSubSet,
    IgnoreOrder,
    ListSubSet,
    RegexMatcher,
    caplog_with_custom_formatter,
    config_overrides,
)
from openeo_driver.util.auth import ClientCredentials
from openeo_driver.util.logging import ExtraLoggingFilter

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


def test_get_ejr_credentials_from_env_legacy(monkeypatch):
    monkeypatch.setenv("OPENEO_EJR_OIDC_ISSUER", "https://id.example")
    monkeypatch.setenv("OPENEO_EJR_OIDC_CLIENT_ID", "c-9876")
    monkeypatch.setenv("OPENEO_EJR_OIDC_CLIENT_SECRET", "!@#$%%")
    creds = get_ejr_credentials_from_env()
    assert creds == ("https://id.example", "c-9876", "!@#$%%")


def test_get_ejr_credentials_from_env_compact(monkeypatch):
    monkeypatch.setenv("OPENEO_EJR_OIDC_CLIENT_CREDENTIALS", "c-534:!f00ba>@https://id2.example")
    creds = get_ejr_credentials_from_env()
    assert creds == ("https://id2.example", "c-534", "!f00ba>")


def test_get_ejr_credentials_from_env_both(monkeypatch):
    monkeypatch.setenv("OPENEO_EJR_OIDC_ISSUER", "https://id.example")
    monkeypatch.setenv("OPENEO_EJR_OIDC_CLIENT_ID", "c-9876")
    monkeypatch.setenv("OPENEO_EJR_OIDC_CLIENT_SECRET", "!@#$%%")
    monkeypatch.setenv("OPENEO_EJR_OIDC_CLIENT_CREDENTIALS", "c-534:!f00ba>@https://id2.example")
    creds = get_ejr_credentials_from_env()
    assert creds == ("https://id2.example", "c-534", "!f00ba>")


def test_get_ejr_credentials_from_env_strictness_legacy(monkeypatch):
    monkeypatch.setenv("OPENEO_EJR_OIDC_ISSUER", "https://id.example")
    monkeypatch.setenv("OPENEO_EJR_OIDC_CLIENT_ID", "c-9876")
    with pytest.raises(
        EjrError,
        match="Failed building ClientCredentials from env: missing {'OPENEO_EJR_OIDC_CLIENT_SECRET'}",
    ):
        _ = get_ejr_credentials_from_env()
    creds = get_ejr_credentials_from_env(strict=False)
    assert creds is None


def test_get_ejr_credentials_from_env_strictness_compact(monkeypatch):
    monkeypatch.setenv("OPENEO_EJR_OIDC_CLIENT_CREDENTIALS", "c-534@https://id2.example")
    with pytest.raises(ValueError, match="Failed parsing ClientCredentials from credentials string"):
        _ = get_ejr_credentials_from_env()
    creds = get_ejr_credentials_from_env(strict=False)
    assert creds is None


def test_get_partial_job_status():
    assert PARTIAL_JOB_STATUS.for_job_status(JOB_STATUS.CREATED) == 'running'
    assert PARTIAL_JOB_STATUS.for_job_status(JOB_STATUS.QUEUED) == 'running'
    assert PARTIAL_JOB_STATUS.for_job_status(JOB_STATUS.RUNNING) == 'running'
    assert PARTIAL_JOB_STATUS.for_job_status(JOB_STATUS.FINISHED) == 'finished'
    assert PARTIAL_JOB_STATUS.for_job_status(JOB_STATUS.ERROR) == 'error'
    assert PARTIAL_JOB_STATUS.for_job_status(JOB_STATUS.CANCELED) == 'canceled'


def test_ejr_job_info_to_metadata():
    job_info = {
        "job_id": "j-123",
        "status": "running",
        "results_metadata_uri": "s3://bucket/path/to/job_metadata.json",
    }

    metadata = ejr_job_info_to_metadata(job_info)
    assert metadata.status == JOB_STATUS.RUNNING


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
    def ejr_credentials(self) -> ClientCredentials:
        return ClientCredentials(
            oidc_issuer=self.OIDC_CLIENT_INFO["oidc_issuer"],
            client_id=self.OIDC_CLIENT_INFO["client_id"],
            client_secret=self.OIDC_CLIENT_INFO["client_secret"],
        )

    @pytest.fixture
    def ejr(self, oidc_mock, ejr_credentials) -> ElasticJobRegistry:
        """ElasticJobRegistry set up with authentication"""
        ejr = ElasticJobRegistry(api_url=self.EJR_API_URL, backend_id="unittests")
        ejr.setup_auth_oidc_client_credentials(ejr_credentials)
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
        credentials = ClientCredentials(
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

    @pytest.mark.parametrize(
        ["preserialize_process", "expected"],
        [
            (False, DUMMY_PROCESS),
            (
                True,
                '{"summary":"calculate 3+5, please","process_graph":{"add":{"process_id":"add","arguments":{"x":3,"y":5},"result":true}}}',
            ),
        ],
    )
    def test_create_job_preserialize_process_graph(
        self, requests_mock, oidc_mock, ejr_credentials, preserialize_process, expected
    ):
        ejr = ElasticJobRegistry(
            api_url=self.EJR_API_URL, backend_id="unittests", preserialize_process=preserialize_process
        )
        ejr.setup_auth_oidc_client_credentials(ejr_credentials)

        posted = []

        def post_jobs(request, context):
            """Handler of `POST /jobs`"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            # TODO: what to return? What does API return?  https://github.com/Open-EO/openeo-job-tracker-elastic-api/issues/3
            context.status_code = 201
            post_data = request.json()
            posted.append(post_data)
            return post_data

        requests_mock.post(f"{self.EJR_API_URL}/jobs", json=post_jobs)

        with time_machine.travel("2020-01-02 03:04:05+00", tick=False):
            ejr.create_job(process=DUMMY_PROCESS, user_id="john")

        assert posted == [
            dirty_equals.IsPartialDict(
                {
                    "job_id": RegexMatcher("j-[0-9a-f]+"),
                    "user_id": "john",
                    "process": expected,
                    "job_options": None,
                }
            )
        ]

    def _build_handler_post_jobs_search_single_job_lookup(
        self,
        *,
        job_id: str = "job-123",
        backend_id: str = "unittests",
        user_id: str = "john",
        status: str = "created",
        source_fields: Sequence[str] = ("job_id", "user_id", "created", "status", "updated", "*"),
        oidc_mock: OidcMock,
        process: Union[dict, str, None] = None,
    ) -> Callable:
        """
        Build handler for 'POST /jobs/search for a single job lookup.
        """

        def handler(request, context):
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            filters = [
                {"term": {"backend_id": backend_id}},
                {"term": {"job_id": job_id}},
            ]
            if user_id is not None:
                filters.append({"term": {"user_id": user_id}})
            assert request.json() == {
                "query": {
                    "bool": {
                        "filter": filters
                    }
                },
                "_source": IgnoreOrder(list(source_fields)),
            }
            job = {"job_id": job_id, "user_id": user_id, "status": status}
            if process:
                job["process"] = process
            return [job]

        return handler

    def test_get_job(self, requests_mock, oidc_mock, ejr):
        requests_mock.post(
            f"{self.EJR_API_URL}/jobs/search",
            json=self._build_handler_post_jobs_search_single_job_lookup(
                job_id="job-123", user_id="john", status="created", oidc_mock=oidc_mock
            ),
        )

        result = ejr.get_job(job_id="job-123", user_id="john")
        assert result == {"job_id": "job-123", "user_id": "john", "status": "created"}

    def test_get_job_not_found(self, requests_mock, oidc_mock, ejr):
        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            return []

        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=post_jobs_search)

        with pytest.raises(JobNotFoundException):
            _ = ejr.get_job(job_id="job-123", user_id="john")

    @pytest.mark.parametrize(
        ["process_payload", "expected"], [(DUMMY_PROCESS, DUMMY_PROCESS), ('{"foo":"bar"}', {"foo": "bar"})]
    )
    def test_get_job_deserialize(self, requests_mock, oidc_mock, ejr, process_payload, expected):
        requests_mock.post(
            f"{self.EJR_API_URL}/jobs/search",
            json=self._build_handler_post_jobs_search_single_job_lookup(
                job_id="job-123", user_id="john", status="created", oidc_mock=oidc_mock, process=process_payload
            ),
        )

        result = ejr.get_job(job_id="job-123", user_id="john")
        assert result == {"job_id": "job-123", "user_id": "john", "status": "created", "process": expected}

    def test_delete_job(self, requests_mock, oidc_mock, ejr):
        search_mock = requests_mock.post(
            f"{self.EJR_API_URL}/jobs/search", [
                {"json": self._build_handler_post_jobs_search_single_job_lookup(
                    job_id="job-123", user_id="john", status="created", oidc_mock=oidc_mock,
                    source_fields=["job_id", "user_id", "created", "status", "updated"]
                )},
                {"json": []}
            ]
        )
        delete_mock = requests_mock.delete(f"{self.EJR_API_URL}/jobs/job-123", status_code=200, content=b"")

        ejr.delete_job(job_id="job-123", user_id="john")
        assert search_mock.call_count == 2
        assert delete_mock.call_count == 1

    def test_delete_their_job(self, requests_mock, oidc_mock, ejr):
        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            return []

        search_mock = requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=post_jobs_search)

        with pytest.raises(JobNotFoundException):
            ejr.delete_job(job_id="job-123", user_id="john")
        assert search_mock.call_count == 1

    def test_delete_job_not_found(self, requests_mock, oidc_mock, ejr):
        # shouldn't happen, really
        search_mock = requests_mock.post(
            f"{self.EJR_API_URL}/jobs/search",
            json=self._build_handler_post_jobs_search_single_job_lookup(
                job_id="job-123", user_id="john", status="created", oidc_mock=oidc_mock,
                source_fields=["job_id", "user_id", "created", "status", "updated"]
            ),
        )
        delete_mock = requests_mock.delete(
            f"{self.EJR_API_URL}/jobs/job-123",
            status_code=404,
            json={"statusCode": 404, "message": "Could not find job with job-123", "error": "Not Found"},
        )
        with pytest.raises(JobNotFoundException):
            ejr.delete_job(job_id="job-123", user_id="john")
        assert search_mock.call_count == 1
        assert delete_mock.call_count == 1

    def test_delete_job_with_verification_direct(self, requests_mock, oidc_mock, ejr, caplog):
        caplog.set_level(logging.DEBUG)

        search_mock = requests_mock.post(f"{self.EJR_API_URL}/jobs/search", [
            {"json": self._build_handler_post_jobs_search_single_job_lookup(  # assert own job
                job_id="job-123", user_id="john", status="created", oidc_mock=oidc_mock,
                source_fields=["job_id", "user_id", "created", "status", "updated"])},
            {"json": []}  # was deleted, empty search response
        ])
        delete_mock = requests_mock.delete(f"{self.EJR_API_URL}/jobs/job-123", status_code=200, content=b"")

        ejr.delete_job(job_id="job-123", user_id="john")
        assert search_mock.call_count == 2
        assert delete_mock.call_count == 1

        log_messages = caplog.messages
        for expected in [
            "EJR deleted job_id='job-123'",
            "_verify_job_existence job_id='job-123' user_id='john' exists=False backoff=0",
        ]:
            assert expected in log_messages

    def test_delete_job_with_verification_backoff(self, requests_mock, oidc_mock, ejr, caplog):
        caplog.set_level(logging.DEBUG)

        delete_mock = requests_mock.delete(f"{self.EJR_API_URL}/jobs/job-123", status_code=200, content=b"")
        search_mock = requests_mock.post(
            f"{self.EJR_API_URL}/jobs/search",
            [
                # First attempt: assert own job
                {
                    "json": self._build_handler_post_jobs_search_single_job_lookup(
                        job_id="job-123",
                        user_id="john",
                        status="created",
                        oidc_mock=oidc_mock,
                        source_fields=["job_id", "user_id", "created", "status", "updated"],
                    )
                },
                # Second attempt: still found
                {
                    "json": self._build_handler_post_jobs_search_single_job_lookup(
                        job_id="job-123",
                        user_id="john",
                        status="created",
                        oidc_mock=oidc_mock,
                        source_fields=["job_id", "user_id", "created", "status", "updated"],
                    )
                },
                # Third attempt: not found anymore
                {"json": []},
            ],
        )

        ejr.delete_job(job_id="job-123", user_id="john")
        assert delete_mock.call_count == 1
        assert search_mock.call_count == 3

        log_messages = caplog.messages
        for expected in [
            "EJR deleted job_id='job-123'",
            "_verify_job_existence job_id='job-123' user_id='john' exists=False backoff=0",
            "_verify_job_existence job_id='job-123' user_id='john' exists=False backoff=0.1",
        ]:
            assert expected in log_messages

    def test_get_job_multiple_results(self, requests_mock, oidc_mock, ejr):
        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            return [
                {"job_id": "job-123", "user_id": "alice"},
                {"job_id": "job-123", "user_id": "alice"},
            ]

        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=post_jobs_search)

        with pytest.raises(
            InternalException, match="Found 2 jobs for job_id='job-123'"
        ):
            _ = ejr.get_job(job_id="job-123", user_id="alice")

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

    def _build_url(self, params: dict):
        return "https://oeo.test/jobs?" + urllib.parse.urlencode(query=dict_no_none(params))

    @pytest.mark.parametrize(
        [
            "limit",
            "request_parameters",
            "expected_query_string",
            "pagination_response",
            "expected_next_url",
            "expected_jobs",
        ],
        [
            (
                3,
                None,
                {"size": ["3"]},
                "auto",
                "https://oeo.test/jobs?limit=3&page=1",
                ["j-1", "j-2", "j-3"],
            ),
            (
                3,
                {ElasticJobRegistry.PAGINATION_URL_PARAM: "1"},
                {"size": ["3"], "page": ["1"]},
                "auto",
                "https://oeo.test/jobs?limit=3&page=2",
                ["j-4", "j-5", "j-6"],
            ),
            (
                3,
                {ElasticJobRegistry.PAGINATION_URL_PARAM: "10"},
                {"size": ["3"], "page": ["10"]},
                None,
                None,
                ["j-31", "j-32", "j-33"],
            ),
            (
                3,
                {ElasticJobRegistry.PAGINATION_URL_PARAM: "10"},
                {"size": ["3"], "page": ["10"]},
                {},
                None,
                ["j-31", "j-32", "j-33"],
            ),
            (
                3,
                {ElasticJobRegistry.PAGINATION_URL_PARAM: "10"},
                {"size": ["3"], "page": ["10"]},
                {"invalid": "stuff"},
                None,
                ["j-31", "j-32", "j-33"],
            ),
            (
                5,
                {ElasticJobRegistry.PAGINATION_URL_PARAM: "111"},
                {"size": ["5"], "page": ["111"]},
                "auto",
                "https://oeo.test/jobs?limit=5&page=112",
                ["j-556", "j-557", "j-558", "j-559", "j-560"],
            ),
        ],
    )
    def test_list_user_jobs_paginated(
        self,
        requests_mock,
        oidc_mock,
        ejr,
        limit,
        request_parameters,
        expected_query_string,
        pagination_response,
        expected_next_url,
        expected_jobs,
    ):
        def post_jobs_search_paginated(request, context):
            """Handler of `POST /jobs/search/paginated"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            assert request.qs == expected_query_string
            this_page = int(request.qs.get("page", ["0"])[0])
            next_page = this_page + 1
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
            nonlocal pagination_response
            if pagination_response == "auto":
                pagination_response = {"next": {"size": limit, "page": next_page}}
            return {
                "jobs": [
                    {"job_id": f"j-{j}", "status": "created", "created": f"2024-12-06T12:00:00Z"}
                    for j in range(1 + this_page * limit, 1 + next_page * limit)
                ],
                "pagination": pagination_response,
            }

        requests_mock.post(f"{self.EJR_API_URL}/jobs/search/paginated", json=post_jobs_search_paginated)

        listing = ejr.list_user_jobs(user_id="john", limit=limit, request_parameters=request_parameters)
        assert isinstance(listing, JobListing)
        if expected_next_url:
            expected_links = [{"href": expected_next_url, "rel": "next"}]
        else:
            expected_links = []
        assert listing.to_response_dict(build_url=self._build_url) == {
            "jobs": [
                {"id": j, "status": "created", "created": "2024-12-06T12:00:00Z", "progress": 0} for j in expected_jobs
            ],
            "links": expected_links,
        }

    @pytest.mark.parametrize(
        ["kwargs", "expected_query"],
        [
            (
                {},
                {
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"backend_id": "unittests"}},
                                {"terms": {"status": ["created", "queued", "running"]}},
                            ]
                        }
                    },
                    "_source": dirty_equals.IsList(
                        "job_id", "user_id", "created", "status", "updated", check_order=False
                    ),
                },
            ),
            (
                {"max_age": 7},
                {
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"backend_id": "unittests"}},
                                {"terms": {"status": ["created", "queued", "running"]}},
                                {"range": {"created": {"gte": "now-7d"}}},
                            ]
                        }
                    },
                    "_source": dirty_equals.IsList(
                        "job_id", "user_id", "created", "status", "updated", check_order=False
                    ),
                },
            ),
            (
                {"fields": ["created", "started"]},
                {
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"backend_id": "unittests"}},
                                {"terms": {"status": ["created", "queued", "running"]}},
                            ]
                        }
                    },
                    "_source": dirty_equals.IsList(
                        "job_id", "user_id", "created", "status", "updated", "started", check_order=False
                    ),
                },
            ),
            (
                {"require_application_id": True},
                {
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"backend_id": "unittests"}},
                                {"terms": {"status": ["created", "queued", "running"]}},
                            ],
                            "must": {"exists": {"field": "application_id"}},
                        }
                    },
                    "_source": dirty_equals.IsList(
                        "job_id", "user_id", "created", "status", "updated", check_order=False
                    ),
                },
            ),
            (
                {"max_updated_ago": 6},
                {
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"backend_id": "unittests"}},
                                {"terms": {"status": ["created", "queued", "running"]}},
                                {"range": {"updated": {"gte": "now-6d"}}},
                            ],
                        }
                    },
                    "_source": dirty_equals.IsList(
                        "job_id", "user_id", "created", "status", "updated", check_order=False
                    ),
                },
            ),
            (
                {"max_age": 14, "fields": ["created", "started"], "require_application_id": True},
                {
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"backend_id": "unittests"}},
                                {"terms": {"status": ["created", "queued", "running"]}},
                                {"range": {"created": {"gte": "now-14d"}}},
                            ],
                            "must": {"exists": {"field": "application_id"}},
                        }
                    },
                    "_source": dirty_equals.IsList(
                        "job_id", "user_id", "created", "status", "updated", "started", check_order=False
                    ),
                },
            ),
            (
                {
                    "max_age": 60,
                    "max_updated_ago": 10,
                    "fields": ["created", "started"],
                    "require_application_id": True,
                },
                {
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"backend_id": "unittests"}},
                                {"terms": {"status": ["created", "queued", "running"]}},
                                {"range": {"created": {"gte": "now-60d"}}},
                                {"range": {"updated": {"gte": "now-10d"}}},
                            ],
                            "must": {"exists": {"field": "application_id"}},
                        }
                    },
                    "_source": dirty_equals.IsList(
                        "job_id", "user_id", "created", "status", "updated", "started", check_order=False
                    ),
                },
            ),
        ],
    )
    def test_list_active_jobs(self, requests_mock, oidc_mock, ejr, kwargs, expected_query):
        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            assert request.json() == expected_query
            return [
                {"job_id": "job-123", "user_id": "alice"},
                {"job_id": "job-456", "user_id": "bob"},
            ]

        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=post_jobs_search)
        result = ejr.list_active_jobs(**kwargs)
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
        path_job = requests_mock.patch(f"{self.EJR_API_URL}/jobs/job-123", json=handler)

        with time_machine.travel("2022-12-14T12:34:56Z"):
            ejr.set_status(job_id="job-123", status=JOB_STATUS.RUNNING)

        assert path_job.call_count == 1

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
        patch_job = requests_mock.patch(f"{self.EJR_API_URL}/jobs/job-123", json=handler)

        ejr.set_status(
            job_id="job-123",
            status=JOB_STATUS.RUNNING,
            updated="2022-12-14T10:00:00",
            started="2022-12-14T10:00:00",
        )
        assert patch_job.call_count == 1

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
        patch_job = requests_mock.patch(f"{self.EJR_API_URL}/jobs/job-123", json=handler)
        with time_machine.travel("2022-12-14T12:34:56Z"):
            ejr.set_status(
                job_id="job-123",
                status=JOB_STATUS.RUNNING,
                finished="2022-12-14T10:00:00",
            )
        assert patch_job.call_count == 1

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

    def test_set_results_metadata_uri(self, requests_mock, oidc_mock, ejr):
        handler = self._handle_patch_jobs(
            oidc_mock=oidc_mock, expected_data={"results_metadata_uri": "s3://bucket/path/to/job_metadata.json"}
        )
        patch_mock = requests_mock.patch(f"{self.EJR_API_URL}/jobs/job-123", json=handler)

        ejr.set_results_metadata_uri(job_id="job-123", results_metadata_uri="s3://bucket/path/to/job_metadata.json")
        assert patch_mock.call_count == 1

    def test_set_results_metadata(self, requests_mock, oidc_mock, ejr):
        handler = self._handle_patch_jobs(
            oidc_mock=oidc_mock, expected_data={
                "costs": 22,
                "usage": {
                    "cpu": {"value": 3283, "unit": "cpu-seconds"},
                    "memory": {"value": 8040202, "unit": "mb-seconds"},
                    "sentinelhub": {"value": 108.33333656191826, "unit": "sentinelhub_processing_unit"}
                },
                "results_metadata": {
                    "unique_process_ids": ["load_stac"]
                },
            }
        )
        patch_mock = requests_mock.patch(
            f"{self.EJR_API_URL}/jobs/job-123", json=handler
        )

        ejr.set_results_metadata(job_id="job-123", costs=22,
                                 usage={
                                     "cpu": {"value": 3283, "unit": "cpu-seconds"},
                                     "memory": {"value": 8040202, "unit": "mb-seconds"},
                                     "sentinelhub": {"value": 108.33333656191826, "unit": "sentinelhub_processing_unit"}
                                 },
                                 results_metadata={"unique_process_ids": ["load_stac"]},
                                 )
        assert patch_mock.call_count == 1

    def test_job_id_logging(self, requests_mock, oidc_mock, ejr, caplog):
        """Check that job_id logging is passed through as logging extra in appropriate places"""
        caplog.set_level(logging.DEBUG)
        caplog.handler.addFilter(ExtraLoggingFilter())

        job_id = "j-123"

        def post_jobs(request, context):
            """Handler of `POST /jobs`"""
            context.status_code = 201
            return {"job_id": request.json()["job_id"]}

        def patch_job(request, context):
            return request.json()

        requests_mock.post(f"{self.EJR_API_URL}/jobs", json=post_jobs)
        requests_mock.patch(f"{self.EJR_API_URL}/jobs/{job_id}", json=patch_job)
        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=[{"job_id": job_id, "user_id": "john"}])
        requests_mock.delete(f"{self.EJR_API_URL}/jobs/{job_id}", status_code=200, content=b"")

        class Formatter:
            def format(self, record: logging.LogRecord):
                job_id = getattr(record, "job_id", None)
                return f"{record.name}:{job_id}:{record.message}"

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

            with time_machine.travel("2020-01-03 12:00:00+00", tick=False):
                ejr.delete_job(job_id=job_id, user_id="john")

        logs = caplog.text.strip().split("\n")

        for expected in [
            # Create
            "openeo_driver.jobregistry.elastic:j-123:EJR creating job_id='j-123' created='2020-01-02T03:04:05Z'",
            "openeo_driver.jobregistry.elastic:j-123:EJR Request `POST /jobs`: start 2020-01-02 03:04:05",
            "openeo_driver.jobregistry.elastic:j-123:Doing EJR request `POST https://ejr.test/jobs` params=None headers.keys()=dict_keys(['Authorization'])",
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
            # delete
            "openeo_driver.jobregistry.elastic:j-123:EJR Request `DELETE /jobs/j-123`: start 2020-01-03 12:00:00",
            "openeo_driver.jobregistry.elastic:j-123:EJR deleted job_id='j-123'",
        ]:
            assert expected in logs

    def test_with_extra_logging(self, requests_mock, oidc_mock, ejr, caplog):
        """Test that "extra logging fields" (like job_id) do not leak outside of context"""
        caplog.set_level(logging.DEBUG)
        caplog.handler.addFilter(ExtraLoggingFilter())

        class Formatter:
            def format(self, record: logging.LogRecord):
                job_id = getattr(record, "job_id", None)
                return f"{record.name} [{job_id}] {record.message}"

        with caplog_with_custom_formatter(caplog=caplog, format=Formatter()):
            # Trigger failure during _with_extra_logging
            requests_mock.post(f"{self.EJR_API_URL}/jobs/search", status_code=500)
            with pytest.raises(EjrApiResponseError):
                _ = ejr.get_job(job_id="job-123", user_id="john")

            # Health check should not be logged with job id in logs
            requests_mock.get(f"{self.EJR_API_URL}/health", json={"ok": "yep"})
            ejr.health_check(use_auth=False, log=True)

        logs = caplog.text.strip().split("\n")
        assert "openeo_driver.jobregistry.elastic [job-123] EJR get job data job_id='job-123' user_id='john'" in logs
        assert "openeo_driver.jobregistry.elastic [None] EJR health check {'ok': 'yep'}" in logs

    def test_connection_error_logging(self, requests_mock, oidc_mock, ejr, caplog):
        """Test connection/timeout errors, which happen before any response is received."""
        requests_mock.post(
            f"{self.EJR_API_URL}/jobs/search",
            exc=requests.ConnectionError("Connection aborted"),
        )

        with pytest.raises(EjrApiError, match="Failed to do EJR API request"):
            _ = ejr.get_job(job_id="job-123", user_id="john")

        assert (
            f"Failed to do EJR API request `POST {self.EJR_API_URL}/jobs/search`: ConnectionError('Connection aborted')"
            in caplog.text
        )

    @pytest.mark.parametrize(
        ["tries", "failures", "success"],
        [
            (2, 1, True),
            (2, 2, False),
            (3, 2, True),
        ],
    )
    def test_retry(self, requests_mock, oidc_mock, ejr, tries, failures, success):
        """Test retry logic (on search)"""

        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            # TODO: what to return? What does API return?  https://github.com/Open-EO/openeo-job-tracker-elastic-api/issues/3
            return [DUMMY_PROCESS]

        requests_mock.post(
            f"{self.EJR_API_URL}/jobs/search",
            [{"exc": requests.exceptions.ConnectTimeout}] * failures + [{"json": post_jobs_search}],
        )

        with config_overrides(ejr_retry_settings={"tries": tries}), mock.patch("time.sleep") as sleep:
            if success:
                result = ejr.list_user_jobs(user_id="john")
                assert result == [DUMMY_PROCESS]
            else:
                with pytest.raises(
                    EjrApiError, match=f"Failed to do EJR API request `POST {self.EJR_API_URL}/jobs/search`"
                ):
                    _ = ejr.list_user_jobs(user_id="john")

        assert sleep.call_count > 0
