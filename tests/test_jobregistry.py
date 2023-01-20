import logging
from typing import Union

import pytest
import requests
import time_machine
from openeo.rest.auth.testing import OidcMock

from openeo_driver.jobregistry import JOB_STATUS, EjrError, ElasticJobRegistry
from openeo_driver.testing import DictSubSet, RegexMatcher

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
            expected_fields={"client_secret": self.OIDC_CLIENT_INFO["client_secret"]},
        )
        return oidc_mock

    @pytest.fixture
    def ejr(self, oidc_mock) -> ElasticJobRegistry:
        """ElasticJobRegistry set up with authentication"""
        ejr = ElasticJobRegistry(backend_id="unittests", api_url=self.EJR_API_URL)
        ejr.setup_auth_oidc_client_credentials(
            oidc_issuer=self.OIDC_CLIENT_INFO["oidc_issuer"],
            client_id=self.OIDC_CLIENT_INFO["client_id"],
            client_secret=self.OIDC_CLIENT_INFO["client_secret"],
        )
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

    def _auth_is_valid(self, oidc_mock: OidcMock, request: requests.Request) -> bool:
        access_token = oidc_mock.state["access_token"]
        return request.headers["Authorization"] == f"Bearer {access_token}"

    def test_health_check(self, requests_mock, oidc_mock, ejr):
        def get_health(request, context):
            if "Authorization" not in request.headers:
                status, state = "down", "missing"
            elif self._auth_is_valid(oidc_mock=oidc_mock, request=request):
                status, state = "up", "ok"
            else:
                status, state = "down", "expired"
            return {"info": {"auth": {"status": status, "state": state}}}

        requests_mock.get(f"{self.EJR_API_URL}/health", json=get_health)

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

    def test_list_user_jobs(self, requests_mock, oidc_mock, ejr):
        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            assert self._auth_is_valid(oidc_mock=oidc_mock, request=request)
            # TODO: what to return? What does API return?  https://github.com/Open-EO/openeo-job-tracker-elastic-api/issues/3
            return [DUMMY_PROCESS]

        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=post_jobs_search)

        result = ejr.list_user_jobs(user_id="john")
        assert result == [DUMMY_PROCESS]

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
