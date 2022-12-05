import pytest
import time_machine

from openeo.rest.auth.testing import OidcMock
from openeo_driver.jobregistry import ElasticJobRegistry
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
        ejr = ElasticJobRegistry(backend_id="test", api_url=self.EJR_API_URL)
        ejr.authenticate_oidc_client_credentials(
            oidc_issuer=self.OIDC_CLIENT_INFO["oidc_issuer"],
            client_id=self.OIDC_CLIENT_INFO["client_id"],
            client_secret=self.OIDC_CLIENT_INFO["client_secret"],
        )
        return ejr

    def test_access_token_caching(self, requests_mock, oidc_mock, ejr):
        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=[])

        def get_token_request_count():
            return sum(
                1
                for r in requests_mock.request_history
                if r.method == "POST" and r.url == oidc_mock.token_endpoint
            )

        with time_machine.travel("2020-01-02 12:00:00+00"):
            result = ejr.list_user_jobs(user_id="john")
            assert result == []
            assert get_token_request_count() == 1

        with time_machine.travel("2020-01-02 12:01:00+00"):
            result = ejr.list_user_jobs(user_id="john")
            assert result == []
            assert get_token_request_count() == 1

        with time_machine.travel("2020-01-03 12:00:00+00"):
            result = ejr.list_user_jobs(user_id="john")
            assert result == []
            assert get_token_request_count() == 2

    def test_create_job(self, requests_mock, oidc_mock, ejr):
        def post_jobs(request, context):
            """Handler of `POST /jobs`"""
            access_token = oidc_mock.state["access_token"]
            assert request.headers["Authorization"] == f"Bearer {access_token}"
            # TODO: what to return? What does API return?  https://github.com/Open-EO/openeo-job-tracker-elastic-api/issues/3
            return request.json()

        requests_mock.post(f"{self.EJR_API_URL}/jobs", json=post_jobs)

        with time_machine.travel("2020-01-02 03:04:05+00", tick=False):
            result = ejr.create_job(process=DUMMY_PROCESS, user_id="john")
        assert result == [
            DictSubSet(
                {
                    "backend_id": "test",
                    "job_id": RegexMatcher("j-[0-9a-f]+"),
                    "user_id": "john",
                    "process": DUMMY_PROCESS,
                    "created": "2020-01-02T03:04:05Z",
                    "updated": "2020-01-02T03:04:05Z",
                    "status": "created",
                    "job_options": None,
                }
            )
        ]

    def test_list_user_jobs(self, requests_mock, oidc_mock, ejr):
        def post_jobs_search(request, context):
            """Handler of `POST /jobs/search"""
            access_token = oidc_mock.state["access_token"]
            assert request.headers["Authorization"] == f"Bearer {access_token}"
            # TODO: what to return? What does API return?  https://github.com/Open-EO/openeo-job-tracker-elastic-api/issues/3
            return [DUMMY_PROCESS]

        requests_mock.post(f"{self.EJR_API_URL}/jobs/search", json=post_jobs_search)

        result = ejr.list_user_jobs(user_id="john")
        assert result == [DUMMY_PROCESS]
