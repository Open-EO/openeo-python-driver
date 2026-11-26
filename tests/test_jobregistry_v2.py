import argparse
import datetime
import json
import urllib.parse
from decimal import Decimal
from unittest import mock

import pytest
import requests
import time_machine
from openeo.rest.auth.testing import OidcMock

from openeo_driver.errors import JobNotFoundException, OpenEOApiException
from openeo_driver.jobregistry import ElasticJobRegistry, EjrApiError, EjrApiResponseError
from openeo_driver.jobregistry_v2 import ElasticJobRegistryV2
from openeo_driver.testing import config_overrides
from openeo_driver.util.auth import ClientCredentials

API = "https://ejr.test"
JOBS = API + "/v2/jobs"
PROCESS = {"process_graph": {"add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}}}
ETAG = '"' + "a" * 64 + '"'
JOB = {"job_id": "j-123", "backend_id": "unittests", "user_id": "john", "status": "created"}
CONTINUATION = {
    "as_of": "2026-01-01T00:00:00Z",
    "fingerprint": "a" * 64,
    "position": {"created": 1767225600000, "job_id": "j-123"},
}


@pytest.fixture
def oidc_mock(requests_mock):
    return OidcMock(
        requests_mock=requests_mock,
        oidc_issuer="https://oidc.test",
        expected_grant_type="client_credentials",
        expected_client_id="ejrclient",
        expected_fields={"client_secret": "secret", "scope": "openid"},
    )


@pytest.fixture
def ejr(oidc_mock, requests_mock):
    requests_mock.get(JOBS + "/j-123", json=JOB, headers={"ETag": ETAG})
    registry = ElasticJobRegistryV2(api_url=API, backend_id="unittests")
    registry.setup_auth_oidc_client_credentials(ClientCredentials("https://oidc.test", "ejrclient", "secret"))
    return registry


def test_create_and_get(requests_mock, ejr):
    create = requests_mock.post(JOBS, status_code=201, json=lambda request, context: request.json())
    result = ejr.create_job(job_id="j-123", user_id="john", process=PROCESS)
    assert result["job_id"] == "j-123"
    assert result["status"] == "created"
    assert create.last_request.json()["backend_id"] == "unittests"
    assert create.last_request.headers["Authorization"]
    get = requests_mock.get(JOBS + "/j-123", json=result)
    assert ejr.get_job("j-123", user_id="john") == result
    assert get.last_request.qs == {}


@pytest.mark.parametrize("method", ["get", "delete"])
def test_not_found(requests_mock, ejr, method):
    requests_mock.get(JOBS + "/missing", status_code=404)
    requests_mock.register_uri(method.upper(), JOBS + "/missing", status_code=404)
    with pytest.raises(JobNotFoundException):
        getattr(ejr, method + "_job")("missing", user_id="john")


def test_user_listing_uses_dedicated_endpoint(requests_mock, ejr):
    listing = requests_mock.post(JOBS + "/user", json={"jobs": [JOB], "next_page": None})
    assert ejr.list_user_jobs("john") == [JOB]
    body = listing.last_request.json()
    assert body["backend_id"] == "unittests"
    assert body["user_id"] == "john"
    assert "filters" not in body


def test_bounded_listing_keeps_page_size(requests_mock, ejr):
    search = requests_mock.post(
        JOBS + "/cleanup-candidates",
        [
            {"json": {"jobs": [{"job_id": str(index)} for index in range(1000)], "next_page": CONTINUATION}},
            {"json": {"jobs": [{"job_id": str(index)} for index in range(1000, 2000)], "next_page": None}},
        ],
    )
    jobs = ejr.get_all_started_jobs_before(datetime.datetime(2026, 1, 1), max_count=1500)
    assert len(jobs) == 1500
    assert [request.json()["limit"] for request in search.request_history] == [1000, 1000]
    assert search.last_request.json()["next_page"] == CONTINUATION


@pytest.mark.parametrize("preserialize_process", [False, True])
def test_create_payload_matches_legacy(requests_mock, oidc_mock, preserialize_process):
    payloads = []
    for registry_class, path in [(ElasticJobRegistry, API + "/jobs"), (ElasticJobRegistryV2, JOBS)]:
        registry = registry_class(api_url=API, backend_id="unittests", preserialize_process=preserialize_process)
        registry.setup_auth_oidc_client_credentials(ClientCredentials("https://oidc.test", "ejrclient", "secret"))
        create = requests_mock.post(path, status_code=201, json=lambda request, context: request.json())
        with time_machine.travel("2026-01-01T12:00:00Z", tick=False):
            registry.create_job(
                job_id="j-123",
                user_id="john",
                process=PROCESS,
                title="Test",
                description="Description",
                parent_id="j-parent",
                api_version="1.2.0",
                job_options={"memory": "2G"},
            )
        payloads.append(create.last_request.json())
    assert payloads[0]["backend_id"] == "unittests"
    assert payloads[0] == payloads[1]
    assert payloads[1]["process"] == (json.dumps(PROCESS, separators=(",", ":")) if preserialize_process else PROCESS)


def test_generated_job_id(requests_mock, ejr):
    requests_mock.post(JOBS, status_code=201, json=lambda request, context: request.json())
    job = ejr.create_job(user_id="john", process=PROCESS)
    assert job["job_id"].startswith("j-")
    assert job["created"] == job["updated"]


@pytest.mark.parametrize("process", [PROCESS, json.dumps(PROCESS)])
@pytest.mark.parametrize("user_id", [None, "john"])
def test_get_deserializes_process(requests_mock, ejr, process, user_id):
    get = requests_mock.get(JOBS + "/j-123", json={**JOB, "process": process})
    assert ejr.get_job("j-123", user_id=user_id)["process"] == PROCESS
    assert get.last_request.qs == {}


def test_get_checks_user_id_locally(requests_mock, ejr):
    get = requests_mock.get(JOBS + "/j-123", json=JOB)
    with pytest.raises(JobNotFoundException):
        ejr.get_job("j-123", user_id="alice")
    assert get.last_request.qs == {}


@pytest.mark.parametrize("operation", ["get", "delete"])
def test_other_user_job_is_not_found(requests_mock, ejr, operation):
    """Other user's job: not found (like legacy API), instead of AssertionError (HTTP 500)."""
    mutation = requests_mock.delete(JOBS + "/j-123", status_code=204)
    with pytest.raises(JobNotFoundException):
        getattr(ejr, operation + "_job")("j-123", user_id="alice")
    assert mutation.call_count == 0


def test_get_conflict_propagates(requests_mock, ejr):
    requests_mock.get(JOBS + "/j-123", status_code=409)
    with pytest.raises(EjrApiResponseError) as error:
        ejr.get_job("j-123")
    assert error.value.status_code == 409


@pytest.mark.parametrize("method", ["get", "delete", "patch"])
def test_server_error_propagates(requests_mock, ejr, method):
    requests_mock.register_uri(method.upper(), JOBS + "/j-123", status_code=503, json={"message": "unavailable"})
    with pytest.raises(EjrApiResponseError) as error:
        if method == "patch":
            ejr.set_application_id("j-123", application_id="app-123")
        else:
            getattr(ejr, method + "_job")("j-123")
    assert error.value.status_code == 503


@pytest.mark.parametrize(
    ["method", "kwargs"],
    [
        ("set_status", {"status": "running"}),
        (
            "set_status",
            {
                "status": "finished",
                "updated": "2026-01-01T12:00:00",
                "started": "2026-01-01T11:00:00",
                "finished": "2026-01-01T12:00:00",
                "infra_id": "infra-123",
            },
        ),
        ("set_dependencies", {"dependencies": [{"id": "dep-123"}]}),
        ("remove_dependencies", {}),
        ("set_dependency_status", {"dependency_status": "awaiting"}),
        ("set_dependency_usage", {"dependency_usage": Decimal("123.4567890123456789")}),
        ("set_proxy_user", {"proxy_user": "proxy-john"}),
        ("set_application_id", {"application_id": "app-123"}),
        ("set_results_metadata_uri", {"results_metadata_uri": "s3://bucket/job_metadata.json"}),
        (
            "set_results_metadata",
            {
                "costs": 22,
                "usage": {"input_pixel": {"value": 2.5, "unit": "mega-pixel"}},
                "results_metadata": {"unique_process_ids": ["load_stac"]},
            },
        ),
        ("set_results_metadata", {"costs": None, "usage": {}}),
    ],
)
def test_update_payload_matches_legacy(requests_mock, oidc_mock, method, kwargs):
    payloads = []
    requests_mock.get(JOBS + "/j-123", json=JOB, headers={"ETag": ETAG})
    for registry_class, path in [(ElasticJobRegistry, API + "/jobs"), (ElasticJobRegistryV2, JOBS)]:
        registry = registry_class(api_url=API, backend_id="unittests")
        registry.setup_auth_oidc_client_credentials(ClientCredentials("https://oidc.test", "ejrclient", "secret"))
        patch = requests_mock.patch(path + "/j-123", json=lambda request, context: request.json())
        with time_machine.travel("2026-01-01T12:00:00Z", tick=False):
            assert getattr(registry, method)(job_id="j-123", **kwargs) is None
        payloads.append(patch.last_request.json())
    assert payloads[0] == payloads[1]


def test_delete_checks_user_id_locally_and_verifies(requests_mock, ejr):
    delete = requests_mock.delete(JOBS + "/j-123", status_code=204)
    get = requests_mock.get(
        JOBS + "/j-123",
        [
            {"json": JOB, "headers": {"ETag": ETAG}},
            {"status_code": 404},
        ],
    )
    ejr.delete_job("j-123", user_id="john")
    assert delete.call_count == 1
    assert get.call_count == 2
    assert delete.last_request.headers["If-Match"] == ETAG
    assert delete.last_request.qs == get.last_request.qs == {}


def test_delete_without_verification(requests_mock, ejr):
    delete = requests_mock.delete(JOBS + "/j-123", status_code=204)
    ejr.delete_job("j-123", verify_deletion=False)
    assert delete.call_count == 1
    assert delete.last_request.headers["If-Match"] == ETAG
    assert delete.last_request.qs == {}


@pytest.mark.parametrize("listing", ["user", "active", "cleanup-candidates"])
def test_non_paginated_listing_consumes_next_pages(requests_mock, ejr, listing):
    search = requests_mock.post(
        JOBS + "/" + listing,
        [
            {"json": {"jobs": [{"job_id": "j-1"}], "next_page": CONTINUATION}},
            {"json": {"jobs": [{"job_id": "j-2"}], "next_page": None}},
        ],
    )
    if listing == "user":
        result = ejr.list_user_jobs("john", fields=["started"])
    elif listing == "active":
        result = ejr.list_active_jobs(fields=["started"])
    else:
        result = ejr.get_all_started_jobs_before(datetime.datetime(2026, 1, 1), field_whitelist=["started"])
    assert result == [{"job_id": "j-1"}, {"job_id": "j-2"}]
    first, second = [request.json() for request in search.request_history]
    assert "next_page" not in first
    assert second.pop("next_page") == CONTINUATION
    assert second == first
    assert set(first["fields"]) == {"job_id", "user_id", "created", "status", "updated", "started"}
    assert first["backend_id"] == "unittests"
    if listing == "user":
        assert first["user_id"] == "john"
    elif listing == "active":
        assert first["require_application_id"] is False
    else:
        assert first["updated_until"] == "2026-01-01T00:00:00Z"
    assert "filters" not in first
    assert "query" not in first


@pytest.mark.parametrize("next_page", [None, CONTINUATION])
def test_paginated_listing(requests_mock, ejr, next_page):
    search = requests_mock.post(
        JOBS + "/user",
        json={
            "jobs": [{"job_id": "j-123", "status": "created", "created": "2026-01-01T12:00:00Z"}],
            "next_page": next_page,
        },
    )
    listing = ejr.list_user_jobs("john", limit=2, request_parameters={"cursor": json.dumps(CONTINUATION)})
    response = listing.to_response_dict(build_url=lambda params: "/jobs?" + urllib.parse.urlencode(params))
    assert response["jobs"] == [{"id": "j-123", "status": "created", "created": "2026-01-01T12:00:00Z", "progress": 0}]
    if next_page:
        assert len(response["links"]) == 1
        assert response["links"][0]["rel"] == "next"
        query = urllib.parse.parse_qs(urllib.parse.urlparse(response["links"][0]["href"]).query)
        assert query["limit"] == ["2"]
        assert json.loads(query["cursor"][0]) == next_page
    else:
        assert response["links"] == []
    assert search.last_request.json()["next_page"] == CONTINUATION
    assert search.last_request.json()["limit"] == 2


@pytest.mark.parametrize("position", [CONTINUATION["position"], {"created": None, "job_id": None}])
def test_next_link_round_trip(requests_mock, ejr, position):
    continuation = {**CONTINUATION, "position": position}
    job = {**JOB, "created": "2026-01-01T00:00:00Z"}
    listing = requests_mock.post(
        JOBS + "/user",
        [
            {"json": {"jobs": [job], "next_page": continuation}},
            {"json": {"jobs": [{**job, "job_id": "j-456"}], "next_page": None}},
        ],
    )
    first_page = ejr.list_user_jobs("john", limit=1, fields=["title"])
    response = first_page.to_response_dict(build_url=lambda params: "/jobs?" + urllib.parse.urlencode(params))
    params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(response["links"][0]["href"]).query))
    second_page = ejr.list_user_jobs("john", limit=int(params["limit"]), fields=["title"], request_parameters=params)
    result = second_page.to_response_dict(build_url=lambda params: "/jobs")
    assert [job["id"] for job in result["jobs"]] == ["j-456"]
    assert result["links"] == []
    first, second = [request.json() for request in listing.request_history]
    assert "next_page" not in first
    assert second.pop("next_page") == continuation
    assert first == second


def test_active_continuation_preserves_age_parameters(requests_mock, ejr):
    listing = requests_mock.post(
        JOBS + "/active",
        [
            {"json": {"jobs": [JOB], "next_page": CONTINUATION}},
            {"json": {"jobs": [], "next_page": None}},
        ],
    )
    assert ejr.list_active_jobs(max_age=7, max_updated_ago=0, require_application_id=True) == [JOB]
    first, second = [request.json() for request in listing.request_history]
    assert second.pop("next_page") == CONTINUATION
    assert first == second
    assert first["max_age"] == 7
    assert first["max_updated_ago"] == 0
    assert first["require_application_id"] is True


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, {}),
        ({"max_age": 7}, {"max_age": 7}),
        ({"max_updated_ago": 6}, {"max_updated_ago": 6}),
        ({"max_age": 0, "max_updated_ago": 0}, {"max_age": 0, "max_updated_ago": 0}),
        ({"max_age": 0.5}, {"max_age": 0.5}),
        ({"require_application_id": True}, {"require_application_id": True}),
        ({"infra_id": "infra-123"}, {"infra_id": "infra-123"}),
        (
            {"max_age": 7, "max_updated_ago": 6, "require_application_id": True, "infra_id": "infra-123"},
            {
                "max_age": 7,
                "max_updated_ago": 6,
                "require_application_id": True,
                "infra_id": "infra-123",
            },
        ),
    ],
)
def test_active_filters(requests_mock, ejr, kwargs, expected):
    search = requests_mock.post(JOBS + "/active", json={"jobs": [], "next_page": None})
    with time_machine.travel("2026-01-08T12:00:00Z", tick=False):
        assert ejr.list_active_jobs(**kwargs) == []
    body = search.last_request.json()
    assert body == {
        "backend_id": "unittests",
        "require_application_id": False,
        "limit": 1000,
        "fields": ["created", "job_id", "status", "updated", "user_id"],
        **expected,
    }


def test_started_jobs_filters(requests_mock, ejr):
    search = requests_mock.post(JOBS + "/cleanup-candidates", json={"jobs": [], "next_page": None})
    assert (
        ejr.get_all_started_jobs_before(
            datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            user_ids=["john", "alice"],
            field_whitelist=["started"],
            max_count=25,
        )
        == []
    )
    body = search.last_request.json()
    assert body == {
        "backend_id": "unittests",
        "updated_until": "2026-01-01T00:00:00Z",
        "user_ids": ["john", "alice"],
        "limit": 25,
        "fields": ["created", "job_id", "started", "status", "updated", "user_id"],
    }
    assert search.last_request.json()["limit"] == 25
    assert "started" in search.last_request.json()["fields"]


def test_zero_max_count_does_not_search(ejr):
    assert ejr.get_all_started_jobs_before(datetime.datetime(2026, 1, 1), max_count=0) == []


def test_wildcard_fields(requests_mock, ejr):
    search = requests_mock.post(JOBS + "/user", json={"jobs": [], "next_page": None})
    assert ejr.list_user_jobs("john", fields=["*"]) == []
    assert "fields" not in search.last_request.json()


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_page_size(requests_mock, ejr, limit):
    search = requests_mock.post(JOBS + "/user", json={"jobs": [], "next_page": None})
    with pytest.raises(OpenEOApiException) as error:
        ejr.list_user_jobs("john", limit=limit)
    assert error.value.status_code == 400
    assert error.value.code == "InvalidPagination"
    assert search.call_count == 0


@pytest.mark.parametrize("limit", [1001, 5000])
def test_large_page_size_is_capped(requests_mock, ejr, limit):
    search = requests_mock.post(
        JOBS + "/user", json={"jobs": [{**JOB, "created": "2026-01-01T00:00:00Z"}], "next_page": CONTINUATION}
    )
    listing = ejr.list_user_jobs("john", limit=limit)
    assert search.last_request.json()["limit"] == 1000
    assert listing._next_parameters["limit"] == 1000


@pytest.mark.parametrize("limit", [0, 1001])
def test_internal_listing_limit_still_validated(ejr, limit):
    with pytest.raises(ValueError, match="between 1 and 1000"):
        ejr._list_jobs("user", {"user_id": "john"}, limit=limit)


@pytest.mark.parametrize(
    "cursor",
    [
        "not json",
        "",
        "123",
        '["a"]',
        "null",
        '{"position": {"created": 1, "job_id": "j"}}',
        '{"as_of": 123, "position": {}}',
        '{"as_of": "2026-01-01T00:00:00Z", "position": "x"}',
    ],
)
def test_malformed_cursor(requests_mock, ejr, cursor):
    search = requests_mock.post(JOBS + "/user", json={"jobs": [], "next_page": None})
    with pytest.raises(OpenEOApiException) as error:
        ejr.list_user_jobs("john", limit=10, request_parameters={"cursor": cursor})
    assert error.value.status_code == 400
    assert error.value.code == "InvalidPagination"
    assert search.call_count == 0


def test_cursor_rejected_by_ejr(requests_mock, ejr):
    requests_mock.post(JOBS + "/user", status_code=400, json={"message": "bad continuation"})
    cursor = json.dumps({"as_of": "2026-01-01T00:00:00Z", "position": {"created": "x", "job_id": None}})
    with pytest.raises(OpenEOApiException) as error:
        ejr.list_user_jobs("john", limit=10, request_parameters={"cursor": cursor})
    assert error.value.status_code == 400
    assert error.value.code == "InvalidPagination"


def test_ejr_400_without_cursor_propagates(requests_mock, ejr):
    requests_mock.post(JOBS + "/user", status_code=400, json={"message": "bad request"})
    with pytest.raises(EjrApiResponseError) as error:
        ejr.list_user_jobs("john", limit=10)
    assert error.value.status_code == 400


@pytest.mark.parametrize("persistent", [False, True])
def test_401_refreshes_token_once(requests_mock, oidc_mock, ejr, persistent):
    search = requests_mock.post(
        JOBS + "/user",
        [
            {"status_code": 401},
            {"status_code": 401} if persistent else {"json": {"jobs": [], "next_page": None}},
        ],
    )
    if persistent:
        with pytest.raises(EjrApiResponseError) as error:
            ejr.list_user_jobs("john")
        assert error.value.status_code == 401
    else:
        assert ejr.list_user_jobs("john") == []
    assert search.call_count == 2
    assert len(oidc_mock.get_request_history(url="/token")) == 2


def test_access_token_caching(requests_mock, oidc_mock, ejr):
    requests_mock.post(JOBS + "/user", json={"jobs": [], "next_page": None})
    for timestamp, expected_tokens in [
        ("2026-01-01T12:00:00Z", 1),
        ("2026-01-01T12:01:00Z", 1),
        ("2026-01-02T12:00:00Z", 2),
    ]:
        with time_machine.travel(timestamp, tick=False):
            assert ejr.list_user_jobs("john") == []
            assert len(oidc_mock.get_request_history(url="/token")) == expected_tokens


def test_listing_connection_retry(requests_mock, ejr):
    search = requests_mock.post(
        JOBS + "/user",
        [
            {"exc": requests.exceptions.ConnectTimeout},
            {"json": {"jobs": [], "next_page": None}},
        ],
    )
    with config_overrides(ejr_retry_settings={"tries": 2}), mock.patch("time.sleep"):
        assert ejr.list_user_jobs("john") == []
    assert search.call_count == 2


def test_connection_error(requests_mock, ejr):
    requests_mock.get(JOBS + "/j-123", exc=requests.ConnectionError("Connection aborted"))
    with pytest.raises(EjrApiError, match="Failed to do EJR API request"):
        ejr.get_job("j-123")


def test_health_uses_shared_endpoint(requests_mock, ejr):
    health = requests_mock.get(API + "/health", json={"info": {"auth": {"status": "up"}}})
    assert ejr.health_check()["info"]["auth"]["status"] == "up"
    assert health.last_request.headers["Authorization"]
    ejr.health_check(use_auth=False)
    assert "Authorization" not in health.last_request.headers


def test_global_job_path_encoding(requests_mock, oidc_mock):
    registry = ElasticJobRegistryV2(api_url=API, backend_id="backend/name")
    registry.setup_auth_oidc_client_credentials(ClientCredentials("https://oidc.test", "ejrclient", "secret"))
    job = {"job_id": "job/name", "backend_id": "backend/name"}
    get = requests_mock.get(JOBS + "/job%2Fname", json=job)
    assert registry.get_job("job/name") == job
    assert get.call_count == 1


@pytest.mark.parametrize("method", ["patch", "delete"])
def test_conditional_write_uses_current_etag(requests_mock, ejr, method):
    mutation = requests_mock.register_uri(
        method.upper(),
        JOBS + "/j-123",
        status_code=204 if method == "delete" else 200,
        json=None if method == "delete" else JOB,
    )
    if method == "patch":
        ejr.set_application_id("j-123", application_id="app-123")
        assert mutation.last_request.json() == {"application_id": "app-123"}
    else:
        ejr.delete_job("j-123", verify_deletion=False)
    assert mutation.last_request.headers["If-Match"] == ETAG


def _do_mutation(ejr, method: str):
    if method == "patch":
        ejr.set_application_id("j-123", application_id="app-123")
    else:
        ejr.delete_job("j-123", verify_deletion=False)


@pytest.mark.parametrize("method", ["patch", "delete"])
def test_stale_etag_is_retried_with_fresh_etag(requests_mock, ejr, method):
    """Concurrent change (412): re-read current ETag and retry (last writer wins, like legacy API)."""
    next_etag = '"' + "b" * 64 + '"'
    get = requests_mock.get(
        JOBS + "/j-123",
        [
            {"json": JOB, "headers": {"ETag": ETAG}},
            {"json": {**JOB, "status": "running"}, "headers": {"ETag": next_etag}},
        ],
    )
    ok = {"status_code": 204} if method == "delete" else {"json": JOB}
    mutation = requests_mock.register_uri(method.upper(), JOBS + "/j-123", [{"status_code": 412}, ok])
    _do_mutation(ejr, method)
    assert get.call_count == 2
    assert [r.headers["If-Match"] for r in mutation.request_history] == [ETAG, next_etag]


@pytest.mark.parametrize("method", ["patch", "delete"])
def test_persistent_stale_etag_gives_up(requests_mock, ejr, method):
    get = requests_mock.get(JOBS + "/j-123", json=JOB, headers={"ETag": ETAG})
    mutation = requests_mock.register_uri(method.upper(), JOBS + "/j-123", status_code=412)
    with pytest.raises(EjrApiResponseError) as error:
        _do_mutation(ejr, method)
    assert error.value.status_code == 412
    assert mutation.call_count == ElasticJobRegistryV2._PRECONDITION_FAILED_ATTEMPTS
    assert get.call_count == ElasticJobRegistryV2._PRECONDITION_FAILED_ATTEMPTS


@pytest.mark.parametrize("method", ["patch", "delete"])
@pytest.mark.parametrize("status_code", [400, 409, 500])
def test_other_write_errors_are_not_retried(requests_mock, ejr, method, status_code):
    mutation = requests_mock.register_uri(method.upper(), JOBS + "/j-123", status_code=status_code)
    with pytest.raises(EjrApiResponseError) as error:
        _do_mutation(ejr, method)
    assert error.value.status_code == status_code
    assert mutation.call_count == 1


def test_delete_retry_after_concurrent_deletion(requests_mock, ejr):
    """Job deleted concurrently between 412 and re-read: not found."""
    requests_mock.get(JOBS + "/j-123", [{"json": JOB, "headers": {"ETag": ETAG}}, {"status_code": 404}])
    mutation = requests_mock.delete(JOBS + "/j-123", status_code=412)
    with pytest.raises(JobNotFoundException):
        ejr.delete_job("j-123", verify_deletion=False)
    assert mutation.call_count == 1


def test_results_metadata_413_propagates(requests_mock, ejr):
    """Job tracker relies on 413 to retry with smaller results metadata."""
    requests_mock.patch(JOBS + "/j-123", status_code=413)
    with pytest.raises(EjrApiResponseError) as error:
        ejr.set_results_metadata("j-123", costs=1.0, usage={}, results_metadata={"links": []})
    assert error.value.status_code == 413


@pytest.mark.parametrize("method", ["patch", "delete"])
def test_missing_etag_prevents_write(requests_mock, ejr, method):
    requests_mock.get(JOBS + "/j-123", json=JOB)
    mutation = requests_mock.register_uri(method.upper(), JOBS + "/j-123", status_code=200)
    with pytest.raises(EjrApiError, match="missing an ETag"):
        if method == "patch":
            ejr.set_application_id("j-123", application_id="app-123")
        else:
            ejr.delete_job("j-123", verify_deletion=False)
    assert mutation.call_count == 0


@pytest.mark.parametrize("operation", ["get", "patch", "delete"])
def test_other_backend_job_is_not_accessible(requests_mock, ejr, operation):
    requests_mock.get(JOBS + "/j-123", json={**JOB, "backend_id": "other-backend"}, headers={"ETag": ETAG})
    with pytest.raises(JobNotFoundException):
        if operation == "patch":
            ejr.set_application_id("j-123", application_id="app-123")
        else:
            getattr(ejr, operation + "_job")("j-123")


@pytest.mark.parametrize("method", ["get", "patch", "delete"])
@pytest.mark.parametrize("persistent", [False, True])
def test_versioned_request_refreshes_token_once(requests_mock, oidc_mock, ejr, method, persistent):
    response = {"status_code": 204} if method == "delete" else {"json": JOB, "headers": {"ETag": ETAG}}
    endpoint = requests_mock.register_uri(
        method.upper(),
        JOBS + "/j-123",
        [
            {"status_code": 401},
            {"status_code": 401} if persistent else response,
        ],
    )

    def call():
        if method == "patch":
            ejr.set_application_id("j-123", application_id="app-123")
        elif method == "delete":
            ejr.delete_job("j-123", verify_deletion=False)
        else:
            assert ejr.get_job("j-123") == JOB

    if persistent:
        with pytest.raises(EjrApiResponseError) as error:
            call()
        assert error.value.status_code == 401
    else:
        call()
    assert endpoint.call_count == 2
    assert len(oidc_mock.get_request_history(url="/token")) == 2
    if method != "get":
        assert [request.headers["If-Match"] for request in endpoint.request_history] == [ETAG, ETAG]


def test_each_update_reads_current_etag(requests_mock, ejr):
    next_etag = '"' + "b" * 64 + '"'
    get = requests_mock.get(
        JOBS + "/j-123",
        [
            {"json": JOB, "headers": {"ETag": ETAG}},
            {"json": JOB, "headers": {"ETag": next_etag}},
        ],
    )
    patch = requests_mock.patch(JOBS + "/j-123", json=JOB)
    ejr.set_application_id("j-123", application_id="app-123")
    ejr.set_proxy_user("j-123", proxy_user="john")
    assert get.call_count == 2
    assert [request.headers["If-Match"] for request in patch.request_history] == [ETAG, next_etag]
    assert all("If-Match" not in request.headers for request in get.request_history)


class TestApiVersionSelection:
    def test_default_is_v1(self):
        from openeo_driver.jobregistry import build_elastic_job_registry, get_elastic_job_registry_class

        assert get_elastic_job_registry_class() is ElasticJobRegistry
        registry = build_elastic_job_registry(api_url=API, backend_id="unittests")
        assert type(registry) is ElasticJobRegistry

    @pytest.mark.parametrize(
        ["api_version", "expected"],
        [("v1", ElasticJobRegistry), ("v2", ElasticJobRegistryV2)],
    )
    def test_explicit_version(self, api_version, expected):
        from openeo_driver.jobregistry import build_elastic_job_registry

        registry = build_elastic_job_registry(api_url=API, backend_id="unittests", api_version=api_version)
        assert type(registry) is expected

    def test_opt_in_through_config(self):
        from openeo_driver.jobregistry import build_elastic_job_registry

        with config_overrides(ejr_api_version="v2"):
            assert type(build_elastic_job_registry(api_url=API, backend_id="unittests")) is ElasticJobRegistryV2
            # Explicit argument takes precedence over config
            assert type(build_elastic_job_registry(api_url=API, backend_id="u", api_version="v1")) is ElasticJobRegistry

    def test_kwargs_are_passed(self):
        from openeo_driver.jobregistry import build_elastic_job_registry

        registry = build_elastic_job_registry(
            api_url=API, backend_id="unittests", api_version="v2", preserialize_process=True
        )
        assert registry._preserialize_process is True
        assert registry.backend_id == "unittests"

    def test_invalid_version(self):
        from openeo_driver.jobregistry import build_elastic_job_registry

        with pytest.raises(ValueError, match="Unsupported EJR API version 'v3'"):
            build_elastic_job_registry(api_url=API, backend_id="unittests", api_version="v3")

    def test_invalid_config_version(self):
        from openeo_driver.config import OpenEoBackendConfig

        with pytest.raises(ValueError):
            OpenEoBackendConfig(ejr_api_version="v3")

    @pytest.mark.parametrize(
        ["environ", "expected"],
        [
            ({}, ElasticJobRegistry),
            ({"OPENEO_EJR_API_VERSION": ""}, ElasticJobRegistry),
            ({"OPENEO_EJR_API_VERSION": "v1"}, ElasticJobRegistry),
            ({"OPENEO_EJR_API_VERSION": "v2"}, ElasticJobRegistryV2),
        ],
    )
    def test_cli(self, environ, expected):
        from openeo_driver.jobregistry import CliApp

        app = CliApp(environ={"OPENEO_EJR_API": API, **environ})
        registry = app._get_job_registry(
            setup_auth=False, cli_args=argparse.Namespace(backend_id="unittests", show_curl=False)
        )
        assert type(registry) is expected
