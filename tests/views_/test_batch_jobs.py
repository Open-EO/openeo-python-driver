import pytest

from openeo_driver.urlsigning import UrlSigner
from openeo_driver.views_.batch_jobs import list_job_results_canonical_link, list_job_results_self_link


def test_list_job_results_self_link(flask_app) -> None:
    with flask_app.test_request_context("openeo/1.2/jobs"):
        assert list_job_results_self_link("job-456") == {
            "rel": "self",
            "href": "http://oeo.net/openeo/1.2/jobs/job-456/results",
            "type": "application/json",
        }


def test_list_job_results_self_link_partial(flask_app) -> None:
    with flask_app.test_request_context("openeo/1.2/jobs"):
        assert list_job_results_self_link("job-456", partial=True) == {
            "rel": "self",
            "href": "http://oeo.net/openeo/1.2/jobs/job-456/results?partial=true",
            "type": "application/json",
        }


@pytest.mark.parametrize(
    ["backend_config_overrides", "expected_href"],
    [
        (
            {},
            "http://oeo.net/openeo/1.2/jobs/job-456/results",
        ),
        (
            {"url_signer": UrlSigner(secret="123&@#")},
            "http://oeo.net/openeo/1.2/jobs/job-456/results/dXNlci0xMjM=/5e684ebdb5865f831a7c1b37bfa47d48",
        ),
    ],
)
def test_list_job_results_canonical_link(flask_app, expected_href) -> None:
    with flask_app.test_request_context("openeo/1.2/jobs"):
        assert list_job_results_canonical_link("job-456", user_id="user-123") == {
            "rel": "canonical",
            "href": expected_href,
            "type": "application/json",
        }
