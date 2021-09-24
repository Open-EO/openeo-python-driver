import pytest

from openeo_driver.testing import ApiTester
from .data import get_path, TEST_DATA_ROOT, load_json


@pytest.fixture
def api100(client) -> ApiTester:
    data_root = TEST_DATA_ROOT / "pg" / "1.0"
    return ApiTester(api_version="1.0.0", client=client, data_root=data_root)


def test_basic_ok(api100):
    pg = {"add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}}
    res = api100.validation(pg)
    assert res.json == {"errors": []}


@pytest.mark.parametrize(["pg", "expected_code", "expected_message"], [
    ({}, "ProcessGraphInvalid", "No result node in process graph: {}"),
    (
            {"add": {"process_id": "fluxbormav", "arguments": {"x": 3, "y": 5}, "result": True}},
            "ProcessUnsupported",
            "Process with identifier 'fluxbormav' is not available in namespace 'backend'.",
    )
])
def test_basic_fail(api100, pg, expected_code, expected_message):
    res = api100.validation(pg)
    errors = res.json["errors"]
    assert errors == [{"code": expected_code, "message": expected_message}]
