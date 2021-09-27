from typing import List, Iterator
from unittest import mock
import pytest

from openeo_driver.dry_run import SourceConstraint
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
    ),
    (
            {"lc": {"process_id": "load_collection", "arguments": {"id": "flehmeh"}, "result": True}},
            "CollectionNotFound", "Collection 'flehmeh' does not exist."
    )
])
def test_basic_fail(api100, pg, expected_code, expected_message):
    res = api100.validation(pg)
    errors = res.json["errors"]
    assert errors == [{"code": expected_code, "message": expected_message}]


def test_load_collection_basic(api100, backend_implementation):
    def extra_validation(process_graph: dict, result, source_constraints: List[SourceConstraint]) -> Iterator[dict]:
        for source_id, constraints in source_constraints:
            if source_id[0] == "load_collection" and source_id[1][0] == "S2_FOOBAR":
                yield {"code": "MissingProduct", "message": "Tile 4322 not available"}

    backend_implementation.extra_validations.append(extra_validation)

    pg = {
        "lc": {
            "process_id": "load_collection",
            "arguments": {
                "id": "S2_FOOBAR",
                "spatial_extent": {"west": 1, "east": 2, "south": 3, "north": 4},
                "temporal_extent": ["2021-02-01", "2021-02-20"],
            },
            "result": True,
        }
    }
    res = api100.validation(pg)
    errors = res.json["errors"]
    assert errors == [{"code": "MissingProduct", "message": "Tile 4322 not available"}]
