import pytest

from openeo_driver.testing import TEST_USER, ApiTester
from openeo_driver.users.auth import HttpAuthHandler
from .data import TEST_DATA_ROOT


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
            "Process with identifier 'fluxbormav' is not available in namespace 'None'.",
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


@pytest.mark.parametrize("namespace", ["user", None, "_undefined"])
def test_validate_user_defined_process(api100, udp_registry, namespace):
    udp_registry.save(
        user_id=TEST_USER,
        process_id="my_add",
        spec={
            "id": "my_add",
            "parameters": [{"name": "x"}],
            "process_graph": {
                "add": {
                    "process_id": "add",
                    "arguments": {"x": {"from_parameter": "x"}, "y": 2},
                    "result": True,
                }
            },
        },
    )
    node = {"process_id": "my_add", "arguments": {"x": 3}, "result": True}
    if namespace != "_undefined":
        node["namespace"] = namespace
    assert api100.validation({"udp": node}).json == {"errors": []}


def test_validate_user_defined_process_is_user_scoped(api100, udp_registry):
    udp_registry.save(
        user_id=TEST_USER,
        process_id="my_add",
        spec={
            "id": "my_add",
            "process_graph": {"add": {"process_id": "add", "arguments": {"x": 1, "y": 2}, "result": True}},
        },
    )
    pg = {"udp": {"process_id": "my_add", "namespace": "user", "arguments": {}, "result": True}}
    api100.set_auth_bearer_token("basic//" + HttpAuthHandler.build_basic_access_token(user_id="Other"))
    assert api100.validation(pg).json["errors"][0]["code"] == "ProcessUnsupported"


def test_validate_user_defined_process_invalid_graph(api100, udp_registry):
    udp_registry.save(
        user_id=TEST_USER,
        process_id="my_process",
        spec={
            "id": "my_process",
            "process_graph": {"unknown": {"process_id": "unknown", "arguments": {}, "result": True}},
        },
    )
    pg = {"udp": {"process_id": "my_process", "namespace": "user", "arguments": {}, "result": True}}
    assert api100.validation(pg).json["errors"][0]["message"] == (
        "Process with identifier 'unknown' is not available in namespace 'None'."
    )


def test_validate_user_defined_process_other_namespace(api100, udp_registry):
    udp_registry.save(user_id=TEST_USER, process_id="my_process", spec={"id": "my_process"})
    pg = {"udp": {"process_id": "my_process", "namespace": "backend", "arguments": {}, "result": True}}
    assert api100.validation(pg).json["errors"][0]["code"] == "ProcessUnsupported"


def test_validation_without_auth(api100):
    pg = {"add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}}
    assert api100.validation(pg, ensure_auth=False).json == {"errors": []}


def test_validation_invalid_auth(api100):
    pg = {"add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}}
    api100.set_auth_bearer_token("invalid")
    api100.post("/validation", json={"process_graph": pg}).assert_error(403, "TokenInvalid")


def test_validation_checks_user_access(api100):
    pg = {"add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}}
    api100.set_auth_bearer_token("basic//" + HttpAuthHandler.build_basic_access_token(user_id="Mark"))
    api100.post("/validation", json={"process_graph": pg}).assert_error(403, "PermissionsInsufficient")


def test_load_collection_basic(api100, backend_implementation):
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


def test_validate_zero_area(api100, backend_implementation):
    pg = {
        "lc": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR", "spatial_extent": {"west": 1, "east": 1, "south": 1, "north": 1}},
            "result": True,
        }
    }
    res = api100.validation(pg)
    errors = res.json["errors"]
    assert errors == []


@pytest.mark.parametrize(
    ["spatial_extent", "expected_message_part"],
    [
        ([1, 2, 3, 4], "Expected dictionary/mapping but got"),
        (
            {"west": [0], "south": 60.11, "east": 25.24, "north": 60.35},
            "'west' must be a number, but got [0].",
        ),
        (
            {"west": 5, "south": 51.215, "east": 4, "north": 51.22},
            "'west' must be smaller than 'east'",
        ),
        (
            {"west": 4, "south": 51.22, "east": 5, "north": 51.215},
            "'south' must be smaller than 'north'",
        ),
        (
            {
                "west": 4329317.717132108,
                "east": 4330615.2810456185,
                "north": 3005295.0854642093,
                "south": 3003997.791438847,
            },
            "outside the valid EPSG:4326 range while no 'crs' was specified",
        ),
    ],
)
def test_validation_load_collection_invalid_spatial_extent(api100, spatial_extent, expected_message_part):
    pg = {
        "lc": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR", "spatial_extent": spatial_extent},
            "result": True,
        }
    }
    res = api100.validation(pg)
    errors = res.json["errors"]
    assert len(errors) == 1
    assert errors[0]["code"] == "ProcessParameterInvalid"
    assert expected_message_part in errors[0]["message"]


@pytest.mark.parametrize(
    "spatial_extent",
    [
        {"west": -200, "south": 51.215, "east": -190, "north": 51.22},
        {"west": 4, "south": 95, "east": 5, "north": 96},
    ],
)
def test_validation_load_collection_spatial_extent_lenient_epsg4326_default_bounds(api100, spatial_extent):
    pg = {
        "lc": {
            "process_id": "load_collection",
            "arguments": {"id": "S2_FOOBAR", "spatial_extent": spatial_extent},
            "result": True,
        }
    }
    res = api100.validation(pg)
    errors = res.json["errors"]
    assert all(error["code"] != "ProcessParameterInvalid" for error in errors)
