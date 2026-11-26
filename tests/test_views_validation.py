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


@pytest.fixture
def udp_definition():
    return {
        "id": "my_add",
        "description": "Add two to a number.",
        "parameters": [{"name": "x", "description": "Input number.", "schema": {"type": "number"}}],
        "returns": {"description": "Sum.", "schema": {"type": "number"}},
        "process_graph": {
            "add": {
                "process_id": "add",
                "arguments": {"x": {"from_parameter": "x"}, "y": 2},
                "result": True,
            }
        },
    }


@pytest.fixture
def udp_call_definition():
    return {
        "id": "call_my_add",
        "description": "Call the saved user-defined process.",
        "parameters": [],
        "returns": {"description": "Sum.", "schema": {"type": "number"}},
        "process_graph": {"udp": {"process_id": "my_add", "arguments": {"x": 3}, "result": True}},
    }


@pytest.mark.parametrize(
    ["namespace", "user_id", "supported"],
    [
        pytest.param("user", TEST_USER, True, id="user-namespace"),
        pytest.param(None, TEST_USER, True, id="null-namespace"),
        pytest.param("_undefined", TEST_USER, True, id="implicit-namespace"),
        pytest.param("user", "Other", False, id="different-user"),
        pytest.param("backend", TEST_USER, False, id="backend-namespace"),
        pytest.param("user", None, False, id="anonymous"),
    ],
)
def test_validate_user_defined_process(api100, udp_definition, udp_call_definition, namespace, user_id, supported):
    api100.set_auth_bearer_token()
    api100.put("/process_graphs/my_add", json=udp_definition).assert_status_code(200)
    if user_id is None:
        api100.default_request_headers.clear()
    else:
        api100.set_auth_bearer_token("basic//" + HttpAuthHandler.build_basic_access_token(user_id=user_id))

    if namespace != "_undefined":
        udp_call_definition["process_graph"]["udp"]["namespace"] = namespace
    expected_errors = (
        []
        if supported
        else [
            {
                "code": "ProcessUnsupported",
                "message": f"Process with identifier 'my_add' is not available in namespace '{namespace}'.",
            }
        ]
    )
    response = api100.post("/validation", json=udp_call_definition).assert_status_code(200)
    assert response.json == {"errors": expected_errors}


def test_validate_user_defined_process_with_unsupported_inner_process(api100, udp_definition, udp_call_definition):
    udp_definition["process_graph"]["add"]["process_id"] = "unknown"
    api100.set_auth_bearer_token()
    api100.put("/process_graphs/my_add", json=udp_definition).assert_status_code(200)
    udp_call_definition["process_graph"]["udp"]["namespace"] = "user"
    response = api100.post("/validation", json=udp_call_definition).assert_status_code(200)
    assert response.json == {
        "errors": [
            {
                "code": "ProcessUnsupported",
                "message": "Process with identifier 'unknown' is not available in namespace 'None'.",
            }
        ]
    }


@pytest.mark.parametrize(
    ["user_id", "expected_status", "expected_code"],
    [
        pytest.param(None, 200, None, id="anonymous"),
        pytest.param(TEST_USER, 200, None, id="authenticated"),
        pytest.param("invalid-token", 403, "TokenInvalid", id="invalid-token"),
        pytest.param("Mark", 403, "PermissionsInsufficient", id="access-denied"),
    ],
)
def test_validation_authentication(api100, user_id, expected_status, expected_code):
    if user_id == "invalid-token":
        api100.set_auth_bearer_token("invalid")
    elif user_id is not None:
        api100.set_auth_bearer_token("basic//" + HttpAuthHandler.build_basic_access_token(user_id=user_id))
    pg = {"add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}}
    response = api100.post("/validation", json={"process_graph": pg})
    if expected_code:
        response.assert_error(expected_status, expected_code)
    else:
        assert response.assert_status_code(expected_status).json == {"errors": []}


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
