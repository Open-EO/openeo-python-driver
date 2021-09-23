import pytest

from openeo_driver.filter_properties import extract_literal_match, PropertyConditionException


def test_extract_literal_match_basic():
    pg = {"process_graph": {"eq": {
        "process_id": "eq",
        "arguments": {"x": {"from_parameter": "value"}, "y": "ASCENDING"},
        "result": True
    }}}
    assert extract_literal_match(pg) == "ASCENDING"


def test_extract_literal_match_reverse():
    pg = {"process_graph": {"eq": {
        "process_id": "eq",
        "arguments": {"x": "ASCENDING", "y": {"from_parameter": "value"}},
        "result": True
    }}}
    assert extract_literal_match(pg) == "ASCENDING"


@pytest.mark.parametrize("arguments", [
    {"x": {"from_parameter": "value"}, "y": {"from_parameter": "value"}},
    {"x": "ASCENDING", "y": "DESCENDING"},
    {"x": "ASCENDING", "y": {"from_parameter": "data"}},
])
def test_extract_literal_match_failures(arguments):
    pg = {"process_graph": {"eq": {
        "process_id": "eq",
        "arguments": arguments,
        "result": True
    }}}
    with pytest.raises(PropertyConditionException):
        extract_literal_match(pg)
