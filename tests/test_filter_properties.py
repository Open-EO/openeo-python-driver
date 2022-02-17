import pytest

from openeo_driver.filter_properties import extract_literal_match, PropertyConditionException


def test_extract_literal_match_basic():
    pg = {"process_graph": {"eq1": {
        "process_id": "eq",
        "arguments": {"x": {"from_parameter": "value"}, "y": "ASCENDING"},
        "result": True
    }}}
    assert extract_literal_match(pg) == {"eq": "ASCENDING"}


def test_extract_literal_match_reverse():
    pg = {"process_graph": {"eq1": {
        "process_id": "eq",
        "arguments": {"x": "ASCENDING", "y": {"from_parameter": "value"}},
        "result": True
    }}}
    assert extract_literal_match(pg) == {"eq": "ASCENDING"}


def test_extract_literal_match_lte():
    pg = {"process_graph": {"lte1": {
        "process_id": "lte",
        "arguments": {"x": {"from_parameter": "value"}, "y": 20},
        "result": True
    }}}
    assert extract_literal_match(pg) == {"lte": 20}


def test_extract_literal_match_gte_reverse():
    pg = {"process_graph": {"gte1": {
        "process_id": "gte",
        "arguments": {"x": 20, "y": {"from_parameter": "value"}},
        "result": True
    }}}
    assert extract_literal_match(pg) == {"lte": 20}


@pytest.mark.parametrize("node", [
    {"process_id": "eq", "arguments": {"x": {"from_parameter": "value"}, "y": {"from_parameter": "value"}}},
    {"process_id": "eq", "arguments": {"x": "ASCENDING", "y": "DESCENDING"}},
    {"process_id": "eq", "arguments": {"x": "ASCENDING", "y": {"from_parameter": "data"}}},
    {"process_id": "between", "arguments": {"x": {"from_parameter": "value"}, "min": 2, "max": 4}},
])
def test_extract_literal_match_failures(node):
    node["result"] = True
    pg = {"process_graph": {"p": node}}
    with pytest.raises(PropertyConditionException):
        extract_literal_match(pg)
