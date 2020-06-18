import pytest
from openeo_driver.ProcessGraphDeserializer import extract_deep
from openeo_driver.errors import ProcessParameterInvalidException


def test_extract_deep():
    args = {
        "data": {"foo": "bar"},
        "axes": {
            "x": {"type": "axis", "kind": "axis", "orientation": "horizontal", "dir": "hor"},
            "y": {"type": "axis", "orientation": "vertical"},
        },
        "color": "red",
        "size": {"x": {"length": 123, "dim": "cm"}, "z": "flat"}
    }
    assert extract_deep(args, "color") == "red"
    assert extract_deep(args, "size", "z") == "flat"
    assert extract_deep(args, "size", "x", "dim") == "cm"
    assert extract_deep(args, "data") == {"foo": "bar"}
    assert extract_deep(args, "data", "foo") == "bar"
    assert extract_deep(args, "axes", "x", ["kind", "type"]) == "axis"
    assert extract_deep(args, "axes", "x", ["wut", "kind", "type"]) == "axis"
    assert extract_deep(args, "axes", "x", ["wut", "orientation", "dir"]) == "horizontal"
    assert extract_deep(args, "axes", "x", ["wut", "dir", "orientation"]) == "hor"

    with pytest.raises(ProcessParameterInvalidException):
        extract_deep(args, "data", "lol")
