import math
import pytest

from openeo.rest.datacube import DataCube, PGNode
from openeo_driver.ProcessGraphDeserializer import extract_deep, extract_args_subset, _period_to_intervals, \
    extract_arg_enum, SimpleProcessing
from openeo_driver.errors import ProcessParameterInvalidException, ProcessParameterRequiredException


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


def test_extract_args_subset():
    assert extract_args_subset({"foo": 3, "bar": 5}, ["foo"]) == {"foo": 3}
    assert extract_args_subset({"foo": 3, "bar": 5}, ["bar"]) == {"bar": 5}
    assert extract_args_subset({"foo": 3, "bar": 5}, ["meh"]) == {}
    assert extract_args_subset({"foo": 3, "bar": 5}, ["foo", "bar"]) == {"foo": 3, "bar": 5}
    assert extract_args_subset({"foo": 3, "bar": 5}, ["foo", "bar", "meh"]) == {"foo": 3, "bar": 5}


def test_extract_args_subset_aliases():
    assert extract_args_subset({"foo": 3, "bar": 5}, ["foo", "mask"], aliases={"bar": "mask"}) == {"foo": 3, "mask": 5}
    assert extract_args_subset({"foo": 3, "bar": 5}, ["foo", "mask"], aliases={"bar": "foo"}) == {"foo": 3}


def test_extract_arg_enum():
    enum_values = {"hour", "minute", "second"}

    assert extract_arg_enum({"unit": "hour"}, "unit", enum_values=enum_values) == "hour"
    assert extract_arg_enum({"unit": "minute"}, "unit", enum_values=enum_values) == "minute"

    with pytest.raises(ProcessParameterRequiredException, match="parameter 'unit' is required."):
        extract_arg_enum({"foo": "hour"}, "unit", enum_values=enum_values)

    with pytest.raises(ProcessParameterInvalidException, match="Invalid enum value 'hours'"):
        extract_arg_enum({"unit": "hours"}, "unit", enum_values=enum_values)
    with pytest.raises(ProcessParameterInvalidException, match="Invalid enum value 'foo'"):
        extract_arg_enum({"unit": "foo"}, "unit", enum_values=enum_values)


def test_period_to_intervals():
    weekly_intervals = _period_to_intervals("2021-06-08", "2021-06-24", "week")
    print(list(weekly_intervals))
    weekly_intervals = [(i[0].isoformat(), i[1].isoformat()) for i in weekly_intervals]
    assert 3 == len(weekly_intervals)
    assert weekly_intervals[0] == ('2021-06-06T00:00:00', '2021-06-13T00:00:00')
    assert weekly_intervals[1] == ('2021-06-13T00:00:00', '2021-06-20T00:00:00')
    assert weekly_intervals[2] == ('2021-06-20T00:00:00', '2021-06-27T00:00:00')


def test_period_to_intervals_monthly():
    intervals = _period_to_intervals("2021-06-08", "2021-08-24", "month")
    print(list(intervals))
    intervals = [(i[0].isoformat(), i[1].isoformat()) for i in intervals]
    assert 3 == len(intervals)
    assert intervals[0] == ('2021-06-01T00:00:00', '2021-07-01T00:00:00')
    assert intervals[1] == ('2021-07-01T00:00:00', '2021-08-01T00:00:00')
    assert intervals[2] == ('2021-08-01T00:00:00', '2021-09-01T00:00:00')

def test_period_to_intervals_yearly():
    intervals = _period_to_intervals("2018-06-08", "2021-08-24", "year")
    print(list(intervals))
    intervals = [(i[0].isoformat(), i[1].isoformat()) for i in intervals]
    assert 4 == len(intervals)
    assert intervals[0] == ('2018-01-01T00:00:00', '2019-01-01T00:00:00')
    assert intervals[1] == ('2019-01-01T00:00:00', '2020-01-01T00:00:00')
    assert intervals[2] == ('2020-01-01T00:00:00', '2021-01-01T00:00:00')
    assert intervals[3] == ('2021-01-01T00:00:00', '2022-01-01T00:00:00')


def test_period_to_intervals_monthly_full_year():
    intervals = _period_to_intervals("2020-01-01", "2021-01-01", "month")
    print(list(intervals))
    intervals = [(i[0].isoformat(), i[1].isoformat()) for i in intervals]
    assert 12 == len(intervals)
    assert intervals[0] == ('2020-01-01T00:00:00', '2020-02-01T00:00:00')
    assert intervals[1] == ('2020-02-01T00:00:00', '2020-03-01T00:00:00')
    assert intervals[2] == ('2020-03-01T00:00:00', '2020-04-01T00:00:00')
    assert intervals[11] == ('2020-12-01T00:00:00', '2021-01-01T00:00:00')


def test_period_to_intervals_daily():
    intervals = _period_to_intervals("2021-06-08", "2021-06-11", "day")
    print(list(intervals))
    intervals = [(i[0].isoformat(), i[1].isoformat()) for i in intervals]
    assert 4 == len(intervals)
    assert intervals[0] == ('2021-06-07T00:00:00', '2021-06-08T00:00:00')
    assert intervals[1] == ('2021-06-08T00:00:00', '2021-06-09T00:00:00')
    assert intervals[2] == ('2021-06-09T00:00:00', '2021-06-10T00:00:00')
    assert intervals[3] == ('2021-06-10T00:00:00', '2021-06-11T00:00:00')


def test_period_to_intervals_dekad():
    intervals = _period_to_intervals("2021-06-08", "2021-07-20", "dekad")
    print(list(intervals))
    intervals = [(i[0].isoformat(), i[1].isoformat()) for i in intervals]
    assert 5 == len(intervals)
    assert intervals[0] == ('2021-06-01T00:00:00', '2021-06-11T00:00:00')
    assert intervals[1] == ('2021-06-11T00:00:00', '2021-06-21T00:00:00')
    assert intervals[2] == ('2021-06-21T00:00:00', '2021-07-01T00:00:00')
    assert intervals[3] == ('2021-07-01T00:00:00', '2021-07-11T00:00:00')
    assert intervals[4] == ('2021-07-11T00:00:00', '2021-07-21T00:00:00')

def test_period_to_intervals_dekad_first_of_month():
    intervals = _period_to_intervals("2021-06-01", "2021-07-20", "dekad")
    print(list(intervals))
    intervals = [(i[0].isoformat(), i[1].isoformat()) for i in intervals]
    assert 5 == len(intervals)
    assert intervals[0] == ('2021-06-01T00:00:00', '2021-06-11T00:00:00')
    assert intervals[1] == ('2021-06-11T00:00:00', '2021-06-21T00:00:00')
    assert intervals[2] == ('2021-06-21T00:00:00', '2021-07-01T00:00:00')
    assert intervals[3] == ('2021-07-01T00:00:00', '2021-07-11T00:00:00')
    assert intervals[4] == ('2021-07-11T00:00:00', '2021-07-21T00:00:00')


class TestSimpleProcessing:

    def test_basic(self):
        processing = SimpleProcessing()
        pg = {"add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}}
        assert processing.evaluate(pg) == 8

    @pytest.mark.parametrize(["pg", "expected"], [
        ({"s": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}}, 8),
        ({"s": {"process_id": "subtract", "arguments": {"x": 3, "y": 5}, "result": True}}, -2),
        ({"s": {"process_id": "multiply", "arguments": {"x": 3, "y": 5}, "result": True}}, 15),
        ({"s": {"process_id": "product", "arguments": {"data": [3, 5, 8]}, "result": True}}, 120),
        ({"s": {"process_id": "count", "arguments": {"data": [3, 5, 8]}, "result": True}}, 3),
        ({"s": {"process_id": "mean", "arguments": {"data": [3, 5, 10]}, "result": True}}, 6),
        ({"s": {"process_id": "median", "arguments": {"data": [3, 5, 10]}, "result": True}}, 5),
        ({"s": {"process_id": "array_element", "arguments": {"data": [3, 5, 8], "index": 2}, "result": True}}, 8),
        ({"s": {"process_id": "gt", "arguments": {"x": 3, "y": 5}, "result": True}}, False),
        ({"s": {"process_id": "lte", "arguments": {"x": 3, "y": 5}, "result": True}}, True),
        ({"s": {"process_id": "and", "arguments": {"x": True, "y": False}, "result": True}}, False),
        ({"s": {"process_id": "or", "arguments": {"x": True, "y": False}, "result": True}}, True),
        ({
             "a": {"process_id": "add", "arguments": {"x": 3, "y": 5}},
             "b": {"process_id": "multiply", "arguments": {"x": {"from_node": "a"}, "y": 10}, "result": True}
         }, 80),
        ({
             "a": {"process_id": "divide", "arguments": {"x": 90, "y": 180}},
             "b": {"process_id": "max", "arguments": {"data": [1, 2, math.pi]}},
             "c": {"process_id": "multiply", "arguments": {"x": {"from_node": "a"}, "y": {"from_node": "b"}}},
             "d": {"process_id": "sin", "arguments": {"x": {"from_node": "c"}}, "result": True}
         }, 1.0),
    ])
    def test_simple(self, pg, expected):
        processing = SimpleProcessing()
        assert processing.evaluate(pg) == expected

    def test_parameter(self):
        processing = SimpleProcessing()
        env = processing.get_basic_env()
        pg = {"add": {"process_id": "add", "arguments": {"x": {"from_parameter": "foo"}, "y": 5}, "result": True}}
        assert processing.evaluate(pg, env=env.push(parameters={"foo": 3})) == 8
        assert processing.evaluate(pg, env=env.push(parameters={"foo": 30})) == 35

        with pytest.raises(ProcessParameterRequiredException):
            processing.evaluate(pg, env=env)

    def test_client_pipeline(self):
        graph = PGNode("add", x=3, y=8)
        cube = DataCube(graph=graph, connection=None)
        pg = cube.flat_graph()
        processing = SimpleProcessing()
        assert processing.evaluate(pg) == 11
