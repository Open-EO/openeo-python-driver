import math
import pytest

from openeo.rest.datacube import DataCube, PGNode
from openeo_driver.ProcessGraphDeserializer import (
    _period_to_intervals,
    SimpleProcessing,
    flatten_children_node_types,
    flatten_children_node_names,
)
from openeo_driver.errors import ProcessParameterRequiredException


def test_period_to_intervals():
    weekly_intervals = _period_to_intervals("2021-06-08", "2021-06-24", "week")
    weekly_intervals = [(i[0].isoformat(), i[1].isoformat()) for i in weekly_intervals]
    assert 3 == len(weekly_intervals)
    assert weekly_intervals[0] == ('2021-06-06T00:00:00', '2021-06-13T00:00:00')
    assert weekly_intervals[1] == ('2021-06-13T00:00:00', '2021-06-20T00:00:00')
    assert weekly_intervals[2] == ('2021-06-20T00:00:00', '2021-06-27T00:00:00')


@pytest.mark.parametrize(["start", "end"], [
    ("2021-06-01", "2021-08-02"),
    ("2021-06-08", "2021-08-24"),
    ("2021-06-30", "2021-08-31"),
])
def test_period_to_intervals_monthly(start, end):
    intervals = _period_to_intervals(start, end, "month")
    assert [(s.isoformat(), e.isoformat()) for (s, e) in intervals] == [
        ('2021-06-01T00:00:00', '2021-07-01T00:00:00'),
        ('2021-07-01T00:00:00', '2021-08-01T00:00:00'),
        ('2021-08-01T00:00:00', '2021-09-01T00:00:00')
    ]


@pytest.mark.parametrize(["start", "end", "expected"], [
    ("2021-01-01", "2021-01-02", ["2021-01-01", "2021-02-01"]),
    ("2021-01-01", "2021-03-02", ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"]),
    ("2021-01-31", "2021-02-28", ["2021-01-01", "2021-02-01", "2021-03-01"]),
    ("2021-03-31", "2021-03-31", ["2021-03-01", "2021-04-01"]),
    ("2021-11-11", "2022-02-01", ["2021-11-01", "2021-12-01", "2022-01-01", "2022-02-01"]),
])
def test_period_to_intervals_monthly_2(start, end, expected):
    intervals = _period_to_intervals(start, end, "month")
    expected = [e + "T00:00:00" if "T" not in e else e for e in expected]
    assert [(s.isoformat(), e.isoformat()) for (s, e) in intervals] == list(zip(expected[:-1], expected[1:]))


def test_period_to_intervals_monthly_tz():
    intervals = _period_to_intervals("2021-06-01T00:00:00+0000", "2021-08-02T00:00:00+0000", "month")
    assert [(s.isoformat(), e.isoformat()) for (s, e) in intervals] == [
        ('2021-06-01T00:00:00', '2021-07-01T00:00:00'),
        ('2021-07-01T00:00:00', '2021-08-01T00:00:00'),
        ('2021-08-01T00:00:00', '2021-09-01T00:00:00')
    ]


def test_period_to_intervals_yearly():
    intervals = _period_to_intervals("2018-06-08", "2021-08-24", "year")
    intervals = [(i[0].isoformat(), i[1].isoformat()) for i in intervals]
    assert 4 == len(intervals)
    assert intervals[0] == ('2018-01-01T00:00:00', '2019-01-01T00:00:00')
    assert intervals[1] == ('2019-01-01T00:00:00', '2020-01-01T00:00:00')
    assert intervals[2] == ('2020-01-01T00:00:00', '2021-01-01T00:00:00')
    assert intervals[3] == ('2021-01-01T00:00:00', '2022-01-01T00:00:00')


def test_period_to_intervals_monthly_full_year():
    intervals = _period_to_intervals("2020-01-01", "2021-01-01", "month")
    intervals = [(i[0].isoformat(), i[1].isoformat()) for i in intervals]
    assert 12 == len(intervals)
    assert intervals[0] == ('2020-01-01T00:00:00', '2020-02-01T00:00:00')
    assert intervals[1] == ('2020-02-01T00:00:00', '2020-03-01T00:00:00')
    assert intervals[2] == ('2020-03-01T00:00:00', '2020-04-01T00:00:00')
    assert intervals[11] == ('2020-12-01T00:00:00', '2021-01-01T00:00:00')


def test_period_to_intervals_daily():
    intervals = _period_to_intervals("2021-06-08", "2021-06-11", "day")
    intervals = [(i[0].isoformat(), i[1].isoformat()) for i in intervals]
    assert 4 == len(intervals)
    assert intervals[0] == ('2021-06-07T00:00:00', '2021-06-08T00:00:00')
    assert intervals[1] == ('2021-06-08T00:00:00', '2021-06-09T00:00:00')
    assert intervals[2] == ('2021-06-09T00:00:00', '2021-06-10T00:00:00')
    assert intervals[3] == ('2021-06-10T00:00:00', '2021-06-11T00:00:00')


def test_period_to_intervals_dekad():
    intervals = _period_to_intervals("2021-06-08", "2021-07-20", "dekad")
    intervals = [(i[0].isoformat(), i[1].isoformat()) for i in intervals]
    assert 5 == len(intervals)
    assert intervals[0] == ('2021-06-01T00:00:00', '2021-06-11T00:00:00')
    assert intervals[1] == ('2021-06-11T00:00:00', '2021-06-21T00:00:00')
    assert intervals[2] == ('2021-06-21T00:00:00', '2021-07-01T00:00:00')
    assert intervals[3] == ('2021-07-01T00:00:00', '2021-07-11T00:00:00')
    assert intervals[4] == ('2021-07-11T00:00:00', '2021-07-21T00:00:00')


def test_period_to_intervals_dekad_first_of_month():
    intervals = _period_to_intervals("2021-06-01", "2021-07-20", "dekad")
    intervals = [(i[0].isoformat(), i[1].isoformat()) for i in intervals]
    assert 5 == len(intervals)
    assert intervals[0] == ('2021-06-01T00:00:00', '2021-06-11T00:00:00')
    assert intervals[1] == ('2021-06-11T00:00:00', '2021-06-21T00:00:00')
    assert intervals[2] == ('2021-06-21T00:00:00', '2021-07-01T00:00:00')
    assert intervals[3] == ('2021-07-01T00:00:00', '2021-07-11T00:00:00')
    assert intervals[4] == ('2021-07-11T00:00:00', '2021-07-21T00:00:00')


@pytest.mark.parametrize(["start", "end"], [
    ("2021-09-01", "2022-03-02"),
    ("2021-09-30", "2022-03-31"),
    ("2021-10-10", "2022-04-20"),
    ("2021-10-31", "2022-04-30"),
    ("2021-11-20", "2022-05-20"),
    ("2021-11-30", "2022-05-31"),
])
def test_period_to_intervals_season(start, end):
    intervals = _period_to_intervals(start, end, "season")
    assert [(s.isoformat(), e.isoformat()) for (s, e) in intervals] == [
        ('2021-09-01T00:00:00', '2021-12-01T00:00:00'),
        ('2021-12-01T00:00:00', '2022-03-01T00:00:00'),
        ('2022-03-01T00:00:00', '2022-06-01T00:00:00')
    ]


@pytest.mark.parametrize(["start", "end", "expected"], [
    ("2021-01-01", "2021-01-01", ["2020-12-01", "2021-03-01"]),
    ("2021-01-01", "2021-05-02", ["2020-12-01", "2021-03-01", "2021-06-01"]),
    ("2021-02-01", "2021-05-02", ["2020-12-01", "2021-03-01", "2021-06-01"]),
    ("2021-03-01", "2021-05-02", ["2021-03-01", "2021-06-01"]),
    ("2021-05-01", "2021-07-02", ["2021-03-01", "2021-06-01", "2021-09-01"]),
    ("2021-10-01", "2022-02-02", ["2021-09-01", "2021-12-01", "2022-03-01"]),
])
def test_period_to_intervals_season_2(start, end, expected):
    intervals = _period_to_intervals(start, end, "season")
    expected = [e + "T00:00:00" if "T" not in e else e for e in expected]
    assert [(s.isoformat(), e.isoformat()) for (s, e) in intervals] == list(zip(expected[:-1], expected[1:]))


@pytest.mark.parametrize(["start", "end"], [
    ("2021-05-01", "2022-03-02"),
    ("2021-05-30", "2022-03-31"),
    ("2021-07-10", "2022-04-20"),
    ("2021-08-31", "2022-03-31"),
    ("2021-09-20", "2022-04-20"),
    ("2021-10-30", "2022-04-30"),
])
def test_period_to_intervals_tropical_season(start, end):
    intervals = _period_to_intervals(start, end, "tropical-season")
    assert [(s.isoformat(), e.isoformat()) for (s, e) in intervals] == [
        ('2021-05-01T00:00:00', '2021-11-01T00:00:00'),
        ('2021-11-01T00:00:00', '2022-05-01T00:00:00'),
    ]


@pytest.mark.parametrize(["start", "end", "expected"], [
    ("2021-01-01", "2021-01-01", ["2020-11-01", "2021-05-01"]),
    ("2021-01-01", "2021-05-02", ["2020-11-01", "2021-05-01", "2021-11-01"]),
    ("2021-02-01", "2021-05-02", ["2020-11-01", "2021-05-01", "2021-11-01"]),
    ("2021-01-01", "2021-04-02", ["2020-11-01", "2021-05-01"]),
    ("2021-05-01", "2021-12-02", ["2021-05-01", "2021-11-01", "2022-05-01"]),
])
def test_period_to_intervals_tropical_season_2(start, end, expected):
    intervals = _period_to_intervals(start, end, "tropical-season")
    expected = [e + "T00:00:00" if "T" not in e else e for e in expected]
    assert [(s.isoformat(), e.isoformat()) for (s, e) in intervals] == list(zip(expected[:-1], expected[1:]))


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
        assert processing.evaluate(pg, env=env.push_parameters({"foo": 3})) == 8
        assert processing.evaluate(pg, env=env.push_parameters({"foo": 30})) == 35

        with pytest.raises(ProcessParameterRequiredException):
            processing.evaluate(pg, env=env)

    def test_client_pipeline(self):
        graph = PGNode("add", x=3, y=8)
        cube = DataCube(graph=graph, connection=None)
        pg = cube.flat_graph()
        processing = SimpleProcessing()
        assert processing.evaluate(pg) == 11

    def test_get_all_dependency_nodes(self):
        process_graph = {
            "from_node": "filter1",
            "node": {
                "arguments": {
                    "bands": [
                        "B02"
                    ],
                    "data": {
                        "from_node": "load_collection1",
                        "node": {
                            "arguments": {
                                "id": "S2_FOOBAR"
                            },
                            "process_id": "load_collection"
                        }
                    }
                },
                "process_id": "filter_bands"
            }
        }
        children_node_types = flatten_children_node_types(process_graph)
        children_node_names = flatten_children_node_names(process_graph)

        assert len(children_node_types) == 2
        assert len(children_node_names) == 2
