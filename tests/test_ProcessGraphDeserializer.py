import itertools

import math
import pytest

from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo.rest.datacube import DataCube, PGNode
from openeo_driver.ProcessGraphDeserializer import (
    _period_to_intervals,
    SimpleProcessing,
    flatten_children_node_types,
    flatten_children_node_names,
    convert_node,
)
from openeo_driver.dummy.dummy_backend import DummyProcessing, DummyBackendImplementation, DummyProcessRegistry
from openeo_driver.errors import ProcessParameterRequiredException
from openeo_driver.processes import ProcessArgs
from openeo_driver.util import UNSET
from openeo_driver.utils import EvalEnv


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


class TestConvertNode:
    """
    Tests for `convert_node` function
    """
    _API_VERSION = "1.1.0"

    @pytest.fixture
    def fancy_add(self):
        """Fixture for process that allows to play with caching behavior"""

        class FancyAdd:
            """Add two numbers and the number of times this process was called before"""

            def __init__(self):
                self.call_history = []

            def __call__(self, args: ProcessArgs, env: EvalEnv) -> int:
                x = args.get_required("x")
                y = args.get_required("y")
                res = x + y + len(self.call_history)
                self.call_history.append((x, y))
                return res

        return FancyAdd()

    @pytest.fixture
    def dummy_process_registry(self, fancy_add) -> DummyProcessRegistry:
        registry = DummyProcessRegistry().with_standard_processes()

        # Register additional processes to play with and inspect caching behavior
        registry.add_process(name="fancy_add", function=fancy_add, spec={"id": "fancy_add"})
        return registry

    @pytest.fixture
    def backend_implementation(self, backend_config, dummy_process_registry) -> DummyBackendImplementation:
        processing = DummyProcessing(process_registry=dummy_process_registry)
        return DummyBackendImplementation(config=backend_config, processing=processing)

    @pytest.fixture
    def env(self, backend_implementation):
        return EvalEnv(
            {
                "backend_implementation": backend_implementation,
                "openeo_api_version": self._API_VERSION,
            }
        )

    def _preprocess_process_graph(self, process_graph: dict) -> dict:
        """
        convert_node expects a process graph node in nested representation
        instead of flat graph representation.
        This helper function takes care of the conversion
        """
        # TODO: improve this poor DX experience and improve abstraction/encapsulation
        node_id = ProcessGraphVisitor.dereference_from_node_arguments(process_graph)
        return process_graph[node_id]

    def test_basic(self, env):
        process_graph = {
            "add35": {
                "process_id": "add",
                "arguments": {"x": 3, "y": 5},
                "result": True,
            },
        }
        result = convert_node(self._preprocess_process_graph(process_graph), env=env)
        assert result == 8
        assert process_graph == {
            "add35": {
                "process_id": "add",
                "arguments": {"x": 3, "y": 5},
                "result": True,
                "result_cache": 8,
            }
        }

    @pytest.mark.parametrize(
        ["node_caching", "expected_result", "expected_calls"],
        [
            (UNSET, 4030, [(1, 2), (1, 2)]),
            (None, 4030, [(1, 2), (1, 2)]),
            (False, 4030, [(1, 2), (1, 2)]),
            ("full", 3030, [(1, 2)]),
            ("geopyspark-conservative", 4030, [(1, 2), (1, 2)]),
            ("geopyspark-progressive", 3030, [(1, 2)]),
        ],
    )
    def test_node_caching_basic(self, env, node_caching, expected_result, fancy_add, expected_calls):
        # Process graph with a `fancy_add` node that produces different results:
        # first call gives correct sum, but subsequent calls increment that each time
        process_graph = {
            "A": {
                "process_id": "fancy_add",
                "arguments": {"x": 1, "y": 2},
            },
            "A10": {
                "process_id": "multiply",
                "arguments": {"x": {"from_node": "A"}, "y": 10},
            },
            "A1000": {
                "process_id": "multiply",
                "arguments": {"x": {"from_node": "A"}, "y": 1000},
            },
            "result": {
                "process_id": "add",
                "arguments": {"x": {"from_node": "A10"}, "y": {"from_node": "A1000"}},
                "result": True,
            },
        }
        if node_caching is not UNSET:
            env = env.push({"node_caching": node_caching})

        result = convert_node(self._preprocess_process_graph(process_graph), env=env)
        assert result == expected_result
        assert fancy_add.call_history == expected_calls

    @pytest.mark.parametrize(
        ["node_caching", "expected"],
        [
            (
                False,
                {"constructed": [1, 2], "evaluated": [1, 2]},
            ),
            (
                "full",
                {"constructed": [1], "evaluated": [1, 1]},
            ),
            (
                "geopyspark-conservative",
                # Multiple results are constructed, but only first is evaluated
                {"constructed": [1, 2], "evaluated": [1, 1]},
            ),
            (
                "geopyspark-progressive",
                {"constructed": [1], "evaluated": [1, 1]},
            ),
        ],
    )
    def test_node_caching_geopyspark_conservative_lazy_and_compare(
        self, env, dummy_process_registry, node_caching, expected
    ):
        class LazyResult:
            """Helper to take care of lazy evaluation aspects"""

            _get_next_id = itertools.count(1).__next__
            stats = {"constructed": [], "evaluated": []}

            def __init__(self, value):
                self.value = value
                self.id = self._get_next_id()
                self.stats["constructed"].append(self.id)

            def evaluate(self):
                self.stats["evaluated"].append(self.id)
                return self.value

            def __eq__(self, other):
                return isinstance(other, LazyResult) and self.value == other.value

        @dummy_process_registry.add_function(spec={"id": "lazy_wrap"})
        def lazy_wrap(args: ProcessArgs, env: EvalEnv):
            data = args.get_required("data")
            return LazyResult(data)

        @dummy_process_registry.add_function(spec={"id": "lazy_collect"})
        def lazy_collect(args: ProcessArgs, env: EvalEnv):
            items = args.get_required("items")
            return [item.evaluate() for item in items]

        process_graph = {
            "source": {
                "process_id": "add",
                "arguments": {"x": 3, "y": 5},
            },
            "lazy_wrap": {
                "process_id": "lazy_wrap",
                "arguments": {"data": {"from_node": "source"}},
            },
            "lazy_collect": {
                "process_id": "lazy_collect",
                "arguments": {"items": [{"from_node": "lazy_wrap"}, {"from_node": "lazy_wrap"}]},
                "result": True,
            },
        }

        if node_caching is not UNSET:
            env = env.push({"node_caching": node_caching})
        result = convert_node(self._preprocess_process_graph(process_graph), env=env)

        assert result == [8, 8]
        assert LazyResult.stats == expected

    def test_node_caching_warnings(self, env, caplog):
        # Process graph with multiple nodes
        # and legacy node_caching setting that should trigger a warning, once
        process_graph = {
            "add12": {
                "process_id": "add",
                "arguments": {"x": 1, "y": 2},
            },
            "add34": {
                "process_id": "add",
                "arguments": {"x": {"from_node": "add12"}, "y": 34},
            },
            "add56": {
                "process_id": "add",
                "arguments": {"x": {"from_node": "add34"}, "y": 56},
                "result": True,
            },
        }
        env = env.push({"node_caching": True})
        caplog.set_level("WARNING")
        result = convert_node(self._preprocess_process_graph(process_graph), env=env)
        assert result == 93
        assert len([m for m in caplog.messages if "node_caching" in m]) == 1
