import dirty_equals

import openeo_driver.processgraph.evaluator
import pytest

from openeo_driver.dummy.dummy_backend import DummyProcessRegistry, DummyProcessing, DummyBackendImplementation
from openeo_driver.save_result import RasterCubeResult
from openeo_driver.testing import TEST_USER
from openeo_driver.users import User
from openeo_driver.utils import EvalEnv


@pytest.fixture
def basic_env(backend_implementation) -> EvalEnv:
    return EvalEnv(
        {
            "backend_implementation": backend_implementation,
            "user": User(user_id=TEST_USER),
        }
    )


def test_evaluate_basic_3_plus_5(basic_env):
    pg = {
        "add35": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True},
    }
    res = openeo_driver.processgraph.evaluator.evaluate(process_graph=pg, env=basic_env)
    assert res == 8


def test_issue534_save_result_leakage_math(udp_registry, basic_env):
    """https://github.com/Open-EO/openeo-python-driver/issues/534"""
    udp_registry.save(
        user_id=TEST_USER,
        process_id="3_plus_5",
        spec={
            "id": "3_plus_5",
            "process_graph": {
                "add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True},
            },
        },
    )

    pg = {
        "3plus5": {"process_id": "3_plus_5", "arguments": {}},
        # Intermediate result
        "sr1": {"process_id": "save_result", "arguments": {"data": {"from_node": "3plus5"}, "format": "JSON"}},
        "add1000": {"process_id": "add", "arguments": {"x": {"from_node": "3plus5"}, "y": 1000}},
        "sr2": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "add1000"}, "format": "JSON"},
            "result": True,
        },
    }
    result = openeo_driver.processgraph.evaluator.evaluate(
        process_graph=pg, env=basic_env, legacy_save_result_handling=False
    )
    assert result == 1008


def test_issue534_save_result_leakage_cubes_case1(udp_registry, basic_env):
    """
    https://github.com/Open-EO/openeo-python-driver/issues/534
    """
    # Wrap load_collection in a UDP to trigger sub-evaluation
    udp_registry.save(
        user_id=TEST_USER,
        process_id="my_load_collection",
        spec={
            "id": "my_load_collection",
            "process_graph": {
                "loadcollection1": {
                    "process_id": "load_collection",
                    "arguments": {"id": "S2_FOOBAR"},
                    "result": True,
                }
            },
        },
    )

    pg = {
        "load1": {
            "process_id": "my_load_collection",
            "arguments": {},
        },
        "filterbbox1": {
            "process_id": "filter_bbox",
            "arguments": {
                "data": {"from_node": "load1"},
                "extent": {"west": 3, "south": 51, "east": 4, "north": 52, "crs": 4326},
            },
        },
        "saveresult1": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "filterbbox1"}, "format": "netCDF"},
        },
        "saveresult2": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "filterbbox1"}, "format": "GTiff"},
            "result": True,
        },
    }
    result = openeo_driver.processgraph.evaluator.evaluate(process_graph=pg, env=basic_env)
    assert result == dirty_equals.IsList(
        dirty_equals.IsInstance(RasterCubeResult),
        dirty_equals.IsInstance(RasterCubeResult),
    )


def test_issue534_save_result_leakage_cubes_case2(udp_registry, basic_env):
    """
    https://github.com/Open-EO/openeo-python-driver/issues/534
    """
    # Wrap load_collection in a UDP to trigger sub-evaluation
    udp_registry.save(
        user_id=TEST_USER,
        process_id="my_load_collection",
        spec={
            "id": "my_load_collection",
            "process_graph": {
                "loadcollection1": {
                    "process_id": "load_collection",
                    "arguments": {"id": "S2_FOOBAR"},
                    "result": True,
                }
            },
        },
    )

    pg = {
        "load1": {
            "process_id": "my_load_collection",
            "arguments": {},
        },
        "saveresult1": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "load1"}, "format": "netCDF"},
        },
        "saveresult2": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "load1"}, "format": "netCDF"},
        },
        "filterbbox1": {
            "process_id": "filter_bbox",
            "arguments": {
                "data": {"from_node": "load1"},
                "extent": {"west": 3, "south": 51, "east": 4, "north": 52, "crs": 4326},
            },
        },
        "saveresult3": {
            "process_id": "save_result",
            "arguments": {"data": {"from_node": "filterbbox1"}, "format": "GTiff"},
            "result": True,
        },
    }
    result = openeo_driver.processgraph.evaluator.evaluate(process_graph=pg, env=basic_env)
    assert result == dirty_equals.IsList(
        dirty_equals.IsInstance(RasterCubeResult),
        dirty_equals.IsInstance(RasterCubeResult),
        dirty_equals.IsInstance(RasterCubeResult),
    )
