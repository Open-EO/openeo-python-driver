import pytest

from openeo_driver.vendor.processes_python import get_process, has_process
from openeo_driver.vendor.processes_python import eval_datatype, isnull
from openeo_driver.processgraph.registry import (
    _OPENEO_PROCESSES_PYTHON_WHITELIST,
    _openeo_processes_extra,
    process_registry_100,
    process_registry_2xx,
)
import numpy as np


def test_has_process():
    assert has_process("add")
    assert has_process("and")
    assert not has_process("and_")
    assert not has_process("foobar")
    assert not has_process("if")  # not part of the retained processes


@pytest.mark.parametrize(
    ["pid", "args", "expected"],
    [
        ("add", (2, 3), 5),
        ("multiply", (2, 3), 6),
        ("sum", ([1, 2, 3, 4, 5, 6],), 21),
        ("median", ([2, 5, 3, 8, 11],), 5),
        ("and", (False, True), False),
        ("or", (False, True), True),
    ],
)
def test_get_process(pid, args, expected):
    assert get_process(pid)(*args) == expected


def test_xarray_not_supported():
    class FakeXarray:
        __module__ = "xarray.core.dataarray"

    with pytest.raises(NotImplementedError):
        get_process("add")(FakeXarray(), 1)


def test_eval_datatype():
    assert eval_datatype(None) == "NoneType"
    assert eval_datatype(np.array([1])) == "numpy"


def test_isnull():
    assert isnull(None) is True
    assert isnull(np.nan) is True
    assert isnull(1) is False
    assert isnull("a") is False
    assert isnull({"a": "b"}) is False
    assert list(isnull(np.array([1.0, np.nan]))) == [False, True]
    assert list(isnull(np.array([None, "a", np.nan], dtype=object))) == [True, False, True]


@pytest.mark.parametrize("registry", [process_registry_100, process_registry_2xx])
def test_whitelisted_processes_have_implementation(registry):
    for pid in set(_OPENEO_PROCESSES_PYTHON_WHITELIST):
        assert pid in _openeo_processes_extra or has_process(pid), pid
        assert registry.contains(pid)

