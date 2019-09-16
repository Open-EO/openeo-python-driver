import pytest

from openeo_driver.processes import ProcessSpec, ProcessRegistry, NoSuchProcessException


def test_process_spec_basic():
    spec = (
        ProcessSpec("mean", "Mean value")
            .param("input", "Input data", schema={"type": "array", "items": {"type": "number"}})
            .param("mask", "The mask", schema=ProcessSpec.RASTERCUBE, required=False)
            .returns("Mean value of data", schema={"type": "number"})
    )
    assert spec.to_dict() == {
        "id": "mean",
        "description": "Mean value",
        "parameters": {
            "input": {
                "description": "Input data",
                "required": True,
                "schema": {"type": "array", "items": {"type": "number"}}
            },
            "mask": {
                "description": "The mask",
                "required": False,
                "schema": {"type": "object", "format": "raster-cube"}
            }
        },
        "parameter_order": ["input", "mask"],
        "returns": {"description": "Mean value of data", "schema": {"type": "number"}},
    }


def test_process_spec_no_params():
    spec = ProcessSpec("foo", "bar").returns("output", schema={"type": "number"})
    with pytest.warns(UserWarning):
        assert spec.to_dict() == {"id": "foo", "description": "bar", "parameters": {}, "parameter_order": [],
                                  "returns": {"description": "output", "schema": {"type": "number"}}}


def test_process_sepc_no_returns():
    spec = ProcessSpec("foo", "bar").param("input", "Input", schema=ProcessSpec.RASTERCUBE)
    with pytest.raises(AssertionError):
        spec.to_dict()


def test_process_registry_add_by_name():
    reg = ProcessRegistry()
    reg.add_by_name("max")
    assert set(reg._specs.keys()) == {"max"}
    spec = reg.get_spec('max')
    assert spec['id'] == 'max'
    assert 'largest value' in spec['description']
    assert all(k in spec for k in ['parameters', 'parameter_order', 'returns'])


def test_process_registry_add_function():
    reg = ProcessRegistry()

    @reg.add_function
    def max(*args):
        return max(*args)

    assert set(reg._specs.keys()) == {"max"}
    spec = reg.get_spec('max')
    assert spec['id'] == 'max'
    assert 'largest value' in spec['description']
    assert all(k in spec for k in ['parameters', 'parameter_order', 'returns'])

    assert reg.get_function('max') is max


def test_process_registry_with_spec():
    reg = ProcessRegistry()

    @reg.add_function_with_spec(
        ProcessSpec("foo", "bar")
            .param("input", "Input", schema=ProcessSpec.RASTERCUBE)
            .returns(description="Output", schema=ProcessSpec.RASTERCUBE)
    )
    def foo(*args):
        return 42

    assert reg.get_spec('foo') == {
        "id": "foo",
        "description": "bar",
        "parameters": {
            "input": {"description": "Input", "schema": {"type": "object", "format": "raster-cube"},
                      "required": True},
        },
        "parameter_order": ["input"],
        "returns": {"description": "Output", "schema": {"type": "object", "format": "raster-cube"}}
    }


def test_process_registry_add_deprecated():
    reg = ProcessRegistry()

    @reg.add_deprecated
    def foo(*args):
        return 42

    new_foo = reg.get_function('foo')
    with pytest.warns(UserWarning, match="deprecated process"):
        assert new_foo() == 42
    with pytest.raises(NoSuchProcessException):
        reg.get_spec('foo')
