import pytest

from openeo_driver.errors import ProcessUnsupportedException, ProcessParameterRequiredException
from openeo_driver.processes import ProcessSpec, ProcessRegistry, ProcessRegistryException


def test_process_spec_basic_040():
    spec = (
        ProcessSpec("mean", "Mean value")
            .param("input", "Input data", schema={"type": "array", "items": {"type": "number"}})
            .param("mask", "The mask", schema=ProcessSpec.RASTERCUBE, required=False)
            .returns("Mean value of data", schema={"type": "number"})
    )
    assert spec.to_dict_040() == {
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


def test_process_spec_basic_100():
    spec = (
        ProcessSpec("mean", "Mean value")
            .param("input", "Input data", schema={"type": "array", "items": {"type": "number"}})
            .param("mask", "The mask", schema=ProcessSpec.RASTERCUBE, required=False)
            .returns("Mean value of data", schema={"type": "number"})
    )
    assert spec.to_dict_100() == {
        "id": "mean",
        "description": "Mean value",
        "parameters": [
            {
                "name": "input",
                "description": "Input data",
                "optional": False,
                "schema": {"type": "array", "items": {"type": "number"}}
            },
            {
                "name": "mask",
                "description": "The mask",
                "optional": True,
                "schema": {"type": "object", "format": "raster-cube"}
            }
        ],
        "returns": {"description": "Mean value of data", "schema": {"type": "number"}},
    }


def test_process_spec_no_params_040():
    spec = ProcessSpec("foo", "bar").returns("output", schema={"type": "number"})
    with pytest.warns(UserWarning):
        assert spec.to_dict_040() == {
            "id": "foo", "description": "bar", "parameters": {}, "parameter_order": [],
            "returns": {"description": "output", "schema": {"type": "number"}}
        }


def test_process_spec_no_params_100():
    spec = ProcessSpec("foo", "bar").returns("output", schema={"type": "number"})
    with pytest.warns(UserWarning):
        assert spec.to_dict_100() == {
            "id": "foo", "description": "bar", "parameters": [],
            "returns": {"description": "output", "schema": {"type": "number"}}
        }


def test_process_spec_no_returns():
    spec = ProcessSpec("foo", "bar").param("input", "Input", schema=ProcessSpec.RASTERCUBE)
    with pytest.raises(AssertionError):
        spec.to_dict_100()


def test_process_spec_extra_100():
    spec = (
        ProcessSpec("incr", "Increment", extra={"experimental": True, "categories":["math"]})
            .param("value", "Input value", schema={"type": "number"})
            .returns("Incremented value", schema={"type": "number"})
    )
    assert spec.to_dict_100() == {
        "id": "incr",
        "description": "Increment",
        "parameters": [
            {
                "name": "value",
                "description": "Input value",
                "optional": False,
                "schema": {"type": "number"}
            },
        ],
        "returns": {"description": "Incremented value", "schema": {"type": "number"}},
        "experimental": True,
        "categories": ["math"],
    }



def test_process_registry_add_by_name():
    reg = ProcessRegistry()
    reg.add_spec_by_name("max")
    assert reg.contains("max")
    spec = reg.get_spec('max')
    assert spec['id'] == 'max'
    assert 'largest value' in spec['description']
    assert all(k in spec for k in ['parameters', 'returns'])


def test_process_registry_add_by_name_and_namespace():
    reg = ProcessRegistry()
    reg.add_spec_by_name("max", namespace="foo")
    assert not reg.contains("max")
    assert reg.contains("max", namespace="foo")
    with pytest.raises(ProcessUnsupportedException):
        reg.get_spec('max')
    spec = reg.get_spec('max', namespace="foo")
    assert spec['id'] == 'max'
    assert 'largest value' in spec['description']
    assert all(k in spec for k in ['parameters', 'returns'])


def test_process_registry_contains():
    reg = ProcessRegistry()
    assert not reg.contains("max")
    reg.add_spec_by_name("max")
    assert reg.contains("max")


def test_process_registry_load_predefined_specs():
    """Test if all spec json files load properly"""
    reg = ProcessRegistry()
    for name in reg.list_predefined_specs().keys():
        spec = reg.load_predefined_spec(name)
        assert spec["id"] == name


def test_process_registry_add_function():
    reg = ProcessRegistry()

    @reg.add_function
    def max(*args):
        return max(*args)

    assert reg.contains("max")
    spec = reg.get_spec('max')
    assert spec['id'] == 'max'
    assert 'largest value' in spec['description']
    assert all(k in spec for k in ['parameters', 'returns'])

    assert reg.get_function('max') is max


def test_process_registry_add_function_other_name():
    reg = ProcessRegistry()

    @reg.add_function(name="max")
    def madmax(*args):
        return max(*args)

    assert reg.contains("max")
    spec = reg.get_spec('max')
    assert spec['id'] == 'max'
    assert 'largest value' in spec['description']
    assert all(k in spec for k in ['parameters', 'returns'])

    assert reg.get_function('max') is madmax


def test_process_registry_add_function_namespace():
    reg = ProcessRegistry()

    @reg.add_function(name="max", namespace="foo")
    def madmax(*args):
        return max(*args)

    assert not reg.contains("max")
    assert reg.contains("max", namespace="foo")
    spec = reg.get_spec('max', namespace="foo")
    assert spec['id'] == 'max'
    assert 'largest value' in spec['description']
    assert all(k in spec for k in ['parameters', 'returns'])

    assert reg.get_function('max', namespace="foo") is madmax


def test_process_registry_add_function_argument_names():
    reg = ProcessRegistry(argument_names=["args", "env"])

    @reg.add_function
    def max(args, env=None):
        return max(*args)

    with pytest.raises(ProcessRegistryException):
        @reg.add_function
        def min(args):
            return min(*args)

    assert reg.contains("max")
    spec = reg.get_spec('max')
    assert spec['id'] == 'max'
    assert 'largest value' in spec['description']
    assert all(k in spec for k in ['parameters', 'returns'])
    assert reg.get_function('max') is max


def test_process_registry_with_spec_040():
    reg = ProcessRegistry()

    def add_function_with_spec(spec: ProcessSpec):
        def decorator(f):
            reg.add_function(f=f, spec=spec.to_dict_040())
            return f

        return decorator

    @add_function_with_spec(
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


def test_process_registry_with_spec_100():
    reg = ProcessRegistry()

    def add_function_with_spec(spec: ProcessSpec):
        def decorator(f):
            reg.add_function(f=f, spec=spec.to_dict_100())
            return f

        return decorator

    @add_function_with_spec(
        ProcessSpec("foo", "bar")
            .param("input", "Input", schema=ProcessSpec.RASTERCUBE)
            .returns(description="Output", schema=ProcessSpec.RASTERCUBE)
    )
    def foo(*args):
        return 42

    assert reg.get_spec('foo') == {
        "id": "foo",
        "description": "bar",
        "parameters": [
            {"name": "input", "description": "Input", "schema": {"type": "object", "format": "raster-cube"},
             "optional": False},
        ],
        "returns": {"description": "Output", "schema": {"type": "object", "format": "raster-cube"}}
    }


def test_process_registry_add_hidden():
    reg = ProcessRegistry()

    @reg.add_hidden
    def foo(*args):
        return 42

    new_foo = reg.get_function('foo')
    assert new_foo() == 42
    with pytest.raises(ProcessUnsupportedException):
        reg.get_spec('foo')


def test_process_registry_add_hidden_with_name():
    reg = ProcessRegistry()

    def bar(*args):
        return 42

    reg.add_hidden(bar, name="boz")

    new_bar = reg.get_function('boz')
    assert new_bar() == 42
    with pytest.raises(ProcessUnsupportedException):
        reg.get_spec('bar')
    with pytest.raises(ProcessUnsupportedException):
        reg.get_spec('boz')


def test_process_registry_add_hidden_with_namespace():
    reg = ProcessRegistry()

    def bar(*args):
        return 42

    reg.add_hidden(bar, name="boz", namespace="secret")

    new_bar = reg.get_function('boz', namespace="secret")
    assert new_bar() == 42
    with pytest.raises(ProcessUnsupportedException):
        reg.get_spec('bar', namespace="secret")
    with pytest.raises(ProcessUnsupportedException):
        reg.get_spec('boz', namespace="secret")


def test_process_registry_add_deprecated():
    reg = ProcessRegistry()

    @reg.add_deprecated
    def foo(*args):
        return 42

    new_foo = reg.get_function('foo')
    with pytest.warns(UserWarning, match="deprecated process"):
        assert new_foo() == 42
    with pytest.raises(ProcessUnsupportedException):
        reg.get_spec('foo')


def test_process_registry_add_deprecated_namespace():
    reg = ProcessRegistry()

    @reg.add_deprecated(namespace="old")
    def foo(*args):
        return 42

    with pytest.raises(ProcessUnsupportedException):
        reg.get_function('foo')
    new_foo = reg.get_function('foo', namespace="old")
    with pytest.warns(UserWarning, match="deprecated process"):
        assert new_foo() == 42
    with pytest.raises(ProcessUnsupportedException):
        reg.get_spec('foo', namespace="old")


def test_process_registry_get_spec():
    reg = ProcessRegistry()
    reg.add_spec_by_name("min")
    reg.add_spec_by_name("max")
    with pytest.raises(ProcessUnsupportedException):
        reg.get_spec('foo')


def test_process_registry_get_specs():
    reg = ProcessRegistry()
    reg.add_spec_by_name("min")
    reg.add_spec_by_name("max")
    reg.add_spec_by_name("sin")
    assert set(p['id'] for p in reg.get_specs()) == {"max", "min", "sin"}
    assert set(p['id'] for p in reg.get_specs('')) == {"max", "min", "sin"}
    assert set(p['id'] for p in reg.get_specs("m")) == {"max", "min"}
    assert set(p['id'] for p in reg.get_specs("in")) == {"min", "sin"}


def test_process_registry_get_specs_namepsaces():
    reg = ProcessRegistry()
    reg.add_spec_by_name("min", namespace="stats")
    reg.add_spec_by_name("max", namespace="stats")
    reg.add_spec_by_name("sin", namespace="math")
    assert set(p['id'] for p in reg.get_specs()) == set()
    assert set(p['id'] for p in reg.get_specs(namespace="stats")) == {"max", "min"}
    assert set(p['id'] for p in reg.get_specs(namespace="math")) == {"sin"}
    assert set(p['id'] for p in reg.get_specs("", namespace="stats")) == {"max", "min"}
    assert set(p['id'] for p in reg.get_specs("m", namespace="math")) == set()
    assert set(p['id'] for p in reg.get_specs("in", namespace="stats")) == {"min"}
    assert set(p['id'] for p in reg.get_specs("in", namespace="math")) == {"sin"}


def test_process_registry_add_simple_function():
    reg = ProcessRegistry(argument_names=["args", "env"])

    @reg.add_simple_function
    def add(x: int, y: int = 100):
        return x + y

    process = reg.get_function("add")

    assert process(args={"x": 2, "y": 3}, env=None) == 5
    assert process(args={"x": 2}, env=None) == 102
    with pytest.raises(ProcessParameterRequiredException):
        _ = process(args={}, env=None)
