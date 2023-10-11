import re

import pytest

from openeo_driver.datacube import DriverDataCube
from openeo_driver.errors import (
    FileTypeInvalidException,
    OpenEOApiException,
    ProcessParameterInvalidException,
    ProcessParameterRequiredException,
    ProcessUnsupportedException,
)
from openeo_driver.processes import ProcessArgs, ProcessRegistry, ProcessRegistryException, ProcessSpec


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
    with pytest.raises(ProcessParameterRequiredException, match="Process 'add' parameter 'x' is required."):
        _ = process(args={}, env=None)


def test_process_registry_add_simple_function_with_name():
    reg = ProcessRegistry(argument_names=["args", "env"])

    @reg.add_simple_function(name="if")
    def if_(value, accept, reject=None):
        return accept if value else reject

    process = reg.get_function("if")

    assert process(args={"value": True, "accept": 3}, env=None) == 3
    assert process(args={"value": False, "accept": 3}, env=None) is None
    assert process(args={"value": False, "accept": 3, "reject": 5}, env=None) == 5
    with pytest.raises(ProcessParameterRequiredException, match="Process 'if' parameter 'value' is required."):
        _ = process(args={}, env=None)


def test_process_registry_add_simple_function_with_spec():
    reg = ProcessRegistry(argument_names=["args", "env"])

    @reg.add_simple_function(spec={"id": "something_custom"})
    def something_custom(x: int, y: int = 123):
        return x + y

    process = reg.get_function("something_custom")

    assert process(args={"x": 5, "y": 3}, env=None) == 8
    assert process(args={"x": 5}, env=None) == 128
    with pytest.raises(
        ProcessParameterRequiredException, match="Process 'something_custom' parameter 'x' is required."
    ):
        _ = process(args={}, env=None)


class TestProcessArgs:
    def test_dict(self):
        args = ProcessArgs({"foo": "bar"}, process_id="wibble")
        assert isinstance(args, dict)

    def test_get_required(self):
        args = ProcessArgs({"foo": "bar"}, process_id="wibble")
        assert args.get_required("foo") == "bar"
        with pytest.raises(ProcessParameterRequiredException, match="Process 'wibble' parameter 'other' is required."):
            _ = args.get_required("other")

    def test_get_required_with_type(self):
        args = ProcessArgs({"color": "red", "size": 5}, process_id="wibble")
        assert args.get_required("color", expected_type=str) == "red"
        assert args.get_required("color", expected_type=(str, int)) == "red"
        assert args.get_required("size", expected_type=int) == 5
        assert args.get_required("size", expected_type=(str, int)) == 5
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape(
                "The value passed for parameter 'color' in process 'wibble' is invalid: Expected <class 'openeo_driver.datacube.DriverDataCube'> but got <class 'str'>."
            ),
        ):
            _ = args.get_required("color", expected_type=DriverDataCube)

    def test_get_required_with_validator(self):
        args = ProcessArgs({"color": "red", "size": 5}, process_id="wibble")
        assert args.get_required("color", expected_type=str, validator=lambda v: len(v) == 3) == "red"
        assert (
            args.get_required(
                "color", expected_type=str, validator=ProcessArgs.validator_one_of(["red", "green", "blue"])
            )
            == "red"
        )
        assert args.get_required("size", expected_type=int, validator=lambda v: v % 3 == 2) == 5
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape(
                "The value passed for parameter 'color' in process 'wibble' is invalid: Failed validation."
            ),
        ):
            _ = args.get_required("color", expected_type=str, validator=lambda v: len(v) == 10)
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape("The value passed for parameter 'size' in process 'wibble' is invalid: Failed validation."),
        ):
            _ = args.get_required("size", expected_type=int, validator=lambda v: v % 3 == 1)
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape(
                "The value passed for parameter 'color' in process 'wibble' is invalid: Must be one of ['yellow', 'violet'] but got 'red'."
            ),
        ):
            _ = args.get_required(
                "color", expected_type=str, validator=ProcessArgs.validator_one_of(["yellow", "violet"])
            )

    def test_get_optional(self):
        args = ProcessArgs({"foo": "bar"}, process_id="wibble")
        assert args.get_optional("foo") == "bar"
        assert args.get_optional("other") is None
        assert args.get_optional("foo", 123) == "bar"
        assert args.get_optional("other", 123) == 123

    def test_get_optional_callable_default(self):
        args = ProcessArgs({"foo": "bar"}, process_id="wibble")
        assert args.get_optional("foo", default=lambda: 123) == "bar"
        assert args.get_optional("other", default=lambda: 123) == 123

        # Possible, but probably a bad idea:
        default = [1, 2, 3].pop
        assert args.get_optional("other", default=default) == 3
        assert args.get_optional("other", default=default) == 2

    def test_get_optional_with_type(self):
        args = ProcessArgs({"foo": "bar"}, process_id="wibble")
        assert args.get_optional("foo", expected_type=str) == "bar"
        assert args.get_optional("foo", expected_type=(str, int)) == "bar"
        assert args.get_optional("other", expected_type=str) is None
        assert args.get_optional("foo", 123, expected_type=(str, int)) == "bar"
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape(
                "The value passed for parameter 'foo' in process 'wibble' is invalid: Expected <class 'openeo_driver.datacube.DriverDataCube'> but got <class 'str'>."
            ),
        ):
            _ = args.get_optional("foo", expected_type=DriverDataCube)

    def test_get_optional_with_validator(self):
        args = ProcessArgs({"foo": "bar"}, process_id="wibble")
        assert args.get_optional("foo", validator=lambda s: all(c.lower() for c in s)) == "bar"
        assert args.get_optional("foo", validator=ProcessArgs.validator_one_of(["bar", "meh"])) == "bar"
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape("The value passed for parameter 'foo' in process 'wibble' is invalid: Failed validation."),
        ):
            _ = args.get_optional("foo", validator=lambda s: all(c.isupper() for c in s))
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape(
                "The value passed for parameter 'foo' in process 'wibble' is invalid: Must be one of ['nope', 'meh'] but got 'bar'."
            ),
        ):
            _ = args.get_optional("foo", validator=ProcessArgs.validator_one_of(["nope", "meh"]))

    def test_get_deep(self):
        args = ProcessArgs({"foo": {"bar": {"color": "red", "size": {"x": 5, "y": 8}}}}, process_id="wibble")
        assert args.get_deep("foo", "bar") == {"color": "red", "size": {"x": 5, "y": 8}}
        assert args.get_deep("foo", "bar", "color") == "red"
        assert args.get_deep("foo", "bar", "size", "y") == 8

        with pytest.raises(
            ProcessParameterInvalidException,
            match="The value passed for parameter 'foo' in process 'wibble' is invalid: step='z'",
        ):
            _ = args.get_deep("foo", "bar", "size", "z")

    def test_get_deep_with_type(self):
        args = ProcessArgs({"foo": {"bar": {"color": "red", "size": {"x": 5, "y": 8}}}}, process_id="wibble")
        assert args.get_deep("foo", "bar", "color", expected_type=str) == "red"
        assert args.get_deep("foo", "bar", "color", expected_type=(str, int)) == "red"
        assert args.get_deep("foo", "bar", "size", "x", expected_type=(str, int)) == 5

        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape(
                "The value passed for parameter 'foo' in process 'wibble' is invalid: Expected (<class 'openeo_driver.datacube.DriverDataCube'>, <class 'str'>) but got <class 'int'>."
            ),
        ):
            _ = args.get_deep("foo", "bar", "size", "x", expected_type=(DriverDataCube, str))

    def test_get_deep_with_validator(self):
        args = ProcessArgs({"foo": {"bar": {"color": "red", "size": {"x": 5, "y": 8}}}}, process_id="wibble")
        assert args.get_deep("foo", "bar", "size", "x", validator=lambda v: v % 5 == 0) == 5

        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape("The value passed for parameter 'foo' in process 'wibble' is invalid: Failed validation."),
        ):
            _ = args.get_deep("foo", "bar", "size", "y", validator=lambda v: v % 5 == 0)

    def test_get_aliased(self):
        args = ProcessArgs({"size": 5, "color": "red"}, process_id="wibble")
        assert args.get_aliased(["size", "dimensions"]) == 5
        assert args.get_aliased(["dimensions", "size"]) == 5
        assert args.get_aliased(["size", "color"]) == 5
        assert args.get_aliased(["color", "size"]) == "red"
        with pytest.raises(
            ProcessParameterRequiredException,
            match=re.escape("Process 'wibble' parameter '['shape', 'height']' is required."),
        ):
            _ = args.get_aliased(["shape", "height"])

    def test_get_subset(self):
        args = ProcessArgs({"size": 5, "color": "red", "shape": "circle"}, process_id="wibble")
        assert args.get_subset(["size", "color"]) == {"size": 5, "color": "red"}
        assert args.get_subset(["size", "height"]) == {"size": 5}
        assert args.get_subset(["meh"]) == {}
        assert args.get_subset(["color"], aliases={"shape": "form"}) == {"color": "red", "form": "circle"}
        assert args.get_subset(["color"], aliases={"foo": "bar"}) == {"color": "red"}

    def test_get_enum(self):
        args = ProcessArgs({"size": 5, "color": "red"}, process_id="wibble")
        assert args.get_enum("color", options=["red", "green", "blue"]) == "red"
        assert args.get_enum("size", options={3, 5, 8}) == 5

        with pytest.raises(ProcessParameterRequiredException, match="Process 'wibble' parameter 'shape' is required."):
            _ = args.get_enum("shape", options=["circle", "square"])

        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape(
                "The value passed for parameter 'color' in process 'wibble' is invalid: Invalid enum value 'red'. Expected one of ['R', 'G', 'B']."
            ),
        ):
            _ = args.get_enum("color", options=["R", "G", "B"])

    def test_validator_generic(self):
        args = ProcessArgs({"size": 11}, process_id="wibble")

        validator = ProcessArgs.validator_generic(lambda v: v > 1, error_message="Should be stricly positive.")
        value = args.get_required("size", expected_type=int, validator=validator)
        assert value == 11

        validator = ProcessArgs.validator_generic(lambda v: v % 2 == 0, error_message="Should be even.")
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape("The value passed for parameter 'size' in process 'wibble' is invalid: Should be even."),
        ):
            _ = args.get_required("size", expected_type=int, validator=validator)

        validator = ProcessArgs.validator_generic(
            lambda v: v % 2 == 0, error_message="Should be even but got {actual}."
        )
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape(
                "The value passed for parameter 'size' in process 'wibble' is invalid: Should be even but got 11."
            ),
        ):
            _ = args.get_required("size", expected_type=int, validator=validator)

    def test_validator_one_of(self):
        args = ProcessArgs({"color": "red", "size": 5}, process_id="wibble")
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape(
                "The value passed for parameter 'color' in process 'wibble' is invalid: Must be one of ['yellow', 'violet'] but got 'red'."
            ),
        ):
            _ = args.get_required(
                "color", expected_type=str, validator=ProcessArgs.validator_one_of(["yellow", "violet"])
            )

    def test_validator_geojson_dict(self):
        polygon = {"type": "Polygon", "coordinates": [[1, 2]]}
        args = ProcessArgs({"geometry": polygon, "color": "red"}, process_id="wibble")

        validator = ProcessArgs.validator_geojson_dict()
        assert args.get_required("geometry", validator=validator) == polygon
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape(
                "The value passed for parameter 'color' in process 'wibble' is invalid: Invalid GeoJSON: JSON object (mapping/dictionary) expected, but got str."
            ),
        ):
            _ = args.get_required("color", validator=validator)

        validator = ProcessArgs.validator_geojson_dict(allowed_types=["FeatureCollection"])
        with pytest.raises(
            ProcessParameterInvalidException,
            match=re.escape(
                "The value passed for parameter 'geometry' in process 'wibble' is invalid: Invalid GeoJSON: Found type 'Polygon', but expects one of ['FeatureCollection']."
            ),
        ):
            _ = args.get_required("geometry", validator=validator)

    @pytest.mark.parametrize(
        ["formats"],
        [
            (["GeoJSON", "CSV"],),
            ({"GeoJSON": {}, "CSV": {}},),
        ],
    )
    def test_validator_file_format(self, formats):
        args = ProcessArgs(
            {"format1": "GeoJSON", "format2": "geojson", "format3": "TooExotic"},
            process_id="wibble",
        )

        validator = ProcessArgs.validator_file_format(formats=formats)

        assert args.get_required("format1", validator=validator) == "GeoJSON"
        assert args.get_required("format2", validator=validator) == "geojson"

        with pytest.raises(
            OpenEOApiException,
            match=re.escape("Invalid file format 'TooExotic'. Allowed formats: GeoJSON, CSV"),
        ) as exc_info:
            _ = args.get_required("format3", validator=validator)

        assert exc_info.value.code == "FormatUnsuitable"
