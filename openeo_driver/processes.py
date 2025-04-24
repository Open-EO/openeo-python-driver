from __future__ import annotations

import typing
import logging
import functools
import inspect
import warnings
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union, Iterable

from openeo_driver.datacube import DriverDataCube
from openeo_driver.errors import (
    OpenEOApiException,
    ProcessParameterInvalidException,
    ProcessParameterRequiredException,
    ProcessUnsupportedException,
)
from openeo_driver.specs import SPECS_ROOT
from openeo_driver.util.geometry import validate_geojson_basic
from openeo_driver.utils import EvalEnv, read_json


_log = logging.getLogger(__name__)

class ProcessParameter:
    """Process Parameter."""

    def __init__(self, name: str, description: str, schema: dict, required: bool = True):
        self.name = name
        self.description = description
        self.schema = schema
        self.required = required


class ProcessSpec:
    """
    Helper object to easily build a process specification with a fluent/chained API.

    Intended for custom processes that are not specified in the official OpenEo Processes listing.
    """

    # Some predefined parameter schema's
    RASTERCUBE = {"type": "object", "format": "raster-cube"}

    def __init__(self, id, description: str, extra: Optional[dict] = None):
        self.id = id
        self.description = description
        self._parameters: List[ProcessParameter] = []
        self._returns = None
        self.extra = extra or {}

    def param(self, name, description, schema, required=True) -> 'ProcessSpec':
        """Add a process parameter"""
        self._parameters.append(
            ProcessParameter(name=name, description=description, schema=schema, required=required)
        )
        return self

    def returns(self, description: str, schema: dict) -> 'ProcessSpec':
        """Define return spec."""
        self._returns = {"description": description, "schema": schema}
        return self

    def to_dict_040(self) -> dict:
        """Generate process spec as (JSON-able) dictionary (API 0.4.0 style)."""
        # TODO #47 drop this
        if len(self._parameters) == 0:
            warnings.warn("Process with no parameters")
        assert self._returns is not None
        return {**self.extra, **{
            "id": self.id,
            "description": self.description,
            "parameters": {
                p.name: {"description": p.description, "schema": p.schema, "required": p.required}
                for p in self._parameters
            },
            "parameter_order": [p.name for p in self._parameters],
            "returns": self._returns
        }}

    def to_dict_100(self) -> dict:
        """Generate process spec as (JSON-able) dictionary (API 1.0.0 style)."""
        if len(self._parameters) == 0:
            warnings.warn("Process with no parameters")
        assert self._returns is not None
        return {**self.extra, **{
            "id": self.id,
            "description": self.description,
            "parameters": [
                {"name": p.name, "description": p.description, "schema": p.schema, "optional": not p.required}
                for p in self._parameters
            ],
            "returns": self._returns
        }}


DEFAULT_NAMESPACE = "backend"


class ProcessData(typing.NamedTuple):
    function: Optional[Callable]
    spec: Optional[dict]


class _ProcessRegKey(typing.NamedTuple):
    namespace: str
    name: str


class ProcessRegistryException(Exception):
    pass


class ProcessRegistry:
    """
    Registry for processes we support in the backend.

    For each process (name or "id" in official openEO terminology), we keep track of:
    - optional: a Python function that implements the process in some way.
        While generally optional, it is a requirement for openEO processes
        that are used in the top level of an openEO process graph.
    - optional: a specification (dictionary construct) in the style of the official
        openEO processes specifications.
        While generally optional, it is a requirement for processes that should
        be visible to standard openEO tooling (e.g. listed under `GET /processes`).
    """

    def __init__(
        self,
        spec_root: Path = SPECS_ROOT / "openeo-processes/1.x",
        argument_names: List[str] = None,
        target_version: Optional[str] = None,
    ):
        self._processes_spec_root = spec_root
        # Dictionary (namespace, process_name) -> ProcessData
        self._processes: Dict[_ProcessRegKey, ProcessData] = {}
        # Expected argument names that process function signature should start with
        self._argument_names = argument_names
        # openeo-processes version targeted by this collection of specs (per https://github.com/Open-EO/openeo-api/pull/549)
        self.target_version = target_version

    def __repr__(self):
        return "<{c} {n} processes>".format(c=self.__class__.__name__, n=len(self._processes))

    def contains(self, name: str, namespace: str = DEFAULT_NAMESPACE) -> bool:
        return _ProcessRegKey(name=name, namespace=namespace) in self._processes

    def _get(self, name: str, namespace: str = DEFAULT_NAMESPACE) -> ProcessData:
        return self._processes[_ProcessRegKey(name=name, namespace=namespace)]

    def load_predefined_spec(self, name: str) -> dict:
        """Get predefined process specification (dict) based on process name."""
        try:
            spec = read_json(self._processes_spec_root / '{n}.json'.format(n=name))
            # Health check: required fields for predefined processes
            assert all(k in spec for k in ['id', 'description', 'parameters', 'returns'])
            return spec
        except Exception:
            raise ProcessRegistryException("Failed to load predefined spec of process {n!r}".format(n=name))

    def list_predefined_specs(self) -> Dict[str, Path]:
        """List all processes with a spec JSON file."""
        return {p.stem: p for p in self._processes_spec_root.glob("*.json")}

    def add_process(
        self,
        name: str,
        *,
        function: Optional[Callable] = None,
        spec: Optional[dict] = None,
        namespace: str = DEFAULT_NAMESPACE,
        allow_override: bool = False,
    ):
        """
        Generic method to add a process to the registry,
        with callable function (optional) and specification dict (optional).

        There are multiple methods to add an entry in the registry.
        Which one to pick depends on the situation or use case:

        - :py:meth:`add_process`: most generic method, used by all other methods,
            but least recommended for general use, because it hides intent and purpose.

        - spec only:
            - :py:meth:`add_spec`: add a process just by a specification dict (e.g. loaded from a JSON file),
                without handling function
            - :py:meth:`add_spec_by_name`: add one or more standard openEO processes just by process id
                (spec will be fetched from collection of official openEO-process specs)

        - function-focussed or function-only:
            - :py:meth:`add_function` add a Python function, and guess the rest (process id, spec, ...) from that (unless given).
                The Python function must accept a `args: ProcessArgs` and `env EvalEnv` argument
            - :py:meth:`add_simple_function`: like `add_function`, but the function can have simple argument,
                e.g. `x` and `y`. The function will be wrapped to automatically extract
                these argument values from the actual runtime `ProcessArgs`.
            - :py:meth:`add_hidden`: add a Python function like `add_function`,
                but don't register any spec to make it hidden from the process listing.
            - :py:meth:`add_deprecated`: add a Python function like `add_hidden`
                and also wrap it so that warnings are triggered when used
        """
        if self.contains(name, namespace):
            if allow_override:
                _log.info(f"Overriding process {name} (namespace {namespace})")
            else:
                raise ProcessRegistryException(f"Process {name!r} already defined in namespace {namespace!r}")
        if spec:
            if name != spec["id"]:
                raise ProcessRegistryException(f"Process {name!r} has unexpected id {spec['id']!r}")
        if function and self._argument_names:
            sig = inspect.signature(function)
            arg_names = [n for n, p in sig.parameters.items() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
            if arg_names[: len(self._argument_names)] != self._argument_names:
                raise ProcessRegistryException(
                    f"Process {name!r} has invalid argument names: {arg_names}. Expected {self._argument_names}"
                )

        self._processes[_ProcessRegKey(name=name, namespace=namespace)] = ProcessData(function=function, spec=spec)

    def add_spec(self, spec: dict, *, namespace: str = DEFAULT_NAMESPACE):
        """
        Add process by specification dictionary (no function).

        Also see :py:meth:`add_process` for alternatives.
        """
        self.add_process(name=spec['id'], spec=spec, namespace=namespace)

    def add_spec_by_name(self, *names: str, namespace: str = DEFAULT_NAMESPACE):
        """
        Add process by name (process id). Multiple processes can be given.

        Also see :py:meth:`add_process` for alternatives.
        """
        for name in set(names):
            self.add_spec(self.load_predefined_spec(name), namespace=namespace)

    def add_function(
        self,
        f: Union[Callable, None] = None,
        *,
        name: Optional[str] = None,
        spec: Optional[dict] = None,
        namespace: str = DEFAULT_NAMESPACE,
        allow_override: bool = False,
    ) -> Callable:
        """
        Register the process corresponding with given function.
        Process name can be specified explicitly, otherwise the function name will be used.
        Process spec can be specified explicitly, otherwise it will be derived from function name.
        Can be used as function decorator.

        Also see :py:meth:`add_process` for alternatives.
        """
        if f is None:
            # Called as parameterized decorator
            return functools.partial(
                self.add_function, name=name, spec=spec, namespace=namespace, allow_override=allow_override
            )

        # TODO check if function arguments correspond with spec
        self.add_process(
            name=name or f.__name__,
            function=f,
            spec=spec or self.load_predefined_spec(name or f.__name__),
            namespace=namespace,
            allow_override=allow_override,
        )
        return f

    def add_simple_function(
        self,
        f: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        spec: Optional[dict] = None,
        namespace: str = DEFAULT_NAMESPACE,
    ):
        """
        Register a simple function that uses normal arguments instead of `args: ProcessArgs, env: EvalEnv`:
        wrap it in a wrapper that automatically extracts these arguments

        :param f:
        :param name: process_id (when guessing from `f.__name__` doesn't work)
        :param spec: optional spec dict

        Also see :py:meth:`add_process` for alternatives.
        """
        if f is None:
            # Called as parameterized decorator
            return functools.partial(self.add_simple_function, name=name, spec=spec)

        process_id = name or f.__name__
        # Detect arguments without and with defaults
        signature = inspect.signature(f)
        required = []
        defaults = {}
        for param in signature.parameters.values():
            if param.default is inspect.Parameter.empty:
                required.append(param.name)
            else:
                defaults[param.name] = param.default

        # TODO: can we generalize this assumption?
        assert self._argument_names == ["args", "env"]

        # TODO: option to also pass `env: EvalEnv` to `f`?
        def wrapped(args: Union[dict, ProcessArgs], env: EvalEnv):
            args = ProcessArgs.cast(args, process_id=process_id)
            # TODO: optional type checks based on type annotations too?
            kwargs = {a: args.get_required(name=a) for a in required}
            kwargs.update({a: args.get_optional(a, default=d) for a, d in defaults.items()})
            return f(**kwargs)

        spec = spec or self.load_predefined_spec(process_id)
        self.add_process(name=process_id, function=wrapped, spec=spec, namespace=namespace)
        return f

    def add_hidden(self, f: Callable, *, name: Optional[str] = None, namespace: str = DEFAULT_NAMESPACE):
        """
        Just register the function, but don't register spec for (public) listing.

        Also see :py:meth:`add_process` for alternatives.
        """
        self.add_process(name=name or f.__name__, function=f, spec=None, namespace=namespace)
        return f

    def add_deprecated(self, f: Callable = None, namespace: str = DEFAULT_NAMESPACE):
        """
        Register a deprecated function (non-public, throwing warnings).

        Also see :py:meth:`add_process` for alternatives.
        """
        if f is None:
            # Called as parameterized decorator
            return functools.partial(self.add_deprecated, namespace=namespace)

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            warnings.warn("Calling deprecated process function {f}".format(f=f.__name__))
            return f(*args, **kwargs)

        return self.add_hidden(wrapped, name=f.__name__, namespace=namespace)

    def get_spec(self, name: str, namespace: str = DEFAULT_NAMESPACE) -> dict:
        """Get spec dict of given process name"""
        if not self.contains(name, namespace) or self._get(name, namespace).spec is None:
            raise ProcessUnsupportedException(process=name, namespace=namespace)
        return self._get(name, namespace).spec

    def get_specs(
        self,
        *,
        substring: Optional[str] = None,
        namespace: str = DEFAULT_NAMESPACE,
        exclusion_list: Optional[Iterable[str]] = None,
    ) -> List[dict]:
        """Get all specs (or subset based on name substring/exclusion list)."""
        exclusion_list = set(exclusion_list or [])

        def id_match(p: ProcessData) -> bool:
            process_id = p.spec["id"]
            if substring and substring not in process_id:
                return False
            if process_id in exclusion_list:
                return False
            return True
        return [
            process_data.spec
            for k, process_data in self._processes.items()
            if k.namespace == namespace and process_data.spec and id_match(process_data)
        ]

    def get_function(self, name: str, namespace: str = DEFAULT_NAMESPACE) -> Callable:
        """Get Python function (if available) corresponding with given process name"""
        if not self.contains(name, namespace) or self._get(name, namespace).function is None:
            raise ProcessUnsupportedException(process=name, namespace=namespace)
        return self._get(name, namespace).function

    def get_processes_listing(self, *, exclusion_list: Optional[Iterable[str]] = None) -> ProcessesListing:
        return ProcessesListing(
            processes=self.get_specs(exclusion_list=exclusion_list),
            target_version=self.target_version,
        )


class ProcessesListing:
    def __init__(self, processes: List[dict], *, target_version: str):
        self.processes = processes
        self.target_version = target_version

    def to_response_dict(self) -> dict:
        return {
            "version": self.target_version,
            "processes": self.processes,
            "links": [],
        }


# Type annotation aliases
ArgumentValue = Any
Validator = Callable[[Any], bool]


class ProcessArgs(dict):
    """
    Wrapper for process argument extraction with proper exception throwing.

    Implemented as `dict` subclass to stay backwards compatible,
    but with additional methods for compact extraction
    """

    def __init__(self, args: dict, process_id: Optional[str] = None):
        super().__init__(args)
        self.process_id = process_id

    @classmethod
    def cast(cls, args: Union[dict, "ProcessArgs"], process_id: Optional[str] = None):
        if isinstance(args, ProcessArgs):
            assert args.process_id == process_id
        else:
            args = ProcessArgs(args=args, process_id=process_id)
        return args

    def get_required(
        self,
        name: str,
        *,
        expected_type: Optional[Union[type, Tuple[type, ...]]] = None,
        validator: Optional[Validator] = None,
    ) -> ArgumentValue:
        """
        Get a required argument by name.

        Originally: `extract_arg`.
        """
        # TODO: add option for type check too
        try:
            value = self[name]
        except KeyError:
            raise ProcessParameterRequiredException(process=self.process_id, parameter=name) from None
        self._check_value(name=name, value=value, expected_type=expected_type, validator=validator)
        return value

    def _check_value(
        self,
        *,
        name: str,
        value: Any,
        expected_type: Optional[Union[type, Tuple[type, ...]]] = None,
        validator: Optional[Validator] = None,
    ):
        if expected_type:
            if not isinstance(value, expected_type):
                if expected_type is DriverDataCube:
                    expected_type = "raster cube"
                raise ProcessParameterInvalidException(
                    parameter=name, process=self.process_id, reason=f"Expected {expected_type} but got {type(value)}."
                )
        if validator:
            reason = None
            try:
                valid = validator(value)
            except OpenEOApiException:
                # Preserve original OpenEOApiException
                raise
            except Exception as e:
                valid = False
                reason = str(e)
            if not valid:
                raise ProcessParameterInvalidException(
                    parameter=name, process=self.process_id, reason=reason or "Failed validation."
                )

    def get_optional(
        self,
        name: str,
        default: Union[Any, Callable[[], Any]] = None,
        *,
        expected_type: Optional[Union[type, Tuple[type, ...]]] = None,
        validator: Optional[Validator] = None,
    ) -> ArgumentValue:
        """
        Get an optional argument with default

        :param name: argument name
        :param default: default value or a function/factory to generate the default value
        :param expected_type: expected class (or list of multiple options) the value should be (unless it's None)
        :param validator: optional validation callable
        """
        if name in self:
            value = self.get(name)
        else:
            value = default() if callable(default) else default
        if value is not None:
            self._check_value(name=name, value=value, expected_type=expected_type, validator=validator)

        return value

    def get_deep(
        self,
        *steps: str,
        expected_type: Optional[Union[type, Tuple[type, ...]]] = None,
        validator: Optional[Validator] = None,
    ) -> ArgumentValue:
        """
        Walk recursively through a dictionary to get to a value.

        Originally: `extract_deep`
        """
        # TODO: current implementation requires the argument. Allow it to be optional too?
        value = self
        for step in steps:
            keys = [step] if not isinstance(step, list) else step
            for key in keys:
                if key in value:
                    value = value[key]
                    break
            else:
                raise ProcessParameterInvalidException(process=self.process_id, parameter=steps[0], reason=f"{step=}")

        self._check_value(name=steps[0], value=value, expected_type=expected_type, validator=validator)
        return value

    def get_aliased(self, names: List[str]) -> ArgumentValue:
        """
        Get argument by list of (legacy/fallback/...) names.

        Originally: `extract_arg_list`.
        """
        # TODO: support for default value?
        for name in names:
            if name in self:
                return self[name]
        raise ProcessParameterRequiredException(process=self.process_id, parameter=str(names))

    def get_subset(self, names: List[str], aliases: Optional[Dict[str, str]] = None) -> Dict[str, ArgumentValue]:
        """
        Extract subset of given argument names (where available) from given dictionary,
        possibly handling legacy aliases

        Originally: `extract_args_subset`

        :param names: keys to extract
        :param aliases: mapping of (legacy) alias to target key
        :return:
        """
        kwargs = {k: self[k] for k in names if k in self}
        if aliases:
            for alias, key in aliases.items():
                if alias in self and key not in kwargs:
                    kwargs[key] = self[alias]
        return kwargs

    def get_enum(
        self, name: str, options: Collection[ArgumentValue], default: Optional[ArgumentValue] = None
    ) -> ArgumentValue:
        """
        Get argument by name and check if it belongs to given set of (enum) values.

        Originally: `extract_arg_enum`
        """
        # TODO: use an "unset" sentinel value instead of None for default?
        if default is None:
            value = self.get_required(name=name)
        else:
            value = self.get_optional(name=name, default=default)
        if value not in options:
            raise ProcessParameterInvalidException(
                parameter=name,
                process=self.process_id,
                reason=f"Invalid enum value {value!r}. Expected one of {options}.",
            )
        return value

    @staticmethod
    def validator_generic(condition: Callable[[Any], bool], error_message: str) -> Validator:
        """
        Build validator function based on a condition (another validator)
        and a custom error message when validation returns False.
        (supports interpolation of actual value with "{actual}").
        """

        def validator(value):
            valid = condition(value)
            if not valid:
                raise ValueError(error_message.format(actual=value))
            return valid

        return validator

    @staticmethod
    def validator_one_of(options: list, show_value: bool = True) -> Validator:
        """Build a validator function that check that the value is in given list"""

        def validator(value) -> bool:
            if value not in options:
                if show_value:
                    message = f"Must be one of {options!r} but got {value!r}."
                else:
                    message = f"Must be one of {options!r}."
                raise ValueError(message)
            return True

        return validator

    @staticmethod
    def validator_file_format(formats: Union[List[str], Dict[str, dict]]) -> Validator:
        """
        Build validator for input/output format (case-insensitive check)

        :param formats list of valid formats, or dictionary with formats as keys
        """
        formats = list(formats)
        options = set(f.lower() for f in formats)

        def validator(value: str) -> bool:
            if value.lower() not in options:
                raise OpenEOApiException(
                    message=f"Invalid file format {value!r}. Allowed formats: {', '.join(formats)}",
                    code="FormatUnsuitable",
                    status_code=400,
                )
            return True

        return validator

    @staticmethod
    def validator_geojson_dict(
        allowed_types: Optional[Collection[str]] = None,
    ) -> Validator:
        """Build validator to verify that provided structure looks like a GeoJSON-style object"""

        def validator(value) -> bool:
            issues = validate_geojson_basic(value=value, allowed_types=allowed_types, raise_exception=False)
            if issues:
                raise ValueError(f"Invalid GeoJSON: {', '.join(issues)}.")
            return True

        return validator
