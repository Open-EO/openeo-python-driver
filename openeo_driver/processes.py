import functools
import inspect
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from openeo_driver.errors import ProcessUnsupportedException
from openeo_driver.specs import SPECS_ROOT
from openeo_driver.utils import read_json


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

    def __init__(self, id, description):
        self.id = id
        self.description = description
        self._parameters: List[ProcessParameter] = []
        self._returns = None

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
        if len(self._parameters) == 0:
            warnings.warn("Process with no parameters")
        assert self._returns is not None
        return {
            "id": self.id,
            "description": self.description,
            "parameters": {
                p.name: {"description": p.description, "schema": p.schema, "required": p.required}
                for p in self._parameters
            },
            "parameter_order": [p.name for p in self._parameters],
            "returns": self._returns
        }

    def to_dict_100(self) -> dict:
        """Generate process spec as (JSON-able) dictionary (API 1.0.0 style)."""
        if len(self._parameters) == 0:
            warnings.warn("Process with no parameters")
        assert self._returns is not None
        return {
            "id": self.id,
            "description": self.description,
            "parameters": [
                {"name": p.name, "description": p.description, "schema": p.schema, "optional": not p.required}
                for p in self._parameters
            ],
            "returns": self._returns
        }


ProcessData = namedtuple("ProcessData", ["function", "spec"])


class ProcessRegistryException(Exception):
    pass


DEFAULT_NAMESPACE = "backend"


class ProcessRegistry:
    """
    Registry for processes we support in the backend.

    Basically a dictionary of process specification dictionaries
    """

    def __init__(self, spec_root: Path = SPECS_ROOT / 'openeo-processes/1.x', argument_names: List[str] = None):
        self._processes_spec_root = spec_root
        # Dictionary (namespace, process_name) -> ProcessData
        self._processes: Dict[Tuple[str, str], ProcessData] = {}
        # Expected argument names that process function signature should start with
        self._argument_names = argument_names

    def _key(self, name: str, namespace: str = DEFAULT_NAMESPACE) -> tuple:
        """Lookup key for in `_processes` dict"""
        return namespace, name

    def contains(self, name: str, namespace: str = DEFAULT_NAMESPACE) -> bool:
        return self._key(name=name, namespace=namespace) in self._processes

    def _get(self, name: str, namespace: str = DEFAULT_NAMESPACE) -> ProcessData:
        return self._processes[self._key(name=name, namespace=namespace)]

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

    def add_process(self, name: str, function: Callable = None, spec: dict = None, namespace: str = DEFAULT_NAMESPACE):
        """
        Add a process to the registry, with callable function (optional) and specification dict (optional)
        """
        if self.contains(name, namespace):
            raise ProcessRegistryException(f"Process {name!r} already defined in namespace {namespace!r}")
        if spec:
            assert name == spec['id']
        if function and self._argument_names:
            sig = inspect.signature(function)
            arg_names = [n for n, p in sig.parameters.items() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
            if arg_names[:len(self._argument_names)] != self._argument_names:
                raise ProcessRegistryException("Process {p!r} has invalid argument names: {a}".format(
                    p=name, a=arg_names
                ))

        self._processes[self._key(name=name, namespace=namespace)] = ProcessData(function=function, spec=spec)

    def add_spec(self, spec: dict, namespace: str = DEFAULT_NAMESPACE):
        """Add process specification dictionary."""
        self.add_process(name=spec['id'], spec=spec, namespace=namespace)

    def add_spec_by_name(self, *names: str, namespace: str = DEFAULT_NAMESPACE):
        """Add process by name. Multiple processes can be given."""
        for name in set(names):
            self.add_spec(self.load_predefined_spec(name), namespace=namespace)

    def add_function(
            self, f: Callable = None, name: str = None, spec: dict = None, namespace: str = DEFAULT_NAMESPACE
    ) -> Callable:
        """
        Register the process corresponding with given function.
        Process name can be specified explicitly, otherwise the function name will be used.
        Process spec can be specified explicitly, otherwise it will be derived from function name.
        Can be used as function decorator.
        """
        if f is None:
            # Called as parameterized decorator
            return functools.partial(self.add_function, name=name, spec=spec, namespace=namespace)

        # TODO check if function arguments correspond with spec
        self.add_process(
            name=name or f.__name__,
            function=f,
            spec=spec or self.load_predefined_spec(name or f.__name__),
            namespace=namespace
        )
        return f

    def add_hidden(self, f: Callable, name: str = None, namespace: str = DEFAULT_NAMESPACE):
        """Just register the function, but don't register spec for (public) listing."""
        self.add_process(name=name or f.__name__, function=f, spec=None, namespace=namespace)
        return f

    def add_deprecated(self, f: Callable = None, namespace: str = DEFAULT_NAMESPACE):
        """Register a deprecated function (non-public, throwing warnings)."""
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

    def get_specs(self, substring: str = None, namespace: str = DEFAULT_NAMESPACE) -> List[dict]:
        """Get all specs (or subset based on name substring)."""
        id_match = (lambda p: True) if substring is None else (lambda p: substring.lower() in p.spec["id"].lower())
        return [
            process_data.spec
            for (ns, n), process_data in self._processes.items()
            if ns == namespace and process_data.spec and id_match(process_data)
        ]

    def get_function(self, name: str, namespace: str = DEFAULT_NAMESPACE) -> Callable:
        """Get Python function (if available) corresponding with given process name"""
        if not self.contains(name, namespace) or self._get(name, namespace).function is None:
            raise ProcessUnsupportedException(process=name, namespace=namespace)
        return self._get(name, namespace).function
