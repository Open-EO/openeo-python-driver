from collections import namedtuple
import functools
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import warnings

from openeo_driver.errors import ProcessUnsupportedException
from openeo_driver.specs import SPECS_ROOT


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


class ProcessRegistry:
    """
    Registry for processes we support in the backend.

    Basically a dictionary of process specification dictionaries
    """

    def __init__(self, spec_root: Path = SPECS_ROOT / 'openeo-processes/1.0'):
        self._processes_spec_root = spec_root
        # Dictionary process_name -> ProcessData
        self._processes: Dict[str, ProcessData] = {}

    def load_predefined_spec(self, name: str) -> dict:
        """Get predefined process specification (dict) based on process name."""
        try:
            with (self._processes_spec_root / '{n}.json'.format(n=name)).open('r', encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            raise RuntimeError("Failed to load predefined spec of process {n!r}".format(n=name))

    def list_predefined_specs(self) -> Dict[str, Path]:
        """List all processes with a spec JSON file."""
        return {p.stem: p for p in self._processes_spec_root.glob("*.json")}

    def add_process(self, name: str, function: Callable = None, spec: dict = None):
        """
        Add a process to the registry, with callable function (optional) and specification dict (optional)
        """
        assert name not in self._processes, name
        if spec:
            # Basic health check
            assert all(k in spec for k in ['id', 'description', 'parameters', 'returns'])
            assert name == spec['id']
        self._processes[name] = ProcessData(function=function, spec=spec)

    def add_spec(self, spec: dict):
        """Add process specification dictionary."""
        self.add_process(name=spec['id'], spec=spec)

    def add_spec_by_name(self, *names: str):
        """Add process by name. Multiple processes can be given."""
        for name in set(names):
            self.add_spec(self.load_predefined_spec(name))

    def add_function(self, f: Callable=None, name: str = None, spec: dict = None) -> Callable:
        """
        Register the process corresponding with given function.
        Process name can be specified explicitly, otherwise the function name will be used.
        Process spec can be specified explicitly, otherwise it will be derived from function name.
        Can be used as function decorator.
        """
        if f is None:
            # Called as parameterized decorator
            return functools.partial(self.add_function, name=name, spec=spec)

        # TODO check if function arguments correspond with spec
        self.add_process(
            name=name or f.__name__,
            function=f,
            spec=spec or self.load_predefined_spec(name or f.__name__)
        )
        return f

    def add_hidden(self, f: Callable, name: str = None):
        """Just register the function, but don't register spec for (public) listing."""
        self.add_process(name=name or f.__name__, function=f, spec=None)
        return f

    def add_deprecated(self, f: Callable):
        """Register a deprecated function (non-public, throwing warnings)."""

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            warnings.warn("Calling deprecated process function {f}".format(f=f.__name__))
            return f(*args, **kwargs)

        return self.add_hidden(wrapped, name=f.__name__)

    def get_spec(self, name: str) -> dict:
        """Get spec dict of given process name"""
        if name not in self._processes or self._processes[name].spec is None:
            raise ProcessUnsupportedException(process=name)
        return self._processes[name].spec

    def get_specs(self, substring: str = None) -> List[dict]:
        """Get all specs (or subset based on name substring)."""
        return [
            process_data.spec
            for process_data in self._processes.values()
            if process_data.spec and (not substring or substring.lower() in process_data.spec['id'])
        ]

    def get_function(self, name: str) -> Callable:
        """Get Python function (if available) corresponding with given process name"""
        if name not in self._processes or self._processes[name].function is None:
            raise ProcessUnsupportedException(process=name)
        return self._processes[name].function


class UserDefinedProcessRegistry:
    """Registry for user defined processes"""

    def __init__(self):
        # Starting simple with in-memory mapping of (username, process_id) -> process_spec
        # TODO: use some persistent storage backend instead of in-memory dict
        self._processes: Dict[Tuple[str, str], dict] = {}

    def add_udp(self, user_id: str, process_id: str, spec: dict):
        self._processes[user_id, process_id] = spec

    def has_udp(self, user_id: str, process_id: str) -> bool:
        return (user_id, process_id) in self._processes

    def get_udp_spec(self, user_id: str, process_id: str) -> dict:
        return self._processes[user_id, process_id]
