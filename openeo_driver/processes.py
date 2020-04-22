from collections import namedtuple
import functools
import json
from pathlib import Path
from typing import Callable, Dict, List
import warnings

from openeo_driver.errors import ProcessUnsupportedException
from openeo_driver.specs import SPECS_ROOT


class ProcessSpec:
    """
    Helper object to easily build a process specification with a fluent/chained API.

    Intended for custom processes that are not specified in the official OpenEo Processes listing.
    """

    # Some predefined parameter schema's
    RASTERCUBE = {"type": "object", "format": "raster-cube"}

    class Parameter:
        """Process Parameter."""

        def __init__(self, name: str, description: str, schema: dict, required: bool = True):
            self.name = name
            self.description = description
            self.schema = schema
            self.required = required

        def to_dict(self):
            return {"description": self.description, "schema": self.schema, "required": self.required}

    def __init__(self, id, description):
        self.id = id
        self.description = description
        self._parameters = []
        self._returns = None

    def param(self, name, description, schema, required=True) -> 'ProcessSpec':
        """Add a process parameter"""
        self._parameters.append(self.Parameter(name, description, schema, required))
        return self

    def returns(self, description: str, schema: dict) -> 'ProcessSpec':
        """Define return spec."""
        self._returns = {"description": description, "schema": schema}
        return self

    def to_dict(self) -> dict:
        """Generate process spec as (JSON-able) dictionary."""
        if len(self._parameters) == 0:
            warnings.warn("Process with no parameters")
        assert self._returns is not None
        return {
            "id": self.id,
            "description": self.description,
            "parameters": {
                p.name: p.to_dict()
                for p in self._parameters
            },
            "parameter_order": [p.name for p in self._parameters],
            "returns": self._returns
        }


ProcessData = namedtuple("ProcessData", ["function", "spec"])


class ProcessRegistry:
    """
    Registry for processes we support in the backend.

    Basically a dictionary of process specification dictionaries
    """

    def __init__(self):
        self._processes_spec_root = SPECS_ROOT / 'openeo-processes/0.4'
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

    def _add_process(self, name: str, function: Callable = None, spec: dict = None):
        """Add ProcessData"""
        assert name not in self._processes
        self._processes[name] = ProcessData(function=function, spec=spec)

    def add_spec(self, spec: dict):
        """Add process specification dictionary."""
        # Basic health check
        assert all(k in spec for k in ['id', 'description', 'parameters', 'returns'])
        self._add_process(name=spec['id'], spec=spec)

    def add_spec_by_name(self, name):
        """Add process by name"""
        self.add_spec(self.load_predefined_spec(name))

    def add_function(self, f: Callable):
        """To be used as function decorator: register the process corresponding with the function name."""
        # TODO check if function arguments correspond with spec
        self._add_process(
            name=f.__name__,
            function=f,
            spec=self.load_predefined_spec(f.__name__)
        )
        return f

    def add_function_with_spec(self, spec: ProcessSpec):
        """To be used as function decorator: register a custom process based on function name and given spec."""

        def decorator(f: Callable):
            assert f.__name__ == spec.id
            self._add_process(name=f.__name__, function=f, spec=spec.to_dict())
            return f

        return decorator

    def add_deprecated(self, f: Callable):
        """To be used as function decorator: just register the function, but don't register spec for (public) listing."""

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            warnings.warn("Calling deprecated process function {f}".format(f=f.__name__))
            return f(*args, **kwargs)

        self._add_process(name=f.__name__, function=wrapped)
        return f

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
