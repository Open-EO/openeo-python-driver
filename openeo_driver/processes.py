import functools
import warnings
from typing import Callable

import requests


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


class ProcessRegistry:
    """
    Registry for processes we support in the backend.

    Basically a dictionary of process specification dictionaries
    """

    def __init__(self):
        self._specs = {}
        self._functions = {}

    def _fetch_spec(self, name: str) -> dict:
        """Get process specification (dict) based on process name."""
        try:
            # TODO use git submodule of openeo-processes project instead of doing HTTP requests
            url = 'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/{n}.json'.format(n=name)
            spec = requests.get(url).json()
        except Exception:
            raise RuntimeError("Failed to get spec of process {n!r}".format(n=name))
        return spec

    def add_spec(self, spec: dict):
        """Add process specification dictionary."""
        # Basic health check
        assert all(k in spec for k in ['id', 'description', 'parameters', 'returns'])
        self._specs[spec['id']] = spec

    def add_by_name(self, name):
        """Add process by name"""
        self.add_spec(self._fetch_spec(name))

    def add_function(self, f: Callable):
        """To be used as function decorator: register the process corresponding the function name."""
        # TODO check if function arguments correspond with spec
        self.add_by_name(f.__name__)
        self._functions[f.__name__] = f
        return f

    def add_function_with_spec(self, spec: ProcessSpec):
        """To be used as function decorator: register a custom process based on function name and given spec."""

        def decorator(f: Callable):
            assert f.__name__ == spec.id
            self.add_spec(spec.to_dict())
            self._functions[f.__name__] = f
            return f

        return decorator

    def add_deprecated(self, f: Callable):
        """To be used as function decorator: just register the function for callback, but don't register the spec."""

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            warnings.warn("Calling deprecated process function {f}".format(f=f.__name__))
            return f(*args, **kwargs)

        self._functions[f.__name__] = wrapped
        return f

    def get_spec(self, name):
        """Get spec of given process name"""
        if name not in self._specs:
            raise NoSuchProcessException(name)
        return self._specs[name]

    def get_function(self, name):
        """Get Python function (if available) corresponding with given process name"""
        if name not in self._functions:
            raise NoSuchProcessException(name)
        return self._functions[name]


class NoSuchProcessException(ValueError):
    pass
