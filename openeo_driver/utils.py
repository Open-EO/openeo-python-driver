"""
Small general utilities and helper functions
"""
import json
from math import isnan
from pathlib import Path
from typing import Union, Any

import shapely.geometry


class EvalEnv:
    """
    Process graph evaluation environment: key-value container for keeping track
    of state/variables during evaluation of a process graph.

    The container itself is immutable and pushing new key-value pairs to it
    creates a new container (referencing the original one as parent).
    This layering of immutable key-value mappings allows
    "overwriting" keys when walking "up" from the result node of a process graph
    and restoring original values when walking "down" again.
    """

    def __init__(self, values: dict = None, parent: 'EvalEnv' = None):
        self._values = dict(values or [])
        self._parent = parent

    def __contains__(self, key) -> bool:
        return key in self._values or (self._parent and key in self._parent)

    def __getitem__(self, key: str) -> Any:
        if key in self._values:
            return self._values[key]
        elif self._parent:
            return self._parent[key]
        else:
            raise KeyError(key)

    def get(self, key: str, default=None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def push(self, values: dict = None, **kwargs) -> 'EvalEnv':
        """Create new EvalStack by pushing new values (as dict argument or through kwargs"""
        merged = {**(values or {}), **kwargs}
        return EvalEnv(values=merged, parent=self)

    def as_dict(self) -> dict:
        if self._parent:
            return {**self._parent.as_dict(), **self._values}
        else:
            return self._values.copy()


def replace_nan_values(o):
    """

    :param o:
    :return:
    """

    if isinstance(o, float) and isnan(o):
        return None

    if isinstance(o, str):
        return o

    if isinstance(o, dict):
        return {replace_nan_values(key): replace_nan_values(value) for key, value in o.items()}

    try:
        return [replace_nan_values(elem) for elem in o]
    except TypeError:
        pass

    return o


def read_json(filename: Union[str, Path]) -> Union[dict, list]:
    """Read a dict or list from a JSON file"""
    with Path(filename).open(encoding='utf-8') as f:
        return json.load(f)


def smart_bool(value):
    """Convert given value to bool, using a bit more interpretation for strings."""
    if isinstance(value, str) and value.lower() in ["0", "no", "off", "false"]:
        return False
    else:
        return bool(value)


def geojson_to_geometry(geojson: dict) -> shapely.geometry.base.BaseGeometry:
    """Convert GeoJSON object to shapely geometry object"""
    if geojson["type"] == "FeatureCollection":
        geojson = {
            'type': 'GeometryCollection',
            'geometries': [feature['geometry'] for feature in geojson['features']]
        }
    return shapely.geometry.shape(geojson)


def to_hashable(obj):
    """
    Convert nested data structure (e.g. with dicts and lists)
    to something immutable and hashable (tuples, ...)
    """
    if isinstance(obj, (int, float, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return tuple(to_hashable(x) for x in obj)
    elif isinstance(obj, dict):
        return tuple((k, to_hashable(v)) for (k, v) in sorted(obj.items()))
    elif isinstance(obj, set):
        return to_hashable(sorted(obj))
    else:
        raise ValueError(obj)
