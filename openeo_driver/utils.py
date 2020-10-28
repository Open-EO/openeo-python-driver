"""
Small general utilities and helper functions
"""
import json
from functools import reduce
from math import isnan
from pathlib import Path
from typing import Union, Any, List, Tuple

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


def bands_union(*args: List[str]) -> List[str]:
    """Take union of given lists/sets of bands"""
    bands = []
    for arg in args:
        for a in arg:
            if a not in bands:
                bands.append(a)
    return bands


def temporal_extent_union(
        *args: Tuple[Union[str, None], Union[str, None]], none_is_infinity=True
) -> Tuple[Union[str, None], Union[str, None]]:
    """Calculate temporal extent covering all given extents"""
    # TODO: handle datetime values as well?
    if len(args) == 0:
        return None, None
    starts, ends = zip(*args)

    if none_is_infinity:
        start = None if None in starts else min(starts)
        end = None if None in ends else max(ends)
    else:
        start = min(s for s in starts if s is not None)
        end = max(s for s in ends if s is not None)
    return start, end


def spatial_extent_union(*args: dict) -> dict:
    """
    Calculate spatial bbox covering all given bounding boxes

    :return: bbox dict (fields "west", "south", "east", "north", "crs")
    """
    # TODO: assuming CRS where west/south is lower and east/north is higher.
    # TODO: smarter CRS handling/combining
    assert len(args) >= 1
    crss = set(a.get("crs", "EPSG:4326") for a in args)
    if len(crss) > 1:
        raise ValueError("Different CRS's: {c}".format(c=crss))
    crs = crss.pop()
    bbox = {
        "west": min(a["west"] for a in args),
        "south": min(a["south"] for a in args),
        "east": max(a["east"] for a in args),
        "north": max(a["north"] for a in args),
        "crs": crs
    }
    return bbox


class dict_item:
    """
    Create an attribute in a custom dict subclass that accesses
    the dict value keyed by the attribute's name:
    
        >>> class UserInfo(dict):
        >>>     name = dict_item()
        >>>     age = dict_item()
        >>> user = UserInfo(name="John")
        >>> print(user.name)
        John
        >>> user.age = 42
        >>> user["color"] = "green"
        >>> print(user)
        {"name":"John", "age": 42, "color": "green"}

    `user` acts as a normal dictionary, but the items under keys "name" and "age"
    can be accessed as attributes too.

    `dict_item` allows to easily create/prototype dict-based data structures
    that have some predefined (but possibly missing) fields as attributes.
    This makes the data structure more self-documenting than a regular dict
    and helps avoiding key typo's (e.g. through code completion features in your
    editor or IDE).

    `dict_item` also allows to specify a default value which will be returned
    when accessing the value as attribute if the item is not set in the dict:

        >>> class UserInfo(dict):
        >>>     name = dict_item(default="John Doe")
        >>> user = UserInfo()
        >>> print(user.name)
        John Doe
        >>> print(user["name"])
        KeyError: 'name'


    This class implements the descriptor protocol.
    """
    _DEFAULT_UNSET = object()

    def __init__(self, default=_DEFAULT_UNSET):
        self.default = default

    def __set_name__(self, owner, name):
        self.key = name

    def __get__(self, instance, owner):
        if self.default is not self._DEFAULT_UNSET:
            return instance.get(self.key, self.default)
        else:
            return instance[self.key]

    def __set__(self, instance, value):
        instance[self.key] = value
