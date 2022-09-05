"""
Small general utilities and helper functions
"""
import datetime
import importlib.metadata
import json
import logging
import time
import typing
import uuid
from deprecated import deprecated
from enum import Enum
from json import JSONEncoder
from math import isnan
from pathlib import Path
from typing import Union, List, Tuple, Any, Optional

import pyproj
import shapely.geometry
import shapely.ops
from shapely.geometry import mapping
from shapely.geometry.base import CAP_STYLE, BaseGeometry

from openeo.util import rfc3339

_log = logging.getLogger(__name__)


class EvalEnvEncoder(JSONEncoder):
    """
    A custom json encoder in support of the __hash__ function. Does not aim to provide a completely representative json encoding.
    """
    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)

        if isinstance(o,BaseGeometry):
            return mapping(o)

        from openeo_driver.backend import OpenEoBackendImplementation
        from openeo_driver.dry_run import DryRunDataTracer
        if isinstance(o,OpenEoBackendImplementation) or isinstance(o,DryRunDataTracer):
            return str(o.__class__.__name__)

        from openeo_driver.datacube import DriverDataCube
        if isinstance(o,DriverDataCube):
            return str(o)

        from openeo_driver.users import User
        if isinstance(o,User):
            return o.user_id

        if isinstance(o, Enum):
            return o.value

        from openeo_driver.delayed_vector import DelayedVector
        if isinstance(o, DelayedVector):
            return o.path

            # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, o)


class EvalEnv:
    """
    Process graph evaluation environment: key-value container for keeping track
    of state/variables during evaluation of a process graph.

    The container itself is immutable and pushing new key-value pairs to it
    creates a new container (referencing the original one as parent).
    This layering of immutable key-value mappings allows
    "overwriting" keys when walking "up" from the result node of a process graph
    and restoring original values when walking "down" again.

    A common key is "parameters" under which the arguments of the
    current process should be pushed to build layered scopes
    of process arguments accessible through "from_parameter" references.
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
        """Create new EvalStack by pushing new values (as dict argument or through kwargs)"""
        merged = {**(values or {}), **kwargs}
        return EvalEnv(values=merged, parent=self)

    def collect(self, key: str) -> dict:
        """
        Walk the parent chain, collect the values (which must be dicts) for given key and combine to a single dict
        """
        d = self.get(key, default={})
        assert isinstance(d, dict)
        if self._parent:
            d = {**self._parent.collect(key=key), **d}
        return d

    def collect_parameters(self) -> dict:
        """Collect single dict of all parameters"""
        return self.collect("parameters")

    def as_dict(self) -> dict:
        if self._parent:
            return {**self._parent.as_dict(), **self._values}
        else:
            return self._values.copy()

    def __str__(self):
        return str(self.as_dict())

    def __hash__(self) -> int:
        return hash(json.dumps(self.as_dict(), sort_keys=True, cls=EvalEnvEncoder))

    @property
    def backend_implementation(self) -> 'OpenEoBackendImplementation':
        return self["backend_implementation"]


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
    with Path(filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def smart_bool(value):
    """Convert given value to bool, using a bit more interpretation for strings."""
    if isinstance(value, str) and value.lower() in ["0", "no", "off", "false"]:
        return False
    else:
        return bool(value)


def geojson_to_geometry(geojson: dict) -> shapely.geometry.base.BaseGeometry:
    """Convert GeoJSON object to shapely geometry object"""
    # TODO #71 #114 EP-3981 standardize on using (FeatureCollection like) vector cubes  instead of GeometryCollection?
    if geojson["type"] == "FeatureCollection":
        geojson = {
            'type': 'GeometryCollection',
            'geometries': [feature['geometry'] for feature in geojson['features']]
        }
    elif geojson["type"] == "Feature":
        geojson = geojson["geometry"]
    try:
        return shapely.geometry.shape(geojson)
    except Exception as e:
        _log.error(e,exc_info=True)
        _log.error(f"Invalid geojson: {json.dumps(geojson)}")
        raise ValueError(f"Invalid geojson object, the shapely library generated this error: {str(e)}. When trying to parse your geojson.")


def geojson_to_multipolygon(
        geojson: dict
) -> Union[shapely.geometry.MultiPolygon, shapely.geometry.Polygon]:
    """
    Convert GeoJSON object (dict) to shapely MultiPolygon (or Polygon where possible/allowed; in particular, this
    means dissolving overlapping polygons into one).
    """
    # TODO: option to also force conversion of Polygon to MultiPolygon?
    # TODO: #71 #114 migrate/centralize all this kind of logic to vector cubes
    if geojson["type"] == "Feature":
        geojson = geojson["geometry"]

    if geojson["type"] in ("MultiPolygon", "Polygon"):
        geometry = shapely.geometry.shape(geojson)
    elif geojson["type"] == "GeometryCollection":
        geometry = shapely.ops.unary_union(shapely.geometry.shape(geojson).geoms)
    elif geojson["type"] == "FeatureCollection":
        geometry = shapely.ops.unary_union([shapely.geometry.shape(f["geometry"]) for f in geojson["features"]])
    else:
        raise ValueError(f"Invalid GeoJSON type for MultiPolygon conversion: {geojson['type']}")

    if not isinstance(geometry, (shapely.geometry.MultiPolygon, shapely.geometry.Polygon)):
        raise ValueError(f"Failed to convert to MultiPolygon ({geojson['type']})")

    return geometry


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


def reproject_bounding_box(bbox: dict, from_crs: str, to_crs: str) -> dict:
    """
    Reproject given bounding box dictionary

    :param bbox: bbox dict with fields "west", "south", "east", "north"
    :param from_crs: source CRS. Specify `None` to use the "crs" field of input bbox dict
    :param to_crs: target CRS
    :return: bbox dict (fields "west", "south", "east", "north", "crs")
    """
    box = shapely.geometry.box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    if from_crs is None:
        from_crs = bbox["crs"]
    tranformer = pyproj.Transformer.from_crs(crs_from=from_crs, crs_to=to_crs, always_xy=True)
    reprojected = shapely.ops.transform(tranformer.transform, box)
    return dict(zip(["west", "south", "east", "north"], reprojected.bounds), crs=to_crs)


def spatial_extent_union(*bboxes: dict, default_crs="EPSG:4326") -> dict:
    """
    Calculate spatial bbox covering all given bounding boxes

    :return: bbox dict (fields "west", "south", "east", "north", "crs")
    """
    assert len(bboxes) >= 1
    crss = set(b.get("crs", default_crs) for b in bboxes)
    if len(crss) > 1:
        # Re-project to CRS of first bbox
        def reproject(bbox, to_crs):
            from_crs = bbox.get("crs", default_crs)
            if from_crs != to_crs:
                return reproject_bounding_box(bbox, from_crs=from_crs, to_crs=to_crs)
            return bbox

        to_crs = bboxes[0].get("crs", default_crs)
        bboxes = [reproject(b, to_crs=to_crs) for b in bboxes]
        crs = to_crs
    else:
        crs = crss.pop()
    bbox = {
        "west": min(b["west"] for b in bboxes),
        "south": min(b["south"] for b in bboxes),
        "east": max(b["east"] for b in bboxes),
        "north": max(b["north"] for b in bboxes),
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


def extract_namedtuple_fields_from_dict(
        d: dict, named_tuple_class: typing.Type[typing.NamedTuple],
        convert_datetime: bool = False, convert_timedelta: bool = False,
) -> dict:
    """
    Extract `typing.NamedTuple` fields from given dictionary,
    silently skipping items not defined as field.

    :param d: dictionary
    :param named_tuple_class: `typing.NamedTuple` subclass
    :return: subset of given dictionary (only containing fields defined by named tuple class)
    """

    field_names = set(named_tuple_class.__annotations__.keys())
    result = {k: v for k, v in d.items() if k in field_names}

    required = set(f for f in field_names if f not in named_tuple_class._field_defaults)
    missing = set(f for f in required if f not in result)
    if missing:
        raise KeyError(
            f"Missing {named_tuple_class.__name__} field{'s' if len(missing) > 1 else ''}: {', '.join(sorted(missing))}."
        )

    # Additional auto-conversions (by type annotation)
    converters = {}
    if convert_datetime:
        converters[datetime.datetime] = lambda v: rfc3339.parse_datetime(v)
    if convert_timedelta:
        converters[datetime.timedelta] = lambda v: datetime.timedelta(seconds=v)

    if converters:
        for k in result:
            converter = converters.get(named_tuple_class.__annotations__.get(k))
            if converter:
                result[k] = converter(result[k])

    return result


def get_package_versions(packages: List[str], na_value="n/a") -> dict:
    """Get (installed) version number of each Python package (where possible)."""
    version_info = {}
    for package in packages:
        try:
            version_info[package] = importlib.metadata.version(distribution_name=package)
        except importlib.metadata.PackageNotFoundError:
            version_info[package] = na_value
    return version_info


class TtlCache:
    """
    Simple memory cache with expiry
    """

    def __init__(self, default_ttl: int = 60, _clock: typing.Callable[[], float] = time.time):
        self._cache = {}
        self.default_ttl = default_ttl
        self._clock = _clock

    def set(self, key, value, ttl: int = None):
        """Add item to cache"""
        self._cache[key] = (value, self._clock() + (ttl or self.default_ttl))

    def contains(self, key) -> bool:
        """Check whether cache contains item under given key"""
        if key in self._cache:
            value, expiration = self._cache[key]
            if self._clock() <= expiration:
                return True
            del self._cache[key]
        return False

    def get(self, key, default=None):
        """Get item from cache and if not available: return default value."""
        return self._cache[key][0] if self.contains(key) else default

    def flush(self):
        self._cache = {}


def buffer_point_approx(point: shapely.geometry.Point, point_crs: str, buffer_distance_in_meters=10.0) -> shapely.geometry.Polygon:
    src_proj = pyproj.Proj("EPSG:3857")
    dst_proj = pyproj.Proj(point_crs)

    def reproject_point(x, y):
        return pyproj.transform(
            src_proj,
            dst_proj,
            x, y,
            always_xy=True
        )

    #this is too approximate in a lot of cases, better to use utm zones?
    left, _ = reproject_point(0.0, 0.0)
    right, _ = reproject_point(buffer_distance_in_meters, 0.0)

    buffer_distance = right - left

    return point.buffer(buffer_distance, cap_style=CAP_STYLE.square)


@deprecated(reason="call generate_unique_id instead")
def generate_uuid(prefix: Optional[str] = None) -> str:
    return generate_unique_id(prefix)


def generate_unique_id(prefix: Optional[str] = None) -> str:
    """
    Generate a random, unique identifier, to be used as job id, request id
    correlation id, error id, ...
    """
    id = uuid.uuid4().hex
    if prefix:
        id = f"{prefix}-{id}"
    return id
