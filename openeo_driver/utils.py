"""
Small general utilities and helper functions
"""
import datetime
import importlib.metadata
import inspect
import json
import logging
import typing
import uuid
from math import isnan
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, Callable

from deprecated import deprecated
from openeo.util import rfc3339, Rfc3339


_log = logging.getLogger(__name__)


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

    def push_parameters(self, parameters: dict) -> "EvalEnv":
        """
        Shortcut method to push parameters, allowing quick discovery of places where parameters are pushed.
        """
        return self.push(parameters=parameters)

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
        return 0  # poorly hashable but load_collection's lru_cache is small anyway

    def __eq__(self, other) -> bool:
        return isinstance(other, EvalEnv) and self.as_dict() == other.as_dict()

    @property
    def backend_implementation(self) -> 'OpenEoBackendImplementation':
        return self["backend_implementation"]

class WhiteListEvalEnv(EvalEnv):
    """
    This copies white-listed values from an evalenv. Trying to retrieve non-whitelisted values results in a clear error,
    to indicate to implementors that an illegal value got extracted from the env.
    """

    def __init__(self, values: EvalEnv, whitelist):
        super().__init__({k: values[k] for k in whitelist if k in values}, None)
        self.whitelist = whitelist

    def __getitem__(self, key: str) -> Any:
        if key not in self.whitelist:
            raise KeyError(f"Your key: {key} was not in the whitelist {self.whitelist}. This error needs to be fixed in the backend.")
        return super().__getitem__(key)

    def get(self, key: str, default=None) -> Any:
        if key not in self.whitelist:
            raise KeyError(f"Your key: {key} was not in the whitelist {self.whitelist}. This error needs to be fixed in the backend.")
        return super().get(key, default)

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


def smart_bool(value: Any) -> bool:
    """
    Convert given value to a boolean value, like `bool()` builtin,
    but in case of strings: interpret some common cases as `False`:
    "0", "no", "off", "false", ...
    """
    if isinstance(value, str) and value.lower() in ["0", "no", "off", "false"]:
        return False
    else:
        return bool(value)




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


class dict_item:
    """
    "Descriptor" trick to easily add attribute-style access
    to standard dictionary items (with optional default values).

    Note: instead of this simple trick, consider using any of the more standard, widely used solutions
    like dataclasses from Python stdlib, or something like attrs (https://www.attrs.org).

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
    and helps to avoid key typo's (e.g. through code completion features in your
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

    # TODO: deprecate usage of this descriptor trick and migrate to dataclasses or attrs

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
        converters[Optional[datetime.datetime]] = lambda v: Rfc3339(propagate_none=True).parse_datetime(v)
    if convert_timedelta:
        converters[datetime.timedelta] = lambda v: datetime.timedelta(seconds=v)  # TODO: assumes always seconds?

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


@deprecated(reason="call generate_unique_id instead")
def generate_uuid(prefix: Optional[str] = None) -> str:
    return generate_unique_id(prefix)


def generate_unique_id(prefix: Optional[str] = None, date_prefix: bool = True) -> str:
    """
    Generate a random, unique identifier, to be used as job id, request id
    correlation id, error id, ...
    """
    id = uuid.uuid4().hex
    if date_prefix:
        date_repr = datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d")
        id = f"{date_repr}{id[len(date_repr):]}"
    if prefix:
        id = f"{prefix}-{id}"
    return id


def filter_supported_kwargs(callable: Callable, **kwargs) -> dict:
    """
    Check a callable's signature and only keep the kwargs that are supported by the callable.

    Helps with calling API functions (e.g. in MicroService subclasses/implementations)
    in a backward/forward compatible way when arguments are being deprecated/added.

    Note that this helper makes function calls less readable (compared to standard arg/kwarg usage),
    so usage should be minimized just to allow migration of all components to a new API version.
    """
    params = inspect.signature(callable).parameters
    return {
        k: v
        for k, v in kwargs.items()
        if k in params and params[k].kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]
    }
