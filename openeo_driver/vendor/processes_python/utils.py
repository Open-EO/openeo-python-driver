# Adapted from openeo-processes-python (https://github.com/Open-EO/openeo-processes-python),
# release 0.0.4, licensed under Apache License 2.0.
# Only the numpy/pure-python code paths are retained (no xarray/dask/pandas).

import functools
import re
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np


def eval_datatype(data):
    """
    Returns a data type tag depending on the data type of `data`.
    This can be:
        - "numpy": `numpy.ndarray`
        - "xarray": `xarray.DataArray`
        - "dask": `dask.array.core.Array`
        - "int", "float", "dict", "list", "set", "tuple", "NoneType": Python builtins
        - "datetime": `datetime.datetime`
        - "function": callable object

    The xarray/dask tags are detected by module name, without importing these libraries.
    """
    package = type(data).__module__
    package_root = package.split(".", 1)[0]
    if package in ("builtins", "datetime"):
        return type(data).__name__
    elif package_root in ("numpy", "xarray", "dask"):
        return package_root
    else:
        return package + "." + type(data).__name__


# Registry of processes (dict mapping process id to wrapped process implementation).
_processes = {}


def process(processor):
    """
    Decorator for empty openEO process definitions, which call a class `processor` defining
    the process implementations for different data types (`exec_num` or `exec_np`).
    """

    @functools.wraps(processor)
    def fun_wrapper(*args, **kwargs):
        cls = processor()

        # Convert lists to numpy arrays
        args = tuple(list2nparray(a) if isinstance(a, list) else a for a in args)
        kwargs = {k: (list2nparray(v) if isinstance(v, list) else v) for k, v in kwargs.items()}

        # retrieve data types of input (keyword) arguments
        datatypes = set(eval_datatype(a) for a in args)
        datatypes.update(eval_datatype(v) for v in kwargs.values())
        if "numpy" in datatypes:
            cls_fun = getattr(cls, "exec_np")
        elif "xarray" in datatypes or "dask" in datatypes:
            raise NotImplementedError("xarray/dask data is not supported.")
        elif datatypes.issubset({"int", "float", "NoneType", "str", "bool", "datetime"}):
            cls_fun = getattr(cls, "exec_num")
        else:
            raise Exception("Datatype unknown.")

        return cls_fun(*args, **kwargs)

    process_id = processor.__name__.rstrip("_")
    _processes[process_id] = fun_wrapper

    return fun_wrapper


def has_process(process_id: str) -> bool:
    """Check if the given process is defined."""
    return process_id in _processes


def get_process(process_id: str) -> Callable:
    """Get the function corresponding with given process id."""
    return _processes[process_id]


def isnull(x):
    """
    Detect missing values (None, NaN, NaT), element-wise for array-likes.
    Minimal numpy-only replacement for `pandas.isnull`.
    """
    if x is None:
        return True
    arr = np.asarray(x)
    kind = arr.dtype.kind
    if kind in "fc":
        result = np.isnan(arr)
    elif kind in "mM":
        result = np.isnat(arr)
    elif kind == "O":

        def _is_null(v):
            return (
                v is None
                or (isinstance(v, (float, np.floating)) and bool(np.isnan(v)))
                or (isinstance(v, (np.datetime64, np.timedelta64)) and bool(np.isnat(v)))
            )

        result = np.asarray(np.frompyfunc(_is_null, 1, 1)(arr), dtype=bool)
    else:
        result = np.zeros(arr.shape, dtype=bool)
    if arr.ndim == 0:
        return bool(result)
    return result


def list2nparray(x):
    """
    Converts a list to a numpy array (string arrays are converted to object arrays).
    """
    x_tmp = np.array(x)
    if x_tmp.dtype.kind in ["U", "S"]:
        x = np.array(x, dtype=object)
    else:
        x = x_tmp

    return x


def create_slices(index, axis=0, n_axes=1):
    """
    Creates a multidimensional slice index.

    Parameters
    ----------
    index : int
        The zero-based index of the element to retrieve.
    axis : int, optional
        Axis of the given index (default is 0).
    n_axes : int, optional
        Number of axes (default is 1).

    Returns
    -------
    tuple of slice:
        Tuple of index slices.
    """
    slices = [slice(None)] * n_axes
    slices[axis] = index

    return tuple(slices)


def str2time(string, allow_24h=False):
    """
    Converts time strings in various formats to a datetime object.
    The datetime formats follow the RFC3339 convention.

    Parameters
    ----------
    string : str
        String representation of time or date.
    allow_24h : bool, optional
        If True, `string` is allowed to contain '24' as hour value.

    Returns
    -------
    datetime.datetime :
        Parsed datetime object (None if no format matched).
    """

    # handle timezone formatting and replace possibly occuring ":" in time zone string
    # handle timezone formatting for +
    if "+" in string:
        string_parts = string.split("+")
        string_parts[-1] = string_parts[-1].replace(":", "")
        string = "+".join(string_parts)

    # handle timezone formatting for -
    if "t" in string.lower():  # a full datetime string is given
        time_string = string[10:]
        if "-" in time_string:
            string_parts = time_string.split("-")
            string_parts[-1] = string_parts[-1].replace(":", "")
            string = string[:10] + "-".join(string_parts)
    else:  # a time string is given
        if "-" in string:
            string_parts = string.split("-")
            string_parts[-1] = string_parts[-1].replace(":", "")
            string = "-".join(string_parts)

    # searches for 24 in hour value
    pattern = re.compile(r"24:\d{2}:\d{2}")
    pattern_match = re.search(pattern, string)
    if pattern_match:
        if allow_24h:  # if the user allows 24 as an hour value, replace 24 by 23 and add a timedelta of one hour later
            old_sub_string = pattern_match.group()
            new_sub_string = "23" + old_sub_string[2:]
            string = string.replace(old_sub_string, new_sub_string)
        else:
            err_msg = (
                "24 is not allowed as an hour value. Hours are only allowed to be given in the range 0 - 23. "
                "Set 'allow_24h' to 'True' if you want to translate 24 as a an hour."
            )
            raise ValueError(err_msg)

    rfc3339_time_formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%Sz",
        "%Y-%m-%dt%H:%M:%SZ",
        "%Y-%m-%dt%H:%M:%Sz",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dt%H:%M:%S%z",
        "%H:%M:%SZ",
        "%H:%M:%S%z",
    ]
    date_time = None
    # loops through each format and takes the one for which the translation succeeded first
    for used_time_format in rfc3339_time_formats:
        try:
            date_time = datetime.strptime(string, used_time_format)
            if date_time.tzinfo is None:
                date_time = date_time.replace(tzinfo=timezone.utc)
            break
        except Exception:
            continue

    # add a timedelta of one hour if 24 is allowed as an hour value
    if date_time and allow_24h:
        date_time += timedelta(hours=1)

    return date_time


