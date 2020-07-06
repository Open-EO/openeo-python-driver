"""
Small general utilities and helper functions
"""
from datetime import datetime
import json
from math import isnan
from pathlib import Path
from typing import Union

from openeo.util import date_to_rfc3339


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


# TODO: use new Rfc3339 object?
date_to_rfc3339 = date_to_rfc3339


def parse_rfc3339(s) -> datetime:
    """Parse RFC-3339 formatted string to a datetime object """
    # TODO: move this to openeo client like date_to_rfc3339?
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')
