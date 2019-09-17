"""
Small general utilities and helper functions
"""

import json
from math import isnan
from pathlib import Path
from typing import Union


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
