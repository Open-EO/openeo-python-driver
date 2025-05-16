"""
Generic helpers to handle/consume/produce STAC items, collections, metadata constructs.
"""
from typing import Any

import collections.abc


def sniff_stac_extension_prefix(data: Any, prefix: str) -> bool:
    """
    Recursively walk through a data structure to
    find a particular STAC extension prefix
    in object keys (e.g. "eo:" in a "eo:bands" field of an asset).

    :param data: data structure to scan
    :param prefix: STAC extension prefix to look for,
        e.g. "eo:", "raster:", "proj:", ...
    """
    if isinstance(data, dict):
        if any(isinstance(k, str) and k.startswith(prefix) for k in data.keys()):
            return True
        return sniff_stac_extension_prefix(data=list(data.values()), prefix=prefix)
    elif isinstance(data, collections.abc.Iterable) and not isinstance(data, (str, bytes)):
        return any(sniff_stac_extension_prefix(data=x, prefix=prefix) for x in data)
    return False
