"""
Helpers to load default config values from environment variables
with some optional transformations.
"""

import os
from typing import Optional, Union, Callable, List, Mapping
import attrs


def _get_env() -> Mapping:
    return os.environ


def from_env(var: str, *, default=None) -> Callable[[], Optional[str]]:
    """
    Attrs default factory to get a value from an env var

    Usage example:

        >>> @attrs.define
        ... class Config:
        ...    colors: List[str] = attrs.field(factory=from_env("COLOR", default="red"))

        >>> Config().color
        "red"
        >>> os.environ["COLOR"] = "blue"
        >>> Config().color
        "blue"
        >>> Config(color="green").color
        "green"

    :param var: env var name
    :param default: fallback value if env var is not set
    :return: callable to be used with `attrs.field(factory=...)` or `attrs.Factory(...)`
    """

    def get():
        value = _get_env().get(var, default=default)
        return value

    return get


def to_list(value: str, *, strip: bool = True, separator: str = ",") -> List[str]:
    """Split a string to a list, properly handling leading/trailing whitespace and empty items"""
    result = value.split(separator)
    if strip:
        result = [s.strip() for s in result]
    result = [s for s in result if s]
    return result


def from_env_list(
    var: str, *, default: Union[str, List[str]] = "", strip: bool = True, separator: str = ","
) -> Callable[[], List[str]]:
    """
    Attrs default factory to get a list from an env var

    Usage example:

        >>> @attrs.define
        ... class Config:
        ...    colors: List[str] = attrs.field(
        ...        factory=from_env_list("COLORS", default="red,blue")
        ...    )

        >>> Config().colors
        ["red", "blue"]
        >>> os.environ["COLORS"] = "blue,black"
        >>> Config().color
        ["blue", "black"]
        >>> Config(colors=["green", "white"]).colors
        ["green", "white"]

    :param var: env var name
    :param default: fallback value to parse if env var is not set
    :param strip: whether to strip surrounding whitespace
    :param separator: item separator
    :return: callable to be used with `attrs.field(factory=...)` or `attrs.Factory(...)`
    """

    def get():
        value = _get_env().get(var, default=default)
        if isinstance(value, str):
            value = to_list(value, strip=strip, separator=separator)
        return value

    return get
