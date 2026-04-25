"""Text operation process implementations."""
from typing import List, Union

from openeo_driver.processgraph.registry import process_registry_100, process_registry_2xx, simple_function
from openeo_driver.specs import read_spec


@simple_function
def text_begins(data: str, pattern: str, case_sensitive: bool = True) -> Union[bool, None]:
    if data is None:
        return None
    if not case_sensitive:
        data = data.lower()
        pattern = pattern.lower()
    return data.startswith(pattern)


@simple_function
def text_contains(data: str, pattern: str, case_sensitive: bool = True) -> Union[bool, None]:
    if data is None:
        return None
    if not case_sensitive:
        data = data.lower()
        pattern = pattern.lower()
    return pattern in data


@simple_function
def text_ends(data: str, pattern: str, case_sensitive: bool = True) -> Union[bool, None]:
    if data is None:
        return None
    if not case_sensitive:
        data = data.lower()
        pattern = pattern.lower()
    return data.endswith(pattern)


@process_registry_100.add_simple_function
def text_merge(
        data: List[Union[str, int, float, bool, None]],
        separator: Union[str, int, float, bool, None] = ""
) -> str:
    # TODO #196 text_merge is deprecated in favor of text_concat
    return str(separator).join(str(d) for d in data)


@process_registry_100.add_simple_function(spec=read_spec("openeo-processes/2.x/text_concat.json"))
@process_registry_2xx.add_simple_function
def text_concat(
    data: List[Union[str, int, float, bool, None]],
    separator: str = "",
) -> str:
    return str(separator).join(str(d) for d in data)
