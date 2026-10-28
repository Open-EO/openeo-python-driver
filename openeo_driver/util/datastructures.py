"""
Generic data structure utilities
"""
from typing import Iterable, Any, List


def unique_by_equality(items: Iterable[Any]) -> List[Any]:
    """
    Return a list of unique items based on regular equality comparison.
    To be used when a classic `set()` approach is not possible
    because items are not guaranteed to be hashable.
    """
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
