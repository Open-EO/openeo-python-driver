from __future__ import annotations
from typing import Container, Callable


class Exclude(Container[str]):
    """
    Exclude list, implementing the `Container` interface,
    based on a callable (instead of an exhausive listing).
    """
    def __init__(self, exclude: Callable[[str], bool]):
        self.exclude = exclude

    def __contains__(self, item) -> bool:
        return self.exclude(item)

    @classmethod
    def by_prefix(cls, prefix: str) -> Exclude:
        return cls(exclude=lambda s: isinstance(s, str) and s.startswith(prefix))

    def union(self, other: Container[str]) -> Exclude:
        """
        Combine with another exclude list in logical OR fashion:
        an item is considered excluded when it is excluded by either this or the other exclude list.

        Note that ther other exclude list does not have to be an Exclude instance,
        it just has to implement the `Container` interface like a list or set.
        """
        return Exclude(exclude=lambda s: s in self or s in other)
