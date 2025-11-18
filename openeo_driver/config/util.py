from __future__ import annotations
from typing import Container, Callable


class Exclude(Container[str]):
    def __init__(self, exclude: Callable[[str], bool]):
        self.exclude = exclude

    def __contains__(self, item) -> bool:
        return self.exclude(item)

    @classmethod
    def by_prefix(cls, prefix: str) -> Exclude:
        return cls(lambda s: isinstance(s, str) and s.startswith(prefix))
