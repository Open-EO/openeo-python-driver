import logging
import time
from typing import Union, Sequence, Tuple, Any, Callable, Dict, Optional

_log = logging.getLogger(__name__)

# Typehint for cache keys: single string or tuple of strings
CacheKey = Union[str, Tuple[str, ...]]


class TtlCache:
    """
    Simple dictionary based, in-memory key-value cache with expiry.
    """

    def __init__(
        self, default_ttl: float = 60, _clock: Callable[[], float] = time.time
    ):
        self._cache: Dict[CacheKey, Tuple[Any, float]] = {}
        self.default_ttl = default_ttl
        self._clock = _clock

    def set(self, key: CacheKey, value: Any, ttl: Optional[float] = None) -> None:
        """Store item in cache"""
        self._cache[key] = (value, self._clock() + (ttl or self.default_ttl))

    def contains(self, key: CacheKey) -> bool:
        """Check whether cache contains item under given key"""
        if key in self._cache:
            value, expiration = self._cache[key]
            if self._clock() <= expiration:
                return True
            del self._cache[key]
        return False

    def get(self, key: CacheKey, default=None) -> Any:
        """Get item from cache and if not available: return default value."""
        # TODO: raise KeyError on cache miss?
        return self._cache[key][0] if self.contains(key) else default

    def get_or_call(
        self, key: CacheKey, callback: Callable[[], Any], ttl: Optional[float] = None
    ) -> Any:
        if self.contains(key):
            value = self._cache[key][0]
        else:
            value = callback()
            self.set(key=key, value=value, ttl=ttl)
        return value

    def flush(self):
        self._cache = {}
