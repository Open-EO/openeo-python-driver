import functools
import logging
import threading
import time
import warnings
from typing import Union, Tuple, Any, Callable, Optional

import cachetools

_log = logging.getLogger(__name__)

# Typehint for cache keys: single string or tuple of strings
CacheKey = Union[str, Tuple[str, ...]]


class TtlCache:
    """
    In-memory key-value cache with TTL expiry and a maximum size, backed by
    :class:`cachetools.TTLCache`. When the cache is full, the least-recently-used
    item is evicted to make room for new entries.

    Cache interactions are thread-safe. The lock is intentionally *not* held while
    a cache-miss callback is executing, so slow callbacks do not block other readers.
    """

    def __init__(
        self,
        default_ttl: float = 60,
        *,
        max_size: int = 1000,
        _clock: Callable[[], float] = time.time,
    ):
        self.default_ttl = default_ttl
        self._cache: cachetools.TTLCache = cachetools.TTLCache(maxsize=max_size, ttl=default_ttl, timer=_clock)
        self._lock = threading.Lock()

    def set(self, key: CacheKey, value: Any, ttl: Optional[float] = None) -> None:
        """Store item in cache"""
        if ttl is not None:
            warnings.warn(
                "Per-item ttl is deprecated and will be ignored; use default_ttl on the cache instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        with self._lock:
            self._cache[key] = value

    def contains(self, key: CacheKey) -> bool:
        """Check whether cache contains a non-expired item under the given key."""
        with self._lock:
            return key in self._cache

    def get(self, key: CacheKey, default=None) -> Any:
        """Get item from cache; return *default* on a cache miss or expiry."""
        with self._lock:
            return self._cache.get(key, default)

    def get_or_call(
        self, key: CacheKey, callback: Callable[[], Any], ttl: Optional[float] = None
    ) -> Any:
        """
        Return the cached value for *key*, or call *callback* to build it on a miss.

        The lock is held only during cache look-up and result storage, **not** while
        *callback* is executing.  This means two concurrent callers may both experience
        a cache miss and both invoke the callback simultaneously; the last one to finish
        wins the store.  This is intentional — it avoids blocking other callers during
        potentially slow work.

        This method allows implementing the typical cache usage pattern in a single call::

            item = cache.get_or_call(
                key="foo",
                callback=lambda: expensive_operation(iterations=10000),
            )

        :param key: cache key (a string or a tuple of strings/ints)
        :param callback: callable invoked on a cache miss to produce the value
        :param ttl: deprecated; has no effect and will emit a :class:`DeprecationWarning`
        :return: the cached or freshly produced value
        """
        if ttl is not None:
            warnings.warn(
                "Per-item ttl is deprecated and will be ignored; use default_ttl on the cache instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        # Lock intentionally released before calling the callback.
        value = callback()
        with self._lock:
            self._cache[key] = value
        return value

    def flush(self) -> None:
        with self._lock:
            self._cache.clear()


def lru_cache_if_simple_args(func=None, *, maxsize: int = 128):
    """
    Decorator similar to `functools.lru_cache`, but only applies caching
    when all positional and keyword arguments are instances of simple
    built-in types (str, int, float, bool, None).

    If any argument is of another type (e.g. dict, list, or other
    complex/mutable objects), the function is executed without caching.
    Note that the standard `functools.lru_cache` would fail at runtime here
    with something like

        TypeError: unhashable type ...

    This is useful when a function is typically called with simple values,
    but occasionally receives more complex inputs that are not worth
    normalizing or converting into cacheable/hashable forms.
    """
    simple = (str, int, float, bool, type(None))

    def decorator(func):
        func_cached = functools.lru_cache(maxsize=maxsize)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if all(isinstance(a, simple) for a in args) and all(isinstance(v, simple) for v in kwargs.values()):
                return func_cached(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    # Support decorator usage both with and without parentheses:
    if func is not None:
        return decorator(func)
    else:
        return decorator
