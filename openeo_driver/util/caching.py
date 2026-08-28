import functools
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cachetools

_log = logging.getLogger(__name__)

# Typehint for cache keys: single string or tuple of strings
CacheKey = Union[str, Tuple[str, ...]]


class TtlCache:
    """
    Simple dictionary based, in-memory key-value cache with expiry.
    """

    def __init__(self, default_ttl: float = 60, _clock: Callable[[], float] = time.time):
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

    def get_or_call(self, key: CacheKey, callback: Callable[[], Any], ttl: Optional[float] = None) -> Any:
        """
        Try to get item from cache. If not available or expired: call callback to build it and store result in cache.

        This method allows to implement typicall cache usage pattern in a single call:

            item = cache.get_or_call(
                key="foo",
                callback=lambda: expensive_operation(iterations=10000)
            )

        :param key: key to store item at (can be a simple string,
            or something more complex like a tuple of strings/ints)
        :param callback: item builder to call when item is not in cache or expired
        :param ttl: optionally override default TTL
        :return: item (from cache or freshly built)
        """
        if self.contains(key):
            value = self._cache[key][0]
        else:
            value = callback()
            self.set(key=key, value=value, ttl=ttl)
        return value

    def flush(self):
        self._cache = {}


class BoundedTtlCache:
    """
    Thread-safe, in-memory key-value cache with TTL expiry and a maximum size,
    backed by :class:`cachetools.TTLCache`.

    When the cache is full the least-recently-used item is evicted to make room
    for new entries.  All constructor arguments are keyword-only.

    Cache interactions are protected by a :class:`threading.Lock`.  The lock is
    intentionally *not* held while a cache-miss callback executes in
    :meth:`get_or_call`, so slow callbacks do not block other readers.  As a
    consequence two concurrent callers may both experience a cache miss and both
    invoke the callback; the last writer wins.
    """

    def __init__(self, *, ttl: float = 60, max_size: int = 1000):
        self._cache: cachetools.TTLCache = cachetools.TTLCache(maxsize=max_size, ttl=ttl, timer=time.time)
        self._lock = threading.Lock()

    def set(self, key: CacheKey, value: Any) -> None:
        """Store *value* under *key*."""
        with self._lock:
            self._cache[key] = value

    def contains(self, key: CacheKey) -> bool:
        """Return ``True`` if *key* is present and not yet expired."""
        with self._lock:
            return key in self._cache

    def get(self, key: CacheKey, default: Any = None) -> Any:
        """Return the cached value for *key*, or *default* on a miss or expiry."""
        with self._lock:
            return self._cache.get(key, default)

    def get_or_call(self, key: CacheKey, callback: Callable[[], Any]) -> Any:
        """
        Return the cached value for *key*, or call *callback* to build it on a miss.

        The lock is held only during cache look-up and result storage, **not**
        while *callback* is executing.

        Usage::

            item = cache.get_or_call(
                key="foo",
                callback=lambda: expensive_operation(iterations=10000),
            )

        :param key: cache key (a string or a tuple of strings/ints)
        :param callback: callable invoked on a cache miss to produce the value
        :return: the cached or freshly produced value
        """
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        value = callback()
        with self._lock:
            self._cache[key] = value
        return value

    def flush(self) -> None:
        """Remove all entries from the cache."""
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
