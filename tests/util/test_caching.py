import pytest
import time_machine

from openeo_driver.util.caching import BoundedTtlCache, TtlCache, lru_cache_if_simple_args


class FakeClock:
    # TODO: migrate to time_machine
    now = 0

    def set(self, now):
        self.now = now

    def __call__(self):
        return self.now


class TestTtlCache:
    def test_basic(self):
        cache = TtlCache()
        assert not cache.contains("foo")
        cache.set("foo", "bar")
        assert cache.contains("foo")
        assert cache.get("foo") == "bar"
        assert cache.get("meh") is None

    def test_get_default(self):
        cache = TtlCache()
        assert cache.get("foo") is None
        assert cache.get("foo", 123) == 123

    def test_default_ttl(self):
        clock = FakeClock()
        cache = TtlCache(default_ttl=10, _clock=clock)
        clock.set(100)
        cache.set("foo", "bar")
        clock.set(105)
        assert cache.get("foo") == "bar"
        clock.set(109)
        assert cache.contains("foo")
        assert cache.get("foo") == "bar"
        clock.set(115)
        assert not cache.contains("foo")
        assert cache.get("foo") is None

    def test_item_ttl(self):
        clock = FakeClock()
        cache = TtlCache(default_ttl=10, _clock=clock)
        clock.set(100)
        cache.set("foo", "bar", ttl=20)
        clock.set(115)
        assert cache.contains("foo")
        assert cache.get("foo") == "bar"
        clock.set(125)
        assert not cache.contains("foo")
        assert cache.get("foo") is None

    def test_get_or_call(self):
        def calculate(_state={"x": 0}):
            _state["x"] += 1
            return _state["x"]

        clock = FakeClock()
        cache = TtlCache(default_ttl=10, _clock=clock)
        clock.set(100)
        assert cache.get("foo") is None
        assert cache.get_or_call("foo", callback=calculate) == 1
        assert cache.get_or_call("foo", callback=calculate) == 1
        clock.set(120)
        assert cache.get_or_call("foo", callback=calculate) == 2
        clock.set(140)
        assert cache.get_or_call("foo", callback=calculate) == 3

    def test_get_or_call_error(self):
        def calculate():
            return 4 / 0

        cache = TtlCache(default_ttl=10)
        assert cache.get("foo") is None
        with pytest.raises(ZeroDivisionError):
            cache.get_or_call("foo", callback=calculate)
        with pytest.raises(ZeroDivisionError):
            cache.get_or_call("foo", callback=calculate)


class TestBoundedTtlCache:
    def test_basic(self):
        cache = BoundedTtlCache()
        assert not cache.contains("foo")
        cache.set("foo", "bar")
        assert cache.contains("foo")
        assert cache.get("foo") == "bar"
        assert cache.get("meh") is None

    def test_ttl(self):
        with time_machine.travel(0, tick=False) as traveller:
            cache = BoundedTtlCache(ttl=10)
            cache.set("foo", "bar")
            traveller.shift(9)
            assert cache.contains("foo")
            assert cache.get("foo") == "bar"
            traveller.shift(2)  # t=11, past ttl
            assert not cache.contains("foo")
            assert cache.get("foo") is None

    def test_max_size(self):
        cache = BoundedTtlCache(ttl=60, max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.contains("a")
        assert cache.contains("b")
        assert cache.contains("c")
        # Adding a fourth item evicts the least-recently-used entry.
        cache.set("d", 4)
        assert cache.contains("d")
        assert sum(cache.contains(k) for k in ("a", "b", "c", "d")) == 3

    def test_get_or_call(self):
        def calculate(_state={"x": 0}):
            _state["x"] += 1
            return _state["x"]

        with time_machine.travel(0, tick=False) as traveller:
            cache = BoundedTtlCache(ttl=10)
            assert cache.get("foo") is None
            assert cache.get_or_call("foo", callback=calculate) == 1
            assert cache.get_or_call("foo", callback=calculate) == 1
            traveller.shift(15)
            assert cache.get_or_call("foo", callback=calculate) == 2
            traveller.shift(15)
            assert cache.get_or_call("foo", callback=calculate) == 3

    def test_get_or_call_error(self):
        cache = BoundedTtlCache(ttl=10)
        assert cache.get("foo") is None
        with pytest.raises(ZeroDivisionError):
            cache.get_or_call("foo", callback=lambda: 1 / 0)
        # Failed callback must not populate the cache.
        with pytest.raises(ZeroDivisionError):
            cache.get_or_call("foo", callback=lambda: 1 / 0)

    def test_flush(self):
        cache = BoundedTtlCache(ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.contains("a")
        cache.flush()
        assert not cache.contains("a")
        assert not cache.contains("b")


class TestLruCacheIfSimpleArgs:
    def test_default_no_parentheses(self):
        _call_history = []

        @lru_cache_if_simple_args
        def fun(x) -> str:
            _call_history.append(x)
            return str(x)

        assert fun(1) == "1"
        assert _call_history == [1]

        assert fun(1) == "1"
        assert _call_history == [1]

        assert fun("one") == "one"
        assert _call_history == [1, "one"]

        assert fun({1: 1}) == "{1: 1}"
        assert _call_history == [1, "one", {1: 1}]

        assert fun(1) == "1"
        assert _call_history == [1, "one", {1: 1}]

        assert fun({1: 1}) == "{1: 1}"
        assert _call_history == [1, "one", {1: 1}, {1: 1}]

    def test_default_with_parentheses(self):
        _call_history = []

        @lru_cache_if_simple_args()
        def fun(x) -> str:
            _call_history.append(x)
            return str(x)

        assert fun(1) == "1"
        assert _call_history == [1]

        assert fun(1) == "1"
        assert _call_history == [1]

        assert fun("one") == "one"
        assert _call_history == [1, "one"]

        assert fun({1: 1}) == "{1: 1}"
        assert _call_history == [1, "one", {1: 1}]

        assert fun(1) == "1"
        assert _call_history == [1, "one", {1: 1}]

        assert fun({1: 1}) == "{1: 1}"
        assert _call_history == [1, "one", {1: 1}, {1: 1}]

    def test_maxsize(self):
        _call_history = []

        @lru_cache_if_simple_args(maxsize=2)
        def fun(x) -> str:
            _call_history.append(x)
            return str(x)

        assert fun(1) == "1"
        assert _call_history == [1]

        assert fun(1) == "1"
        assert _call_history == [1]

        assert fun("one") == "one"
        assert _call_history == [1, "one"]

        assert fun({1: 1}) == "{1: 1}"
        assert _call_history == [1, "one", {1: 1}]

        assert fun(1) == "1"
        assert _call_history == [1, "one", {1: 1}]

        assert fun(2) == "2"
        assert _call_history == [1, "one", {1: 1}, 2]

        assert fun("one") == "one"
        assert _call_history == [1, "one", {1: 1}, 2, "one"]

    @pytest.mark.parametrize(
        ["args", "kwargs", "expected_caching"],
        [
            ((1, 2), {}, True),
            ((1, "two", 3.3, True, None), {}, True),
            ((1, {2: 2}, 3), {}, False),
            ((1, (2, 22), 3), {}, False),
            ((1, [2, 22], 3), {}, False),
            ((0,), {"a": 1, "b": "two", "c": 3.3, "d": True, "e": None}, True),
            ((0,), {"a": 1, "b": {2: 2}}, False),
            ((0,), {"a": 1, "b": (2, 22)}, False),
            ((0,), {"a": 1, "b": [2, 22]}, False),
        ],
    )
    def test_caching_from_argument_types(self, args, kwargs, expected_caching):
        _call_history = []

        @lru_cache_if_simple_args
        def fun(*args, **kwargs) -> str:
            _call_history.append((args, kwargs))
            return str((args, kwargs))

        fun(*args, **kwargs)
        assert _call_history == [(args, kwargs)]

        fun(*args, **kwargs)
        if expected_caching:
            assert _call_history == [(args, kwargs)]
        else:
            assert _call_history == [(args, kwargs), (args, kwargs)]
