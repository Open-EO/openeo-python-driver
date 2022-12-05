from openeo_driver.util.caching import TtlCache


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
        clock.set(110)
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
