import re
import typing

import pytest

from openeo_driver.testing import RegexMatcher
from openeo_driver.utils import (
    smart_bool,
    EvalEnv,
    to_hashable,
    bands_union,
    temporal_extent_union,
    dict_item,
    extract_namedtuple_fields_from_dict,
    get_package_versions,
    TtlCache,
    generate_unique_id,
    WhiteListEvalEnv,
)


def test_smart_bool():
    for value in [1, 100, [1], (1,), "x", "1", "on", "ON", "yes", "YES", "true", "True", "TRUE", ]:
        assert smart_bool(value) == True
    for value in [0, [], (), {}, False, "0", "off", "OFF", "no", "No", "NO", "False", "false", "FALSE"]:
        assert smart_bool(value) == False


def test_eval_stack_empty():
    s = EvalEnv()
    assert s.get("foo") is None
    assert s.get("foo", "default") == "default"
    with pytest.raises(KeyError):
        _ = s["foo"]


def test_eval_stack_init_value():
    s = EvalEnv({"foo": "bar"})
    assert s.get("foo") == "bar"
    assert s["foo"] == "bar"


def test_eval_stack_init_value_copy():
    d = {"foo": "bar"}
    s = EvalEnv(d)
    assert d["foo"] == "bar"
    assert s["foo"] == "bar"
    d["foo"] = "meh"
    assert d["foo"] == "meh"
    assert s["foo"] == "bar"


def test_eval_stack_push():
    s1 = EvalEnv()
    s2 = s1.push({"foo": "bar", "xev": "lol"})
    assert s2["foo"] == "bar"
    assert s2["xev"] == "lol"
    assert s2.get("foo") == "bar"
    assert s2.get("xev") == "lol"
    assert s1.get("foo") is None
    assert s1.get("xev") is None
    with pytest.raises(KeyError):
        _ = s1["foo"]
    with pytest.raises(KeyError):
        _ = s1["xev"]


def test_eval_stack_overwrite():
    s1 = EvalEnv({"foo": "bar"})
    assert s1["foo"] == "bar"
    s2 = s1.push({"foo": "yoo"})
    assert s1["foo"] == "bar"
    assert s2["foo"] == "yoo"
    s3 = s2.push(foo="meh")
    assert s1["foo"] == "bar"
    assert s2["foo"] == "yoo"
    assert s3["foo"] == "meh"


def test_eval_env_get_deep():
    s1 = EvalEnv({"foo": "bar"})
    s2 = s1.push({})
    s3 = s2.push({})
    assert s3.get("foo") == "bar"
    assert s3["foo"] == "bar"
    assert s3.get("meh", default="jop") == "jop"
    with pytest.raises(KeyError):
        _ = s3["meh"]


def test_eval_stack_contains():
    s1 = EvalEnv({"foo": "bar"})
    assert "foo" in s1
    s2 = s1.push({"meh": "moh"})
    assert "foo" in s2
    assert "meh" in s2
    assert "meh" not in s1


def test_eval_stack_as_dict():
    s1 = EvalEnv({"foo": "bar"})
    s2 = s1.push({"foo": "meh", "xev": "lol"})
    s3 = s2.push({"xev": "zup", 1: 2, 3: 4})
    assert s1.as_dict() == {"foo": "bar"}
    assert s2.as_dict() == {"foo": "meh", "xev": "lol"}
    assert s3.as_dict() == {"foo": "meh", "xev": "zup", 1: 2, 3: 4}


def test_eval_stack_parameters():
    s0 = EvalEnv()
    s1 = s0.push(parameters={"color": "red", "size": 1})
    s2 = s1.push({"parameters": {"size": 3}})
    s3 = s2.push(user="alice")
    s4 = s3.push(parameters={"color": "green", "height": 88})
    assert s0.collect_parameters() == {}
    assert s1.collect_parameters() == {"color": "red", "size": 1}
    assert s2.collect_parameters() == {"color": "red", "size": 3}
    assert s3.collect_parameters() == {"color": "red", "size": 3}
    assert s4.collect_parameters() == {"color": "green", "size": 3, "height": 88}


def test_eval_whitelist():
    s1 = EvalEnv()
    s2 = s1.push({"foo": "bar", "xev": "lol"})
    s3 = WhiteListEvalEnv(s2, ["xev"])
    assert s3["xev"] == "lol"
    assert s3.get("xev") == "lol"
    assert s2.get("foo") == "bar"

    with pytest.raises(KeyError, match='.*Your key: foo.*'):
        bla = s3["foo"]
    with pytest.raises(KeyError):
        bla = s3.get("foo",None)


@pytest.mark.parametrize(["obj", "result"], [
    (123, 123),
    (23.45, 23.45),
    ("foo", "foo"),
    ((1, 2, 3), (1, 2, 3)),
    ([1, 2, 3], (1, 2, 3)),
    ({3, 2, 2, 1}, (1, 2, 3)),
    (
            {"foo": ["b", "a"], "faa": ["bar", {"li": {"s", "p"}}]},
            (("faa", ("bar", (("li", ("p", "s")),))), ("foo", ("b", "a")))
    )
])
def test_to_hashable(obj, result):
    assert to_hashable(obj) == result


def test_bands_union():
    assert bands_union() == []
    assert bands_union(["red", "blue"]) == ["red", "blue"]
    assert bands_union(["red"], ["blue"]) == ["red", "blue"]
    assert bands_union(["red"], [], ["blue"]) == ["red", "blue"]
    assert bands_union(["r", "b"], ["g"], ["b", "r"]) == ["r", "b", "g"]


def test_temporal_extent_union():
    assert temporal_extent_union() == (None, None)
    assert temporal_extent_union(("2020-01-01", "2020-12-31")) == ("2020-01-01", "2020-12-31")
    assert temporal_extent_union(
        ("2020-01-01", "2020-02-02"), ("2020-03-03", "2020-04-04")
    ) == ("2020-01-01", "2020-04-04")
    assert temporal_extent_union(
        ("2020-01-01", "2020-03-03"), ("2020-02-02", "2020-04-04")
    ) == ("2020-01-01", "2020-04-04")
    assert temporal_extent_union(
        ("2020-01-01", "2020-02-02"), ("2020-05-05", "2020-06-06"), ("2020-03-03", "2020-04-04"),
    ) == ("2020-01-01", "2020-06-06")
    assert temporal_extent_union(
        (None, "2020-02-02"), ("2020-05-05", "2020-06-06"), ("2020-03-03", "2020-04-04"),
    ) == (None, "2020-06-06")
    assert temporal_extent_union(
        ("2020-01-01", "2020-02-02"), ("2020-05-05", "2020-06-06"), (None, "2020-04-04"),
        none_is_infinity=False
    ) == ("2020-01-01", "2020-06-06")
    assert temporal_extent_union(
        ("2020-01-01", "2020-02-02"), ("2020-05-05", "2020-06-06"), ("2020-03-03", None),
    ) == ("2020-01-01", None)
    assert temporal_extent_union(
        ("2020-01-01", "2020-02-02"), ("2020-05-05", "2020-06-06"), ("2020-03-03", None),
        none_is_infinity=False
    ) == ("2020-01-01", "2020-06-06")


def test_dict_item():
    class UserInfo(dict):
        name = dict_item()
        age = dict_item()

    user = UserInfo(name="John")

    assert user["name"] == "John"
    assert user.name == "John"
    user.name = "Alice"
    assert user["name"] == "Alice"
    assert user.name == "Alice"
    user["name"] = "Bob"
    assert user["name"] == "Bob"
    assert user.name == "Bob"

    with pytest.raises(KeyError):
        age = user.age
    user.age = 42
    assert user["age"] == 42
    assert user.age == 42

    user["color"] = "green"

    assert user == {"name": "Bob", "age": 42, "color": "green"}


def test_dict_item_defaults():
    class UserInfo(dict):
        name = dict_item(default="John Doe")
        age = dict_item()

    user = UserInfo()
    assert user.name == "John Doe"
    with pytest.raises(KeyError):
        _ = user["name"]
    with pytest.raises(KeyError):
        _ = user["age"]
    with pytest.raises(KeyError):
        _ = user.age

    user.name = "Alice"
    user.age = 32
    assert user["name"] == "Alice"
    assert user.name == "Alice"
    assert user["age"] == 32
    assert user.age == 32


def test_extract_namedtuple_fields_from_dict():
    class Foo(typing.NamedTuple):
        id: str
        size: int
        color: str = "red"

    with pytest.raises(KeyError, match=r"Missing Foo fields: id, size."):
        extract_namedtuple_fields_from_dict({}, Foo)
    with pytest.raises(KeyError, match=r"Missing Foo field: size."):
        extract_namedtuple_fields_from_dict({"id": "bar"}, Foo)
    with pytest.raises(KeyError, match=r"Missing Foo field: id."):
        extract_namedtuple_fields_from_dict({"size": 3}, Foo)

    assert extract_namedtuple_fields_from_dict(
        {"id": "b", "size": 3}, Foo
    ) == {"id": "b", "size": 3}
    assert extract_namedtuple_fields_from_dict(
        {"id": "bar", "size": 3, "color": "blue"}, Foo
    ) == {"id": "bar", "size": 3, "color": "blue"}
    assert extract_namedtuple_fields_from_dict(
        {"id": "bar", "size": 3, "height": 666}, Foo
    ) == {"id": "bar", "size": 3}


def test_get_package_versions_basic():
    versions = get_package_versions(["flask", "requests"])
    assert versions == {
        "flask": RegexMatcher(r"\d+\.\d+\.\d+"),
        "requests": RegexMatcher(r"\d+\.\d+\.\d+"),
    }


def test_get_package_versions_na():
    versions = get_package_versions(["foobarmeh"])
    assert versions == {
        "foobarmeh": "n/a",
    }
    versions = get_package_versions(["foobarmeh"], na_value=None)
    assert versions == {
        "foobarmeh": None,
    }


class FakeClock:
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


def test_generate_uuid():
    assert re.match("^[0-9a-f]{32}$", generate_unique_id())
    assert re.match("^j-[0-9a-f]{32}$", generate_unique_id("j"))
