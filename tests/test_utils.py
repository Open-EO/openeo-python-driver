import pytest

from openeo_driver.utils import smart_bool, EvalEnv, to_hashable, bands_union, temporal_extent_union, \
    spatial_extent_union


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


def test_spatial_extent_union():
    assert spatial_extent_union(
        {"west": 1, "south": 51, "east": 2, "north": 52}
    ) == {"west": 1, "south": 51, "east": 2, "north": 52, "crs": "EPSG:4326"}
    assert spatial_extent_union(
        {"west": 1, "south": 51, "east": 2, "north": 52},
        {"west": 3, "south": 53, "east": 4, "north": 54},
    ) == {"west": 1, "south": 51, "east": 4, "north": 54, "crs": "EPSG:4326"}
    assert spatial_extent_union(
        {"west": 1, "south": 51, "east": 2, "north": 52},
        {"west": -5, "south": 50, "east": 3, "north": 53},
        {"west": 3, "south": 53, "east": 4, "north": 54},
    ) == {"west": -5, "south": 50, "east": 4, "north": 54, "crs": "EPSG:4326"}
