import pytest

from openeo_driver.utils import smart_bool, EvalEnv, to_hashable, bands_union, temporal_extent_union, \
    spatial_extent_union, dict_item, reproject_bounding_box


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


def test_spatial_extent_union_mixed_crs():
    bbox1 = {"west": 640860.0, "south": 5676170.0, "east": 642140.0, "north": 5677450.0, "crs": "EPSG:32631"}
    bbox2 = {"west": 5.017, "south": 51.21, "east": 5.035, "north": 51.23, "crs": "EPSG:4326"}
    assert spatial_extent_union(bbox1, bbox2) == {
        "west": 640824.9450876965, "south": 5675111.354480841, "north": 5677450.0, "east": 642143.1739784481,
        "crs": "EPSG:32631",
    }
    assert spatial_extent_union(bbox2, bbox1) == {
        "west": 5.017, "south": 51.21, "east": 5.035868037661473, "north": 51.2310228422003,
        "crs": "EPSG:4326",
    }


@pytest.mark.parametrize(["crs", "bbox"], [
    ("EPSG:32631", {"west": 640800, "south": 5676000, "east": 642200, "north": 5677000}),
    ("EPSG:4326", {"west": 5.01, "south": 51.2, "east": 5.1, "north": 51.5}),
])
def test_reproject_bounding_box_same(crs, bbox):
    reprojected = reproject_bounding_box(bbox, from_crs=crs, to_crs=crs)
    assert reprojected == dict(crs=crs, **bbox)


def test_reproject_bounding_box():
    bbox = {"west": 640800, "south": 5676000, "east": 642200.0, "north": 5677000.0}
    reprojected = reproject_bounding_box(bbox, from_crs="EPSG:32631", to_crs="EPSG:4326")
    assert reprojected == {
        "west": 5.016118467277098, "south": 51.217660146353246,
        "east": 5.036548264535997, "north": 51.22699369149726,
        "crs": "EPSG:4326",
    }


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
