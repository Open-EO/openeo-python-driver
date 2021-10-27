import flask
import pytest

from openeo_driver.testing import preprocess_check_and_replace, IgnoreOrder, ApiTester, RegexMatcher, DictSubSet


def test_api_tester_url():
    app = flask.Flask(__name__)
    api = ApiTester("1.0", client=app.test_client())
    assert api.url("foo/bar") == "/openeo/1.0/foo/bar"
    assert api.url("/foo/bar") == "/openeo/1.0/foo/bar"


@pytest.mark.parametrize(["url_root", "expected"], [
    ("", "/foo/bar"),
    ("/", "/foo/bar"),
    ("/oeo-v{api_version}", "/oeo-v1.0/foo/bar"),
    ("oeo", "/oeo/foo/bar"),
])
def test_api_tester_url_root(url_root, expected):
    app = flask.Flask(__name__)
    api = ApiTester("1.0", client=app.test_client(), url_root=url_root)
    assert api.url("foo/bar") == expected
    assert api.url("/foo/bar") == expected


def test_preprocess_check_and_replace():
    preprocess = preprocess_check_and_replace("foo", "bar")
    assert preprocess("foobar") == "barbar"
    assert preprocess("foobarfoobar") == "barbarbarbar"
    with pytest.raises(AssertionError):
        preprocess("bazbar")


def test_ignore_order_basic_lists():
    assert [1, 2, 3] == IgnoreOrder([1, 2, 3])
    assert IgnoreOrder([1, 2, 3]) == [1, 2, 3]
    assert [1, 2, 3] == IgnoreOrder([3, 1, 2])
    assert IgnoreOrder([3, 1, 2]) == [1, 2, 3]
    assert [1, 2, 3, 3] == IgnoreOrder([3, 1, 2, 3])
    assert IgnoreOrder([3, 1, 2, 3]) == [1, 2, 3, 3]


def test_ignore_order_basic_tuples():
    assert (1, 2, 3) == IgnoreOrder((1, 2, 3))
    assert IgnoreOrder((1, 2, 3)) == (1, 2, 3)
    assert (1, 2, 3) == IgnoreOrder((3, 1, 2))
    assert IgnoreOrder((3, 1, 2)) == (1, 2, 3)
    assert (1, 2, 3, 3) == IgnoreOrder((3, 1, 2, 3))
    assert IgnoreOrder((3, 1, 2, 3)) == (1, 2, 3, 3)


def test_ignore_order_basic_list_vs_tuples():
    assert [1, 2, 3] != IgnoreOrder((1, 2, 3))
    assert (1, 2, 3) != IgnoreOrder([1, 2, 3])
    assert IgnoreOrder((1, 2, 3)) != [1, 2, 3]
    assert IgnoreOrder([1, 2, 3]) != (1, 2, 3)


def test_ignore_order_nesting():
    assert {"foo": [1, 2, 3]} == {"foo": IgnoreOrder([3, 2, 1])}
    assert {"foo": [1, 2, 3], "bar": [4, 5]} == {"foo": IgnoreOrder([3, 2, 1]), "bar": [4, 5]}
    assert {"foo": [1, 2, 3], "bar": [4, 5]} != {"foo": IgnoreOrder([3, 2, 1]), "bar": [5, 4]}


def test_ignore_order_key():
    assert [{"f": "b"}, {"x": "y:"}] == IgnoreOrder([{"x": "y:"}, {"f": "b"}], key=repr)


def test_regex_matcher():
    assert {"foo": "baaaaa"} == {"foo": RegexMatcher("ba+")}


def test_dict_subset():
    expected = DictSubSet({"foo": "bar"})
    assert {"foo": "bar"} == expected
    assert {"foo": "nope"} != expected
    assert {"foo": "bar", "meh": 4} == expected
    assert {"foo": "nope", "meh": 4} != expected
    assert {"meh": 4} != expected


def test_dict_subset_nesting():
    assert {1: 2, 3: 4, 5: {6: 7, 8: 9}} == DictSubSet({})
    assert {1: 2, 3: 4, 5: {6: 7, 8: 9}} == DictSubSet({1: DictSubSet({})})
    assert {1: 2, 3: 4, 5: {6: 7, 8: 9}} == DictSubSet({3: 4, 5: DictSubSet({8: 9})})
    assert {1: {2: 3, 4: 5}} == {1: DictSubSet({4: 5})}
    assert {1: {2: {3: {4: 5, 6: 7}, 8: 9}}} == {1: {2: DictSubSet({3: DictSubSet({})})}}
