import logging
import re
import urllib.error
import urllib.request

import flask
import pytest

from openeo_driver.testing import (
    preprocess_check_and_replace,
    IgnoreOrder,
    ApiTester,
    RegexMatcher,
    DictSubSet,
    ListSubSet,
    approxify,
    caplog_with_custom_formatter,
    UrllibMocker,
)


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
    assert "foo" != expected
    assert 3 != expected


def test_dict_subset_init():
    assert {"foo": "bar"} == DictSubSet({"foo": "bar"})
    assert {"foo": "bar"} == DictSubSet(foo="bar")
    assert {"foo": "bar"} == DictSubSet({"foo": "meh"}, foo="bar")


def test_dict_subset_nesting():
    assert {1: 2, 3: 4, 5: {6: 7, 8: 9}} == DictSubSet({})
    assert {1: 2, 3: 4, 5: {6: 7, 8: 9}} == DictSubSet({5: DictSubSet({})})
    assert {1: 2, 3: 4, 5: {6: 7, 8: 9}} == DictSubSet({3: 4, 5: DictSubSet({8: 9})})
    assert {1: {2: 3, 4: 5}} == {1: DictSubSet({4: 5})}
    assert {1: {2: {3: {4: 5, 6: 7}, 8: 9}}} == {1: {2: DictSubSet({3: DictSubSet({})})}}


def test_dict_subset_repr():
    actual = {1: 2, 3: 4, 5: 6}
    expected = DictSubSet({3: 4, 5: 66, 7: 8})
    assert actual != expected
    assert repr(expected) == (
        "{3: 4, 5: 66, 7: 8}\n"
        "    # Missing: {7: 8}\n"
        "    # Differing: {5: (66, 6)}"
    )


def test_list_subset():
    assert [] == ListSubSet([])
    assert [2, 3, 5] == ListSubSet([])
    assert [2, 3, 5] == ListSubSet([2])
    assert 2 != ListSubSet([2])
    assert (2, 3, 5) != ListSubSet([2])
    assert [2, 3, 5] != ListSubSet([-100])
    assert [2, 3, 5] == ListSubSet([2, 5])
    assert [2, 3, 5] == ListSubSet([2, 5])
    assert [2, 3, 5] == ListSubSet([5, 3])
    assert [2, 3, 5] != ListSubSet([5, 3, 8])
    assert [2, 3, 5] != ListSubSet([5, 3, None])
    assert [2, 2, 3, 3, 3] == ListSubSet([2, 3])


def test_list_subset_nesting():
    assert [1, 2, [3, 4]] == ListSubSet([])
    assert [1, 2, [3, 4]] == ListSubSet([2])
    assert [1, 2, [3, 4]] == ListSubSet([1, ListSubSet([])])
    assert [1, 2, [3, 4]] == ListSubSet([1, ListSubSet([4])])


def test_list_and_dict_subset_nesting():
    assert [1, {2: 3}, {5: 6, 7: [8, 9]}] == ListSubSet([])
    assert [1, {2: 3}, {5: 6, 7: [8, 9]}] == ListSubSet([1])
    assert [1, {2: 3}, {5: 6, 7: [8, 9]}] == ListSubSet([{2: 3}])
    assert [1, {2: 3}, {5: 6, 7: [8, 9]}] == ListSubSet([1, {2: 3}])
    assert [1, {2: 3}, {5: 6, 7: [8, 9]}] == ListSubSet([1, DictSubSet({})])
    assert [1, {2: 3}, {5: 6, 7: [8, 9]}] == ListSubSet([1, DictSubSet({2: 3})])
    assert [1, {2: 3}, {5: 6, 7: [8, 9]}] == ListSubSet([1, DictSubSet({7: [8, 9]})])
    assert [1, {2: 3}, {5: 6, 7: [8, 9]}] == ListSubSet([1, DictSubSet({7: ListSubSet([8])})])


class TestUrllibMocker:

    def test_get(self, urllib_mock: UrllibMocker):
        urllib_mock.get("http://a.test/foo", data="hello world")

        r = urllib.request.urlopen("http://a.test/foo")
        assert r.read() == b"hello world"

    def test_get_not_found(self, urllib_mock: UrllibMocker):
        with pytest.raises(urllib.error.HTTPError, match="404: Not Found"):
            urllib.request.urlopen("http://a.test/bar")

    def test_context_manager_get(self, urllib_mock: UrllibMocker):
        urllib_mock.get("http://a.test/foo", data="hello world")

        with urllib.request.urlopen("http://a.test/foo") as f:
            data = f.read()

        assert data == b"hello world"


def test_approxify_basic():
    assert {"a": 1.2345} == approxify({"a": 1.2345})
    assert {"a": 1.2345} != approxify({"a": 1.23466666})
    assert {"a": 1.2345, "b": "foo"} == approxify({"a": 1.2345, "b": "foo"})
    assert {"a": 1.2345, "b": "foo"} != approxify({"a": 1.2345})
    assert {"a": 1000.0, "b": "foo"} == approxify({"a": 1000.0001, "b": "foo"})
    assert {"a": 1000.0, "b": "foo"} != approxify({"a": 1000.01, "b": "foo"})
    assert {"a": [10.00001, 2.299999]} == approxify({"a": [10, 2.3]})
    assert {"a": [10.00001, 2.299999]} == approxify({"a": [10, 2.3]})
    assert [{"a": [10.00001, 2.299999]}, ([4.9999999, 6], -123.000001)] == approxify(
        [{"a": [10, 2.3]}, ([5, 6], -123)],
    )
    assert [
        {"a": [10.00001, 2.299999, 6666]},
        ([4.9999999, 6], -123.000001),
    ] != approxify(
        [{"a": [10, 2.3]}, ([5, 6], -123)],
    )


def test_approxify_tolerance_abs():
    assert {"a": [10.1, 2.2]} == approxify({"a": [10, 2.3]}, abs=0.1)
    assert {"a": [10.1, 2.2]} != approxify({"a": [10, 2.3]}, abs=0.09)
    assert {"a": [10.1, 2.2]} != approxify({"a": [10, 2.3]}, abs=0.001)
    assert {"a": [10.001, 2.299]} == approxify({"a": [10, 2.3]}, abs=0.001)


def test_approxify_tolerance_rel():
    assert {"a": [10.1, 2.0]} != approxify({"a": [10, 2.3]}, rel=0.1)
    assert {"a": [10.1, 2.1]} == approxify({"a": [10, 2.3]}, rel=0.1)
    assert {"a": [10.1, 2.1]} != approxify({"a": [10, 2.3]}, rel=0.01)


@pytest.mark.parametrize(
    "format",
    [
        "[%(levelname)s] %(message)s (%(name)s)",
        logging.Formatter("[%(levelname)s] %(message)s (%(name)s)"),
    ],
)
def test_caplog_with_custom_formatter(caplog, format):
    # Assumes default formatter of caplog (see DEFAULT_LOG_FORMAT in _pytest.logging)
    logging.warning("not good")
    # Switch to custom formatter
    with caplog_with_custom_formatter(caplog, format=format):
        logging.warning("still not good")
    # Original formatter should be restored
    logging.warning("hmm bad times")

    logs = caplog.text.strip()
    # Get rid of unstable line numbers
    logs = re.sub(r".py:(\d+)", ".py:XXX", logs)
    assert logs.split("\n") == [
        "WARNING  root:test_testing.py:XXX not good",
        "[WARNING] still not good (root)",
        "WARNING  root:test_testing.py:XXX hmm bad times",
    ]
