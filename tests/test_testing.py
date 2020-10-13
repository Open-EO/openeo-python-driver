import pytest

from openeo_driver.testing import preprocess_check_and_replace, IgnoreOrder


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
