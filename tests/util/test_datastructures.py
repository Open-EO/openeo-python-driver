from openeo_driver.util.datastructures import unique_by_equality


def test_unique_by_equality_basic():
    assert unique_by_equality([1, 2, 3, 2, 1]) == [1, 2, 3]
    assert unique_by_equality(["a", "b", "a", "c", "d"]) == ["a", "b", "c", "d"]
    assert unique_by_equality(["a", 1, "b", 3, 2, 1]) == ["a", 1, "b", 3, 2]


def test_unique_by_equality_non_hashable():
    assert unique_by_equality([[1, 22], [22, 333], [1, 22], [4444]]) == [[1, 22], [22, 333], [4444]]
    assert unique_by_equality(
        [
            {"color": "red"},
            {"color": "green"},
            {"color": "red"},
            {"flavor": "chocolate"},
        ]
    ) == [
        {"color": "red"},
        {"color": "green"},
        {"flavor": "chocolate"},
    ]
