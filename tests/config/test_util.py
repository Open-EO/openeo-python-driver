import pytest

from openeo_driver.config.util import Exclude


class TestExclude:
    def test_by_prefix(self):
        exclude = Exclude.by_prefix("_")
        assert "foo" not in exclude
        assert "_foo" in exclude
        assert "__foo" in exclude
        assert 123 not in exclude
        assert ("_", "foo") not in exclude

    def test_union_with_exclude(self):
        exclude1 = Exclude.by_prefix("_")
        exclude2 = Exclude.by_prefix("temp_")
        combined = exclude1.union(exclude2)

        assert "foo" not in combined
        assert "_foo" in combined
        assert "temp_foo" in combined
        assert "__foo" in combined
        assert "temp__bar" in combined
        assert 123 not in combined
        assert ("_", "foo") not in combined

    @pytest.mark.parametrize(
        ["other"],
        [
            (["exclude-me"],),
            ({"exclude-me"},),
        ],
    )
    def test_union_with_container(self, other):
        exclude1 = Exclude.by_prefix("_")
        combined = exclude1.union(other)

        assert "foo" not in combined
        assert "_foo" in combined
        assert "exclude-me" in combined
        assert "exclude-me-not" not in combined
        assert 123 not in combined
        assert ("_", "foo") not in combined
