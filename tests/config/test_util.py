from openeo_driver.config.util import Exclude


class TestExclude:
    def test_by_prefix(self):
        exclude = Exclude.by_prefix("_")
        assert "foo" not in exclude
        assert "_foo" in exclude
        assert "__foo" in exclude
        assert 123 not in exclude
        assert ("_", "foo") not in exclude
