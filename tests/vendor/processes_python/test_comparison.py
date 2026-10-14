# Adapted from openeo-processes-python (release 0.0.4, Apache License 2.0).
# xarray/dask based assertions have been removed.
import unittest

import numpy as np
from openeo_driver.vendor import processes_python as oeop
import pytest


@pytest.mark.parametrize(["value", "expected"], [
    # Based on https://github.com/Open-EO/openeo-processes/issues/189
    (None, True),
    (np.nan, True),
    (0, False),
    (1, False),
    (np.inf, False),
    ("a string", True),
])
def test_is_nan(value, expected):
    """ Tests `is_nan` function. """
    assert oeop.is_nan(value) == expected


@pytest.mark.parametrize(["value", "expected"], [
    # Based on https://github.com/Open-EO/openeo-processes/issues/189
    (None, True),
    (np.nan, False),
    (0, False),
    (1, False),
    (np.inf, False),
    ("a string", False),
])
def test_is_nodata(value, expected):
    """ Tests `is_nodata` function. """
    assert oeop.is_nodata(value) == expected


@pytest.mark.parametrize(["value", "expected"], [
    # Based on https://github.com/Open-EO/openeo-processes/issues/189
    (None, False),
    (np.nan, False),
    (0, True),
    (1, True),
    (np.inf, False),
    ("a string", True),
])
def test_is_valid(value, expected):
    """ Tests `is_valid` function. """
    assert oeop.is_valid(value) == expected

class ComparisonTester(unittest.TestCase):
    """ Tests all comparison functions. """

    def test_eq(self):
        """ Tests `eq` function. """
        assert oeop.eq(1, None) is None
        assert oeop.eq(1, 1)
        assert not oeop.eq(1, "1")
        assert not oeop.eq(1.02, 1, delta=0.01)
        assert oeop.eq(-1, -1.001, delta=0.01)
        assert oeop.eq(115, 110, delta=10)
        assert not oeop.eq("Test", "test")
        assert oeop.eq("Test", "test", case_sensitive=False)
        assert oeop.eq("Ä", "ä", case_sensitive=False)
        assert oeop.eq("00:00:00+00:00", "00:00:00Z")
        assert not oeop.eq("2018-01-01T12:00:00Z", "2018-01-01T12:00:00")
        assert oeop.eq("2018-01-01T00:00:00Z", "2018-01-01T01:00:00+01:00")

    def test_neq(self):
        """ Tests `neq` function. """
        assert oeop.neq(1, None) is None
        assert not oeop.neq(1, 1)
        assert oeop.neq(1, "1")
        assert oeop.neq(1.02, 1, delta=0.01)
        assert not oeop.neq(-1, -1.001, delta=0.01)
        assert not oeop.neq(115, 110, delta=10)
        assert oeop.neq("Test", "test")
        assert not oeop.neq("Test", "test", case_sensitive=False)
        assert not oeop.neq("Ä", "ä", case_sensitive=False)
        assert not oeop.neq("00:00:00+00:00", "00:00:00Z")
        assert oeop.neq("2018-01-01T12:00:00Z", "2018-01-01T12:00:00")
        assert not oeop.neq("2018-01-01T00:00:00Z", "2018-01-01T01:00:00+01:00")

    def test_gt(self):
        """ Tests `gt` function. """
        assert oeop.gt(1, None) is None
        assert not oeop.gt(0, 0)
        assert oeop.gt(2, 1)
        assert oeop.gt(-0.5, -0.6)
        assert oeop.gt("00:00:00Z", "00:00:00+01:00")
        assert not oeop.gt("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z")
        assert not oeop.gt("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z")

    def test_gte(self):
        """ Tests `gte` function. """
        assert oeop.gte(1, None) is None
        assert oeop.gte(0, 0)
        assert not oeop.gte(1, 2)
        assert oeop.gte(-0.5, -0.6)
        assert oeop.gte("00:00:00Z", "00:00:00+01:00")
        assert not oeop.gte("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z")
        assert oeop.gte("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z")

    def test_lt(self):
        """ Tests `lt` function. """
        assert oeop.lt(1, None) is None
        assert not oeop.lt(0, 0)
        assert oeop.lt(1, 2)
        assert not oeop.lt(-0.5, -0.6)
        assert oeop.lt("00:00:00+01:00", "00:00:00Z")
        assert oeop.lt("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z")
        assert not oeop.lt("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z")

    def test_lte(self):
        """ Tests `lte` function. """
        assert oeop.lte(1, None) is None
        assert oeop.lte(0, 0)
        assert oeop.lte(1, 2)
        assert not oeop.lte(-0.5, -0.6)
        assert oeop.lte("00:00:00+01:00", "00:00:00Z")
        assert oeop.lte("1950-01-01T00:00:00Z", "2018-01-01T12:00:00Z")
        assert oeop.lte("2018-01-01T12:00:00+00:00", "2018-01-01T12:00:00Z")

    def test_between(self):
        """ Tests `between` function. """
        assert oeop.between(None, min=0, max=1) is None
        assert not oeop.between(0.5, min=1, max=0)
        assert not oeop.between(-0.5, min=0, max=-1)
        assert oeop.between("00:59:59Z", min="01:00:00+01:00", max="01:00:00Z")
        assert oeop.between("2018-07-23T17:22:45Z", min="2018-01-01T00:00:00Z", max="2018-12-31T23:59:59Z")
        assert not oeop.between("2000-01-01", min="2018-01-01", max="2020-01-01")
        assert not oeop.between("2018-12-31T17:22:45Z", min="2018-01-01", max="2018-12-31", exclude_max=True)

if __name__ == '__main__':
    unittest.main()
