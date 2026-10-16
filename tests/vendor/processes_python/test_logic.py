# Adapted from openeo-processes-python (release 0.0.4, Apache License 2.0).
# xarray/dask based assertions have been removed.
import unittest
import numpy as np
from openeo_driver.vendor import processes_python as oeop


class LogicTester(unittest.TestCase):
    """ Tests all logic functions. """

    def test_not_(self):
        """ Tests `not_` function. """
        assert not oeop.not_(True)
        assert oeop.not_(False)
        assert oeop.not_(None) is None

    def test_and_(self):
        """ Tests `and_` function. """
        assert not oeop.and_(False, None)
        assert oeop.and_(True, None) is None
        assert not oeop.and_(False, False)
        assert not oeop.and_(True, False)
        assert oeop.and_(True, True)

    def test_or_(self):
        """ Tests `or_` function. """
        assert oeop.or_(False, None) is None
        assert oeop.or_(True, None)
        assert not oeop.or_(False, False)
        assert oeop.or_(True, False)
        assert oeop.or_(True, True)

    def test_xor(self):
        """ Tests `xor` function. """
        assert oeop.xor(False, None) is None
        assert oeop.xor(True, None) is None
        assert not oeop.xor(False, False)
        assert oeop.xor(True, False)
        assert not oeop.xor(True, True)


    def test_any_(self):
        """ Tests `any_` function. """
        assert not oeop.any_([False, np.nan])
        assert oeop.any_([True, np.nan])
        assert np.isnan(oeop.any_([False, np.nan], ignore_nodata=False))
        assert oeop.any_([True, np.nan], ignore_nodata=False)
        assert oeop.any_([True, False, True, False])
        assert oeop.any_([True, False])
        assert not oeop.any_([False, False])
        assert oeop.any_([True])
        assert np.isnan(oeop.any_([np.nan], ignore_nodata=False))
        assert np.isnan(oeop.any_([]))

    def test_all_(self):
        """ Tests `all_` function. """
        assert not oeop.all_([False, np.nan])
        assert oeop.all_([True, np.nan])
        assert not oeop.all_([False, np.nan], ignore_nodata=False)
        assert np.isnan(oeop.all_([True, np.nan], ignore_nodata=False))
        assert not oeop.all_([True, False, True, False])
        assert not oeop.all_([True, False])
        assert oeop.all_([True, True])
        assert oeop.all_([True])
        assert np.isnan(oeop.all_([np.nan], ignore_nodata=False))
        assert np.isnan(oeop.all_([]))

if __name__ == '__main__':
    unittest.main()
