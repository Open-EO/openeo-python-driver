# Adapted from openeo-processes-python (release 0.0.4, Apache License 2.0).
# xarray/dask based assertions have been removed.
"""
Most tests are in alignment with:
https://openeo.org/documentation/1.0/processes.html
"""

import numpy as np
from openeo_driver.vendor import processes_python as oeop
import pytest
from unittest import TestCase
import math


class ArrayTester(TestCase):
    """ Tests all array functions. """


    def test_array_contains(self):
        """ Tests `array_contains` function. """
        assert oeop.array_contains([1, 2, 3], value=2)
        assert not oeop.array_contains(["A", "B", "C"], value="b")
        assert not oeop.array_contains([1, 2, 3], value="2")
        assert oeop.array_contains([1, 2, 3], value=2)
        assert oeop.array_contains([1, 2, np.nan], value=np.nan)
        assert oeop.array_contains([[1, 2], [3, 4]], value=[1, 2])
        assert not oeop.array_contains([[1, 2], [3, 4]], value=2)
        assert oeop.array_contains([{"a": "b"}, {"c": "d"}], value={"a": "b"})

    def test_array_element(self):
        """ Tests `array_element` function. """
        # numpy tests
        assert oeop.array_element([9, 8], label="B", labels=np.array(["A", "B"])) == 8
        assert oeop.array_element([9, 8, 7, 6, 5], index=2) == 7
        assert oeop.array_element(["A", "B", "C"], index=0) == "A"
        assert np.isnan(oeop.array_element([], index=0, return_nodata=True))

        # multi-dim
        test_array = np.empty((3, 2, 2))
        test_array[0, :, :] = np.array([[1, 2], [3, 4]])
        test_array[1, :, :] = np.array([[1, 2], [3, 4]]) * 20
        test_array[2, :, :] = np.array([[1, 2], [3, 4]]) * 500
        array_i2_d0 = np.array([
                                [500, 1000],
                                [1500, 2000]
                                ])
        array_i0_d1 = np.array([
                                [1, 2],
                                [20, 40],
                                [500, 1000]
                                ])
        assert np.isclose(oeop.array_element(test_array, index=2, dimension=0), array_i2_d0, equal_nan=True).all()
        assert np.isclose(oeop.array_element(test_array, index=0, dimension=1), array_i0_d1, equal_nan=True).all()

        # xarray tests
        # Assert raised errors?
        # ArrayElementNotAvailable
        # oeop.array_element(self.xr_data_4d, dimension='s', label="B09")
        # oeop.array_element(self.xr_data_4d, dimension='s', index=4)

    def test_count(self):
        """ Tests `count` function. """
        assert oeop.count([]) == 0
        assert oeop.count([], condition=True) == 0
        assert oeop.count([1, 0, 3, 2]) == 4
        assert oeop.count(["ABC", np.nan]) == 1
        assert oeop.count([False, np.nan], condition=True) == 2
        assert oeop.count([0, 1, 2, 3, 4, 5, np.nan], condition=oeop.gt, context={'y': 2}) == 3
        assert oeop.count([0, 1, 2, 3, 4, 5, np.nan], condition=oeop.lte, context={'y': 2}) == 3


    def test_array_filter(self):
        """ Tests `array_filter` function. """
        assert oeop.array_filter([1, 0, 3, 2], condition=oeop.lte, context={'y':0.5}) == 0
        assert (oeop.array_filter([1, 0, 3, 2, np.nan, 3], condition=oeop.gt, context={'y': 1}) == [3, 2, 3]).all()
        assert (oeop.array_filter([0, 1, 2, 3, 4, 5, np.nan], condition=oeop.gt, context={'y': 2}) == [3, 4, 5]).all()

    def test_array_find(self):
        """ Tests `array_find` function. """
        assert oeop.array_find([1, 0, 3, 2], value= 3) == 2
        assert np.isnan(oeop.array_find([1, 0, 3, 2, np.nan, 3], value = np.nan))

    def test_array_labels(self):
        """ Tests `array_labels` function. """
        assert (oeop.array_labels([1, 0, 3, 2]) == np.array([0, 1, 2, 3])).all()
        assert (oeop.array_labels([[1, 0, 3, 2]], dimension = 0) == np.array([0]))


    def test_first(self):
        """ Tests `first` function. """
        assert oeop.first([1, 0, 3, 2]) == 1
        assert oeop.first([np.nan, "A", "B"]) == "A"
        assert np.isnan(oeop.first([np.nan, 2, 3], ignore_nodata=False))
        assert np.isnan(oeop.first([]))

        # 2D test
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
        first_elem_ref = np.array([[[3., 2.], [1., 2.]]])
        first_elem = oeop.first(test_arr)
        assert np.isclose(first_elem, first_elem_ref, equal_nan=True).all()
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
        first_elem_ref = np.array([[[np.nan, 2.], [1., 2.]]])
        first_elem = oeop.first(test_arr, ignore_nodata=False)
        assert np.isclose(first_elem, first_elem_ref, equal_nan=True).all()

    def test_last(self):
        """ Tests `last` function. """
        assert oeop.last([1, 0, 3, 2]) == 2
        assert oeop.last(["A", "B", np.nan]) == "B"
        assert np.isnan(oeop.last([0, 1, np.nan], ignore_nodata=False))
        assert np.isnan(oeop.last([]))

        # 2D test
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 3]], [[1, 2], [1, np.nan]]])
        last_elem_ref = np.array([[[1., 2.], [1., 3.]]])
        last_elem = oeop.last(test_arr)
        assert np.isclose(last_elem, last_elem_ref, equal_nan=True).all()
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
        last_elem_ref = np.array([[[1., 2.], [1., np.nan]]])
        last_elem = oeop.last(test_arr, ignore_nodata=False)
        assert np.isclose(last_elem, last_elem_ref, equal_nan=True).all()

    def test_order(self):
        """ Tests `order` function. """
        self.assertListEqual(oeop.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9]).tolist(),
                             [1, 2, 8, 5, 0, 4, 7, 9, 10])
        self.assertListEqual(oeop.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], nodata=True).tolist(),
                             [1, 2, 8, 5, 0, 4, 7, 9, 10, 3, 6])
        self.assertListEqual(oeop.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=True).tolist(),
                             [9, 10, 7, 4, 0, 5, 8, 2, 1, 3, 6])
        self.assertListEqual(oeop.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=False).tolist(),
                             [6, 3, 9, 10, 7, 4, 0, 5, 8, 2, 1])

    def test_rearrange(self):
        """ Tests `rearrange` function. """
        self.assertListEqual(oeop.rearrange([5, 4, 3], [2, 1, 0]).tolist(), [3, 4, 5])
        self.assertListEqual(oeop.rearrange([5, 4, 3, 2], [0, 2, 1, 3]).tolist(), [5, 3, 4, 2])
        self.assertListEqual(oeop.rearrange([5, 4, 3, 2], [1, 3]).tolist(), [4, 2])

    def test_sort(self):
        """ Tests `sort` function. """
        self.assertListEqual(oeop.sort([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9]).tolist(),
                             [-1, 2, 3, 4, 6, 7, 8, 9, 9])
        assert np.isclose(oeop.sort([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=True),
                          [9, 9, 8, 7, 6, 4, 3, 2, -1, np.nan, np.nan], equal_nan=True).all()
