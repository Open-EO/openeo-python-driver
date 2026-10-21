# Adapted from openeo-processes-python (https://github.com/Open-EO/openeo-processes-python),
# release 0.0.4, licensed under Apache License 2.0.
# Only the numpy/pure-python implementations of a limited set of processes are retained.
# xarray/dask specific code and the pandas dependency have been removed.

import numpy as np

from openeo_driver.vendor.processes_python.utils import create_slices
from openeo_driver.vendor.processes_python.utils import process, isnull
from openeo_driver.vendor.processes_python.comparison import is_valid
from openeo_driver.vendor.processes_python.comparison import is_empty

from openeo_driver.vendor.processes_python.errors import ArrayElementNotAvailable
from openeo_driver.vendor.processes_python.errors import ArrayElementParameterMissing
from openeo_driver.vendor.processes_python.errors import ArrayElementParameterConflict
from openeo_driver.vendor.processes_python.errors import GenericError

########################################################################################################################
# Array Contains Process
########################################################################################################################

@process
def array_contains():
    """
    Returns class instance of `ArrayContains`.
    For more details, please have a look at the implementations inside `ArrayContains`.

    Returns
    -------
    ArrayContains :
        Class instance implementing all 'array_contains' processes.

    """
    return ArrayContains()


class ArrayContains:
    """
    Class implementing all 'array_contains' processes.

    """

    @staticmethod
    def exec_num():
        pass

    # TODO: refine this implementation for larger arrays
    @staticmethod
    def exec_np(data, value):
        """
        Checks whether the array specified for `data` contains the value specified in `value`.
        Returns `True` if there's a match, otherwise `False`.

        Parameters
        ----------
        data : np.array
            Array to find the value in.
        value : object
            Value to find in `data`.

        Returns
        -------
        bool :
            Returns `True` if the list contains the value, `False` otherwise.

        Notes
        -----
        `in` is not working because this process checks only for the first level.

        """
        for elem in data:
            if np.array(isnull(value)).all() and np.isnan(elem):  # special handling for nan values
                return True
            elif np.array(elem == value).all():
                return True
        return False


########################################################################################################################
# Array Element Process
########################################################################################################################

@process
def array_element():
    """
    Returns class instance of `ArrayElement`.
    For more details, please have a look at the implementations inside `ArrayElement`.

    Returns
    -------
    ArrayElement :
        Class instance implementing all 'array_element' processes.

    """
    return ArrayElement()


class ArrayElement:
    """
    Class implementing all 'array_element' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, index=None, label=None, dimension=0, return_nodata=False, labels=None):
        """
        Returns the element with the specified index or label from the array. Either the parameter `index` or `label`
        must be specified, otherwise the `ArrayElementParameterMissing` exception is thrown. If both parameters are set
        the `ArrayElementParameterConflict` exception is thrown.

        Parameters
        ----------
        data : np.array
            An array.
        index : int, optional
            The zero-based index of the element to retrieve (default is 0).
        label : int or str, optional
            The label of the element to retrieve.
        dimension : int, optional
            Defines the index dimension (default is 0).
        return_nodata : bool, optional
            By default this process throws an `ArrayElementNotAvailable` exception if the index or label is invalid.
            If you want to return np.nan instead, set this flag to `True`.
        labels : np.array, optional
            The available labels.

        Returns
        -------
        object
            The value of the requested element.

        Raises
        ------
        ArrayElementNotAvailable :
            The array has no element with the specified index or label.
        ArrayElementParameterMissing :
            Either `index` or `labels` must be set.
        ArrayElementParameterConflict :
            Only `index` or `labels` allowed to be set.

        """
        ArrayElement._check_input(index, label, labels)
        if label:
            # Convert label to index, using labels
            index = labels.tolist().index(label)
        if index >= data.shape[dimension]:
            if not return_nodata:
                raise ArrayElementNotAvailable()
            else:
                array_elem = np.nan
        else:
            idx = create_slices(index, axis=dimension, n_axes=len(data.shape))
            array_elem = data[idx]

        return array_elem


    @staticmethod
    def _check_input(index, label, labels):
        """
        Checks if `index` and `label` are given correctly.

        Either the parameter `index` or `label` must be specified, otherwise the `ArrayElementParameterMissing`
        exception is thrown. If both parameters are set the `ArrayElementParameterConflict `exception is thrown.

        Parameters
        ----------
        index : int, optional
            The zero-based index of the element to retrieve (default is 0).
        label : int or str, optional
            The label of the element to retrieve.
        labels : np.array, optional
            The available labels.

        Raises
        ------
        ArrayElementParameterMissing :
            Either `index` or `labels` must be set.
        ArrayElementParameterConflict :
            Only `index` or `labels` allowed to be set.

        """
        if (index is not None) and (label is not None):
            raise ArrayElementParameterConflict()

        if index is None and label is None:
            raise ArrayElementParameterMissing()
        
        if label and labels is None:
            msg = "Parameter 'labels' is needed when specifying input parameter 'label'."
            raise GenericError(msg)


########################################################################################################################
# Count Process
########################################################################################################################

@process
def count():
    """
    Returns class instance of `Count`.
    For more details, please have a look at the implementations inside `Count`.

    Returns
    -------
    Count :
        Class instance implementing all 'count' processes.

    """
    return Count()


class Count:
    """
    Class instance implementing all 'count' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, condition=None, context=None, dimension=0):
        """
        Gives the number of elements in an array that matches the specified condition.
        Remarks:
            - Counts the number of valid elements by default (condition is set to None).
              A valid element is every element for which is_valid returns True.
            - To count all elements in a list set the `condition` parameter to `True`.

        Parameters
        ----------
        data : np.array
            An array.
        condition : obj, optional
            A condition consists of one ore more processes, which in the end return a boolean value.
            It is evaluated against each element in the array. An element is counted only if the condition
            returns `True`. Defaults to count valid elements in an array (see is_valid). Setting this parameter
            to `True` counts all elements in the array. The following arguments are valid:
                - None : Counts all valid elements, i.e. `is_valid` must yield `True`.
                - `True` : Counts all elements in the array along the specified dimension.
                - object : The following parameters are passed to the process:
                    - `x` : The value of the current element being processed.
                    - `context` : Additional data passed by the user.
        context : dict, optional
            Additional data/keyword arguments to be passed to the condition.
        dimension : int, optional
            Defines the dimension along to count the elements (default is 0).

        Returns
        -------
        count: int
            Count of the data.

        Notes
        -----
        The condition/expression must be able to deal with NumPy arrays.

        """
        if condition is None:
            count = np.sum(is_valid(data), axis=dimension)
        elif condition is True: # explicit check needed
            count = data.shape[dimension]
        elif callable(condition):
            context = context if context is not None else {}
            data = condition(data, **context)
            count = np.sum(data, axis=dimension)
        else:
            err_msg = "Data type of condition is not supported."
            raise ValueError(err_msg)

        return count


########################################################################################################################
# Array Filter Process
########################################################################################################################

@process
def array_filter():
    """
    Returns class instance of `ArrayFilter`.
    For more details, please have a look at the implementations inside `ArrayFilter`.

    Returns
    -------
    ArrayFilter :
        Class instance implementing all 'array_filter' processes.

    """
    return ArrayFilter()


class ArrayFilter:
    """
    Class implementing all 'array_filter' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, condition, context=None):
        """
        Filters the array elements based on a logical expression so that afterwards an array is returned that only
        contains the values conforming to the condition.

        Parameters
        ----------
        data : np.array
            An array.
        condition : callable
            A condition that is evaluated against each value in the array. Only the array elements where the
            condition returns `True` are preserved.
            The following parameters are passed to the process:
                - `x` : The value of the current element being processed.
                - `context` : Additional data passed by the user.
        context : dict, optional
            Additional data/keyword arguments to be passed to the condition.

        Returns
        -------
        np.array :
            An array filtered by the specified condition. The number of elements are less than or equal compared to
            the original array.

        Notes
        -----
        - The condition must be able to deal with NumPy arrays.
        - additional arguments `index` and `label` are ignored as condition arguments

        """

        context = context if context is not None else {}
        return data[condition(data, **context)]


########################################################################################################################
# Array Find Process
########################################################################################################################

@process
def array_find():
    """
    Returns class instance of `ArrayFind`.
    For more details, please have a look at the implementations inside `ArrayFind`.

    Returns
    -------
    ArrayFind :
        Class instance implementing all 'array_find' processes.

    """
    return ArrayFind()


class ArrayFind:
    """
    Class implementing all 'array_find' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, value, dimension=0):
        """
        Checks whether the array specified for `data` contains the value specified in `value` and returns the
        zero-based index for the first match. If there's no match, np.nan is returned..
        Remarks:
            - All definitions for the process `eq` regarding the comparison of values apply here as well.
              A np.nan return value from eq is handled exactly as `False` (no match).
            - Temporal strings are treated as normal strings and are not interpreted.
            - If the specified value is np.nan, the process always returns np.nan.


        Parameters
        ----------
        data : np.array
            An array to find the value in.
        value : object
            Value to find in `data`.
        dimension : int, optional
            Defines the dimension along to find the value (default is 0).

        Returns
        -------
        int :
            Returns the index of the first element with the specified value.
            If no element was found, np.nan is returned.

        Notes
        -----
        Own implementation, since np.argwhere does not fulfil the requirements.

        """
        if np.isnan(value) or is_empty(data):
            return np.nan
        else:
            bool_idxs = (data == value)
            idxs = np.argmax(bool_idxs, axis=dimension)
            return idxs


########################################################################################################################
# Array Labels Process
########################################################################################################################

@process
def array_labels():
    """
    Returns class instance of `ArrayLabels`.
    For more details, please have a look at the implementations inside `ArrayLabels`.

    Returns
    -------
    ArrayLabels :
        Class instance implementing all 'array_labels' processes.

    """
    return ArrayLabels()


class ArrayLabels:
    """
    Class implementing all 'array_labels' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0):
        """
        Returns all labels for a labeled array in the data cube. The labels have the same order as in the array.

        Parameters
        ----------
        data : np.array
            An array with labels.
        dimension : int, optional
            Defines the dimension along to find the labels of the array (default is 0).

        Returns
        -------
        np.array :
            The labels as an array.

        """
        n_vals = data.shape[dimension]
        return np.arange(n_vals)


########################################################################################################################
# First Process
########################################################################################################################

@process
def first():
    """
    Returns class instance of `First`.
    For more details, please have a look at the implementations inside `First`.

    Returns
    -------
    First :
        Class instance implementing all 'first' processes.

    """
    return First()


class First:
    """
    Class implementing all 'first' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, ignore_nodata=True):
        """
        Gives the first element of an array. For an empty array np.nan is returned.

        Parameters
        ----------
        data : np.array
            An array. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to select the first element along (default is 0).

        Returns
        -------
        np.array :
            The first element of the input array.

        """
        if is_empty(data):
            return np.nan

        n_dims = len(data.shape)
        if ignore_nodata:  # skip np.nan values
            nan_mask = ~isnull(data)  # create mask for valid values (not np.nan)
            idx_first = np.argmax(nan_mask, axis=dimension)
            first_elem = np.take_along_axis(data, np.expand_dims(idx_first, axis=dimension), axis=dimension)
        else:  # take the first element, no matter np.nan values are in the array
            idx_first = create_slices(0, axis=dimension, n_axes=n_dims)
            first_elem = data[idx_first]

        return first_elem


########################################################################################################################
# Last Process
########################################################################################################################

@process
def last():
    """
    Returns class instance of `Last`.
    For more details, please have a look at the implementations inside `Last`.

    Returns
    -------
    Last :
        Class instance implementing all 'last' processes.

    """
    return Last()


class Last:
    """
    Class implementing all 'last' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, ignore_nodata=True):
        """
        Gives the last element of an array. For an empty array np.nan is returned.

        Parameters
        ----------
        data : np.array
            An array. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to select the last element along (default is 0).

        Returns
        -------
        np.array :
            The last element of the input array.

        """
        if is_empty(data):
            return np.nan

        n_dims = len(data.shape)
        if ignore_nodata:  # skip np.nan values
            data = np.flip(data, axis=dimension)  # flip data to retrieve the first valid element (thats the only way it works with argmax)
            last_elem = first(data, ignore_nodata=ignore_nodata, dimension=dimension)
        else:  # take the first element, no matter np.nan values are in the array
            idx_last = create_slices(-1, axis=dimension, n_axes=n_dims)
            last_elem = data[idx_last]

        return last_elem


########################################################################################################################
# Order Process
########################################################################################################################

@process
def order():
    """
    Returns class instance of `Order`.
    For more details, please have a look at the implementations inside `Order`.

    Returns
    -------
    Order :
        Class instance implementing all 'order' processes.

    """
    return Order()


# TODO: can nodata algorithm be simplified/enhanced?
class Order:
    """
    Class implementing all 'order' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, asc=True, nodata=None):
        """
        Computes a permutation which allows rearranging the data into ascending or descending order.
        In other words, this process computes the ranked (sorted) element positions in the original list.
        Remarks:
            - The positions in the result are zero-based.
            - Ties will be left in their original ordering.

        Parameters
        ----------
        data : np.array
            An array to compute the order for.
        dimension : int, optional
            Defines the dimension to order along (default is 0).
        asc : bool, optional
            The default sort order is ascending, with smallest values first. To sort in reverse (descending) order,
            set this parameter to `False`.
        nodata : obj, optional
            Controls the handling of no-data values (np.nan). By default they are removed. If `True`, missing values
            in the data are put last; if `False`, they are put first.

        Returns
        -------
        np.array :
            The computed permutation.

        Notes
        -----
        - the case with nodata=False is complicated, since a simple nan masking destroys the structure of the array
        - due to the flipping, the order of the np.nan values is wrong, but this is ignored, since this order should
          not be relevant
        """

        if asc:
            permutation_idxs = np.argsort(data, kind='mergesort', axis=dimension)
        else:  # [::-1] not possible
            permutation_idxs = np.argsort(-data, kind='mergesort', axis=dimension)  # to get the indizes in descending order, the sign of the data is changed

        if nodata is None:  # ignore np.nan values
            # sort the original data first, to get correct position of no data values
            sorted_data = data[permutation_idxs]
            return permutation_idxs[~isnull(sorted_data)]
        elif nodata is False:  # put location/index of np.nan values first
            # sort the original data first, to get correct position of no data values
            sorted_data = data[permutation_idxs]
            nan_idxs = isnull(sorted_data)

            # flip permutation and nan mask
            permutation_idxs_flip = np.flip(permutation_idxs, axis=dimension)
            nan_idxs_flip = np.flip(nan_idxs, axis=dimension)

            # flip causes the nan.values to be first, however the order of all other values is also flipped
            # therefore the non np.nan values (i.e. the wrong flipped order) is replaced by the right order given by
            # the original permutation values
            permutation_idxs_flip[~nan_idxs_flip] = permutation_idxs[~nan_idxs]

            return permutation_idxs_flip
        elif nodata is True:  # default argsort behaviour, np.nan values are put last
            return permutation_idxs
        else:
            err_msg = "Data type of 'nodata' argument is not supported."
            raise Exception(err_msg)


########################################################################################################################
# Rearrange Process
########################################################################################################################

@process
def rearrange():
    """
    Returns class instance of `Rearrange`.
    For more details, please have a look at the implementations inside `Rearrange`.

    Returns
    -------
    Rearrange :
        Class instance implementing all 'rearrange' processes.

    """
    return Rearrange()


class Rearrange:
    """
    Class implementing all 'rearrange' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, order):
        """
        Rearranges an array based on a permutation, i.e. a ranked list of element positions in the original list.
        The positions must be zero-based.

        Parameters
        ----------
        data : np.array
            The array to rearrange.
        order : np.array
            The permutation used for rearranging.

        Returns
        -------
        np.array :
            The rearranged array.

        """

        return data[order]


########################################################################################################################
# Sort Process
########################################################################################################################

@process
def sort():
    """
    Returns class instance of `Sort`.
    For more details, please have a look at the implementations inside `Sort`.

    Returns
    -------
    Sort :
        Class instance implementing all 'sort' processes.

    """
    return Sort()


# TODO: can nodata=False algorithm be simplified?
class Sort:
    """
    Class implementing all 'sort' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, asc=True, nodata=None):
        """
        Sorts an array into ascending (default) or descending order.
        Remarks:
            - Ties will be left in their original ordering.

        Parameters
        ----------
        data : np.array
            An array with data to sort.
        dimension : int, optional
            Defines the dimension to sort along (default is 0).
        asc : bool, optional
            The default sort order is ascending, with smallest values first. To sort in reverse (descending) order,
            set this parameter to `False`.
        nodata : obj, optional
            Controls the handling of no-data values (np.nan). By default they are removed. If `True`, missing values
            in the data are put last; if `False`, they are put first.

        Returns
        -------
        np.array :
            The sorted array.

        """
        if asc:
            data_sorted = np.sort(data, axis=dimension)
        else:  # [::-1] not possible
            data_sorted = -np.sort(-data, axis=dimension)  # to get the indexes in descending order, the sign of the data is changed

        if nodata is None:  # ignore np.nan values
            nan_idxs = isnull(data_sorted)
            return data_sorted[~nan_idxs]
        elif nodata == False:  # put np.nan values first
            nan_idxs = isnull(data_sorted)
            data_sorted_flip = np.flip(data_sorted, axis=dimension)
            nan_idxs_flip = isnull(data_sorted_flip)
            data_sorted_flip[~nan_idxs_flip] = data_sorted[~nan_idxs]
            return data_sorted_flip
        elif nodata == True:  # default sort behaviour, np.nan values are put last
            return data_sorted
        else:
            err_msg = "Data type of 'nodata' argument is not supported."
            raise Exception(err_msg)
