# Adapted from openeo-processes-python (https://github.com/Open-EO/openeo-processes-python),
# release 0.0.4, licensed under Apache License 2.0.
# Only the numpy/pure-python implementations of a limited set of processes are retained.
# xarray/dask specific code and the pandas dependency have been removed.

import builtins
import numpy as np

from openeo_driver.vendor.processes_python.utils import process
from openeo_driver.vendor.processes_python.comparison import is_empty

from openeo_driver.vendor.processes_python.errors import QuantilesParameterConflict
from openeo_driver.vendor.processes_python.errors import QuantilesParameterMissing


########################################################################################################################
# Argumentless Functions/Constants
########################################################################################################################

def e():
    """
    The real number e is a mathematical constant that is the base of the natural logarithm such that ln(e) = 1.
    The numerical value is approximately 2.71828.

    Returns
    -------
    float :
        The numerical value of Euler's number.
    """
    return np.e


def pi():
    """
    The real number Pi (π) is a mathematical constant that is the ratio of the circumference of a circle to its
    diameter. The numerical value is approximately 3.14159.

    Returns
    -------
    float :
        The numerical value of Pi.

    """
    return np.pi


########################################################################################################################
# Floor Process
########################################################################################################################

@process
def floor():
    """
    Returns class instance of `Floor`.
    For more details, please have a look at the implementations inside `Floor`.

    Returns
    -------
    Floor
        Class instance implementing all 'floor' processes.

    """
    return Floor()


class Floor:
    """
    Class implementing all 'floor' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        The greatest integer less than or equal to the number `x`. This process is not an alias for the 'int' process as
        defined by some mathematicians. See the examples for negative numbers in both processes for differences.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            A number to round down.

        Returns
        -------
        float :
            The number rounded down.

        """
        return np.floor(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        The greatest integer less than or equal to the numbers `x`. This process is not an alias for the 'int' process as
        defined by some mathematicians. See the examples for negative numbers in both processes for differences.
        The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Numbers to round down.

        Returns
        -------
        np.array :
            Numbers rounded down.

        """
        return np.floor(x)


########################################################################################################################
# Ceil Process
########################################################################################################################

@process
def ceil():
    """
    Returns class instance of `Ceil`.
    For more details, please have a look at the implementations inside `Ceil`.

    Returns
    -------
    Ceil
        Class instance implementing all 'ceil' processes.

    """
    return Ceil()


class Ceil:
    """
    Class implementing all 'ceil' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        The least integer greater than or equal to the given number.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            A number to round up.

        Returns
        -------
        float :
            The number rounded up.

        """
        return np.ceil(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        The least integer greater than or equal to the given numbers.
        The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Numbers to round up.

        Returns
        -------
        np.array :
            Numbers rounded up.

        """
        return np.ceil(x)


########################################################################################################################
# Int Process
########################################################################################################################

@process
def int():
    """
    Returns class instance of `Int`.
    For more details, please have a look at the implementations inside `Int`.

    Returns
    -------
    Int
        Class instance implementing all 'int' processes.

    """
    return Int()


class Int:
    """
    Class implementing all 'int' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        The integer part of the real number `x`. This process is not an alias for the floor process as defined by
        some mathematicians, see the examples for negative numbers in both processes for differences.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            A number.

        Returns
        -------
        int :
            Integer part of the number.

        """
        return builtins.int(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        The integer part of the real number `x`. This process is not an alias for the floor process as defined by
        some mathematicians, see the examples for negative numbers in both processes for differences.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Numbers.

        Returns
        -------
        int :
            Integer part of the numbers.

        """
        int_x = x.astype(int)
        obj_x = int_x.astype(object)  # convert array to object type to enable storing None values
        obj_x[np.isnan(x)] = None

        return obj_x


########################################################################################################################
# Round Process
########################################################################################################################

@process
def round():
    """
    Returns class instance of `Round`.
    For more details, please have a look at the implementations inside `Round`.

    Returns
    -------
    Round :
        Class instance implementing all 'round' processes.

    """
    return Round()


class Round:
    """
    Class implementing all 'round' processes.

    """

    @staticmethod
    def exec_num(x, p=0):
        """
        Rounds a real number `x` to specified precision `p`.
        If the fractional part of `x` is halfway between two integers, one of which is even and the other odd,
        then the even number is returned. This behaviour follows IEEE Standard 754.
        This kind of rounding is also called "rounding to nearest" or "banker's rounding".
        It minimizes rounding errors that result from consistently rounding a midpoint value in a single direction.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            A number to round.
        p : int, optional
            A positive number specifies the number of digits after the decimal point to round to.
            A negative number means rounding to a power of ten, so for example -2 rounds to the nearest hundred.
            Defaults to 0.

        Returns
        -------
        int or float :
            The rounded number.

        """
        return builtins.round(x, p)

    @staticmethod
    def exec_np(x, p=0):
        """
        Rounds real numbers `x` to specified precision `p`.
        If the fractional part of `x` is halfway between two integers, one of which is even and the other odd,
        then the even number is returned. This behaviour follows IEEE Standard 754.
        This kind of rounding is also called "rounding to nearest" or "banker's rounding".
        It minimizes rounding errors that result from consistently rounding a midpoint value in a single direction.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Numbers to round.
        p : int, optional
            A positive number specifies the number of digits after the decimal point to round to.
            A negative number means rounding to a power of ten, so for example -2 rounds to the nearest hundred.
            Defaults to 0.

        Returns
        -------
        np.array :
            The rounded numbers.

        """
        return np.around(x, p)


########################################################################################################################
# Exp Process
########################################################################################################################

@process
def exp():
    """
    Returns class instance of `Exp`.
    For more details, please have a look at the implementations inside `Exp`.

    Returns
    -------
    Exp
        Class instance implementing all 'exp' processes.

    """
    return Exp()


class Exp:
    """
    Class implementing all 'exp' processes.

    """

    @staticmethod
    def exec_num(p):
        """
        Exponential function to the base e raised to the power of `p`.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        p : int or float
            The numerical exponent.

        Returns
        -------
        float :
            The computed value for e raised to the power of `p`.

        """
        return np.exp(p) if p is not None else p

    @staticmethod
    def exec_np(p):
        """
        Exponential function to the base e raised to the power of `p`.
        The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        p : np.array
            The numerical exponent.

        Returns
        -------
        np.array :
            The computed values for e raised to the power of `p`.

        """
        return np.exp(p)


########################################################################################################################
# Log Process
########################################################################################################################

@process
def log():
    """
    Returns class instance of `Log`.
    For more details, please have a look at the implementations inside `Log`.

    Returns
    -------
    Log :
        Class instance implementing all 'log' processes.

    """
    return Log()


class Log:
    """
    Class implementing all 'log' processes.

    """

    @staticmethod
    def exec_num(x, base):
        """
        Logarithm to the base `base` of the number `x` is defined to be the inverse function of taking `base` to the
        power of `x`. The no-data value None is passed through and therefore gets propagated if any of the arguments is
        None. The computations follow IEEE Standard 754 whenever the processing environment supports it.
        Therefore, exec_num(0, 2) results in ±infinity.

        Parameters
        ----------
        x : int or float
            A number to compute the logarithm for.
        base : int or float
            The numerical base.

        Returns
        -------
        float :
            The computed logarithm.

        """
        return Log.exec_np(x, base) if x is not None and base is not None else None

    @staticmethod
    def exec_np(x, base):
        """
        Logarithm to the base `base` of the numbers `x` is defined to be the inverse function of taking `base` to the
        powers of `x`. The no-data value np.nan is passed through and therefore gets propagated if any of the arguments
        is None. The computations follow IEEE Standard 754 whenever the processing environment supports it.
        Therefore, exec_np(0, 2) results in ±infinity.

        Parameters
        ----------
        x : np.array
            Numbers to compute the logarithm for.
        base : int or float
            The numerical base.

        Returns
        -------
        np.array :
            The computed logarithm.

        """
        return np.log(x)/np.log(base)


########################################################################################################################
# Ln Process
########################################################################################################################

@process
def ln():
    """
    Returns class instance of `Ln`.
    For more details, please have a look at the implementations inside `Ln`.

    Returns
    -------
    Ln :
        Class instance implementing all 'ln' processes.

    """
    return Ln()


class Ln:
    """
    Class implementing all 'ln' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        The natural logarithm is the logarithm to the base e of the number `x`, which equals to using the log process
        with the base set to e. The natural logarithm is the inverse function of taking e to the power `x`. The no-data
        value None is passed through. The computations follow IEEE Standard 754.
        Therefore, exec_num(0) results in ±infinity.

        Parameters
        ----------
        x : int or float
            A number to compute the natural logarithm for.

        Returns
        -------
        float :
            The computed natural logarithm.

        """
        return np.log(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        The natural logarithm is the logarithm to the base e of the numbers `x`, which equals to using the log process
        with the base set to e. The natural logarithm is the inverse function of taking e to the powers `x`. The no-data
        value np.nan is passed through. The computations follow IEEE Standard 754.
        Therefore, exec_np(0) results in ±infinity.

        Parameters
        ----------
        x : np.array
            Numbers to compute the natural logarithm for.

        Returns
        -------
        np.array :
            The computed natural logarithms.

        """
        return np.log(x)


########################################################################################################################
# Cos Process
########################################################################################################################

@process
def cos():
    """
    Returns class instance of `Cos`.
    For more details, please have a look at the implementations inside `Cos`.

    Returns
    -------
    Cos :
        Class instance implementing all 'cos' processes.

    """
    return Cos()


class Cos:
    """
    Class implementing all 'cos' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the cosine of `x`.
        Works on radians only. The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            An angle in radians.

        Returns
        -------
        float :
            The computed cosine of `x`.

        """
        return np.cos(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the cosine of `x`.
        Works on radians only. The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Angles in radians.

        Returns
        -------
        np.array :
            The computed cosines of `x`.

        """
        return np.cos(x)


########################################################################################################################
# Arccos Process
########################################################################################################################

@process
def arccos():
    """
    Returns class instance of `Arccos`.
    For more details, please have a look at the implementations inside `Arccos`.

    Returns
    -------
    Arccos :
        Class instance implementing all 'arccos' processes.

    """
    return Arccos()


class Arccos:
    """
    Class implementing all 'arccos' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the arc cosine of `x`. The arc cosine is the inverse function of the cosine so that
        `arccos(cos(x)) = x`. Works on radians only. The no-data value None is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : int or float
            A number.

        Returns
        -------
        float :
            The computed angle in radians.

        """
        return np.arccos(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the arc cosine of `x`. The arc cosine is the inverse function of the cosine so that
        `arccos(cos(x)) = x`. Works on radians only. The no-data value np.nan is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : np.array
            Numbers.

        Returns
        -------
        np.array :
            The computed angles in radians.

        """
        return np.arccos(x)


########################################################################################################################
# Cosh Process
########################################################################################################################

@process
def cosh():
    """
    Returns class instance of `Cosh`.
    For more details, please have a look at the implementations inside `Cosh`.

    Returns
    -------
    Cosh :
        Class instance implementing all 'cosh' processes.

    """
    return Cosh()


class Cosh:
    """
    Class implementing all 'cosh' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the hyperbolic cosine of `x`. Works on radians only.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            An angle in radians.

        Returns
        -------
        float :
            The computed hyperbolic cosine of `x`.

        """
        return np.cosh(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the hyperbolic cosine of `x`. Works on radians only.
        The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Angles in radians.

        Returns
        -------
        np.array :
            The computed hyperbolic cosines of `x`.

        """
        return np.cosh(x)


########################################################################################################################
# Arcosh Process
########################################################################################################################

@process
def arcosh():
    """
    Returns class instance of `Arcosh`.
    For more details, please have a look at the implementations inside `Arcosh`.

    Returns
    -------
    Arcosh :
        Class instance implementing all 'arcosh' processes.

    """
    return Arcosh()


class Arcosh:
    """
    Class implementing all 'arcosh' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the inverse hyperbolic cosine of `x`. It is the inverse function of the hyperbolic cosine so that
        `arcosh(cosh(x)) = x`. Works on radians only. The no-data value None is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : int or float
            A number.

        Returns
        -------
        float :
            The computed angle in radians.

        """
        return np.arccosh(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the inverse hyperbolic cosine of `x`. It is the inverse function of the hyperbolic cosine so that
        `arcosh(cosh(x)) = x`. Works on radians only. The no-data value np.nan is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : np.array
            Numbers.

        Returns
        -------
        np.array :
            The computed angles in radians.

        """
        return np.arccosh(x)


########################################################################################################################
# Sin Process
########################################################################################################################

@process
def sin():
    """
    Returns class instance of `Sin`.
    For more details, please have a look at the implementations inside `Sin`.

    Returns
    -------
    Sin :
        Class instance implementing all 'sin' processes.

    """
    return Sin()


class Sin:
    """
    Class implementing all 'sin' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the sine of `x`.
        Works on radians only. The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            An angle in radians.

        Returns
        -------
        float :
            The computed sine of `x`.

        """
        return np.sin(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the sine of `x`.
        Works on radians only. The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Angles in radians.

        Returns
        -------
        np.array :
            The computed sines of `x`.

        """
        return np.sin(x)


########################################################################################################################
# Arcsin Process
########################################################################################################################

@process
def arcsin():
    """
    Returns class instance of `Arcsin`.
    For more details, please have a look at the implementations inside `Arcsin`.

    Returns
    -------
    Arcsin :
        Class instance implementing all 'arcsin' processes.

    """
    return Arcsin()


class Arcsin:
    """
    Class implementing all 'arcsin' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the arc sine of `x`. The arc sine is the inverse function of the sine so that
        `arcsin(sin(x)) = x`. Works on radians only. The no-data value None is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : int or float
            A number.

        Returns
        -------
        float :
            The computed angle in radians.

        """
        return np.arcsin(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the arc sine of `x`. The arc sine is the inverse function of the sine so that
        `arcsin(sin(x)) = x`. Works on radians only. The no-data value np.nan is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : np.array
            Numbers.

        Returns
        -------
        np.array :
            The computed angles in radians.

        """
        return np.arcsin(x)


########################################################################################################################
# Sinh Process
########################################################################################################################

@process
def sinh():
    """
    Returns class instance of `Sinh`.
    For more details, please have a look at the implementations inside `Sinh`.

    Returns
    -------
    Sinh :
        Class instance implementing all 'sinh' processes.

    """
    return Sinh()


class Sinh:
    """
    Class implementing all 'sinh' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the hyperbolic sine of `x`. Works on radians only.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            An angle in radians.

        Returns
        -------
        float :
            The computed hyperbolic sine of `x`.

        """
        return np.sinh(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the hyperbolic sine of `x`. Works on radians only.
        The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Angles in radians.

        Returns
        -------
        np.nan :
            The computed hyperbolic sines of `x`.

        """
        return np.sinh(x)


########################################################################################################################
# Arsinh Process
########################################################################################################################

@process
def arsinh():
    """
    Returns class instance of `Arsinh`.
    For more details, please have a look at the implementations inside `Arsinh`.

    Returns
    -------
    Arsinh :
        Class instance implementing all 'arsinh' processes.

    """
    return Arsinh()


class Arsinh:
    """
    Class implementing all 'arsinh' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the inverse hyperbolic sine of `x`. It is the inverse function of the hyperbolic sine so that
        `arsinh(sinh(x)) = x`. Works on radians only. The no-data value None is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : int or float
            A number.

        Returns
        -------
        float :
            The computed angle in radians.

        """
        return np.arcsinh(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the inverse hyperbolic sine of `x`. It is the inverse function of the hyperbolic sine so that
        `arsinh(sinh(x)) = x`. Works on radians only. The no-data value np.nan is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : np.array
            Numbers.

        Returns
        -------
        np.array :
            The computed angles in radians.

        """
        return np.arcsinh(x)


########################################################################################################################
# Tan Process
########################################################################################################################

@process
def tan():
    """
    Returns class instance of `Tan`.
    For more details, please have a look at the implementations inside `Tan`.

    Returns
    -------
    Tan :
        Class instance implementing all 'tan' processes.

    """
    return Tan()


class Tan:
    """
    Class implementing all 'tan' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the tangent of `x`. The tangent is defined to be the sine of `x` divided by the cosine of `x`.
        Works on radians only. The no-data value None is passed through and therefore gets propagated.


        Parameters
        ----------
        x : int or float
            An angle in radians.

        Returns
        -------
        float :
            The computed tangent of `x`.

        """
        return np.tan(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the tangent of `x`. The tangent is defined to be the sine of `x` divided by the cosine of `x`.
        Works on radians only. The no-data value np.nan is passed through and therefore gets propagated.


        Parameters
        ----------
        x : np.array
            Angles in radians.

        Returns
        -------
        np.array :
            The computed tangents of `x`.

        """
        return np.tan(x)


########################################################################################################################
# Arctan Process
########################################################################################################################

@process
def arctan():
    """
    Returns class instance of `Arctan`.
    For more details, please have a look at the implementations inside `Arctan`.

    Returns
    -------
    Arctan :
        Class instance implementing all 'arctan' processes.

    """
    return Arctan()


class Arctan:
    """
    Class implementing all 'arctan' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the arc tangent of `x`. The arc tangent is the inverse function of the tangent so that
        `arctan(tan(x)) = x`. Works on radians only. The no-data value None is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : int or float
            A number.

        Returns
        -------
        float :
            The computed angle in radians.

        """
        return np.arctan(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the arc tangent of `x`. The arc tangent is the inverse function of the tangent so that
        `arctan(tan(x)) = x`. Works on radians only. The no-data value np.nan is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : np.array
            Numbers.

        Returns
        -------
        np.array :
            The computed angles in radians.

        """
        return np.arctan(x)


########################################################################################################################
# Tanh Process
########################################################################################################################

@process
def tanh():
    """
    Returns class instance of `Tanh`.
    For more details, please have a look at the implementations inside `Tanh`.

    Returns
    -------
    Tanh :
        Class instance implementing all 'tanh' processes.

    """
    return Tanh()


class Tanh:
    """
    Class implementing all 'tanh' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the hyperbolic tangent of `x`.  The hyperbolic tangent is defined to be the hyperbolic sine of `x`
        divided by the hyperbolic cosine of `x`. Works on radians only.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            An angle in radians.

        Returns
        -------
        float :
            The computed hyperbolic sine of `x`.

        """
        return np.tanh(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the hyperbolic tangent of `x`.  The hyperbolic tangent is defined to be the hyperbolic sine of `x`
        divided by the hyperbolic cosine of `x`. Works on radians only.
        The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Angles in radians.

        Returns
        -------
        np.array :
            The computed hyperbolic tangents of `x`.

        """
        return np.tanh(x)


########################################################################################################################
# Artanh Process
########################################################################################################################

@process
def artanh():
    """
    Returns class instance of `Artanh`.
    For more details, please have a look at the implementations inside `Artanh`.

    Returns
    -------
    Artanh :
        Class instance implementing all 'artanh' processes.

    """
    return Artanh()


class Artanh:
    """
    Class implementing all 'artanh' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the inverse hyperbolic tangent of `x`. It is the inverse function of the hyperbolic tangent so that
        `artanh(tanh(x)) = x`. Works on radians only. The no-data value None is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : int or float
            A number.

        Returns
        -------
        float :
            The computed angle in radians.

        """
        return np.arctanh(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the inverse hyperbolic tangent of `x`. It is the inverse function of the hyperbolic tangent so that
        `artanh(tanh(x)) = x`. Works on radians only. The no-data value np.nan is passed through and therefore gets
        propagated.

        Parameters
        ----------
        x : np.array
            Numbers.

        Returns
        -------
        np.array :
            The computed angles in radians.

        """
        return np.arctanh(x)


########################################################################################################################
# Arctan2 Process
########################################################################################################################

@process
def arctan2():
    """
    Returns class instance of `Arctan2`.
    For more details, please have a look at the implementations inside `Arctan2`.

    Returns
    -------
    Arctan2 :
        Class instance implementing all 'arctan2' processes.

    """
    return Arctan2()


class Arctan2:
    """
    Class implementing all 'arctan2' processes.

    """

    @staticmethod
    def exec_num(y, x):
        """
        Computes the arc tangent of two numbers `x` and `y`. It is similar to calculating the arc tangent of `y/x`,
        except that the signs of both arguments are used to determine the quadrant of the result. Works on radians only.
        The no-data value None is passed through and therefore gets propagated if any of the arguments is null.

        Parameters
        ----------
        y : int or float
            A number to be used as dividend.
        x : int or float
            A number to be used as divisor.

        Returns
        -------
        float :
            The computed angle in radians.

        """
        return np.arctan2(y, x) if x is not None and y is not None else None

    @staticmethod
    def exec_np(y, x):
        """
        Computes the arc tangent of two arrays `x` and `y`. It is similar to calculating the arc tangent of `y/x`,
        except that the signs of both arguments are used to determine the quadrant of the result. Works on radians only.
        The no-data value np.nan is passed through and therefore gets propagated if any of the arguments is null.

        Parameters
        ----------
        y : np.array
            Numbers to be used as dividend.
        x : np.array
            Numbers to be used as divisor.

        Returns
        -------
        np.array :
            The computed angles in radians.

        """
        return np.arctan2(y, x)


########################################################################################################################
# Mod Process
########################################################################################################################

@process
def mod():
    """
    Returns class instance of `Mod`.
    For more details, please have a look at the implementations inside `Mod`.

    Returns
    -------
    Mod :
        Class instance implementing all 'mod' processes.

    """
    return Mod()


class Mod:
    """
    Class implementing all 'mod' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Remainder after division of `x` by `y`. The result of a modulo operation has the sign of the divisor.
        The no-data value None is passed through and therefore gets propagated if any of the arguments is None.

        Parameters
        ----------
        x : int or float
            A number to be used as dividend.
        y : int or float
            A number to be used as divisor.

        Returns
        -------
        float :
            The remainder after division.

        """
        return x % y if x is not None and y is not None else None

    @staticmethod
    def exec_np(x, y):
        """
        Remainder after division of `x` by `y`. The result of a modulo operation has the sign of the divisor.
        The no-data value None is passed through and therefore gets propagated if any of the arguments is None.

        Parameters
        ----------
        x : np.array
            Numbers to be used as dividend.
        y : np.array
            Numbers to be used as divisor.

        Returns
        -------
        np.array :
            The remainders after division.

        """
        return np.mod(x, y)


########################################################################################################################
# Absolute Process
########################################################################################################################

@process
def absolute():
    """
    Returns class instance of `Absolute`.
    For more details, please have a look at the implementations inside `Absolute`.

    Returns
    -------
    Mod :
        Class instance implementing all 'absolute' processes.

    """
    return Absolute()


class Absolute:
    """
    Class implementing all 'absolute' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the absolute value of a real number `x`, which is the "unsigned" portion of `x` and
        often denoted as `|x|`. The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            A number.

        Returns
        -------
        int or float :
            The computed absolute value.

        """
        return abs(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        Computes the absolute value of real numbers `x`, which is the "unsigned" portion of `x` and
        often denoted as `|x|`. The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Numbers.

        Returns
        -------
        np.array :
            The computed absolute values.

        """
        return np.abs(x)


########################################################################################################################
# Sgn Process
########################################################################################################################

@process
def sgn():
    """
    Returns class instance of `Sgn`.
    For more details, please have a look at the implementations inside `Sgn`.

    Returns
    -------
    Sgn :
        Class instance implementing all 'sgn processes.

    """
    return Sgn()


class Sgn:
    """
    Class implementing all 'sgn' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        The signum (also known as sign) of `x` is defined as:

            - 1 if x > 0
            - 0 if x = 0
            - -1 if x < 0

        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            A number.

        Returns
        -------
        int :
            The computed signum value of `x`.

        """
        return np.sign(x) if x is not None else x

    @staticmethod
    def exec_np(x):
        """
        The signum (also known as sign) of `x` is defined as:

            - 1 if x > 0
            - 0 if x = 0
            - -1 if x < 0

        The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Numbers.

        Returns
        -------
        np.array :
            The computed signum values of `x`.

        """
        return np.sign(x)


########################################################################################################################
# Sqrt Process
########################################################################################################################

@process
def sqrt():
    """
    Returns class instance of `Sqrt`.
    For more details, please have a look at the implementations inside `Sqrt`.

    Returns
    -------
    Sqrt :
        Class instance implementing all 'sqrt' processes.

    """
    return Sqrt()


class Sqrt:
    """
    Class implementing all 'sqrt' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Computes the square root of a real number `x`, which is equal to calculating `x` to the power of 0.5.
        A square root of `x` is a number `a` such that `a^2 = x`. Therefore, the square root is the inverse function
        of `a` to the power of 2, but only for `a >= 0`.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            A number.

        Returns
        -------
        float :
            The computed square root.

        """
        return np.sqrt(x)

    @staticmethod
    def exec_np(x):
        """
        Computes the square root of real numbers `x`, which is equal to calculating `x` to the power of 0.5.
        Square roots of `x` are numbers `a` such that `a^2 = x`. Therefore, the square root is the inverse function
        of `a` to the power of 2, but only for `a >= 0`.
        The no-data value np.nan is passed through and therefore gets propagated.

        Parameters
        ----------
        x : np.array
            Numbers.

        Returns
        -------
        np.array :
            The computed square roots.

        """
        return np.sqrt(x)


########################################################################################################################
# Power Process
########################################################################################################################

@process
def power():
    """
    Returns class instance of `Power`.
    For more details, please have a look at the implementations inside `Power`.

    Returns
    -------
    Power :
        Class instance implementing all 'power' processes.

    """
    return Power()


class Power:
    """
    Class implementing all 'power' processes.

    """

    @staticmethod
    def exec_num(base, p):
        """
        Computes the exponentiation for the base `base` raised to the power of `p`.
        The no-data value None is passed through and therefore gets propagated if any of the arguments is None.

        Parameters
        ----------
        base : int or float
            The numerical base.
        p : int or float
            The numerical exponent.

        Returns
        -------
        int or float :
            The computed value for `base` raised to the power of `p`.

        """
        return np.power(base, float(p)) if base is not None and p is not None else None  # float(p) because of error message in NumPy: ValueError: Integers to negative integer powers are not allowed.

    @staticmethod
    def exec_np(base, p):
        """
        Computes the exponentiation for the bases `base` raised to the power of `p`.
        The no-data value np.nan is passed through and therefore gets propagated if any of the arguments is np.nan.

        Parameters
        ----------
        base : np.array
            The numerical bases.
        p : int or float
            The numerical exponent.

        Returns
        -------
        np.array :
            The computed values for `base` raised to the power of `p`.

        """

        return np.power(base, float(p))  # float(p) because of error message in NumPy: ValueError: Integers to negative integer powers are not allowed.


########################################################################################################################
# Mean Process
########################################################################################################################

@process
def mean():
    """
    Returns class instance of `Mean`.
    For more details, please have a look at the implementations inside `Mean`.

    Returns
    -------
    Mean :
        Class instance implementing all 'mean' processes.

    """
    return Mean()


class Mean:
    """
    Class implementing all 'mean' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        The arithmetic mean of an array of numbers is the quantity commonly called the average.
        It is defined as the sum of all elements divided by the number of elements.

        Parameters
        ----------
        data : np.array
            An array of numbers. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to false considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to calculate the mean along (default is 0).

        Returns
        -------
        np.array :
            The computed arithmetic mean values.

        """
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.mean(data, axis=dimension)
        else:
            return np.nanmean(data, axis=dimension)


########################################################################################################################
# Min Process
########################################################################################################################

@process
def min():
    """
    Returns class instance of `Min`.
    For more details, please have a look at the implementations inside `Min`.

    Returns
    -------
    Min :
        Class instance implementing all 'min' processes.

    """
    return Min()


class Min:
    """
    Class implementing all 'min' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Computes the smallest value of an array of numbers, which is is equal to the last element of a sorted
        (i.e., ordered) version the array.

        Parameters
        ----------
        data : np.array
            An array of numbers. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to calculate the minimum along (default is 0).

        Returns
        -------
        np.array :
            The minimum values.

        """
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.min(data, axis=dimension)
        else:
            return np.nanmin(data, axis=dimension)


########################################################################################################################
# Max Process
########################################################################################################################

@process
def max():
    """
    Returns class instance of `Max`.
    For more details, please have a look at the implementations inside `Max`.

    Returns
    -------
    Max :
        Class instance implementing all 'max' processes.

    """
    return Max()


class Max():

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Computes the largest value of an array of numbers, which is is equal to the first element of a sorted
        (i.e., ordered) version the array.

        Parameters
        ----------
        data : np.array
            An array of numbers. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to calculate the maximum along (default is 0).

        Returns
        -------
        np.array :
            The maximum values.

        """
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.max(data, axis=dimension)
        else:
            return np.nanmax(data, axis=dimension)


########################################################################################################################
# Median Process
########################################################################################################################

@process
def median():
    """
    Returns class instance of `Median`.
    For more details, please have a look at the implementations inside `Median`.

    Returns
    -------
    Median :
        Class instance implementing all 'median' processes.

    """
    return Median()


class Median:
    """
    Class implementing all 'median' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        The statistical median of an array of numbers is the value separating the higher half from the lower half of
        the data. Remarks:

            - For a symmetric arrays, the result is equal to the mean.
            - The median can also be calculated by computing the quantile (see process `quantiles`) with the
              probability of 0.5: quantiles(data, [0.5]).
            - An empty input array returns np.nan.

        Parameters
        ----------
        data : np.array
            An array of numbers. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to calculate the median along (default is 0).

        Returns
        -------
        np.array :
            The computed statistical medians.

        """

        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.median(data, axis=dimension)
        else:
            return np.nanmedian(data, axis=dimension)


########################################################################################################################
# Standard Deviation Process
########################################################################################################################

@process
def sd():
    """
    Returns class instance of `Sd`.
    For more details, please have a look at the implementations inside `Sd`.

    Returns
    -------
    Sd :
        Class instance implementing all 'sd' processes.

    """
    return Sd()


class Sd:
    """
    Class implementing all 'sd' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Computes the sample standard deviation, which quantifies the amount of variation of an array of numbers.
        It is defined to be the square root of the corresponding variance (see `variance`). A low standard deviation
        indicates that the values tend to be close to the expected value, while a high standard deviation indicates
        that the values are spread out over a wider range.


        Parameters
        ----------
        data : np.array
            An array of numbers. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to calculate the standard deviation along (default is 0).

        Returns
        -------
        np.array :
            The computed sample standard deviations.

        Notes
        -----
        The standard deviation is computed with 1 as a degree of freedom.

        """
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.std(data, axis=dimension, ddof=1)
        else:
            return np.nanstd(data, axis=dimension, ddof=1)


########################################################################################################################
# Variance Process
########################################################################################################################

@process
def variance():
    """
    Returns class instance of `Variance`.
    For more details, please have a look at the implementations inside `Variance`.

    Returns
    -------
    Variance :
        Class instance implementing all 'variance' processes.

    """
    return Variance()


class Variance:
    """
    Class implementing all 'variance' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Computes the sample variance of an array of numbers by calculating the square of the standard deviation
        (see `sd`). It is defined to be the expectation of the squared deviation of a random variable from its
        expected value. Basically, it measures how far the numbers in the array are spread out from their average value.

        Parameters
        ----------
        data : np.array
            An array of numbers. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to calculate the variance along (default is 0).

        Returns
        -------
        np.array :
            The computed sample variances.

        """
        if is_empty(data):
            return np.nan

        if not ignore_nodata:
            return np.var(data, axis=dimension, ddof=1)
        else:
            return np.nanvar(data, axis=dimension, ddof=1)


########################################################################################################################
# Extrema Process
########################################################################################################################

@process
def extrema():
    """
    Returns class instance of `Extrema`.
    For more details, please have a look at the implementations inside `Extrema`.

    Returns
    -------
    Extrema :
        Class instance implementing all 'extrema' processes.

    """
    return Extrema()


class Extrema:
    """
    Class implementing all 'extrema' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0, ignore_nodata=True):
        """
        Two element array containing the minimum and the maximum values of data. This process is basically an alias
        for calling both `min` and `max`.

        Parameters
        ----------
        data : np.array
            An array of numbers. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to calculate the variance along (default is 0).

        Returns
        -------
        list of np.arrays :
            A list containing the minimum and maximum values for the specified numbers. The first element is the
            minimum, the second element is the maximum. If the input array is empty both elements are set to np.nan.

        """
        if is_empty(data):
            return [np.nan, np.nan]

        return [min(data, dimension=dimension, ignore_nodata=ignore_nodata),
                max(data, dimension=dimension, ignore_nodata=ignore_nodata)]


########################################################################################################################
# Clip Process
########################################################################################################################

@process
def clip():
    """
    Returns class instance of `Clip`.
    For more details, please have a look at the implementations inside `Clip`.

    Returns
    -------
    Clip :
        Class instance implementing all 'clip' processes.

    """
    return Clip()


class Clip:
    """
    Class implementing all 'clip' processes.

    """

    @staticmethod
    def exec_num(x, min, max):
        """
        Clips a number between specified minimum and maximum values. A value larger than the maximal value will have
        the maximal value, a value lower than minimal value will have the minimal value.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        x : int or float
            A number.
        min : int or float
            Minimum value. If `x` is lower than this value, the process will return the value of this parameter.
        max : int or float
            Maximum value. If `x` is greater than this value, the process will return the value of this parameter.

        Returns
        -------
        float or int :
            The value clipped to the specified range.

        """
        if x is None:
            pass
        elif x < min:
            x = min
        elif x > max:
            x = max

        return x

    @staticmethod
    def exec_np(x, min, max):
        """
        Clips a number between specified minimum and maximum values. A value larger than the maximal value will have
        the maximal value, a value lower than minimal value will have the minimal value.
        The no-data value None is passed through and therefore gets propagated.

        Parameters
        ----------
        np.array : int or float
            An array.
        min : int or float
            Minimum value. If `x` is lower than this value, the process will return the value of this parameter.
        max : int or float
            Maximum value. If `x` is greater than this value, the process will return the value of this parameter.

        Returns
        -------
        np.array :
            The value clipped to the specified range.

        """
        x = np.where(x < min, min, x)
        x = np.where(x > max, max, x)

        return x


# TODO: quantiles with nans are not working properly/really slow -> own implementation (e.g. like in SGRT)?
########################################################################################################################
# Quantiles Process
########################################################################################################################

@process
def quantiles():
    """
    Returns class instance of `Quantiles`.
    For more details, please have a look at the implementations inside `Quantiles`.

    Returns
    -------
    Quantiles :
        Class instance implementing all 'quantiles' processes.

    """
    return Quantiles()


class Quantiles:
    """
    Class implementing all 'quantiles' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, probabilities=None, q=None, dimension=0, ignore_nodata=True):
        """
        Calculates quantiles, which are cut points dividing the range of a probability distribution into either

            - intervals corresponding to the given probabilities or
            - (nearly) equal-sized intervals (q-quantiles based on the parameter q).

        Either the parameter `probabilities` or `q` must be specified, otherwise the `QuantilesParameterMissing`
        exception is thrown. If both parameters are set the `QuantilesParameterConflict `exception is thrown.

        Parameters
        ----------
        data : np.array
            An array of numbers.
        probabilities : list, optional
            A list of probabilities to calculate quantiles for. The probabilities must be between 0 and 1.
        q : int, optional
            A number of intervals to calculate quantiles for. Calculates q-quantiles with (nearly) equal-sized
            intervals.
        dimension : int, optional
            Defines the dimension to calculate the quantiles along (default is 0).
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.

        Returns
        -------
        list of np.arrays :
            An array with the computed quantiles. The list has either
                - as many elements as the given list of probabilities had or
                - q-1 elements.
            If the input array is empty the resulting array is filled with as many np.nan values as required according
            to the list above.

        Raises
        ------
        QuantilesParameterMissing :
            If both parameters `probabilities` and `q` are None.
        QuantilesParameterConflict :
            If both parameters `probabilities` and `q` are set.

        """
        Quantiles._check_input(probabilities, q)

        # convert quantiles and probabilities to percentiles
        if probabilities is not None:
            probabilities = list(np.array(probabilities) * 100.)
        elif q is not None:
            probabilities = list(np.arange(0, 100, 100. / q))[1:]

        if is_empty(data):
            return [np.nan] * len(probabilities)

        if not ignore_nodata:
            return np.percentile(data, probabilities, axis=dimension)
        else:
            return np.nanpercentile(data, probabilities, axis=dimension)


    @staticmethod
    def _check_input(probabilities, q):
        """
        Checks if the probabilities `probabilities` and quantiles `q` are given correctly.

        Either the parameter `probabilities` or `q` must be specified, otherwise the `QuantilesParameterMissing`
        exception is thrown. If both parameters are set the `QuantilesParameterConflict `exception is thrown.

        Parameters
        ----------
        probabilities : list, optional
            A list of probabilities to calculate quantiles for. The probabilities must be between 0 and 1.
        q : int, optional
            A number of intervals to calculate quantiles for. Calculates q-quantiles with (nearly) equal-sized
            intervals.

        Raises
        ------
        QuantilesParameterMissing :
            If both parameters `probabilities` and `q` are None.
        QuantilesParameterConflict :
            If both parameters `probabilities` and `q` are set.

        """
        if (probabilities is not None) and (q is not None):
            raise QuantilesParameterConflict()

        if probabilities is None and q is None:
            raise QuantilesParameterMissing()


########################################################################################################################
# Sum Process
########################################################################################################################

@process
def sum():
    """
    Returns class instance of `Sum`.
    For more details, please have a look at the implementations inside `Sum`.

    Returns
    -------
    Sum :
        Class instance implementing all 'sum' processes.

    """
    return Sum()


class Sum:
    """
    Class implementing all 'sum' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0, extra_values=None):
        """
        Sums up all elements in a sequential array of numbers and returns the computed sum. By default no-data values
        are ignored. Setting `ignore_nodata` to false considers no-data values so that np.nan is returned if any element
        is such a value. The computations follow IEEE Standard 754.


        Parameters
        ----------
        data : np.array
            An array of numbers. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to calculate the sum along (default is 0).
        extra_values: list, optional
            Offers to add additional elements to the computed sum.

        Returns
        -------
        np.array :
            The computed sum of the sequence of numbers.

        Notes
        -----
        `extra_values` have been introduced to handle np.array and single value interaction.
        It is more efficient to add the additional summands after computing the sum along the dimension of the array.

        """
        extra_values = extra_values if extra_values is not None else []

        if is_empty(data) and len(extra_values) == 0:
            return np.nan

        if not ignore_nodata:
            summand = np.sum(extra_values)
            return np.sum(data, axis=dimension) + summand
        else:
            summand = np.nansum(extra_values)
            return np.nansum(data, axis=dimension) + summand


########################################################################################################################
# Product Process
########################################################################################################################

@process
def product():
    """
    Returns class instance of `Product`.
    For more details, please have a look at the implementations inside `Product`.

    Returns
    -------
    Product :
        Class instance implementing all 'product' processes.

    """
    return Product()


class Product:
    """
    Class implementing all 'product' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0, extra_values=None):
        """
        Multiplies all elements in a sequential array of numbers and returns the computed product. By default no-data
        values are ignored. Setting `ignore_nodata` to False considers no-data values so that np.nan is returned if any
        element is such a value. The computations follow IEEE Standard 754.

        Parameters
        ----------
        data : np.array
            An array of numbers. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to calculate the product along (default is 0).
        extra_values: list, optional
            Offers to add additional elements to the computed product.

        Returns
        -------
        np.array :
            The computed product of the sequence of numbers.

        Notes
        -----
        `extra_values` have been introduced to handle np.array and single value interaction.
        It is more efficient to multiply the additional multiplicands after computing the product along the given
        dimension of the array.

        """
        extra_values = extra_values if extra_values is not None else []

        if is_empty(data) and len(extra_values) == 0:
            return np.nan

        if ignore_nodata:
            data[np.isnan(data)] = 1.

        if len(extra_values) > 0:
            multiplicand = np.prod(extra_values)
        else:
            multiplicand = 1.

        return np.prod(data, axis=dimension, initial=multiplicand)


########################################################################################################################
# Add Process
########################################################################################################################

@process
def add():
    """
    Returns class instance of `Add`.
    For more details, please have a look at the implementations inside `Add`.

    Returns
    -------
    Add :
        Class instance implementing all 'add' processes.

    """
    return Add()


class Add:
    """
    Class implementing all 'add' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Sums up the two numbers `x` and `y` (`x + y`) and returns the computed sum. No-data values are taken into
        account so that None is returned if any element is such a value. The computations follow IEEE Standard 754.

        Parameters
        ----------
        x : int or float
            The first summand.
        y : int or float
            The second summand.

        Returns
        -------
        float or int :
            The computed sum of the two numbers.

        """

        return x + y if x is not None and y is not None else None

    @staticmethod
    def exec_np(x, y):
        """
        Sums up the two numbers `x` and `y` (`x + y`) and returns the computed sum. No-data values are taken into
        account so that np.nan is returned if any element is such a value. The computations follow IEEE Standard 754.

        Parameters
        ----------
        x : np.array or int or float
            The first summand.
        y : np.array or int or float
            The second summand.

        Returns
        -------
        np.array or float or int :
            The computed sum.

        """
        return x + y


########################################################################################################################
# Subtract Process
########################################################################################################################

@process
def subtract():
    """
    Returns class instance of `Subtract`.
    For more details, please have a look at the implementations inside `Subtract`.

    Returns
    -------
    Subtract :
        Class instance implementing all 'subtract' processes.

    """
    return Subtract()


class Subtract:
    """
    Class implementing all 'subtract' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Subtracts argument `y` from the argument `x` (`x - y`) and returns the computed result. No-data values are
        taken into account so that None is returned if any element is such a value. The computations follow
        IEEE Standard 754.

        Parameters
        ----------
        x : int or float
            The minuend.
        y : int or float
            The subtrahend.

        Returns
        -------
        float or int :
            The computed result.

        """

        return x - y if x is not None and y is not None else None

    @staticmethod
    def exec_np(x, y):
        """
        Subtracts argument `y` from the argument `x` (`x - y`) and returns the computed result. No-data values are
        taken into account so that np.nan is returned if any element is such a value. The computations follow
        IEEE Standard 754.

        Parameters
        ----------
        x : np.array or int or float
            The minuend.
        y : np.array or int or float
            The subtrahend.

        Returns
        -------
        np.array or float or int :
            The computed result.

        """
        return x - y


########################################################################################################################
# Multiply Process
########################################################################################################################

@process
def multiply():
    """
    Returns class instance of `Multiply`.
    For more details, please have a look at the implementations inside `Multiply`.

    Returns
    -------
    Multiply :
        Class instance implementing all 'multiply' processes.

    """
    return Multiply()


class Multiply:
    """
    Class implementing all 'multiply' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Multiplies the two numbers `x` and `y` (`x * y`) and returns the computed product.
        No-data values are taken into account so that None is returned if any element is such a value.
        The computations follow IEEE Standard 754.

        Parameters
        ----------
        x : int or float
            The multiplier.
        y : int or float
            The multiplicand.

        Returns
        -------
        float or int :
            The computed product of the two numbers.

        """
        return x * y if x is not None and y is not None else None

    @staticmethod
    def exec_np(x, y):
        """
        Multiplies the two numbers `x` and `y` (`x * y`) and returns the computed product.
        No-data values are taken into account so that np.nan is returned if any element is such a value.
        The computations follow IEEE Standard 754.

        Parameters
        ----------
        x : np.array or int or float
            The multiplier.
        y : np.array or int or float
            The multiplicand.

        Returns
        -------
        np.array or float or int :
            The computed product.

        """
        return x * y


########################################################################################################################
# Divide Process
########################################################################################################################

@process
def divide():
    """
    Returns class instance of `Divide`.
    For more details, please have a look at the implementations inside `Divide`.

    Returns
    -------
    Divide :
        Class instance implementing all 'divide' processes.

    """
    return Divide()


class Divide:
    """
    Class implementing all 'divide' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Divides argument `x` by the argument `y` (`x / y`) and returns the computed result. No-data values are taken
        into account so that None is returned if any element is such a value. The computations follow IEEE Standard 754.
        Therefore, a division by zero results in ±infinity.

        Parameters
        ----------
        x : int or float
            The dividend.
        y : int or float
            The divisor.

        Returns
        -------
        float or int :
            The computed result.

        """
        return x / y if x is not None and y is not None else None

    @staticmethod
    def exec_np(x, y):
        """
        Divides argument `x` by the argument `y` (`x / y`) and returns the computed result. No-data values are taken
        into account so that None is returned if any element is such a value. The computations follow IEEE Standard 754.
        Therefore, a division by zero results in ±infinity.

        Parameters
        ----------
        x : np.array or int or float
            The dividend.
        y : np.array or int or float
            The divisor.

        Returns
        -------
        np.array or float or int :
            The computed result.

        """
        return x / y
