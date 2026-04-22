import inspect

from typing import Callable


def function_has_argument(function: Callable, argument: str) -> bool:
    """Does function support given argument?"""
    signature = inspect.signature(function)
    return argument in signature.parameters
