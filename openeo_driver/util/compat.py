import inspect
from typing import Callable


def function_has_argument(function: Callable, argument: str) -> bool:
    """Does function support given argument?"""
    signature = inspect.signature(function)
    return argument in signature.parameters


def filter_supported_kwargs(callable: Callable, **kwargs) -> dict:
    """
    Check a callable's signature and only keep the kwargs that are supported by the callable.

    Helps with calling API functions (e.g. in MicroService subclasses/implementations)
    in a backward/forward compatible way when arguments are being deprecated/added.

    Usage example:

        result = some_function(
            x=5,
            # Conditional usage of argument `y`:
            # don't provide it when `some_function`
            # doesn't support it for some reason.
            **filter_supported_kwargs(
                callable=some_function,
                y=4
            )
        )

    Note that this helper makes function calls less readable (compared to standard arg/kwarg usage),
    so usage should be minimized just to allow migration of all components to a new API version.
    """
    params = inspect.signature(callable).parameters
    return {
        k: v
        for k, v in kwargs.items()
        if k in params and params[k].kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]
    }
