from openeo_driver.util.compat import filter_supported_kwargs, function_has_argument


def test_function_has_argument():
    def fun(x: int, name: str, **kwargs):
        return f"{x} {name}"

    assert function_has_argument(fun, "x") is True
    assert function_has_argument(fun, "y") is False
    assert function_has_argument(fun, "name") is True


def test_filter_supported_kwargs_basic():
    def fun(x, y: int, foo=None):
        return x + y

    assert filter_supported_kwargs(fun) == {}
    assert filter_supported_kwargs(fun, x=1, y=2) == {"x": 1, "y": 2}
    assert filter_supported_kwargs(fun, x=1, y=2, z=3, foo=4, bar=5) == {"x": 1, "y": 2, "foo": 4}


def test_filter_supported_kwargs_parameter_types():
    def fun(x, /, y, *args, z=None, **kwargs):
        return x + y + z

    assert filter_supported_kwargs(fun) == {}
    assert filter_supported_kwargs(fun, x=1, y=2, z=3) == {"y": 2, "z": 3}
    assert filter_supported_kwargs(fun, x=1, y=2, z=3, args=(4, 44), kwargs={"foo": 5}) == {"y": 2, "z": 3}
