from openeo_driver.util.compat import function_has_argument


def test_function_has_argument():
    def fun(x: int, name: str, **kwargs):
        return f"{x} {name}"

    assert function_has_argument(fun, "x") is True
    assert function_has_argument(fun, "y") is False
    assert function_has_argument(fun, "name") is True
