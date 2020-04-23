import warnings
from inspect import isclass
from itertools import groupby
from typing import Dict, Type, List

import pytest

import openeo_driver.errors
from openeo_driver.errors import OpenEOApiException, OpenEOApiErrorSpecHelper


def get_defined_exceptions(mod=openeo_driver.errors) -> Dict[str, List[Type[OpenEOApiException]]]:
    # Get defined errors
    defined_exceptions = [
        value
        for name, value in mod.__dict__.items()
        if isclass(value) and issubclass(value, OpenEOApiException) and value is not OpenEOApiException
    ]
    # Group by OpenEo error code
    return {
        code: list(cases)
        for code, cases in groupby(sorted(defined_exceptions, key=lambda e: e.code), key=lambda e: e.code)
    }


def test_extract_placeholders():
    message = "the quick {color} fox {verb} over the lazy {animal}"
    assert OpenEOApiErrorSpecHelper.extract_placeholders(message) == {"color", "verb", "animal"}


def test_unknown_error_codes():
    from_spec = set(OpenEOApiErrorSpecHelper().get_error_codes())
    defined = set(get_defined_exceptions().keys())
    defined_but_not_in_spec = defined.difference(from_spec)
    assert defined_but_not_in_spec == set()


def test_generate_exception_class():
    spec = {"OutOfTea": {
        "description": "No more tea.",
        "message": "The {color} tea pot is empty.",
        "http": 418,
        "tags": ["foo"]
    }}
    src_lines = OpenEOApiErrorSpecHelper(spec).generate_exception_class("OutOfTea").split("\n")
    assert "class OutOfTeaException(OpenEOApiException):" in src_lines
    assert "    status_code = 418" in src_lines
    assert "    code = 'OutOfTea'" in src_lines
    assert "    message = 'The {color} tea pot is empty.'" in src_lines
    assert "    def __init__(self, color: str):" in src_lines
    assert "        super().__init__(self.message.format(color=color))"


@pytest.mark.parametrize("error_code", OpenEOApiErrorSpecHelper().get_error_codes())
def test_error_code(error_code):
    """
    Check implementation of given OpenEO error code.
    If no implemented exception: fail and print suggested implementation
    If there is an implementation (or multiple): check correctness (status code, ...)
    """
    spec_helper = OpenEOApiErrorSpecHelper()
    error_spec = spec_helper.get(error_code)
    exceptions = get_defined_exceptions().get(error_code, [])
    if len(exceptions) == 0:
        print('Suggested implementation:')
        print(spec_helper.generate_exception_class(error_code))
        raise Exception("No implemented exception class for error code {e!r}".format(e=error_code))
    for exception_cls in exceptions:
        # Check that OpenEO error code is in exception class name. # TODO: is this really necessary?
        assert error_code.lower() in exception_cls.__name__.lower()
        assert exception_cls.status_code == error_spec["http"]
        assert set(exception_cls._tags) == set(error_spec["tags"])
        # Check placeholder usage in message
        placeholders_spec = spec_helper.extract_placeholders(error_spec["message"])
        placeholders_actual = spec_helper.extract_placeholders(exception_cls.message)
        assert placeholders_spec == placeholders_actual
        if exception_cls.message != error_spec["message"]:
            warnings.warn("Exception class {e}: message field {m!r} is different from spec {s!r}".format(
                e=exception_cls.__name__, m=exception_cls.message, s=error_spec["message"]
            ))
        using_default_init = exception_cls.__init__ is OpenEOApiException.__init__
        if placeholders_actual and using_default_init:
            raise Exception("Exception class {c} has placeholder in message but no custom __init__".format(
                c=exception_cls.__name__))
