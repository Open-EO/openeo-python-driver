"""
Reusable helpers and fixtures for testing
"""
import json
import re
from pathlib import Path
from typing import Union, Callable, Pattern

from flask import Response
from flask.testing import FlaskClient
from werkzeug.datastructures import Headers

from openeo.capabilities import ComparableVersion
from openeo_driver.users import HttpAuthHandler

TEST_USER = "Mr.Test"
TEST_USER_BEARER_TOKEN = "basic//" + HttpAuthHandler.build_basic_access_token(user_id=TEST_USER)
TEST_USER_AUTH_HEADER = {
    "Authorization": "Bearer " + TEST_USER_BEARER_TOKEN
}

TIFF_DUMMY_DATA = b'T1f7D6t6l0l' * 1000


def read_file(path: Union[Path, str], mode='r') -> str:
    """Get contents of given file as text string."""
    with Path(path).open(mode) as f:
        return f.read()


def load_json(path: Union[Path, str], preprocess: Callable[[str], str] = None) -> dict:
    """Parse data from JSON file"""
    data = read_file(path)
    if preprocess:
        data = preprocess(data)
    return json.loads(data)


def preprocess_check_and_replace(old: str, new: str) -> Callable[[str], str]:
    """
    Create a preprocess function that replaces `old` with `new` (and fails when there is no `old`).
    """

    def preprocess(orig: str) -> str:
        assert old in orig
        return orig.replace(old, new)

    return preprocess


def preprocess_regex_check_and_replace(pattern: str, replacement: str) -> Callable[[str], str]:
    """
    Create a regex based replacement preprocess function.
    """

    def preprocess(orig: str) -> str:
        new = re.sub(pattern, replacement, orig, flags=re.DOTALL)
        assert orig != new
        return new

    return preprocess


class ApiException(Exception):
    pass


class ApiResponse:
    """
    Thin wrapper around flask `Response` to simplify
    writing unit test (API status/error code) assertions
    """

    def __init__(self, response: Response):
        self.response = response

    @property
    def status_code(self) -> int:
        return self.response.status_code

    @property
    def data(self) -> bytes:
        return self.response.data

    @property
    def json(self) -> dict:
        return self.response.json

    @property
    def text(self) -> str:
        return self.response.get_data(as_text=True)

    @property
    def headers(self) -> Headers:
        return self.response.headers

    def assert_status_code(self, status_code: int) -> 'ApiResponse':
        """Check HTTP status code"""
        if self.status_code != status_code:
            error = self.json if self.status_code >= 400 else "unknown"
            raise ApiException("Expected response with status code {s} but got {a}. Error: {e!r}".format(
                s=status_code, a=self.status_code, e=error
            ))
        return self

    def assert_content(self) -> 'ApiResponse':
        """Assert that the response body is not empty"""
        # TODO: also check content type? also check (prefix of) data?
        assert self.response.content_length > 0
        return self

    def assert_error_code(self, code: str) -> 'ApiResponse':
        """Check OpenEO error code"""
        if self.status_code < 400:
            raise ApiException("Expected response with status < 400 but got {a}".format(a=self.status_code))
        error = self.json
        actual = error.get("code")
        if actual != code:
            raise ApiException("Expected response with error code {c!r} but got {a!r}. Error: {e!r}".format(
                c=code, a=actual, e=error
            ))
        return self

    def assert_substring(self, key, expected: Union[str, Pattern]):
        actual = self.json[key]
        if isinstance(expected, str):
            if expected in actual:
                return self
        elif expected.search(actual):
            return self
        raise ApiException("Expected {e!r} at {k!r}, but got {a!r}".format(e=expected, k=key, a=actual))

    def assert_error(self, status_code: int, error_code: str,
                     message: Union[str, Pattern] = None) -> 'ApiResponse':
        resp = self.assert_status_code(status_code).assert_error_code(error_code)
        if message:
            resp.assert_substring("message", message)
        return resp


class ApiTester:
    """
    Helper container class for compact writing of api version aware API tests
    """
    data_root = None

    def __init__(self, api_version: str, client: FlaskClient, data_root: Path = None):
        self.api_version = api_version
        self.api_version_compare = ComparableVersion(self.api_version)
        self.client = client
        if data_root:
            self.data_root = Path(data_root)
        self.default_request_headers = {}

    def url(self, path):
        """Get versioned url from non-versioned path"""
        return "/openeo/{v}/{p}".format(v=self.api_version, p=path.lstrip("/"))

    def _request_headers(self, headers: dict = None) -> dict:
        return {**self.default_request_headers, **(headers or {})}

    def set_auth_bearer_token(self, token: str = TEST_USER_BEARER_TOKEN):
        """Authentication: set bearer token header for all requests."""
        self.default_request_headers["Authorization"] = "Bearer " + token

    def get(self, path: str, headers: dict = None) -> ApiResponse:
        """Do versioned GET request, given non-versioned path"""
        return ApiResponse(self.client.get(path=self.url(path), headers=self._request_headers(headers)))

    def post(self, path: str, json: dict = None, headers: dict = None) -> ApiResponse:
        """Do versioned POST request, given non-versioned path"""
        return ApiResponse(self.client.post(
            path=self.url(path),
            json=json or {},
            content_type='application/json',
            headers=self._request_headers(headers),
        ))

    def delete(self, path: str, headers: dict = None) -> ApiResponse:
        return ApiResponse(self.client.delete(path=self.url(path), headers=self._request_headers(headers)))

    def put(self, path: str, json: dict = None, headers: dict = None) -> ApiResponse:
        """Do versioned PUT request, given non-versioned path"""
        return ApiResponse(self.client.put(
            path=self.url(path),
            json=json or {},
            content_type='application/json',
            headers=self._request_headers(headers),
        ))

    def data_path(self, filename: str) -> Path:
        """Get absolute pat to a test data file"""
        return self.data_root / filename

    def read_file(self, filename, mode='r') -> str:
        """Get contents of test file, given by relative path."""
        return read_file(self.data_path(filename), mode=mode)

    def load_json(self, filename, preprocess: Callable = None) -> dict:
        """Load test process graph from json file"""
        return load_json(path=self.data_path(filename), preprocess=preprocess)

    def get_process_graph_dict(self, process_graph: dict) -> dict:
        """
        Build dict containing process graph (e.g. to POST, or to expect in metadata),
        according to API version
        """
        if ComparableVersion("1.0.0").or_higher(self.api_version):
            data = {"process": {'process_graph': process_graph}}
        else:
            data = {'process_graph': process_graph}
        return data


class IgnoreOrder:
    """
    pytest helper to test equality of lists/tuples ignoring item order

    E.g., these asserts pass:
    >>> assert [1, 2, 3, 3] == IgnoreOrder([3, 1, 2, 3])
    >>> assert {"foo": [1, 2, 3]} == {"foo": IgnoreOrder([3, 2, 1])}
    """

    def __init__(self, items: Union[list, tuple], key=None):
        self.items = items
        self.key = key

    def __eq__(self, other):
        return type(other) == type(self.items) and sorted(other, key=self.key) == sorted(self.items, key=self.key)

    def __repr__(self):
        return '{c}({v!r})'.format(c=self.__class__.__name__, v=self.items)
