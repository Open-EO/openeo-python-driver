"""
Reusable helpers and fixtures for testing
"""
import base64
import contextlib
import http.server
import json
import logging
import math
import multiprocessing
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, Collection, Dict, Optional, Pattern, Tuple, Union
from unittest import mock

import attrs
import openeo
import openeo.processes
import pytest
import shapely.geometry.base
import shapely.wkt
from flask import Response
from flask.testing import FlaskClient
from openeo.capabilities import ComparableVersion
from werkzeug.datastructures import Headers

from openeo_driver.config.load import ConfigGetter, _backend_config_getter
from openeo_driver.users.auth import HttpAuthHandler
from openeo_driver.util.geometry import as_geojson_feature, as_geojson_feature_collection
from openeo_driver.utils import generate_unique_id

_log = logging.getLogger(__name__)

TEST_USER = "Mr.Test"
TEST_USER_BEARER_TOKEN = "basic//" + HttpAuthHandler.build_basic_access_token(user_id=TEST_USER)
TEST_USER_AUTH_HEADER = {
    "Authorization": "Bearer " + TEST_USER_BEARER_TOKEN
}

TIFF_DUMMY_DATA = b'T1f7D6t6l0l' * 1000


class DummyUser:
    __slots__ = ["user_id", "bearer_token", "auth_header"]

    def __init__(self, user_id: str = "alice2000"):
        self.user_id = user_id
        self.bearer_token = "basic//" + HttpAuthHandler.build_basic_access_token(user_id=self.user_id)
        self.auth_header = {"Authorization": f"Bearer {self.bearer_token}"}


def read_file(path: Union[Path, str], mode='r') -> str:
    """Get contents of given file as text string."""
    # TODO deprecated, just use Path(path).read_text(encoding="utf-8")
    with Path(path).open(mode) as f:
        return f.read()


def load_json(path: Union[Path, str], preprocess: Callable[[str], str] = None) -> dict:
    """Parse data from JSON file"""
    data = Path(path).read_text(encoding="utf-8")
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
            message = f"Expected response with status code {status_code} but got {self.status_code}."
            if self.status_code >= 400:
                message += f" Error: {self.json}"
            raise ApiException(message)
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

    def __init__(
            self, api_version: str, client: FlaskClient, data_root: Path = None,
            url_root: str = "/openeo/{api_version}"
    ):
        self.api_version = api_version
        self.api_version_compare = ComparableVersion(self.api_version)
        self.client = client
        if data_root:
            self.data_root = Path(data_root)
        self.default_request_headers = {}
        self.url_root = url_root.format(api_version=api_version)

    def url(self, path):
        """Build URL based on (possibly versioned) root URL."""
        return re.sub("/+", "/", f"/{self.url_root}/{path}")

    def _request_headers(self, headers: dict = None) -> dict:
        return {**self.default_request_headers, **(headers or {})}

    def set_auth_bearer_token(self, token: str = TEST_USER_BEARER_TOKEN):
        """Authentication: set bearer token header for all requests."""
        self.default_request_headers["Authorization"] = f"Bearer {token}"

    def ensure_auth_header(self):
        """Set a default authorization header if none set up yet."""
        if not self.default_request_headers.get("Authorization"):
            self.set_auth_bearer_token()

    def get(self, path: str, headers: dict = None) -> ApiResponse:
        """Do versioned GET request, given non-versioned path"""
        return ApiResponse(self.client.get(path=self.url(path), headers=self._request_headers(headers)))

    def head(self, path: str, headers: dict = None) -> ApiResponse:
        """Do versioned GET request, given non-versioned path"""
        return ApiResponse(self.client.head(path=self.url(path), headers=self._request_headers(headers)))

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

    def patch(self, path: str, json: dict = None, headers: dict = None) -> ApiResponse:
        """Do versioned PATH request, given non-versioned path"""
        return ApiResponse(self.client.patch(
            path=self.url(path),
            json=json or {},
            content_type='application/json',
            headers=self._request_headers(headers),
        ))

    def data_path(self, filename: str) -> Path:
        """Get absolute pat to a test data file"""
        return self.data_root / filename

    def read_bytes(self, filename) -> bytes:
        return self.data_path(filename).read_bytes()

    def read_text(self, filename) -> str:
        return self.data_path(filename).read_text(encoding="utf-8")

    def read_file(self, filename, mode="r") -> Union[str, bytes]:
        """Get contents of test file, given by relative path."""
        if mode in {"rb", "br"}:
            return self.read_bytes(filename)
        else:
            return self.read_text(filename)

    def load_json(self, filename, preprocess: Callable = None) -> dict:
        """Load test process graph from json file"""
        return load_json(path=self.data_path(filename), preprocess=preprocess)

    def get_process_graph_dict(self, process_graph: dict, title: str = None, description: str = None) -> dict:
        """
        Build dict containing process graph (e.g. to POST, or to expect in metadata),
        according to API version
        """
        if "process_graph" in process_graph:
            process_graph = process_graph["process_graph"]
        if ComparableVersion("1.0.0").or_higher(self.api_version):
            data = {"process": {'process_graph': process_graph}}
        else:
            data = {'process_graph': process_graph}
        if title:
            data["title"] = title
        if description:
            data["description"] = description
        return data

    def result(
        self,
        process_graph: Union[dict, str, openeo.DataCube, openeo.processes.ProcessBuilderBase],
        path="/result",
        preprocess: Callable = None,
    ) -> ApiResponse:
        """
        Post a process_graph (as dict, by filename or as DataCube),
        and get response.
        """
        if isinstance(process_graph, str):
            # Assume it is a file name
            process_graph = self.load_json(process_graph, preprocess=preprocess)
        elif hasattr(process_graph, "flat_graph"):
            # process graph API
            # TODO: make this a more explicit API (e.g. with mixin)
            process_graph = process_graph.flat_graph()
        assert isinstance(process_graph, dict)
        data = self.get_process_graph_dict(process_graph)
        self.ensure_auth_header()
        response = self.post(path=path, json=data)
        return response

    def check_result(
        self,
        process_graph: Union[dict, str, openeo.DataCube, openeo.processes.ProcessBuilderBase],
        path="/result",
        preprocess: Callable = None,
    ) -> ApiResponse:
        """
        Post a process_graph (as dict, by filename or as DataCube),
        get response and do basic checks (e.g. 200 status).
        """
        response = self.result(process_graph=process_graph, path=path, preprocess=preprocess)
        return response.assert_status_code(200).assert_content()

    def validation(
        self, process_graph: Union[dict, str], preprocess: Callable = None, ensure_auth: bool = True
    ) -> ApiResponse:
        """Post a process_graph (as dict or by filename) and run validation."""
        if isinstance(process_graph, str):
            # Assume it is a file name
            process_graph = self.load_json(process_graph, preprocess=preprocess)
        data = {'process_graph': process_graph}
        if ensure_auth:
            self.ensure_auth_header()
        response = self.post(path="/validation", json=data)
        # "Please note that a validation always returns with HTTP status code 200."
        response.assert_status_code(200)
        return response


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


class RegexMatcher:
    """
    pytest helper to check a string against a regex, especially in nested structures, e.g.:

        >>> assert {"foo": "baaaaa"} == {"foo": RegexMatcher("ba+")}
    """

    def __init__(self, pattern: str, flags=0):
        self.regex = re.compile(pattern=pattern, flags=flags)

    def __eq__(self, other):
        return isinstance(other, str) and bool(self.regex.match(other))

    def __repr__(self):
        return self.regex.pattern


class DictSubSet:
    """
    pytest helper to check if a dictionary contains a subset of items, e.g.:

    >> assert {"foo": "bar", "meh": 4} == DictSubSet({"foo": "bar"})
    """

    __slots__ = ["items", "_missing", "_differing"]

    # TODO rename/alias to `a_dict_with()` to be more self-explanatory

    def __init__(self, items: dict = None, **kwargs):
        self.items = {**(items or {}), **kwargs}
        self._missing = None
        self._differing = None

    def __eq__(self, other):
        if not isinstance(other, type(self.items)):
            return False
        self._missing = {k: v for k, v in self.items.items() if k not in other}
        self._differing = {
            k: (v, other[k])
            for k, v in self.items.items()
            if k in other and other[k] != v
        }
        return not (self._missing or self._differing)

    def __repr__(self):
        msg = repr(self.items)
        if self._missing:
            msg += f"\n    # Missing: {self._missing}"
        if self._differing:
            msg += f"\n    # Differing: {self._differing}"
        return msg


class ListSubSet:
    """
    pytest helper to check if a list contains a subset of items, e.g.:

    >> assert [1, 2, 3, 666] == ListSubSet([1, 666])
    """
    # TODO: also take item counts into account?
    def __init__(self, items: list):
        self.items = items

    def __eq__(self, other):
        return isinstance(other, type(self.items)) and all(
            any(y == x for y in other)
            for x in self.items
        )

    def __repr__(self):
        return repr(self.items)


def generate_unique_test_process_id():
    # Because the process registries are global variables we can not mock easily
    # we'll add new test processes with a (random) unique name.
    return generate_unique_id(prefix="_test_process")


def build_basic_http_auth_header(username: str, password: str) -> str:
    """Build HTTP header for Basic HTTP authentication"""
    # Note: this is not the custom basic bearer token used in openEO API.
    return "Basic " + base64.b64encode("{u}:{p}".format(u=username, p=password).encode("utf-8")).decode('ascii')


class UrllibMocker:
    """
    Poor man's urllib mocker (inspired by requests_mock).
    """

    class Response:
        """Dummy http.client.HTTPResponse"""

        def __init__(self, data: Union[bytes, str, Path] = b"", code: int = 200, msg=None):
            if isinstance(data, str):
                data = data.encode("utf8")
            elif isinstance(data, Path):
                data = data.read_bytes()
            self.data: bytes = data
            self.code = code
            self.msg = msg

        def read(self) -> bytes:
            return self.data

        def info(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def __init__(self):
        self.response_callbacks: Dict[Tuple[str, str], Callable[[urllib.request.Request], Response]] = {}

    def register(self, method: str, url: str,
                 response: Union[Response, Callable[[urllib.request.Request], Response]]):
        response_callback = response if isinstance(response, Callable) else lambda req: response
        self.response_callbacks[method, url] = response_callback
        self.response_callbacks[method, self._drop_query_string(url)] = response_callback

    def get(self, url, data: Union[bytes, str, Path], code=200):
        """Register a response for a GET request"""
        self.register(method="GET", url=url, response=self.Response(data=data, code=code))

    def _http_open(self, req: urllib.request.Request):
        """Request handler mock"""
        for match_url in [req.full_url, self._drop_query_string(req.full_url)]:
            key = (req.get_method(), match_url)

            if key in self.response_callbacks:
                return self.response_callbacks[key](req)

        return self.Response(code=404, msg="Not Found")

    @staticmethod
    def _drop_query_string(url: str):
        return url.split("?")[0]

    @contextlib.contextmanager
    def patch(self):
        with mock.patch("urllib.request.HTTPHandler.http_open", new=self._http_open), \
                mock.patch("urllib.request.HTTPSHandler.https_open", new=self._http_open):
            yield self


def approxify(x: Any, rel: Optional = None, abs: Optional[float] = None) -> Any:
    """
    Test helper to approximately check (nested) dict/list/tuple constructs containing floats/ints,
    (using `pytest.approx`).

    >>> assert {"foo": [10.001, 2.3001]} == approxify({"foo": [10, 2.3]}, abs=0.1)
    """
    if isinstance(x, dict):
        return {k: approxify(v, rel=rel, abs=abs) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(approxify(v, rel=rel, abs=abs) for v in x)
    elif isinstance(x, str):
        return x
    elif isinstance(x, (float, int)):
        return pytest.approx(x, rel=rel, abs=abs)
    elif x is None:
        return x
    else:
        # TODO: support more types
        raise ValueError(x)


class IsNan:
    def __eq__(self, other):
        return isinstance(other, float) and math.isnan(other)


class ApproxGeometry:
    """Helper to compactly and approximately compare geometries."""

    __slots__ = ["geometry", "rel", "abs"]

    def __init__(
        self,
        geometry: shapely.geometry.base.BaseGeometry,
        rel: Optional[float] = None,
        abs: Optional[float] = None,
    ):
        self.geometry = geometry
        self.rel = rel
        self.abs = abs

    @classmethod
    def from_wkt(
        cls,
        wkt: str,
        rel: Optional[float] = None,
        abs: Optional[float] = None,
    ) -> "ApproxGeometry":
        return cls(shapely.wkt.loads(wkt), rel=rel, abs=abs)

    def to_geojson(self) -> dict:
        result = shapely.geometry.mapping(self.geometry)
        return approxify(result, rel=self.rel, abs=self.abs)

    def to_geojson_feature(self, properties: Optional[dict] = None) -> dict:
        result = as_geojson_feature(self.geometry, properties=properties)
        result = approxify(result, rel=self.rel, abs=self.abs)
        return result

    def to_geojson_feature_collection(self) -> dict:
        result = as_geojson_feature_collection(self.geometry)
        return approxify(result, rel=self.rel, abs=self.abs)


class ApproxGeoJSONByBounds:
    """
    pytest assert helper to build a matcher to check if a certain GeoJSON construct has expected bounds

    Usage example:

        >>> geometry = {"type": "Polygon",  "coordinates": [...]}
        # Check that this geometry has bounds (1, 2, 6, 5) with some absolute tolerance
        >>> assert geometry == ApproxGeoJSONByBounds(1, 2, 6, 5, abs=0.1)
    """

    def __init__(
        self,
        *args,
        types: Collection[str] = ("Polygon", "MultiPolygon"),
        rel: Optional[float] = None,
        abs: Optional[float] = None,
    ):
        bounds = args[0] if len(args) == 1 else args
        bounds = [float(b) for b in bounds]
        assert len(bounds) == 4
        self.expected_bounds = bounds
        self.rel = rel
        self.abs = abs
        self.expected_types = set(types)
        self.actual_info = []

    def __eq__(self, other):
        try:
            assert isinstance(other, dict), "Not a dict"
            assert "type" in other, "No 'type' field"
            assert other["type"] in self.expected_types, f"Wrong type {other['type']!r}"
            assert "coordinates" in other, "No 'coordinates' field"

            actual_bounds = shapely.geometry.shape(other).bounds
            matching = actual_bounds == pytest.approx(self.expected_bounds, rel=self.rel, abs=self.abs)
            if not matching:
                self.actual_info.append(f"expected bounds {self.expected_bounds} != actual bounds: {actual_bounds}")
            return matching
        except Exception as e:
            self.actual_info.append(str(e))
        return False

    def __repr__(self):
        msg = f"<{type(self).__name__} types={self.expected_types} bounds={self.expected_bounds} rel={self.rel}, abs={self.abs}>"
        if self.actual_info:
            msg += "\n" + "\n".join(f"    # {i}" for i in self.actual_info)
        return msg


def caplog_with_custom_formatter(caplog: pytest.LogCaptureFixture, format: Union[str, logging.Formatter]):
    """
    Context manager to set a custom formatter on the caplog fixture.

    Naive doing `caplog.handler.setFormatter()` is not cleaned up on teardown,
    so extra care must be given. For example with this helper:

        def test_my_function(caplog)
            with caplog_with_custom_formatter(caplog=caplog, format="[%(levelname)s] %(message)s"):
                ...

    or something like:

        def test_my_function(caplog, monkeypatch)
            monkeypatch.setattr(caplog.handler, "formatter", MyCustomFormatter())
            ...

    Also see https://github.com/pytest-dev/pytest/issues/2987#issuecomment-1460509126
    """
    if isinstance(format, str):
        format = logging.Formatter(format)
    # assuming `format` is now a valid formatter:
    # an object with a method `format(self, record: logging.LogRecord)`
    return mock.patch.object(caplog.handler, "formatter", new=format)


@contextlib.contextmanager
def ephemeral_fileserver(path: Union[Path, str], host: str = "localhost", port: int = 0) -> str:
    """
    Context manager to run a short-lived (static) file HTTP server, serving files from a given local test data folder.

    This is an alternative to traditional mocking of HTTP requests (e.g. with requests_mock)
    for situations where that doesn't work (requests are done in a subprocess or at the level of a C-extension/library).

    Usage example:

        >>> # create temp file with `tmp_path` fixture
        >>> (tmp_path / "hello.txt").write_text("Hello world")
        >>> with ephemeral_fileserver(tmp_path) as fileserver_root:
        ...      res = subprocess.check_output(["curl", f"{fileserver_root}/hello.txt"])
        >>> assert res.strip() == "Hello world"

    :param path: root path of the local files to serve
    :return: root URL of the ephemeral file server (e.g. "http://localhost:21342")
    """

    def run(queue: multiprocessing.Queue):
        server = http.server.HTTPServer(
            server_address=(host, port),
            RequestHandlerClass=lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
                *args, directory=path, **kwargs
            ),
        )
        url = f"http://{server.server_address[0]}:{server.server_port}"
        _log.info(f"ephemeral_fileserver: started server at {url=}")
        queue.put(url)
        server.serve_forever()

    queue = multiprocessing.Queue()
    server_process = multiprocessing.Process(target=run, args=(queue,))
    _log.debug("ephemeral_fileserver: starting server process")
    server_process.start()
    _log.info(f"ephemeral_fileserver: started pid={server_process.pid}")
    url = queue.get(timeout=2)
    _log.info(f"ephemeral_fileserver: detected {url=}")
    try:
        yield url
    finally:
        _log.debug(f"ephemeral_fileserver: terminating")
        server_process.terminate()
        server_process.join(timeout=2)
        _log.info(f"ephemeral_fileserver: terminated with exitcode={server_process.exitcode}")
        server_process.close()


def config_overrides(config_getter: ConfigGetter = _backend_config_getter, **kwargs):
    """
    *Only to be used in unit tests*

    `mock.patch` based mocker to override the config returned by `get_backend_config()` at run time

    Can be used as context manager

        >>> with config_overrides(id="foobar"):
        ...     ...

    in a fixture (as context manager):

        >>> @pytest.fixture
        ... def custom_setup()
        ...     with config_overrides(id="foobar"):
        ...         yield

    or as test function decorator

        >>> @config_overrides(id="foobar")
        ... def test_stuff():
    """
    orig_config = config_getter.get()
    config_kwargs = {
        **attrs.asdict(orig_config, recurse=False),
        **kwargs,
    }
    overriden_config = config_getter.expected_class(**config_kwargs)
    return mock.patch.object(config_getter, "_config", new=overriden_config)
