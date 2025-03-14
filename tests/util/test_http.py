import logging
import re
import base64
import pytest
import requests.exceptions
from re_assert import Matches

from openeo_driver.util.http import UrlSafeStructCodec, requests_with_retry, is_http_url


def test_requests_with_retry(caplog):
    """Simple test for retrying using an invalid domain."""
    caplog.set_level(logging.DEBUG)

    session = requests_with_retry(total=2, backoff_factor=0.1)
    with pytest.raises(
        requests.exceptions.ConnectionError, match="Max retries exceeded"
    ):
        _ = session.get("https://example.test")

    assert caplog.messages == [
        "Starting new HTTPS connection (1): example.test:443",
        Matches("Incremented Retry.*Retry\(total=1"),
        Matches("Retrying.*total=1.*Failed to (establish a new connection|resolve)"),
        "Starting new HTTPS connection (2): example.test:443",
        Matches("Incremented Retry.*Retry\(total=0"),
        Matches("Retrying.*total=0.*Failed to (establish a new connection|resolve)"),
        "Starting new HTTPS connection (3): example.test:443",
    ]


def test_requests_with_retry_zero(caplog):
    """Simple test for retrying using an invalid domain."""
    caplog.set_level(logging.DEBUG)

    session = requests_with_retry(total=0)
    with pytest.raises(
        requests.exceptions.ConnectionError, match="Max retries exceeded"
    ):
        _ = session.get("https://example.test")

    assert caplog.messages == [
        "Starting new HTTPS connection (1): example.test:443",
    ]


class TestUrlSafeStructCodec:
    def test_simple(self):
        codec = UrlSafeStructCodec(signature_field="_usc")
        assert codec.encode("foo") == "ImZvbyI="
        assert codec.encode([1, 2, "three"]) == "WzEsMiwidGhyZWUiXQ=="
        assert codec.encode((1, 2, "three")) == "eyJfdXNjIjoidHVwbGUiLCJkIjpbMSwyLCJ0aHJlZSJdfQ=="

    @pytest.mark.parametrize(
        "data",
        [
            "foo",
            "unsafe?str!n@:*&~%^&foo=bar&baz[]",
            1234,
            ["apple", "banana", "coconut"],
            {"type": "banana", "color": "blue"},
            b"bytes",
            bytes([0, 2, 128]),
            ("tuple", 123),
            {"set", "set", "go"},
            ("nest", (1, 2), [3, 4], {5, 6}),
            {"nest", (1, 2), 777},
            {"deep": ["nested", ("data", [1, 2, (3, 4), {"foo": ("bar", ["x", "y"])}])]},
        ],
    )
    def test_basic(self, data):
        codec = UrlSafeStructCodec()
        encoded = codec.encode(data)
        assert isinstance(encoded, str)
        assert re.fullmatch("[a-zA-Z0-9_/=-]+", encoded)
        assert codec.decode(encoded) == data

    def test_decode_b64_garbage(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            _ = UrlSafeStructCodec().decode(data="g@rbag%?$$!")

    def test_decode_json_garbage(self):
        garbage = base64.b64encode(b"nope, n0t J$ON here").decode("utf8")
        with pytest.raises(ValueError, match="Failed to decode"):
            _ = UrlSafeStructCodec().decode(data=garbage)


def test_is_http_url():
    assert is_http_url("http://example.com") == True
    assert is_http_url("http//example.com") == False
    assert is_http_url("https://example.com") == True
    assert is_http_url("https:/example.com") == False
    assert is_http_url("click https://example.com") == False
    assert is_http_url("httpz://example.com") == False


def test_is_http_url_allow_http():
    assert is_http_url("http://example.com") == True
    assert is_http_url("http://example.com", allow_http=True) == True
    assert is_http_url("http://example.com", allow_http=False) == False
    assert is_http_url("https://example.com", allow_http=True) == True
    assert is_http_url("https://example.com", allow_http=False) == True
