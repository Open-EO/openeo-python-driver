import base64
import json
from typing import Set, Union

import requests
import requests.adapters

from openeo.util import repr_truncate


def requests_with_retry(
    total: int = 3,  # the number of retries, not attempts (which is retries + 1)
    backoff_factor: float = 1,
    status_forcelist: Set[int] = frozenset([429, 500, 502, 503, 504]),
    **kwargs,
) -> requests.Session:
    """
    Create a `requests.Session` with automatic retrying

    Inspiration and references:
    - https://requests.readthedocs.io/en/latest/api/#requests.adapters.HTTPAdapter
    - https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html#urllib3.util.Retry
    - https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/#retry-on-failure
    """
    session = requests.Session()
    retry = requests.adapters.Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        **kwargs,
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class UrlSafeStructCodec:
    """
    Utility to encode common data structures (including tuples, bytes, sets)
    in a URL-safe way, based on enriched JSON serialization and base64 encoding
    """

    # TODO: add compression too?
    # TODO: support dicts with tuple keys

    def __init__(self, signature_field: str = "_custjson"):
        """
        :param signature_field: field to use as indicator for custom object constructs
        """
        self._signature_field = signature_field

    def encode(self, data) -> str:
        # JSON serialization (with extra preprocessing for tuples)
        data = json.dumps(self._json_prepare(data), separators=(",", ":"))
        # Use base64 to make get URL-safe string
        data = base64.urlsafe_b64encode(data.encode("utf8")).decode("utf8")
        return data

    def _json_prepare(self, data):
        """Prepare data before passing to `json.dumps`"""
        # Note that we do it as preprocessing instead of using
        # the json.JSONEncoder `default` feature,
        # which does not allow to override the handling of tuples.
        if isinstance(data, tuple):
            return {self._signature_field: "tuple", "d": [self._json_prepare(x) for x in data]}
        elif isinstance(data, set):
            return {self._signature_field: "set", "d": [self._json_prepare(x) for x in data]}
        elif isinstance(data, bytes):
            return {self._signature_field: "bytes", "d": base64.b64encode(data).decode("utf8")}
        elif isinstance(data, list):
            return [self._json_prepare(x) for x in data]
        elif isinstance(data, dict):
            return {k: self._json_prepare(v) for k, v in data.items()}
        return data

    def decode(self, data: str) -> Union[dict, list, tuple, set]:
        try:
            data = base64.urlsafe_b64decode(data.encode("utf8")).decode("utf8")
            data = json.loads(data, object_hook=self._json_object_hook)
            return data
        except Exception as e:
            raise ValueError(f"Failed to decode {repr_truncate(data)}") from e


    def _json_object_hook(self, d: dict):
        """Implementation of `object_hook` of `json.JSONDecoder` API."""
        if d.get(self._signature_field) == "tuple":
            return tuple(d["d"])
        elif d.get(self._signature_field) == "set":
            return set(d["d"])
        elif d.get(self._signature_field) == "bytes":
            return base64.b64decode(d["d"].encode("utf8"))
        return d
