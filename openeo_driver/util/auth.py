from __future__ import annotations

import logging
import re
import time
from typing import Mapping, NamedTuple, Optional, Union

import requests
from openeo.rest.auth.oidc import (
    OidcClientCredentialsAuthenticator,
    OidcClientInfo,
    OidcProviderInfo,
    AccessTokenResult,
)
from openeo.util import str_truncate

_log = logging.getLogger(__name__)


class ClientCredentials(NamedTuple):
    """
    Necessary bits for doing OIDC client credentials flow:
    issuer URL, client id and secret.
    """

    oidc_issuer: str
    client_id: str
    client_secret: str

    def __repr__(self):
        return f"{self.__class__.__name__}(oidc_issuer={self.oidc_issuer!r}, client_id={self.client_id!r})"

    __str__ = __repr__

    @classmethod
    def from_mapping(cls, data: Mapping, *, strict: bool = True) -> Union[ClientCredentials, None]:
        """Build from mapping/dict/config"""
        keys = {"oidc_issuer", "client_id", "client_secret"}
        try:
            kwargs = {k: data[k] for k in keys}
        except KeyError:
            if strict:
                missing = keys.difference(data.keys())
                raise ValueError(f"Failed building {cls.__name__} from mapping: missing {missing!r}") from None
        else:
            return cls(**kwargs)

    @classmethod
    def from_credentials_string(cls, credentials: str, *, strict: bool = True) -> Union[ClientCredentials, None]:
        """
        Parse a credentials string of the form `{client_id}:{client_secret}@{oidc_issuer}` (old school basic auth URL style))
        This single string format simplifies specifying credentials through env vars
        as only one env var is necessary instead of three.
        """
        match = re.match(r"^(?P<client_id>[^:]+):(?P<client_secret>[^:@]+)@(?P<oidc_issuer>https?://.+)$", credentials)
        if match:
            return ClientCredentials(
                oidc_issuer=match.group("oidc_issuer"),
                client_id=match.group("client_id"),
                client_secret=match.group("client_secret"),
            )
        elif strict:
            # Avoid logging of secret
            creds_repr = str_truncate(credentials, width=min(8, len(credentials) // 4))
            _log.error(f"Failed parsing {cls.__name__} from credentials string {creds_repr!r} {len(credentials)=}")
            raise ValueError(f"Failed parsing {cls.__name__} from credentials string")


class _AccessTokenCache(NamedTuple):
    access_token: str
    expires_at: float


class ClientCredentialsAccessTokenHelper:
    """
    Helper to get OIDC access tokens using client credentials flow, e.g. to interact with an API (like EJR, ETL, ...)
    Caches access token too.

    Usage:
    - add an `OidcClientCredentialsHelper` instance to your class (e.g. in __init__)
    - call `setup_credentials()` with `ClientCredentials` instance (or do this directly from __init__)
    - call `get_access_token()` to get an access token where necessary
    """

    __slots__ = ("_authenticator", "_session", "_cache", "_default_ttl", "_expiration_threshold")

    def __init__(
        self,
        *,
        credentials: Optional[ClientCredentials] = None,
        session: Optional[requests.Session] = None,
        default_ttl: float = 5 * 60,
        expiration_threshold: float = 300,
    ):
        self._session = session
        self._authenticator: Optional[OidcClientCredentialsAuthenticator] = None
        self._cache = _AccessTokenCache("", 0)
        self._default_ttl = default_ttl
        self._expiration_threshold = expiration_threshold

        if credentials:
            self.setup_credentials(credentials)

    def setup_credentials(self, credentials: ClientCredentials) -> None:
        """
        Set up an `OidcClientCredentialsAuthenticator`
        (that allows to fetch access tokens)
        using the given client credentials and OIDC issuer configuration.
        """
        # TODO: eliminate need for this separate `setup` and just do it always from `__init__`?
        self._cache = _AccessTokenCache("", 0)
        # _log.debug(f"Setting up {self.__class__.__name__} with {credentials!r}")
        oidc_provider = OidcProviderInfo(
            issuer=credentials.oidc_issuer,
            requests_session=self._session,
        )
        client_info = OidcClientInfo(
            client_id=credentials.client_id,
            provider=oidc_provider,
            client_secret=credentials.client_secret,
        )
        self._authenticator = OidcClientCredentialsAuthenticator(
            client_info=client_info, requests_session=self._session
        )

    def _get_access_token(self) -> AccessTokenResult:
        """Get an access token using the configured authenticator."""
        if not self._authenticator:
            raise RuntimeError("No authentication set up")
        _log.debug(f"{self.__class__.__name__} getting access token")
        access_token_response = self._authenticator.get_tokens()
        return access_token_response

    def get_access_token(self) -> str:
        """Get an access token using the configured authenticator."""
        if time.time() > self._cache.expires_at:
            access_token_response = self._get_access_token()
            access_token = access_token_response.access_token
            self._cache = _AccessTokenCache(access_token, self._get_access_token_expiry_time(access_token_response))
        return self._cache.access_token

    def _get_access_token_expiry_time(self, access_token_response: AccessTokenResult) -> float:
        if access_token_response.expires_in is None:
            return time.time() + self._default_ttl
        else:
            # Expire the cache entry before the entry actually expires
            return time.time() + (access_token_response.expires_in - self._expiration_threshold)
