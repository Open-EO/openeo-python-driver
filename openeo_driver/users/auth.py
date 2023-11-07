"""

User authentication and management

"""
import base64
import functools
import hashlib
import logging
from typing import Callable, Tuple, List, Dict, Optional

import flask
import requests
import requests.exceptions

from openeo.rest.auth.auth import BearerAuth
from openeo_driver.config import OpenEoBackendConfig, get_backend_config
from openeo_driver.users import User
from openeo_driver.users.oidc import OidcProvider
from openeo_driver.errors import AuthenticationRequiredException, \
    AuthenticationSchemeInvalidException, TokenInvalidException, CredentialsInvalidException, OpenEOApiException
from openeo_driver.util.logging import FlaskUserIdLogging, user_id_trim
from openeo_driver.util.caching import TtlCache

_log = logging.getLogger(__name__)


class OidcProviderUnavailableException(OpenEOApiException):
    status_code = 503
    code = "OidcProviderUnavailable"
    message = "OIDC Provider is unavailable"


class HttpAuthHandler:
    """Handler for processing HTTP authentication in a Flask app context"""

    def __init__(
        self,
        oidc_providers: List[OidcProvider],
        user_access_validation: Optional[Callable[[User, flask.Request], User]] = None,
        config: Optional[OpenEoBackendConfig] = None,
    ):
        self._config: OpenEoBackendConfig = config or get_backend_config()
        # TODO: handle `oidc_providers` and `user_access_validation` through OpenEoBackendConfig
        self._oidc_providers: Dict[str, OidcProvider] = {p.id: p for p in oidc_providers}
        self._user_access_validation = user_access_validation
        self._cache = TtlCache(default_ttl=10 * 60)

    def public(self, f: Callable):
        """
        Decorator for public request handler: no authorization is required
        """
        return f

    def requires_http_basic_auth(self, f: Callable):
        """
        Decorator for a flask request handler that requires HTTP basic auth (header)
        """

        @functools.wraps(f)
        def decorated(*args, **kwargs):
            # Try to authenticate user from HTTP basic auth headers (failure will raise appropriate exception).
            self.authenticate_basic(flask.request)
            # TODO: optionally pass access_token and user_id from authentication result?
            return f(*args, **kwargs)

        return decorated

    def requires_bearer_auth(self, f: Callable):
        """
        Decorator for flask request handler that requires a valid bearer auth (header).

        When function has a `user` argument, the User object will be passed
        """

        @functools.wraps(f)
        def decorated(*args, **kwargs):
            # Try to load user info from request (failure will raise appropriate exception).
            user = self.get_user_from_bearer_token(flask.request)
            # TODO: is first 8 chars of user id enough?
            # TODO: use events/signals instead of hardcoded coupling (e.g. https://flask.palletsprojects.com/en/2.1.x/signals/)
            FlaskUserIdLogging.set_user_id(user.user_id)
            if self._user_access_validation:
                user = self._user_access_validation(user, flask.request)
            # If handler function expects a `user` argument: pass the user object
            if 'user' in f.__code__.co_varnames:
                kwargs['user'] = user
            return f(*args, **kwargs)

        return decorated

    def get_auth_token(self, request: flask.Request, type="Bearer") -> str:
        """Get bearer/basic token from Authorization header in request"""
        if "Authorization" not in request.headers:
            raise AuthenticationRequiredException
        try:
            auth_type, auth_code = request.headers["Authorization"].split(' ')
            assert auth_type == type
        except Exception:
            raise AuthenticationSchemeInvalidException
        return auth_code

    def get_user_from_bearer_token(self, request: flask.Request) -> User:
        """Get User object from bearer token of request."""
        bearer = self.get_auth_token(request, "Bearer")

        cache_key = ("bearer", bearer)
        if not self._cache.contains(cache_key):
            user = self._get_user_from_bearer_token(bearer=bearer)
            self._cache.set(cache_key, value=user, ttl=15 * 60)
        return self._cache.get(cache_key)

    def _get_user_from_bearer_token(self, bearer: str) -> User:
        """Get User object from bearer token of request."""
        try:
            bearer_type, provider_id, access_token = bearer.split('/')
        except ValueError:
            _log.warning("Invalid bearer token {b!r}".format(b=bearer))
            raise TokenInvalidException
        if bearer_type == 'basic':
            if not self._config.enable_basic_auth:
                raise AuthenticationSchemeInvalidException(message="Basic authentication is not supported.")
            return self.resolve_basic_access_token(access_token=access_token)
        elif bearer_type == 'oidc' and provider_id in self._oidc_providers:
            if not self._config.enable_oidc_auth:
                raise AuthenticationSchemeInvalidException(message="OIDC authentication is not supported.")
            oidc_provider = self._oidc_providers[provider_id]
            return self.resolve_oidc_access_token(oidc_provider=oidc_provider, access_token=access_token)
        else:
            _log.warning("Invalid bearer token {b!r}".format(b=bearer))
            raise TokenInvalidException

    def parse_basic_auth_header(self, request: flask.Request) -> Tuple[str, str]:
        """
        Parse user and password from request with Basic HTTP authorization header.

        :returns: (username, password)
        """
        token = self.get_auth_token(request, "Basic")
        try:
            username, password = base64.b64decode(token.encode('ascii')).decode('utf-8').split(':')
        except Exception:
            raise TokenInvalidException
        return username, password

    def authenticate_basic(self, request: flask.Request) -> Tuple[str, str]:
        """
        Basic authentication:
        parse a request with Basic HTTP authorization, authenticate user and return access token

        :returns: (access_token, user_id)
        """
        username, password = self.parse_basic_auth_header(request)
        _log.info(f"Handling basic auth for user {username!r}")
        if not (self._config.enable_basic_auth and self._config.valid_basic_auth):
            raise AuthenticationSchemeInvalidException(message="Basic authentication is not supported.")
        if not self._config.valid_basic_auth(username, password):
            raise CredentialsInvalidException
        # TODO real resolving of given user name to user_id?
        user_id = username
        access_token = self.build_basic_access_token(user_id)
        return access_token, user_id

    @staticmethod
    def build_basic_access_token(user_id: str) -> str:
        # TODO: generate real verifiable access token and link to user in some key value store
        return base64.urlsafe_b64encode(user_id.encode("utf-8")).decode("ascii")

    def resolve_basic_access_token(self, access_token: str) -> User:
        try:
            # Resolve token to user id
            # TODO: verify that access token is valid, has not expired, etc.
            user_id = base64.urlsafe_b64decode(access_token.encode('ascii')).decode('utf-8')
        except Exception:
            raise TokenInvalidException
        return User(
            user_id=user_id,
            internal_auth_data={
                "authentication_method": "basic",
            },
        )

    @staticmethod
    def _oidc_provider_request(
            url: str, method="get", timeout=10, raise_for_status=True, **kwargs
    ) -> requests.Response:
        """Helper to do  OIDC provider request with some extra safe-guarding and error handling"""
        try:
            resp = requests.request(method=method, url=url, timeout=timeout, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            _log.error(f"OIDC provider unavailable for request {url!r}", exc_info=True)
            raise OidcProviderUnavailableException

        if 500 <= resp.status_code < 600:
            _log.error(f"OIDC provider server error on request {url!r}: {resp!r}")
            raise OidcProviderUnavailableException
        if raise_for_status:
            resp.raise_for_status()
        return resp

    def _get_oidc_provider_config(self, oidc_provider: OidcProvider) -> dict:
        """Cached retrieval of OIDC config of a provider (/.well-known/openid-configuration)."""
        return self._cache.get_or_call(
            key=("oidc-conf", oidc_provider.issuer),
            callback=lambda: self._oidc_provider_request(
                oidc_provider.discovery_url
            ).json(),
            ttl=15 * 60,
        )

    def resolve_oidc_access_token(self, oidc_provider: OidcProvider, access_token: str) -> User:
        try:
            internal_auth_data = {
                "authentication_method": "OIDC",
                "provider_id": oidc_provider.id,  # TODO: deprecated
                "oidc_provider_id": oidc_provider.id,
                "oidc_provider_title": oidc_provider.title,  # TODO necessary to have title here?
                "oidc_issuer": oidc_provider.issuer,
                # used for e.g. access to SHub APIs on CDSE
                "access_token": access_token,
            }

            userinfo = self._get_userinfo(oidc_provider=oidc_provider, access_token=access_token)

            # The "sub" claim is the only claim in the response that is guaranteed per OIDC spec
            token_sub = userinfo["sub"]
            internal_auth_data["oidc_token_sub"] = token_sub

            # Do token inspection for additional info
            is_client_credentials_token = None
            if self._config.oidc_token_introspection:
                token_data = self._token_introspection(oidc_provider=oidc_provider, access_token=access_token)
                internal_auth_data["oidc_access_token_introspection"] = token_data
                # TODO: better/more robust guessing if this is a client creds based access token
                if any(
                    token_data[n].startswith("service-account-")
                    for n in ["preferred_username", "username"]
                    if n in token_data
                ):
                    _log.info(f"Assuming client credentials based access token for 'sub' {token_sub!r}")
                    internal_auth_data["oidc_client_credentials_access_token"] = True
                    is_client_credentials_token = True

            # Map token "sub" to user id
            oidc_user_map_data = self._config.oidc_user_map.get((oidc_provider.id, token_sub))
            if oidc_user_map_data:
                _log.debug(f"oidc_user_map user mapping {token_sub=} -> {oidc_user_map_data=}")
                internal_auth_data["oidc_user_map_data"] = oidc_user_map_data
                user_id = oidc_user_map_data["user_id"]
            elif is_client_credentials_token:
                raise AccessTokenException(
                    message=f"Client credentials access token without user mapping: sub={token_sub!r}."
                )
            else:
                # Default: normal user access token
                # TODO: do something more than directly using "sub"?
                user_id = token_sub

            return User(user_id=user_id, info={"oidc_userinfo": userinfo}, internal_auth_data=internal_auth_data)

        except OpenEOApiException:
            raise
        except Exception as e:
            _log.error("Unexpected error while resolving OIDC access token.", exc_info=True)
            raise OpenEOApiException(
                message=f"Unexpected error while resolving OIDC access token: {type(e).__name__}."
            ) from e

    def _get_userinfo(self, oidc_provider: OidcProvider, access_token: str) -> dict:
        oidc_config = self._get_oidc_provider_config(oidc_provider=oidc_provider)
        userinfo_url = oidc_config["userinfo_endpoint"]
        resp = self._oidc_provider_request(
            userinfo_url,
            auth=BearerAuth(bearer=access_token),
            raise_for_status=False,
        )
        if resp.status_code == 200:
            userinfo = resp.json()
            return userinfo
        elif resp.status_code in (401, 403):
            # HTTP status `401 Unauthorized`/`403 Forbidden`: token was not accepted.
            raise TokenInvalidException
        else:
            # Unexpected response status, probably a server side issue, not necessarily end user's fault.
            _log.error(f"Unexpected '/userinfo' response {resp.status_code}: {resp.text!r}.")
            raise AccessTokenException(message=f"Unexpected '/userinfo' response: {resp.status_code}.")

    def _token_introspection(self, oidc_provider: OidcProvider, access_token: str) -> dict:
        if not oidc_provider.service_account:
            # TODO: possible to relax the requirement to have a service account for token introspection?
            raise AccessTokenException(message=f"No service account for {oidc_provider.id}") from None

        oidc_config = self._get_oidc_provider_config(oidc_provider=oidc_provider)
        try:
            introspection_endpoint = oidc_config["introspection_endpoint"]
        except KeyError:
            raise AccessTokenException(message=f"No introspection endpoint for {oidc_provider.id}") from None

        # Do introspection request
        resp = self._oidc_provider_request(
            url=introspection_endpoint,
            method="POST",
            data={"token": access_token},
            auth=oidc_provider.service_account,
            raise_for_status=True,
        )
        token_data = resp.json()

        # Basic checks
        if not token_data.get("active"):
            raise AccessTokenException(message=f"Access token not active.")
        scope = set(token_data.get("scope", "").split())
        if not scope.issuperset(oidc_provider.scopes):
            raise AccessTokenException(
                message=f"Token scope {scope} not covering expected scope {oidc_provider.scopes}"
            )

        return token_data


class AccessTokenException(OpenEOApiException):
    status_code = 500
    code = "AccessTokenError"
    message = "Unspecified access token handling error"
