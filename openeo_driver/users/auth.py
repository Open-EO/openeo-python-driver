"""

User authentication and management

"""
import base64
import functools
import hashlib
import logging
from typing import Callable, Tuple, List, Dict, Optional

from flask import request, Request
import requests
import requests.exceptions

from openeo.rest.auth.auth import BearerAuth
from openeo_driver.users import User
from openeo_driver.users.oidc import OidcProvider
from openeo_driver.errors import AuthenticationRequiredException, \
    AuthenticationSchemeInvalidException, TokenInvalidException, CredentialsInvalidException, OpenEOApiException
from openeo_driver.util.logging import UserIdLogging
from openeo_driver.utils import TtlCache

_log = logging.getLogger(__name__)


class OidcProviderUnavailableException(OpenEOApiException):
    status_code = 503
    code = "OidcProviderUnavailable"
    message = "OIDC Provider is unavailable"


class HttpAuthHandler:
    """Handler for processing HTTP authentication in a Flask app context"""

    # Access token prefix for 0.4-style basic auth
    # TODO: get rid of this prefix once 0.4 support is not necessary anymore
    _BASIC_ACCESS_TOKEN_PREFIX = 'basic.'

    def __init__(
            self,
            oidc_providers: List[OidcProvider],
            user_access_validation: Optional[Callable[[User, Request], User]] = None
    ):
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
            self.authenticate_basic(request)
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
            user = self.get_user_from_bearer_token(request)
            # TODO: is first 8 chars of user id enough?
            # TODO: use events/signals instead of hardcoded coupling (e.g. https://flask.palletsprojects.com/en/2.1.x/signals/)
            UserIdLogging.set_user_id(user.user_id[:8])
            if self._user_access_validation:
                user = self._user_access_validation(user, request)
            # If handler function expects a `user` argument: pass the user object
            if 'user' in f.__code__.co_varnames:
                kwargs['user'] = user
            return f(*args, **kwargs)

        return decorated

    def get_auth_token(self, request: Request, type="Bearer") -> str:
        """Get bearer/basic token from Authorization header in request"""
        if "Authorization" not in request.headers:
            raise AuthenticationRequiredException
        try:
            auth_type, auth_code = request.headers["Authorization"].split(' ')
            assert auth_type == type
        except Exception:
            raise AuthenticationSchemeInvalidException
        return auth_code

    def get_user_from_bearer_token(self, request: Request) -> User:
        """Get User object from bearer token of request."""
        bearer = self.get_auth_token(request, "Bearer")
        # Support for 0.4-style basic auth
        if bearer.startswith(self._BASIC_ACCESS_TOKEN_PREFIX):
            return self.resolve_basic_access_token(access_token=bearer)
        # 1.0-style basic and OIDC auth
        try:
            bearer_type, provider_id, access_token = bearer.split('/')
        except ValueError:
            _log.warning("Invalid bearer token {b!r}".format(b=bearer))
            raise TokenInvalidException
        if bearer_type == 'basic':
            return self.resolve_basic_access_token(access_token=access_token)
        elif bearer_type == 'oidc' and provider_id in self._oidc_providers:
            oidc_provider = self._oidc_providers[provider_id]
            return self.resolve_oidc_access_token(oidc_provider=oidc_provider, access_token=access_token)
        else:
            _log.warning("Invalid bearer token {b!r}".format(b=bearer))
            raise TokenInvalidException

    def parse_basic_auth_header(self, request: Request) -> Tuple[str, str]:
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

    def authenticate_basic(self, request: Request) -> Tuple[str, str]:
        """
        Basic authentication:
        parse a request with Basic HTTP authorization, authenticate user and return access token

        :returns: (access_token, user_id)
        """
        username, password = self.parse_basic_auth_header(request)
        password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
        _log.info("Handling basic auth for user {u!r} with sha256(pwd)={p})".format(u=username, p=password_hash))
        # TODO: do real password check
        # move password checking to backend_implementation?
        if password != username + '123':
            raise CredentialsInvalidException
        # TODO real resolving of given user name to user_id
        user_id = username
        access_token = self.build_basic_access_token(user_id)
        return access_token, user_id

    @staticmethod
    def build_basic_access_token(user_id: str) -> str:
        # TODO: generate real access token and link to user in some key value store
        prefix = HttpAuthHandler._BASIC_ACCESS_TOKEN_PREFIX
        return prefix + base64.urlsafe_b64encode(user_id.encode('utf-8')).decode('ascii')

    def resolve_basic_access_token(self, access_token: str) -> User:
        try:
            # Resolve token to user id
            head, _, access_token = access_token.partition(self._BASIC_ACCESS_TOKEN_PREFIX)
            assert head == '' and len(access_token) > 0
            user_id = base64.urlsafe_b64decode(access_token.encode('ascii')).decode('utf-8')
        except Exception:
            raise TokenInvalidException
        return User(
            user_id=user_id,
            internal_auth_data={"authentication_method": "basic"},
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
            _log.error(f"OIDC provider server error on request {url!r}", exc_info=True)
            raise OidcProviderUnavailableException
        if raise_for_status:
            resp.raise_for_status()
        return resp

    def _get_userinfo_endpoint(self, oidc_provider: OidcProvider) -> str:
        key = ("userinfo_endpoint", oidc_provider.issuer)
        if not self._cache.contains(key):
            resp = self._oidc_provider_request(oidc_provider.discovery_url)
            userinfo_url = resp.json()["userinfo_endpoint"]
            self._cache.set(key, value=userinfo_url, ttl=10 * 60)
        return self._cache.get(key)

    def resolve_oidc_access_token(self, oidc_provider: OidcProvider, access_token: str) -> User:
        try:
            userinfo_url = self._get_userinfo_endpoint(oidc_provider=oidc_provider)
            auth = BearerAuth(bearer=access_token)
            resp = self._oidc_provider_request(userinfo_url, auth=auth, raise_for_status=False)
            if resp.status_code == 200:
                # Access token was successfully accepted
                userinfo = resp.json()
                # The "sub" claim is the only claim in the response that is guaranteed per OIDC spec
                # TODO: do we have better options?
                user_id = userinfo["sub"]
                return User(
                    user_id=user_id,
                    info={"oidc_userinfo": userinfo},
                    internal_auth_data={
                        "authentication_method": "OIDC",
                        "provider_id": oidc_provider.id,  # TODO: deprecated
                        "oidc_provider_id": oidc_provider.id,
                        "oidc_provider_title": oidc_provider.title,
                        "oidc_issuer": oidc_provider.issuer,
                        "userinfo_url": userinfo_url,
                        "access_token": access_token,
                    }
                )
            elif resp.status_code in (401, 403):
                # HTTP status `401 Unauthorized`/`403 Forbidden`: token was not accepted.
                raise TokenInvalidException
            else:
                # Unexpected response status, probably a server side issue, not necessarily end user's fault.
                _log.error(f"Unexpected '/userinfo' response {resp.status_code}: {resp.text!r}.")
                raise OpenEOApiException(message=f"Unexpected '/userinfo' response: {resp.status_code}.")

        except OpenEOApiException:
            raise
        except Exception as e:
            _log.error("Unexpected error while resolving OIDC access token.", exc_info=True)
            raise OpenEOApiException(message=f"Unexpected error while resolving OIDC access token: {type(e).__name__}.")
