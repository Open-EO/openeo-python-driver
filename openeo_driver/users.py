"""

User authentication and management

"""
import base64
import functools
import hashlib
import logging
from typing import Callable, Tuple

import requests
from flask import request, current_app, Request

from openeo.rest.auth.auth import BearerAuth
from openeo_driver.errors import AuthenticationRequiredException, \
    AuthenticationSchemeInvalidException, TokenInvalidException, CredentialsInvalidException

_log = logging.getLogger(__name__)


class User:
    # TODO more fields
    def __init__(self, user_id: str, info: dict = None):
        self.user_id = user_id
        self.info = info


class HttpAuthHandler:
    """Handler for processing HTTP authentication in a Flask app context"""

    _BASIC_ACCESS_TOKEN_PREFIX = 'basic.'

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
        token = self.get_auth_token(request, "Bearer")
        if token.startswith(self._BASIC_ACCESS_TOKEN_PREFIX):
            return self.resolve_basic_access_token(token)
        elif len(token) > 16:
            # Assume token is OpenID Connect access token
            return self.resolve_oidc_access_token(token)
        else:
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
        if password != username + '123':
            raise CredentialsInvalidException
        # TODO real resolving of given user name to user_id
        user_id = username
        access_token = self.build_basic_access_token(user_id)
        return access_token, user_id

    def build_basic_access_token(self, user_id:str) -> str:
        # TODO: generate real access token and link to user in some key value store
        return self._BASIC_ACCESS_TOKEN_PREFIX + base64.urlsafe_b64encode(user_id.encode('utf-8')).decode('ascii')

    def resolve_basic_access_token(self, access_token: str) -> User:
        try:
            head, sep, tail = access_token.partition(self._BASIC_ACCESS_TOKEN_PREFIX)
            assert head == '' and sep == self._BASIC_ACCESS_TOKEN_PREFIX and len(tail) > 0
            # Resolve token to user id
            user_id = base64.urlsafe_b64decode(tail.encode('ascii')).decode('utf-8')
        except Exception:
            raise TokenInvalidException
        return User(user_id=user_id, info={"authentication": "basic"})

    def resolve_oidc_access_token(self, access_token: str) -> User:
        try:
            resp = requests.get(current_app.config["OPENID_CONNECT_CONFIG_URL"])
            resp.raise_for_status()
            userinfo_url = resp.json()["userinfo_endpoint"]
            resp = requests.get(userinfo_url, auth=BearerAuth(bearer=access_token))
            resp.raise_for_status()
            userinfo = resp.json()
            # The "sub" claim is the only claim in the response that is guaranteed per OIDC spec
            # TODO: do we have better options?
            user_id = userinfo["sub"]
            return User(user_id=user_id, info=userinfo)
        except Exception:
            raise TokenInvalidException
