import flask
import json
import pytest
import requests.exceptions
from flask import Flask, jsonify, Response, request

from openeo_driver.backend import OidcProvider
from openeo_driver.errors import OpenEOApiException, PermissionsInsufficientException, TokenInvalidException
from openeo_driver.testing import build_basic_http_auth_header, DictSubSet
from openeo_driver.users import User
from openeo_driver.users.auth import HttpAuthHandler, OidcProviderUnavailableException, AccessTokenException


@pytest.fixture()
def oidc_provider(requests_mock) -> OidcProvider:
    oidc_issuer = "https://oeo.example.com"
    oidc_discovery_url = oidc_issuer + "/.well-known/openid-configuration"
    oidc_userinfo_url = oidc_issuer + "/userinfo"
    requests_mock.get(oidc_discovery_url, json={"userinfo_endpoint": oidc_userinfo_url})
    return OidcProvider(id="testoidc", issuer=oidc_issuer, scopes=['openid'], title='Test OIDC provider')


@pytest.fixture()
def app(oidc_provider, backend_config):
    """Fixture for a flask app with some public and some auth requiring handlers"""
    app = Flask("__test__")
    auth = HttpAuthHandler(oidc_providers=[oidc_provider], config=backend_config)
    app.config["auth_handler"] = auth

    @app.route("/public/hello")
    @auth.public
    def public_hello():
        return "hello everybody"

    @app.route("/basic/hello")
    @auth.requires_http_basic_auth
    def basic_hello():
        return "hello basic"

    @app.route("/basic/auth")
    @auth.requires_http_basic_auth
    def basic_auth():
        access_token, user_id = auth.authenticate_basic(request)
        return jsonify({"access_token": access_token, "user_id": user_id})

    @app.route("/private/hello")
    @auth.requires_bearer_auth
    def private_hello():
        return "hello you"

    @app.route("/personal/hello")
    @auth.requires_bearer_auth
    def personal_hello(user: User):
        return "hello {u}".format(u=user.user_id)

    @app.route("/inspect")
    @auth.requires_bearer_auth
    def inspect(user: User):
        return jsonify({"info": user.info, "internal_auth_data": user.internal_auth_data})

    @app.errorhandler(OpenEOApiException)
    def handle_openeoapi_exception(error: OpenEOApiException):
        return jsonify(error.to_dict()), error.status_code

    return app


def test_public(app):
    with app.test_client() as client:
        assert client.get("/public/hello").data == b"hello everybody"


def assert_authentication_required_failure(response: Response):
    assert response.status_code == 401
    error = response.json
    assert error["code"] == 'AuthenticationRequired'
    assert "unauthorized" in error["message"].lower()


def assert_invalid_authentication_method_failure(response: Response):
    assert response.status_code == 403
    error = response.json
    assert error["code"] == 'AuthenticationSchemeInvalid'
    assert "authentication method not supported" in error["message"].lower()


def assert_invalid_credentials_failure(response: Response):
    assert response.status_code == 403
    error = response.json
    assert error["code"] == 'CredentialsInvalid'
    assert "credentials are not correct" in error["message"].lower()


def assert_invalid_token_failure(response: Response):
    assert response.status_code == 403
    error = response.json
    assert error["code"] == 'TokenInvalid'
    # TODO: check message too?


def test_basic_auth_no_auth(app):
    with app.test_client() as client:
        response = client.get("/basic/hello")
        assert_authentication_required_failure(response)


def test_basic_auth_invalid_auth_type(app):
    with app.test_client() as client:
        headers = {"Authorization": "Foo Bar"}
        response = client.get("/basic/hello", headers=headers)
        assert_invalid_authentication_method_failure(response)


def test_basic_auth_invalid_password(app):
    with app.test_client() as client:
        headers = {"Authorization": build_basic_http_auth_header("testuser", "wrongpassword")}
        response = client.get("/basic/hello", headers=headers)
        assert_invalid_credentials_failure(response)


@pytest.mark.parametrize(
    ["backend_config_overrides", "expected"],
    [
        (
            {"enable_basic_auth": False},
            (
                403,
                DictSubSet(
                    {"code": "AuthenticationSchemeInvalid", "message": "Basic authentication is not supported."}
                ),
            ),
        ),
        (
            {"enable_basic_auth": True},
            (200, b"hello basic"),
        ),
    ],
)
def test_basic_auth_disabled(app, backend_config_overrides, expected):
    with app.test_client() as client:
        headers = {"Authorization": build_basic_http_auth_header("Alice", "alice123")}
        resp = client.get("/basic/hello", headers=headers)
        assert (resp.status_code, resp.json or resp.data) == expected


@pytest.mark.parametrize(
    ["username", "password"],
    [
        ("Alice", "alice123"),
        ("Bob", "BOB!!!"),
    ],
)
def test_basic_auth_success(app, username, password):
    with app.test_client() as client:
        headers = {"Authorization": build_basic_http_auth_header(username, password)}
        response = client.get("/basic/hello", headers=headers)
        assert response.status_code == 200
        assert response.data == b"hello basic"


@pytest.mark.parametrize("url", ["/private/hello", "/personal/hello"])
def test_bearer_auth_no_auth(app, url):
    with app.test_client() as client:
        response = client.get(url)
        assert_authentication_required_failure(response)


@pytest.mark.parametrize("url", ["/private/hello", "/personal/hello"])
def test_bearer_auth_invalid_auth_type(app, url):
    with app.test_client() as client:
        headers = {"Authorization": "Foo Bar"}
        response = client.get(url, headers=headers)
        assert_invalid_authentication_method_failure(response)


@pytest.mark.parametrize("url", ["/private/hello", "/personal/hello"])
def test_bearer_auth_empty(app, url):
    with app.test_client() as client:
        headers = {"Authorization": "Bearer "}
        response = client.get(url, headers=headers)
        assert_invalid_token_failure(response)


@pytest.mark.parametrize("url", ["/private/hello", "/personal/hello"])
def test_bearer_auth_basic_invalid_token(app, url):
    with app.test_client() as client:
        headers = {"Authorization": "Bearer basic//blehrff"}
        response = client.get(url, headers=headers)
        assert_invalid_token_failure(response)


@pytest.mark.parametrize("url", ["/private/hello", "/personal/hello"])
def test_bearer_auth_basic_invalid_token_prefix(app, url):
    with app.test_client() as client:
        headers = {"Authorization": "Bearer basic//blehrff"}
        response = client.get(url, headers=headers)
        assert_invalid_token_failure(response)


@pytest.mark.parametrize(
    ["backend_config_overrides", "expected"],
    [
        (
            {"enable_basic_auth": False},
            (
                403,
                DictSubSet(
                    {"code": "AuthenticationSchemeInvalid", "message": "Basic authentication is not supported."}
                ),
            ),
        ),
        (
            {"enable_basic_auth": True},
            (200, b"hello Alice"),
        ),
    ],
)
def test_bearer_auth_basic_disabled(app, backend_config_overrides, expected):
    with app.test_client() as client:
        access_token = HttpAuthHandler.build_basic_access_token(user_id="Alice")
        headers = {"Authorization": f"Bearer basic//{access_token}"}
        resp = client.get("/personal/hello", headers=headers)
        assert (resp.status_code, resp.json or resp.data) == expected

@pytest.mark.parametrize(
    ["url", "expected_data"],
    [
        ("/private/hello", b"hello you"),
        ("/personal/hello", b"hello Alice"),
    ],
)
def test_bearer_auth_basic_token_success(app, url, expected_data):
    with app.test_client() as client:
        headers = {"Authorization": build_basic_http_auth_header("Alice", "alice123")}
        resp = client.get("/basic/auth", headers=headers)
        assert resp.status_code == 200
        access_token = resp.json["access_token"]

        headers = {"Authorization": "Bearer basic//" + access_token}
        resp = client.get(url, headers=headers)
        assert resp.status_code == 200
        assert resp.data == expected_data


@pytest.mark.parametrize("url", ["/private/hello", "/personal/hello"])
def test_bearer_auth_oidc_invalid_token(app, url, requests_mock, oidc_provider):
    requests_mock.get(oidc_provider.issuer + "/userinfo", json={"error": "meh"}, status_code=401)

    with app.test_client() as client:
        oidc_access_token = "kcneududhey8rmxje3uhoe9djdndjeu3rkrnmlxpds834r"
        headers = {"Authorization": "Bearer oidc/{p}/{a}".format(p=oidc_provider.id, a=oidc_access_token)}
        resp = client.get(url, headers=headers)
        assert_invalid_token_failure(resp)


@pytest.mark.parametrize(
    ["resp_status", "body", "api_error"],
    [
        (401, {"error": "meh"}, TokenInvalidException),
        (403, {"error": "meh"}, TokenInvalidException),
        (
            200,
            "inval:d j$on",
            OpenEOApiException(message="Unexpected error while resolving OIDC access token: TypeError."),
        ),
        (204, {"error": "meh"}, AccessTokenException(message="Unexpected '/userinfo' response: 204.")),
        (400, {"error": "meh"}, AccessTokenException(message="Unexpected '/userinfo' response: 400.")),
        (404, {"error": "meh"}, AccessTokenException(message="Unexpected '/userinfo' response: 404.")),
        (500, {"error": "meh"}, OidcProviderUnavailableException),
        (503, {"error": "meh"}, OidcProviderUnavailableException),
    ],
)
def test_bearer_auth_oidc_token_resolve_problems(app, requests_mock, oidc_provider, resp_status, body, api_error):
    requests_mock.get(oidc_provider.issuer + "/userinfo", json=body, status_code=resp_status)

    with app.test_client() as client:
        oidc_access_token = "kcneududhey8rmxje3uhoe9djdndjeu3rkrnmlxpds834r"
        headers = {"Authorization": "Bearer oidc/{p}/{a}".format(p=oidc_provider.id, a=oidc_access_token)}
        resp = client.get("/private/hello", headers=headers)
        assert resp.status_code == api_error.status_code
        assert resp.json["code"] == api_error.code
        assert resp.json["message"] == api_error.message


@pytest.mark.parametrize(
    ["backend_config_overrides", "expected"],
    [
        (
            {"enable_oidc_auth": False},
            (
                403,
                DictSubSet({"code": "AuthenticationSchemeInvalid", "message": "OIDC authentication is not supported."}),
            ),
        ),
        (
            {"enable_oidc_auth": True},
            (200, b"hello oidcuser"),
        ),
    ],
)
def test_bearer_auth_oidc_disabled(app, requests_mock, oidc_provider, expected):
    def userinfo(request, context):
        """Fake OIDC /userinfo endpoint handler"""
        _, _, token = request.headers["Authorization"].partition("Bearer ")
        user_id = token.split(".")[1]
        return json.dumps({"sub": user_id})

    requests_mock.get(oidc_provider.issuer + "/userinfo", text=userinfo)

    with app.test_client() as client:
        # Note: user id is "hidden" in access token
        oidc_access_token = "kcneududhey8rmxje3uhs.oidcuser.o94h4oe9djdndjeu3rkrnmlxpds834r"
        headers = {"Authorization": "Bearer oidc/{p}/{a}".format(p=oidc_provider.id, a=oidc_access_token)}
        resp = client.get("/personal/hello", headers=headers)
        assert (resp.status_code, resp.json or resp.data) == expected


@pytest.mark.parametrize(["url", "expected_data"], [
    ("/private/hello", b"hello you"),
    ("/personal/hello", b"hello oidcuser"),
])
def test_bearer_auth_oidc_success(app, url, expected_data, requests_mock, oidc_provider):
    def userinfo(request, context):
        """Fake OIDC /userinfo endpoint handler"""
        _, _, token = request.headers["Authorization"].partition("Bearer ")
        user_id = token.split(".")[1]
        return json.dumps({"sub": user_id})

    requests_mock.get(oidc_provider.issuer + "/userinfo", text=userinfo)

    with app.test_client() as client:
        # Note: user id is "hidden" in access token
        oidc_access_token = "kcneududhey8rmxje3uhs.oidcuser.o94h4oe9djdndjeu3rkrnmlxpds834r"
        headers = {"Authorization": "Bearer oidc/{p}/{a}".format(p=oidc_provider.id, a=oidc_access_token)}
        resp = client.get(url, headers=headers)
        assert resp.status_code == 200
        assert resp.data == expected_data


def test_bearer_auth_oidc_inspect(app, requests_mock, oidc_provider):
    def userinfo(request, context):
        """Fake OIDC /userinfo endpoint handler"""
        _, _, token = request.headers["Authorization"].partition("Bearer ")
        user_id = token.split(".")[1]
        return json.dumps({"sub": user_id})

    requests_mock.get(oidc_provider.issuer + "/userinfo", text=userinfo)

    with app.test_client() as client:
        # Note: user id is "hidden" in access token
        oidc_access_token = "kcneududhey8rmxje3uhs.oidcuser.o94h4oe9djdndjeu3rkrnmlxpds834r"
        headers = {"Authorization": "Bearer oidc/{p}/{a}".format(p=oidc_provider.id, a=oidc_access_token)}
        resp = client.get("/inspect", headers=headers)
        assert resp.status_code == 200
        userinfo = resp.json["info"]["oidc_userinfo"]
        assert userinfo["sub"] == "oidcuser"
        internal_auth_data = resp.json["internal_auth_data"]
        assert internal_auth_data["authentication_method"] == "OIDC"
        assert internal_auth_data["oidc_provider_id"] == "testoidc"
        assert internal_auth_data["oidc_issuer"] == 'https://oeo.example.com'
        assert internal_auth_data["access_token"] == oidc_access_token


def test_bearer_auth_oidc_caching(app, requests_mock, oidc_provider):
    def userinfo(request, context):
        """Fake OIDC /userinfo endpoint handler"""
        _, _, token = request.headers["Authorization"].partition("Bearer ")
        user_id = token.split(".")[1]
        return json.dumps({"sub": user_id})

    userinfo = requests_mock.get(oidc_provider.issuer + "/userinfo", text=userinfo)

    def set_time(time):
        # TODO reusable time mocking
        app.config["auth_handler"]._cache._clock = lambda: time

    with app.test_client() as client:
        # Note: user id is "hidden" in access token
        headers = {"Authorization": f"Bearer oidc/{oidc_provider.id}/rmxje3uhs.oidcuser.o94h4oe"}
        headers_other = {"Authorization": f"Bearer oidc/{oidc_provider.id}/trwe35.otheruser.fg34fsf"}

        set_time(10)
        resp = client.get("/personal/hello", headers=headers)
        assert (resp.status_code, resp.data) == (200, b"hello oidcuser")
        assert userinfo.call_count == 1
        resp = client.get("/personal/hello", headers=headers_other)
        assert (resp.status_code, resp.data) == (200, b"hello otheruser")
        assert userinfo.call_count == 2

        set_time(100)
        resp = client.get("/personal/hello", headers=headers)
        assert (resp.status_code, resp.data) == (200, b"hello oidcuser")
        assert userinfo.call_count == 2

        set_time(10000)
        resp = client.get("/personal/hello", headers=headers)
        assert (resp.status_code, resp.data) == (200, b"hello oidcuser")
        assert userinfo.call_count == 3


def test_userinfo_url_caching(app, requests_mock, oidc_provider):
    oidc_discovery_url = oidc_provider.issuer + "/.well-known/openid-configuration"
    oidc_userinfo_url = oidc_provider.issuer + "/userinfo"
    discovery_mock = requests_mock.get(oidc_discovery_url, json={"userinfo_endpoint": oidc_userinfo_url})
    requests_mock.get(oidc_provider.issuer + "/userinfo", json={"sub": "foo"})

    def set_time(time):
        # TODO reusable time mocking
        app.config["auth_handler"]._cache._clock = lambda: time

    with app.test_client() as client:
        assert discovery_mock.call_count == 0

        set_time(10)
        resp = client.get("/private/hello", headers={"Authorization": f"Bearer oidc/{oidc_provider.id}/dfergef"})
        assert resp.status_code == 200
        assert discovery_mock.call_count == 1

        set_time(60)
        resp = client.get("/private/hello", headers={"Authorization": f"Bearer oidc/{oidc_provider.id}/ftreyer"})
        assert resp.status_code == 200
        assert discovery_mock.call_count == 1

        set_time(30 * 60)
        resp = client.get("/private/hello", headers={"Authorization": f"Bearer oidc/{oidc_provider.id}/th56te"})
        assert resp.status_code == 200
        assert discovery_mock.call_count == 2


@pytest.fixture
def app_with_user_access_validation(oidc_provider) -> Flask:
    def user_access_validation(user: User, request: flask.Request) -> User:
        """User check that only allows users with a "title case" user name"""
        user_id = user.user_id
        if user_id == user_id.title():
            return User(user_id=f"{user.user_id} (verified)", info=user.info,
                        internal_auth_data=user.internal_auth_data)
        else:
            raise PermissionsInsufficientException(f"Invalid user id {user_id}: expected {user_id.title()}.")

    app = Flask("__test__")
    auth = HttpAuthHandler(oidc_providers=[oidc_provider], user_access_validation=user_access_validation)

    @app.route("/basic/auth")
    @auth.requires_http_basic_auth
    def basic():
        access_token, user_id = auth.authenticate_basic(request)
        return jsonify({"access_token": access_token, "user_id": user_id})

    @app.route("/bearer/hello")
    @auth.requires_bearer_auth
    def personal_hello(user: User):
        return "hello {u}".format(u=user.user_id)

    @app.errorhandler(OpenEOApiException)
    def handle_openeoapi_exception(error: OpenEOApiException):
        return jsonify(error.to_dict()), error.status_code

    return app


@pytest.mark.parametrize(["user_id", "success", "message"], [
    ("John", True, b"hello John (verified)"),
    ("fluffYbeAr93", False, "Invalid user id fluffYbeAr93: expected Fluffybear93."),
])
def test_user_access_validation_basic_auth(app_with_user_access_validation, user_id, success, message):
    with app_with_user_access_validation.test_client() as client:
        headers = {"Authorization": build_basic_http_auth_header(user_id, f"{user_id.upper()}!!!")}
        resp = client.get("/basic/auth", headers=headers)
        assert resp.status_code == 200
        access_token = resp.json["access_token"]

        headers = {"Authorization": "Bearer basic//" + access_token}
        resp = client.get("/bearer/hello", headers=headers)

        if success:
            assert (resp.status_code, resp.data) == (200, message)
        else:
            assert resp.status_code == PermissionsInsufficientException.status_code
            assert resp.json["code"] == "PermissionsInsufficient"
            assert resp.json["message"] == message


@pytest.mark.parametrize(["user_id", "success", "message"], [
    ("John", True, b"hello John (verified)"),
    ("fluffYbeAr93", False, "Invalid user id fluffYbeAr93: expected Fluffybear93."),
])
def test_user_access_validation_oidc(
        app_with_user_access_validation, oidc_provider, requests_mock, user_id, success, message
):
    def userinfo(request, context):
        """Fake OIDC /userinfo endpoint handler"""
        _, _, token = request.headers["Authorization"].partition("Bearer ")
        user_id = token.split(".")[1]
        return json.dumps({"sub": user_id})

    requests_mock.get(oidc_provider.issuer + "/userinfo", text=userinfo)

    with app_with_user_access_validation.test_client() as client:
        # Note: user id is "hidden" in access token
        oidc_access_token = f"kcneududhey8rmxje3uhs.{user_id}.o94h4oe9djdndjeu3rkrnmlxpds834r"
        headers = {"Authorization": "Bearer oidc/{p}/{a}".format(p=oidc_provider.id, a=oidc_access_token)}
        resp = client.get("/bearer/hello", headers=headers)

        if success:
            assert (resp.status_code, resp.data) == (200, message)
        else:
            assert resp.status_code == PermissionsInsufficientException.status_code
            assert resp.json["code"] == "PermissionsInsufficient"
            assert resp.json["message"] == message


@pytest.mark.parametrize("fail_kwargs", [
    {"exc": requests.exceptions.ConnectionError()},
    {"exc": requests.exceptions.ProxyError()},
    {"exc": requests.exceptions.ConnectTimeout()},
    {"exc": requests.exceptions.Timeout()},
    {"exc": requests.exceptions.ReadTimeout()},
    {"status_code": 500, "text": "Internal server error"},
    {"status_code": 502, "text": "Bad Gateway"},
])
@pytest.mark.parametrize("fail_url", [
    "/.well-known/openid-configuration",
    "/userinfo",
])
def test_oidc_provider_down(app, requests_mock, oidc_provider, fail_kwargs, fail_url):
    # Setup connection failure in OIDC provider
    requests_mock.get(oidc_provider.get_issuer() + fail_url, **fail_kwargs)

    with app.test_client() as client:
        headers = {"Authorization": f"Bearer oidc/{oidc_provider.id}/f00b6r"}
        resp = client.get("/personal/hello", headers=headers)
        assert (resp.status_code, resp.json) == (
            503,
            DictSubSet(code="OidcProviderUnavailable", message="OIDC Provider is unavailable")
        )
