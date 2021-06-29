import base64
import json
import re

import pytest
from flask import Flask, jsonify, Response, request

from openeo_driver.backend import OidcProvider
from openeo_driver.errors import OpenEOApiException
from openeo_driver.users import HttpAuthHandler, User
from openeo_driver.users import user_id_b64_encode, user_id_b64_decode


@pytest.mark.parametrize("user_id", [
    "John", "John D", "John Do", "John Doe", "John Drop Tables",
    "Jøhñ Δö€",
    r"J()h&n |>*% $<{}@!\\:,^ #=!,.`=-_+°º¤ø,¸¸,ø¤º°»-(¯`·.·´¯)->¯\_(ツ)_/¯0(╯°□°）╯ ︵ ┻━┻ ",
    "Pablo Diego José Francisco de Paula Juan Nepomuceno María de los Remedios Cipriano de la Santísima Trinidad Ruiz y Picasso"
])
def test_user_id_b64_encode(user_id):
    encoded = user_id_b64_encode(user_id)
    assert isinstance(encoded, str)
    assert re.match("^[A-Za-z0-9_=-]*$", encoded)
    decoded = user_id_b64_decode(encoded)
    assert isinstance(decoded, str)
    assert decoded == user_id


@pytest.fixture()
def oidc_provider(requests_mock) -> OidcProvider:
    oidc_issuer = "https://oeo.example.com"
    oidc_discovery_url = oidc_issuer + "/.well-known/openid-configuration"
    oidc_userinfo_url = oidc_issuer + "/userinfo"
    requests_mock.get(oidc_discovery_url, json={"userinfo_endpoint": oidc_userinfo_url})
    return OidcProvider(id="testoidc", issuer=oidc_issuer, scopes=['openid'], title='Test OIDC provider')


@pytest.fixture()
def app(oidc_provider):
    """Fixture for a flask app with some public and some auth requiring handlers"""
    app = Flask("__test__")
    auth = HttpAuthHandler(oidc_providers=[oidc_provider])

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


def _build_basic_http_auth_header(username: str, password: str) -> str:
    """Build HTTP header for Basic HTTP authentication"""
    # Note: this is not the basic bearer token
    return "Basic " + base64.b64encode("{u}:{p}".format(u=username, p=password).encode("utf-8")).decode('ascii')


def test_basic_auth_invalid_password(app):
    with app.test_client() as client:
        headers = {"Authorization": _build_basic_http_auth_header("testuser", "wrongpassword")}
        response = client.get("/basic/hello", headers=headers)
        assert_invalid_credentials_failure(response)


def test_basic_auth_success(app):
    with app.test_client() as client:
        headers = {"Authorization": _build_basic_http_auth_header("testuser", "testuser123")}
        response = client.get("/basic/hello", headers=headers)
        assert response.status_code == 200
        assert response.data == b"hello basic"


@pytest.mark.parametrize("url", ["/private/hello", "/personal/hello"])
def test_bearer_auth__no_auth(app, url):
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
        headers = {"Authorization": "Bearer basic//{p}blehrff".format(p=HttpAuthHandler._BASIC_ACCESS_TOKEN_PREFIX)}
        response = client.get(url, headers=headers)
        assert_invalid_token_failure(response)


@pytest.mark.parametrize(["url", "expected_data"], [
    ("/private/hello", b"hello you"),
    ("/personal/hello", b"hello testuser"),
])
def test_bearer_auth_basic_token_success(app, url, expected_data):
    with app.test_client() as client:
        headers = {"Authorization": _build_basic_http_auth_header("testuser", "testuser123")}
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
