import pytest

from openeo_driver.users.oidc import OidcProvider


def test_oidc_provider_from_dict_empty():
    with pytest.raises(KeyError, match="Missing OidcProvider fields: id, issuer, title"):
        _ = OidcProvider.from_dict({})


def test_oidc_provider_from_dict_basic():
    p = OidcProvider.from_dict({"id": "foo", "issuer": "https://oidc.foo.test/", "title": "Foo ID"})
    assert p.id == "foo"
    assert p.issuer == "https://oidc.foo.test/"
    assert p.title == "Foo ID"
    assert p.scopes == ["openid"]
    assert p.description is None
    assert p.discovery_url == "https://oidc.foo.test/.well-known/openid-configuration"

    assert p.prepare_for_json() == {
        "id": "foo", "issuer": "https://oidc.foo.test/", "title": "Foo ID", "scopes": ["openid"]
    }
    assert p.get_issuer() == "https://oidc.foo.test"


def test_oidc_provider_from_dict_more():
    p = OidcProvider.from_dict({
        "id": "foo", "issuer": "https://oidc.foo.test/", "title": "Foo ID",
        "scopes": ["openid", "email"],
        "default_clients": {"id": "dcf0e6384", "grant_types": ["refresh_token"]},
    })
    assert p.id == "foo"
    assert p.issuer == "https://oidc.foo.test/"
    assert p.title == "Foo ID"
    assert p.scopes == ["openid", "email"]
    assert p.description is None
    assert p.default_clients == {"id": "dcf0e6384", "grant_types": ["refresh_token"]}
    assert p.discovery_url == "https://oidc.foo.test/.well-known/openid-configuration"

    assert p.prepare_for_json() == {
        "id": "foo", "issuer": "https://oidc.foo.test/", "title": "Foo ID",
        "scopes": ["openid", "email"],
        "default_clients": {"id": "dcf0e6384", "grant_types": ["refresh_token"]},
    }
    assert p.get_issuer() == "https://oidc.foo.test"


def test_oidc_provider_get_issuer():
    assert OidcProvider("d", "https://oidc.test", "t").get_issuer() == "https://oidc.test"
    assert OidcProvider("d", "https://oidc.test/", "t").get_issuer() == "https://oidc.test"
    assert OidcProvider("d", "https://oidc.test//", "t").get_issuer() == "https://oidc.test"
    assert OidcProvider("d", "https://oidc.test//", "t").get_issuer() == "https://oidc.test"
    assert OidcProvider("d", "https://OIDC.test/", "t").get_issuer() == "https://oidc.test"
    assert OidcProvider("d", "https://oidc.test/foo", "t").get_issuer() == "https://oidc.test/foo"
    assert OidcProvider("d", "https://oidc.test/foo/", "t").get_issuer() == "https://oidc.test/foo"
    assert OidcProvider("d", "https://oidc.test/foo//", "t").get_issuer() == "https://oidc.test/foo"
