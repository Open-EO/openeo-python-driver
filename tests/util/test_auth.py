import logging
import re

import pytest
from openeo.rest.auth.testing import OidcMock

from openeo_driver.util.auth import ClientCredentials, ClientCredentialsAccessTokenHelper


class TestClientCredentials:
    def test_basic(self):
        creds = ClientCredentials(oidc_issuer="https://oidc.test/", client_id="c123", client_secret="@#$")
        assert creds.oidc_issuer == "https://oidc.test/"
        assert creds.client_id == "c123"
        assert creds.client_secret == "@#$"
        assert creds == ("https://oidc.test/", "c123", "@#$")

    def test_repr(self):
        creds = ClientCredentials(oidc_issuer="https://oidc.test/", client_id="c123", client_secret="@#$")
        expected = "ClientCredentials(oidc_issuer='https://oidc.test/', client_id='c123')"
        assert repr(creds) == expected
        assert str(creds) == expected

    def test_get_from_mapping(self):
        creds = ClientCredentials.from_mapping(
            {"oidc_issuer": "https://oidc.test/", "client_id": "c456789", "client_secret": "s3cr3t"},
        )
        assert creds == ("https://oidc.test/", "c456789", "s3cr3t")

    def test_get_from_mapping_strictness(self):
        data = {"oidc_issuer": "https://oidc.test/", "client_id": "c456789"}
        with pytest.raises(
            ValueError, match="Failed building ClientCredentials from mapping: missing {'client_secret'}"
        ):
            _ = ClientCredentials.from_mapping(data)
        creds = ClientCredentials.from_mapping(data, strict=False)
        assert creds is None

    def test_get_from_credentials_string(self):
        creds = ClientCredentials.from_credentials_string("Klient:s3cr3t@https://oidc.test/")
        assert creds == ("https://oidc.test/", "Klient", "s3cr3t")

    @pytest.mark.parametrize(
        "data",
        [
            "Klient@https://oidc.test/auth/realms/test",
            "Klient:$ecret!:https://oidc.test/auth/realms/test",
            "Klient@$ecret!@https://oidc.test/auth/realms/test",
            "Klient:Klient:$ecret!@https://oidc.test/auth/realms/test",
            "SECRET",
        ],
    )
    def test_get_from_credentials_string_strictness(self, data, caplog):
        with pytest.raises(
            ValueError, match=re.compile("Failed parsing ClientCredentials from credentials string$")
        ) as exc_info:
            _ = ClientCredentials.from_credentials_string(data)
        assert "$ecret!" not in repr(exc_info.value)
        assert "$ecret!" not in caplog.text

        assert ClientCredentials.from_credentials_string(data, strict=False) is None


class TestClientCredentialsAccessTokenHelper:
    @pytest.fixture
    def credentials(self) -> ClientCredentials:
        return ClientCredentials(oidc_issuer="https://oidc.test", client_id="client123", client_secret="s3cr3t")

    @pytest.fixture
    def oidc_mock(self, requests_mock, credentials) -> OidcMock:
        oidc_mock = OidcMock(
            requests_mock=requests_mock,
            oidc_issuer=credentials.oidc_issuer,
            expected_grant_type="client_credentials",
            expected_client_id=credentials.client_id,
            expected_fields={"client_secret": credentials.client_secret, "scope": "openid"},
        )
        return oidc_mock

    def test_basic(self, credentials, oidc_mock: OidcMock):
        helper = ClientCredentialsAccessTokenHelper(credentials=credentials)
        assert helper.get_access_token() == oidc_mock.state["access_token"]

    def test_caching(self, credentials, oidc_mock: OidcMock):
        helper = ClientCredentialsAccessTokenHelper(credentials=credentials)
        assert oidc_mock.mocks["token_endpoint"].call_count == 0
        assert helper.get_access_token() == oidc_mock.state["access_token"]
        assert oidc_mock.mocks["token_endpoint"].call_count == 1
        assert helper.get_access_token() == oidc_mock.state["access_token"]
        assert oidc_mock.mocks["token_endpoint"].call_count == 1

    def skip_secret_logging(self, credentials, oidc_mock: OidcMock, caplog):
        """Check that secret is not logged"""
        caplog.set_level(logging.DEBUG)
        helper = ClientCredentialsAccessTokenHelper(credentials=credentials)
        assert helper.get_access_token() == oidc_mock.state["access_token"]
        (setup_log,) = [
            log for log in caplog.messages if log.startswith("Setting up ClientCredentialsAccessTokenHelper")
        ]
        assert (
            setup_log
            == "Setting up ClientCredentialsAccessTokenHelper with ClientCredentials(oidc_issuer='https://oidc.test', client_id='client123')"
        )
