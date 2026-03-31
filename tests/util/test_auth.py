import logging
import re
import time
from typing import Optional

import pytest
import time_machine
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
    def access_token_expires_in(self) -> Optional[int]:
        """By default we let access tokens of the mock expire in 1 hour"""
        return 3600

    @pytest.fixture
    def local_cache_ttl(self) -> int:
        """By default we let the local cache expire in 30 minutes"""
        return 1800

    @pytest.fixture
    def oidc_mock(self, requests_mock, credentials, access_token_expires_in) -> OidcMock:
        oidc_mock = OidcMock(
            requests_mock=requests_mock,
            oidc_issuer=credentials.oidc_issuer,
            expected_grant_type="client_credentials",
            expected_client_id=credentials.client_id,
            expected_fields={"client_secret": credentials.client_secret, "scope": "openid"},
            access_token_expires_in=access_token_expires_in,
        )
        return oidc_mock

    def test_basic(self, credentials, oidc_mock: OidcMock, local_cache_ttl):
        helper = ClientCredentialsAccessTokenHelper(credentials=credentials, default_ttl=local_cache_ttl)
        assert helper.get_access_token() == oidc_mock.state["access_token"]

    @pytest.mark.parametrize(
        ["desc", "local_cache_ttl", "access_token_expires_in", "no_cache_at_30m", "no_cache_at_50m"],
        [
            ("Long caching", 3600, 3600, False, False),
            ("No/very short caching", 0, 0, True, True),
            ("Local cache expires after 30m but before 50m, no server expiry", 40 * 60, None, False, True),
            ("Server cache shortest and cause expiry after 30m but before 50m", 3600, 40 * 60, False, True),
            ("Local cache expires after 30m but access token does not", 40 * 60, 7200, False, False),
        ],
    )
    def test_caching(
        self,
        credentials,
        oidc_mock: OidcMock,
        local_cache_ttl,
        access_token_expires_in,
        no_cache_at_30m,
        no_cache_at_50m,
        desc: str,
    ):
        """
        Test caching by requesting an access token at start time and at the 30 and 50 minute mark.
        """
        now = time.time()
        helper = ClientCredentialsAccessTokenHelper(credentials=credentials, default_ttl=local_cache_ttl)

        expected_chache_misses = 0
        with time_machine.travel(now):
            assert oidc_mock.mocks["token_endpoint"].call_count == expected_chache_misses
            assert helper.get_access_token() == oidc_mock.state["access_token"]
            expected_chache_misses += 1  # First request is always a miss
            assert oidc_mock.mocks["token_endpoint"].call_count == expected_chache_misses

        with time_machine.travel(now + 30 * 60):
            assert helper.get_access_token() == oidc_mock.state["access_token"]
            if no_cache_at_30m:
                expected_chache_misses += 1
            assert oidc_mock.mocks["token_endpoint"].call_count == expected_chache_misses

        with time_machine.travel(now + 50 * 60):
            assert helper.get_access_token() == oidc_mock.state["access_token"]
            if no_cache_at_50m:
                expected_chache_misses += 1
            assert oidc_mock.mocks["token_endpoint"].call_count == expected_chache_misses

    @pytest.mark.skip(reason="Logging was removed for eu-cdse/openeo-cdse-infra#476")
    def test_secret_logging(self, credentials, oidc_mock: OidcMock, caplog):
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
