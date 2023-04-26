from openeo_driver.config import OpenEoBackendConfig
from openeo_driver.users.oidc import OidcProvider

oidc_providers = [
    OidcProvider(
        id="testprovider",
        issuer="https://oidc.test",
        scopes=["openid"],
        title="Test",
    ),
    OidcProvider(
        id="eoidc",
        issuer="https://eoidc.test",
        scopes=["openid"],
        title="e-OIDC",
        default_clients=[
            {
                "id": "badcafef00d",
                "grant_types": [
                    "urn:ietf:params:oauth:grant-type:device_code+pkce",
                    "refresh_token",
                ],
            }
        ],
    ),
    # Allow testing with Keycloak setup running in docker on localhost.
    OidcProvider(
        id="local",
        title="Local Keycloak",
        issuer="http://localhost:9090/auth/realms/master",
        scopes=["openid"],
    ),
    # Allow testing the dummy backend with EGI
    OidcProvider(
        id="egi",
        issuer="https://aai.egi.eu/auth/realms/egi/",
        scopes=[
            "openid",
            "email",
            "eduperson_entitlement",
            "eduperson_scoped_affiliation",
        ],
        title="EGI Check-in",
    ),
    OidcProvider(
        id="egi-dev",
        issuer="https://aai-dev.egi.eu/auth/realms/egi",
        scopes=[
            "openid",
            "email",
            "eduperson_entitlement",
            "eduperson_scoped_affiliation",
        ],
        title="EGI Check-in (dev)",
    ),
]


config = OpenEoBackendConfig(
    id="dummy",
    oidc_providers=oidc_providers,
)
