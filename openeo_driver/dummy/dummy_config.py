from openeo_driver.config import OpenEoBackendConfig
from openeo_driver.server import build_backend_deploy_metadata
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


def _valid_basic_auth(username: str, password: str) -> bool:
    # Next generation password scheme!!1!
    if username[:1].lower() in "aeiou":
        return password == f"{username.lower()}123"
    else:
        return password == f"{username.upper()}!!!"


config = OpenEoBackendConfig(
    id="openeo-python-driver-dummy",
    capabilities_title="Dummy openEO Backend",
    capabilities_description="Dummy openEO backend provided by [openeo-python-driver](https://github.com/Open-EO/openeo-python-driver).",
    capabilities_backend_version="1.2.3-foo",
    capabilities_deploy_metadata=build_backend_deploy_metadata(packages=["openeo", "openeo_driver"]),
    processing_facility="Dummy openEO API",
    oidc_providers=oidc_providers,
    enable_basic_auth=True,
    valid_basic_auth=_valid_basic_auth,
)
