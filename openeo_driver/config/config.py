from typing import Callable, Dict, List, Optional, Tuple

import attrs

import openeo_driver
from openeo_driver.server import build_backend_deploy_metadata
from openeo_driver.users.oidc import OidcProvider


class ConfigException(ValueError):
    pass


@attrs.frozen(
    # Note: `kw_only=True` enforces "kwargs" based construction (which is good for readability/maintainability)
    # and allows defining mandatory fields (fields without default) after optional fields.
    kw_only=True
)
class OpenEoBackendConfig:
    """
    Configuration for openEO backend.
    """

    # identifier for this config
    id: Optional[str] = None

    capabilities_service_id: Optional[str] = None
    capabilities_title: str = "Untitled openEO Backend"
    capabilities_description: str = "This is a generic openEO Backend, powered by [openeo-python-driver](https://github.com/Open-EO/openeo-python-driver)."
    capabilities_backend_version: str = openeo_driver.__version__
    capabilities_deploy_metadata: dict = attrs.Factory(
        lambda: build_backend_deploy_metadata(packages=["openeo", "openeo_driver"])
    )

    processing_facility: str = "openEO"
    processing_software: str = "openeo-python-driver"

    oidc_providers: List[OidcProvider] = attrs.Factory(list)

    oidc_token_introspection: bool = False

    # Mapping of `(oidc_provider id, token_sub)`
    # to user info, as a dictionary with at least a "user_id" field.
    # `token_sub` is the  OIDC token "sub" field, which usually identifies a user,
    # but could also identify a OIDC client authenticated through the client credentials grant.
    # TODO: allow it to be a callable instead of a dictionary?
    oidc_user_map: Dict[Tuple[str, str], dict] = attrs.Factory(dict)

    # TODO #90 #186: eliminate simple password scheme
    valid_basic_auth: Callable[[str, str], bool] = lambda u, p: p == f"{u}123"

    # General Flask related settings
    # (e.g. see https://flask.palletsprojects.com/en/2.3.x/config/#builtin-configuration-values)
    flask_settings: dict = attrs.Factory(
        lambda: {
            "MAX_CONTENT_LENGTH": 1024 * 1024,  # bytes
        }
    )
