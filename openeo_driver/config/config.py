from typing import List, Optional, Tuple, Dict

import attrs

from openeo_driver.users.oidc import OidcProvider


class ConfigException(ValueError):
    pass


@attrs.frozen
class OpenEoBackendConfig:
    """
    Configuration for openEO backend.
    """

    # identifier for this config
    id: Optional[str] = None

    oidc_providers: List[OidcProvider] = attrs.Factory(list)

    oidc_token_introspection: bool = False

    # Mapping of OIDC provider id to client credentials tuples (client_id, client_secret)
    # to use as service accounts (for example for OIDC access token introspection).
    oidc_service_accounts: Dict[str, Tuple[str, str]] = attrs.Factory(dict)

    # Mapping of client identifier (OIDC "sub") to corresponding user id
    oidc_client_user_map: Dict[str, str] = attrs.Factory(dict)
