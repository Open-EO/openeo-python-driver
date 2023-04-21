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

    # Mapping of OIDC token "sub" to corresponding user id
    # (e.g. to map a client credentials access token to a real user)
    # TODO: allow it to be a callable instead of a dictionary?
    oidc_user_map: Dict[str, str] = attrs.Factory(dict)
