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

    # Mapping of OIDC token "sub" (which usually identifies a user,
    # but could also identify a OIDC client authenticated through the client credentials grant)
    # to user info, as a dictionary with at least a "user_id" field
    # TODO: allow it to be a callable instead of a dictionary?
    oidc_user_map: Dict[str, dict] = attrs.Factory(dict)
