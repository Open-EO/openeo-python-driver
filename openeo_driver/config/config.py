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

    # Mapping of `(oidc_provider id, token_sub)`
    # to user info, as a dictionary with at least a "user_id" field.
    # `token_sub` is the  OIDC token "sub" field, which usually identifies a user,
    # but could also identify a OIDC client authenticated through the client credentials grant.
    # TODO: allow it to be a callable instead of a dictionary?
    oidc_user_map: Dict[Tuple[str, str], dict] = attrs.Factory(dict)
