from typing import List, Optional

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
