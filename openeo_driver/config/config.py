import os
from typing import Callable, Dict, List, Optional, Tuple, Type

import attrs

import openeo_driver
from openeo_driver.config.base import (
    ConfigException,
    _ConfigBase,
    check_config_definition,
    openeo_backend_config_class,
)
from openeo_driver.server import build_backend_deploy_metadata
from openeo_driver.urlsigning import UrlSigner
from openeo_driver.users.oidc import OidcProvider
from openeo_driver.workspace import Workspace
from openeo_driver.asset_urls import AssetUrl

__all__ = ["OpenEoBackendConfig", "openeo_backend_config_class", "ConfigException", "check_config_definition"]


@openeo_backend_config_class
class OpenEoBackendConfig(_ConfigBase):
    """
    Configuration for openEO backend.
    """

    # identifier for this config
    id: Optional[str] = None

    # Generic indicator describing the environment the code is deployed in
    # (e.g. "prod", "dev", "staging", "test", "integration", ...)
    deploy_env: str = os.environ.get("OPENEO_DEPLOY_ENV") or os.environ.get("OPENEO_ENV") or "dev"

    capabilities_service_id: Optional[str] = None
    capabilities_title: str = "Untitled openEO Backend"
    capabilities_description: str = "This is a generic openEO Backend, powered by [openeo-python-driver](https://github.com/Open-EO/openeo-python-driver)."
    capabilities_backend_version: str = openeo_driver.__version__
    capabilities_deploy_metadata: dict = attrs.Factory(
        lambda: build_backend_deploy_metadata(packages=["openeo", "openeo_driver"])
    )

    processing_facility: str = "openEO"
    processing_software: str = "openeo-python-driver"

    # TODO: merge `enable_basic_auth` and `valid_basic_auth` into a single config field.
    enable_basic_auth: bool = False
    # `valid_basic_auth`: function that takes a username and password and returns a boolean indicating if password is correct.
    valid_basic_auth: Optional[Callable[[str, str], bool]] = None

    enable_oidc_auth: bool = True

    oidc_providers: List[OidcProvider] = attrs.Factory(list)

    oidc_token_introspection: bool = False

    # Mapping of `(oidc_provider id, token_sub) to extra user info dictionary, with:
    # - `token_sub`: OIDC token "sub" field, identifying a user (or client in case of client credentials grant).
    # Example use case: specifying the YARN proxy user to run batch jobs with for service accounts (client credentials).
    oidc_user_map: Dict[Tuple[str, str], dict] = attrs.Factory(dict)

    # General Flask related settings
    # (e.g. see https://flask.palletsprojects.com/en/2.3.x/config/#builtin-configuration-values)
    flask_settings: dict = attrs.Factory(
        lambda: {
            "MAX_CONTENT_LENGTH": 2 * 1024 * 1024,  # bytes
        }
    )

    url_signer: Optional[UrlSigner] = None
    asset_url: AssetUrl = attrs.Factory(AssetUrl)

    """
    Collection exclusion list: mapping of API version to collections to exclude
    e.g. {"1.1.0": ["my_collection_id"]}
    """
    collection_exclusion_list: Dict[str, List[str]] = attrs.Factory(dict)

    """
    Process exclusion list: mapping of API version to processes to exclude
    e.g. {"1.1.0": ["my_process_id"]}
    """
    processes_exclusion_list: Dict[str, List[str]] = attrs.Factory(dict)

    workspaces: Dict[str, Workspace] = attrs.Factory(dict)

    ejr_retry_settings: dict = attrs.Factory(lambda: dict(tries=4, delay=2, backoff=2))

    "Experimental: simple job progress fallback estimation. Specify average batch job completion time (wall clock) in seconds."
    simple_job_progress_estimation: Optional[float] = None
