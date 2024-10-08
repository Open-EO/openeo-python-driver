import inspect
import os
from typing import Callable, Dict, List, Optional, Tuple, Type

import attrs

import openeo_driver
from openeo_driver.server import build_backend_deploy_metadata
from openeo_driver.urlsigning import UrlSigner
from openeo_driver.users.oidc import OidcProvider
from openeo_driver.workspace import Workspace


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

    collection_exclusion_list: Dict[str, List[str]] = {}  # e.g. {"1.1.0":["my_collection_id"]}
    processes_exclusion_list: Dict[str, List[str]] = {}  # e.g. {"1.1.0":["my_process_id"]}

    workspaces: Dict[str, Workspace] = attrs.Factory(dict)

    ejr_retry_settings: dict = attrs.Factory(lambda: dict(tries=4, delay=2, backoff=2))


def check_config_definition(config_class: Type[OpenEoBackendConfig]):
    """
    Verify that the config class definition is correct (e.g. all fields have type annotations).
    """
    annotated = set(k for cls in config_class.__mro__ for k in cls.__dict__.get("__annotations__", {}).keys())
    members = set(k for k, v in inspect.getmembers(config_class) if not k.startswith("_"))

    if annotated != members:
        missing_annotations = members.difference(annotated)
        raise ConfigException(f"{config_class.__name__}: fields without type annotation: {missing_annotations}.")
