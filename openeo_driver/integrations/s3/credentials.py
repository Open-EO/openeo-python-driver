import logging
import os
import re
from typing import TypedDict, Optional

_log = logging.getLogger(__name__)


class Boto3Credentials(TypedDict):
    aws_access_key_id: str
    aws_secret_access_key: str


def get_credentials(region_name: str, provider_name: str) -> Boto3Credentials:
    """
    Credential resolving should always go from most specific to least specific. So let's say we are checking region
    waw3-1 of the cloudferro cloud we must check in order:
    - WAW3_1_ACCESS_KEY_ID && WAW3_1_SECRET_ACCESS_KEY
    - CF_ACCESS_KEY_ID && CF_SECRET_ACCESS_KEY
    - SWIFT_ACCESS_KEY_ID && SWIFT_SECRET_ACCESS_KEY  # for backwards compatibility
    - AWS_ACCESS_KEY_ID && AWS_SECRET_ACCESS_KEY
    """
    for prefix in [f"{region_name}", f"{provider_name}", "SWIFT", "AWS"]:
        credential_candidate = get_credential_for_prefix(prefix)
        if credential_candidate is not None:
            return credential_candidate
    raise EnvironmentError(f"Cannot get credentials for {region_name}")


def _sanitize_env_name(env_name: str) -> str:
    return re.sub("[^A-Z0-9]+", "_", env_name.upper())


def get_credential_for_prefix(prefix: str) -> Optional[Boto3Credentials]:
    """
    When getting credentials from the environment they should have both parts to be valid (access_key_id and secret_key)
    """
    access_key_id = os.environ.get(_sanitize_env_name(f"{prefix}_ACCESS_KEY_ID"))
    secret_key = os.environ.get(_sanitize_env_name(f"{prefix}_SECRET_ACCESS_KEY"))
    if access_key_id is None and secret_key is None:
        return None
    elif access_key_id is None or secret_key is None:
        # Credentials must come in pairs so having only 1 part smells like environment misconfiguration
        _log.warning(f"Likely invalid backend config for managing credentials for {prefix}")
        return None
    return {
        "aws_access_key_id": access_key_id,
        "aws_secret_access_key": secret_key,
    }
