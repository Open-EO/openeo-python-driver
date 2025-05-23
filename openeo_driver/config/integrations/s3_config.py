"""
This file ties regions to cloud providers.
"""

import logging
import os
from typing import Dict, List, TypedDict


_log = logging.getLogger(__name__)


UNKNOWN = "unknown"  # unknown provider uses the legacy behavior


class ProviderDetails(TypedDict):
    """
    When platform operators configure the runtime it is easiest to have configuration per
    object storage provider. Per provider we track regions and an endpoint. If the endpoint has a substring
    `{region}` then that will be replaced with the actual region name. The endpoint must include the protocol.
    """

    regions: List[str]
    endpoint: str


class RegionDetails(TypedDict):
    provider: str
    endpoint: str


class S3ProvidersConfig:
    def __init__(self, cfg: Dict[str, ProviderDetails]):
        """region config maps each lower-cased region on its provider name and the endpoint to be used."""
        region_cfg = {}
        for provider_name, provider_details in cfg.items():
            try:
                endpoint_template = provider_details["endpoint"]
                for region in provider_details["regions"]:
                    region_cfg[region.lower()] = {"provider": provider_name, "endpoint": endpoint_template}
            except KeyError as ke:
                _log.warning(f"Skipping part of config for {provider_name} ({provider_details}) due to {ke}")
        self.cfg: Dict[str, RegionDetails] = region_cfg

    def get_provider(self, region: str) -> str:
        region_lc = region.lower()
        if region_lc not in self.cfg:
            return UNKNOWN
        return self.cfg[region_lc]["provider"]

    def get_endpoint(self, region: str) -> str:
        try:
            return self.cfg[region.lower()]["endpoint"].replace("{region}", region)
        except KeyError:
            legacy_fallback = os.environ.get("SWIFT_URL")
            if legacy_fallback is None:
                raise EnvironmentError(f"Unsupported region {region} and no fallback via SWIFT_URL")
            return legacy_fallback
