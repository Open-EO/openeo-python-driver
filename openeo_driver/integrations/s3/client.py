from __future__ import annotations

import logging
import os

from typing import TYPE_CHECKING, Optional
from openeo_driver.integrations.s3.credentials import get_credentials
from openeo_driver.config import get_backend_config

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

_log = logging.getLogger(__name__)


class S3ClientBuilder:
    @classmethod
    def from_region(cls, region_name: Optional[str]) -> S3Client:
        if region_name is None:
            region_name = os.environ.get("AWS_REGION", "UNKNOWN")
            _log.warning(
                "Building S3 client without explicit region."
                f" Falling back to {region_name!r}."
                " Avoid depending on this brittle fallback mechanism and explicitly specify the region.",
                exc_info=True,
            )
        s3_config = get_backend_config().s3_provider_config
        provider_name = s3_config.get_provider(region_name)
        endpoint = s3_config.get_endpoint(region_name)

        return cls._s3_client(
            region_name=region_name, endpoint_url=endpoint, **get_credentials(region_name, provider_name)
        )

    @classmethod
    def _s3_client(cls, *args, **kwargs) -> S3Client:
        """
        Have a separate client creating class. This will be mocked for tests. This is required because the mocking
        library we use in tests (moto) patches before-send hook for botocore and checks the urls to figure out which
        service it is handling but unfortunately this means custom S3 urls are ignored and the request is actually
        sent over the wire to the defined endpoint.
        """
        # Keep this import inside the method so we kan keep installing boto3 as optional,
        # because we don't always use objects storage.
        # We want to avoid unnecessary dependencies. (And dependencies  of dependencies!)
        import boto3
        return boto3.client("s3", *args, **kwargs)
