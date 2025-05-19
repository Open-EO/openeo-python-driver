from __future__ import annotations

import logging
import os

import boto3

from typing import TYPE_CHECKING, Optional
from openeo_driver.integrations.s3.endpoint import get_endpoint
from openeo_driver.integrations.s3.credentials import get_credentials


if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

_log = logging.getLogger(__name__)


class S3ClientBuilder:
    @classmethod
    def from_region(cls, region_name: Optional[str]) -> S3Client:
        if region_name is None:
            _log.warning(
                "Building S3 client without region. This should be avoided as much as possible because these operations"
                "will be limited to a single endpoint for which the platform is configured."
            )
            region_name = os.environ.get("AWS_REGION", "UNKNOWN")
        return cls._s3_client(
            region_name=region_name, endpoint_url=get_endpoint(region_name), **get_credentials(region_name)
        )

    @classmethod
    def _s3_client(cls, *args, **kwargs) -> S3Client:
        """
        Have a separate client creating class. This will be mocked for tests. This is required because the mocking
        library we use in tests (moto) patches before-send hook for botocore and checks the urls to figure out which
        service it is handling but unfortunately this means custom S3 urls are ignored and the request is actually
        sent over the wire to the defined endpoint.
        """
        return boto3.client("s3", *args, **kwargs)
