from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


_log = logging.getLogger(__name__)
# Sentinel value for unset values
_UNSET = object()


def create_presigned_url(
    client: S3Client, bucket_name: str, object_name: str, expiration: int = 3600, default: Optional[str] = _UNSET
) -> Optional[str]:
    from botocore.exceptions import ClientError

    """
    Generate a presigned URL to share an S3 object
    If a default value is provided rather then that value is returned instead of throwing an exception
    """
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration,
        )
    except ClientError as e:
        logging.info(f"Failed to create presigned url: {e}")
        if default is not _UNSET:
            return default
        return None
