from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


_log = logging.getLogger(__name__)


def create_presigned_url(client: S3Client, bucket_name: str, object_name: str, expiration: int = 3600) -> str:
    """
    Generate a presigned URL to share an S3 object

    :return: Presigned URL as string.
    """
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": object_name},
        ExpiresIn=expiration,
    )


def attempt_create_presigned_url(
    client: S3Client, bucket_name: str, object_name: str, expiration: int = 3600
) -> Optional[str]:
    from botocore.exceptions import ClientError
    """
    Generate a presigned URL to share an S3 object

    :return: Presigned URL as string. If error, returns None.
    """
    try:
        return create_presigned_url(client, bucket_name=bucket_name, object_name=object_name, expiration=expiration)
    except ClientError as e:
        logging.info(f"Failed to create presigned url: {e}, continuing without.")
        return None
