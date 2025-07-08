from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlencode

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


_log = logging.getLogger(__name__)
# Sentinel value for unset values
_UNSET = object()


def _allow_custom_params_for_get_object(s3_client: S3Client) -> None:
    """
    Allow passing custom parameters into the S3 client get_object api calls.

    Custom parameters should start with "x-" to be considered a custom parameter.

    This call is idempotent and can be done multiple times on a single client.
    """
    CUSTOM_PARAMS = "custom_params"

    def client_param_handler(*, params, context, **_kw):
        def is_custom(k):
            return k.lower().startswith("x-")

        # Store custom parameters in context for later event handlers
        context[CUSTOM_PARAMS] = {k: v for k, v in params.items() if is_custom(k)}
        # Remove custom parameters from client parameters,
        # because validation would fail on them
        return {k: v for k, v in params.items() if not is_custom(k)}

    def request_param_injector(*, request, **_kw):
        """https://stackoverflow.com/questions/59056522/create-a-presigned-s3-url-for-get-object-with-custom-logging-information-using-b"""
        if request.context[CUSTOM_PARAMS]:
            request.url += "&" if "?" in request.url else "?"
            request.url += urlencode(request.context[CUSTOM_PARAMS])

    provide_client_params = "provide-client-params.s3.GetObject"
    s3_client.meta.events.register(provide_client_params, client_param_handler, f"{provide_client_params}-paramhandler")
    before_sign = "before-sign.s3.GetObject"
    s3_client.meta.events.register(before_sign, request_param_injector, f"{before_sign}-paraminjector")


def create_presigned_url(
    client: S3Client,
    bucket_name: str,
    object_name: str,
    expiration: int = 3600,
    default: Optional[str] = _UNSET,
    parameters: Optional[dict] = None,
) -> Optional[str]:
    from botocore.exceptions import ClientError
    """
    Generate a presigned URL to share an S3 object
    If a default value is provided rather then that value is returned instead of throwing an exception
    """
    parameters = parameters or {}
    if parameters:
        _allow_custom_params_for_get_object(client)
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name, **parameters},
            ExpiresIn=expiration,
        )
    except ClientError as e:
        logging.info(f"Failed to create presigned url: {e}")
        if default is not _UNSET:
            return default
        raise e
