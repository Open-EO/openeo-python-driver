import logging

import flask

import openeo_driver.config
from openeo_driver.users import user_id_b64_encode

_log = logging.getLogger(__name__)


class AssetUrl:
    """
    This class helps to convert internal storage locations to asset urls


    If assets are not stored on local storage it is possible to come up with other implementations that extend this
    class and go straight to the external storage. If the driver always has access to the artifacts then it is
     recommended fallback to the default implementation by calling the method from its base class.
    """

    def build_url(self, *, asset_metadata: dict, asset_name: str, job_id: str, user_id: str) -> str:
        """
        The default implementation will create urls that go to the driver application.
        If an url_signer is defined these urls will be signed.

        """
        signer = openeo_driver.config.get_backend_config().url_signer
        if signer:
            expires = signer.get_expires()
            secure_key = signer.sign_job_asset(job_id=job_id, user_id=user_id, filename=asset_name, expires=expires)
            user_base64 = user_id_b64_encode(user_id)
            return flask.url_for(
                ".download_job_result_signed",
                job_id=job_id,
                user_base64=user_base64,
                filename=asset_name,
                expires=expires,
                secure_key=secure_key,
                _external=True,
            )
        else:
            return flask.url_for(".download_job_result", job_id=job_id, filename=asset_name, _external=True)
