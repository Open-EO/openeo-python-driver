import base64
import logging
import time
from hashlib import md5
from typing import Union

from openeo_driver.errors import CredentialsInvalidException, ResultLinkExpiredException
from openeo_driver.utils import smart_bool

_log = logging.getLogger(__name__)


class Signer:
    def __init__(self, secret: str, expiration: int = None):
        self._secret = secret
        self._expiration = int(expiration or 0)

    @classmethod
    def from_config(cls, config: dict) -> 'Signer':
        assert smart_bool(config.get('SIGNED_URL'))
        return cls(
            secret=config.get('SIGNED_URL_SECRET'),
            expiration=config.get('SIGNED_URL_EXPIRATION')
        )

    def get_expires(self, now: int = None) -> Union[int, None]:
        if self._expiration:
            return int((now or time.time()) + self._expiration)
        return None

    def sign_job_asset(self, job_id: str, user_id: str, filename: str, expires: Union[int, None]) -> str:
        """Generate a signature for given job result asset"""
        token_key = job_id + user_id + filename + str(expires) + self._secret
        return md5(token_key.encode("utf8")).hexdigest()

    def verify_job_asset(self, signature: str, job_id: str, user_id: str, filename: str, expires: Union[int, None]):
        """Verify signature"""
        expected_signature = self.sign_job_asset(job_id=job_id, user_id=user_id, filename=filename, expires=expires)
        if signature != expected_signature:
            _log.warning(f"URL Signature mismatch for user {user_id} (job {job_id}, filename {filename})")
            raise CredentialsInvalidException()
        if expires and int(expires) < time.time():
            _log.warning(f"URL Signature expired for user {user_id} (job {job_id}, filename {filename})")
            raise ResultLinkExpiredException()
