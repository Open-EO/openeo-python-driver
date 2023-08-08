import logging
import operator
import time
from functools import reduce
from hashlib import md5
from typing import Optional

from openeo_driver.errors import CredentialsInvalidException, ResultLinkExpiredException
from openeo_driver.utils import smart_bool

_log = logging.getLogger(__name__)


class UrlSigner:
    def __init__(self, secret: str, expiration: int = None):
        self._secret = secret
        self._expiration = int(expiration or 0)

    def get_expires(self, now: int = None) -> Optional[int]:
        if self._expiration:
            return int((now or time.time()) + self._expiration)
        return None

    def sign_job_asset(self, job_id: str, user_id: str, filename: str, expires: Optional[int]) -> str:
        """Generate a signature for given job result asset"""
        return self._sign_with_secret(job_id, user_id, filename, str(expires))

    def verify_job_asset(self, signature: str, job_id: str, user_id: str, filename: str, expires: Optional[int]):
        """Verify signature"""
        expected_signature = self.sign_job_asset(job_id=job_id, user_id=user_id, filename=filename, expires=expires)
        if signature != expected_signature:
            _log.warning(f"URL Signature mismatch for user {user_id} (job {job_id}, filename {filename})")
            raise CredentialsInvalidException()
        if expires and int(expires) < time.time():
            _log.warning(f"URL Signature expired for user {user_id} (job {job_id}, filename {filename})")
            raise ResultLinkExpiredException()

    def sign_job_results(self, job_id: str, user_id: str, expires: Optional[int]) -> str:
        return self._sign_with_secret(job_id, user_id, str(expires))

    def verify_job_results(self, signature: str, job_id: str, user_id: str, expires: Optional[int]):
        # TODO: reduce code duplication
        expected_signature = self.sign_job_results(job_id=job_id, user_id=user_id, expires=expires)
        if signature != expected_signature:
            _log.warning(f"URL Signature mismatch for user {user_id} (job {job_id})")
            raise CredentialsInvalidException()
        if expires and int(expires) < time.time():
            _log.warning(f"URL Signature expired for user {user_id} (job {job_id})")
            raise ResultLinkExpiredException()

    def sign_job_item(self, job_id: str, user_id: str, item_id: str, expires: Optional[int]) -> str:
        return self._sign_with_secret(job_id, user_id, item_id, str(expires))

    def verify_job_item(self, signature: str, job_id: str, user_id: str, item_id: str, expires: Optional[int]):
        # TODO: reduce code duplication
        expected_signature = self.sign_job_item(job_id=job_id, user_id=user_id, item_id=item_id, expires=expires)
        if signature != expected_signature:
            _log.warning(f"URL Signature mismatch for user {user_id} (job {job_id}, item {item_id})")
            raise CredentialsInvalidException()
        if expires and int(expires) < time.time():
            _log.warning(f"URL Signature expired for user {user_id} (job {job_id}, item {item_id})")
            raise ResultLinkExpiredException()

    def _sign_with_secret(self, *token_key_parts: str) -> str:
        token_key = reduce(operator.add, list(token_key_parts) + [self._secret], "")
        return md5(token_key.encode("utf8")).hexdigest()
