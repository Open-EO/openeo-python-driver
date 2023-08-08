import time

import pytest

from openeo_driver.errors import CredentialsInvalidException, ResultLinkExpiredException
from openeo_driver.urlsigning import UrlSigner


@pytest.mark.parametrize("expiration", [None, 10])
def test_signer_basic(expiration):
    signer = UrlSigner(secret="$s3cr3t", expiration=expiration)
    expires = signer.get_expires()
    signature = signer.sign_job_asset(job_id="job1", user_id="john", filename="res.tiff", expires=expires)

    signer.verify_job_asset(signature=signature, job_id="job1", user_id="john", filename="res.tiff", expires=expires)

    with pytest.raises(CredentialsInvalidException):
        signer.verify_job_asset(
            signature=signature[:-1], job_id="job1", user_id="john", filename="res.tiff", expires=expires
        )
    with pytest.raises(CredentialsInvalidException):
        signer.verify_job_asset(
            signature=signature, job_id="job2", user_id="john", filename="res.tiff", expires=expires
        )
    with pytest.raises(CredentialsInvalidException):
        signer.verify_job_asset(
            signature=signature, job_id="job1", user_id="johnny", filename="res.tiff", expires=expires
        )
    with pytest.raises(CredentialsInvalidException):
        signer.verify_job_asset(
            signature=signature, job_id="job1", user_id="johny", filename="result.tiff", expires=expires
        )


def test_signer_expiration():
    signer = UrlSigner(secret="$s3cr3t", expiration=1)
    expires = signer.get_expires()
    signature = signer.sign_job_asset(job_id="job1", user_id="john", filename="res.tiff", expires=expires)

    signer.verify_job_asset(signature=signature, job_id="job1", user_id="john", filename="res.tiff", expires=expires)
    time.sleep(1.1)
    with pytest.raises(ResultLinkExpiredException):
        signer.verify_job_asset(
            signature=signature, job_id="job1", user_id="john", filename="res.tiff", expires=expires
        )
