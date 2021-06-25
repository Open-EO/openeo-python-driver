import re
import time

import pytest

from openeo_driver.errors import CredentialsInvalidException, ResultLinkExpiredException
from openeo_driver.urlsigning import user_id_b64_encode, user_id_b64_decode, Signer


@pytest.mark.parametrize("user_id", [
    "John", "John D", "John Do", "John Doe", "John Drop Tables",
    "Jøhñ Δö€",
    r"J()h&n |>*% $<{}@!\\:,^ #=!,.`=-_+°º¤ø,¸¸,ø¤º°»-(¯`·.·´¯)->¯\_(ツ)_/¯0(╯°□°）╯ ︵ ┻━┻ ",
    "Pablo Diego José Francisco de Paula Juan Nepomuceno María de los Remedios Cipriano de la Santísima Trinidad Ruiz y Picasso"
])
def test_user_id_b64_encode(user_id):
    encoded = user_id_b64_encode(user_id)
    assert isinstance(encoded, str)
    assert re.match("^[A-Za-z0-9_=-]*$", encoded)
    decoded = user_id_b64_decode(encoded)
    assert isinstance(decoded, str)
    assert decoded == user_id


@pytest.mark.parametrize("expiration", [None, 10])
def test_signer_basic(expiration):
    signer = Signer(secret="$s3cr3t", expiration=expiration)
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
    signer = Signer(secret="$s3cr3t", expiration=1)
    expires = signer.get_expires()
    signature = signer.sign_job_asset(job_id="job1", user_id="john", filename="res.tiff", expires=expires)

    signer.verify_job_asset(signature=signature, job_id="job1", user_id="john", filename="res.tiff", expires=expires)
    time.sleep(1.1)
    with pytest.raises(ResultLinkExpiredException):
        signer.verify_job_asset(
            signature=signature, job_id="job1", user_id="john", filename="res.tiff", expires=expires
        )
