import pytest
import re

from openeo_driver.users import user_id_b64_encode, user_id_b64_decode, User


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


class TestUser:

    @pytest.mark.parametrize(["oidc_userinfo", "expected"], [
        ({}, "u123"),
        ({"name": "john"}, "john"),
        ({"email": "u123@domain.test"}, "u123@domain.test"),
        ({"voperson_verified_email": ["u123@domain.test"]}, "u123@domain.test"),
        ({"name": "john", "voperson_verified_email": ["u123@domain.test"]}, "john"),
    ])
    def test_get_name_from_oidc_userinfo(self, oidc_userinfo, expected):
        user = User("u123", info={"oidc_userinfo": oidc_userinfo})
        assert user.get_name() == expected
