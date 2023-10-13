import pytest
import re

from openeo_driver.users.user import user_id_b64_encode, user_id_b64_decode, User


@pytest.mark.parametrize(
    "user_id",
    [
        "John",
        "John D",
        "John Do",
        "John Doe",
        "John Drop Tables",
        "Jøhñ Δö€",
        r"J()h&n |>*% $<{}@!\\:,^ #=!,.`=-_+°º¤ø,¸¸,ø¤º°»-(¯`·.·´¯)->¯\_(ツ)_/¯0(╯°□°）╯ ︵ ┻━┻ ",
        "Pablo Diego José Francisco de Paula Juan Nepomuceno María de los Remedios Cipriano de la Santísima Trinidad Ruiz y Picasso",
    ],
)
def test_user_id_b64_encode(user_id):
    encoded = user_id_b64_encode(user_id)
    assert isinstance(encoded, str)
    assert re.match("^[A-Za-z0-9_=-]*$", encoded)
    decoded = user_id_b64_decode(encoded)
    assert isinstance(decoded, str)
    assert decoded == user_id


class TestUser:
    @pytest.mark.parametrize(
        ["oidc_userinfo", "expected"],
        [
            ({}, "u123"),
            ({"name": "john"}, "john"),
            ({"email": "u123@domain.test"}, "u123@domain.test"),
            ({"voperson_verified_email": ["u123@domain.test"]}, "u123@domain.test"),
            ({"name": "john", "voperson_verified_email": ["u123@domain.test"]}, "john"),
        ],
    )
    def test_get_name_from_oidc_userinfo(self, oidc_userinfo, expected):
        user = User("u123", info={"oidc_userinfo": oidc_userinfo})
        assert user.get_name() == expected

    def test_roles(self):
        user = User("john")
        assert user.get_roles() == set([])
        user.add_role("trial")
        assert user.get_roles() == {"trial"}
        user.add_roles(["trial", "newby", "student"])
        assert user.get_roles() == {"trial", "newby", "student"}

    def test_default_plan(self):
        user = User("alice")
        assert user.get_default_plan() is None
        user.set_default_plan("premium")
        assert user.get_default_plan() == "premium"

    def test_user_eq(self):
        assert User("alice") == User("alice")
        assert User("alice") != User("bob")
        assert User("a1", info={"fullname": "Alice"}) == User("a1", info={"fullname": "Alice"})
        assert User("a1", info={"fullname": "Alice"}) != User("a1", info={"color": "red"})

        u1 = User("Alice")
        u2 = User("Alice")
        assert u1 == u2
        u1.add_role("admin")
        assert u1 != u2
        u2.add_role("admin")
        assert u1 == u2

        u1 = User("Alice")
        u2 = User("Alice")
        assert u1 == u2
        u1.set_default_plan("premium")
        assert u1 != u2
        u2.set_default_plan("premium")
        assert u1 == u2
