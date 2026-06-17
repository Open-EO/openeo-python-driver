from typing import Union, Set, Optional, Iterable

import base64


class User:
    __slots__ = ("user_id", "info", "internal_auth_data", "_roles", "_default_plan")
    # TODO more fields
    def __init__(
        self,
        user_id: str,
        info: Optional[dict] = None,
        internal_auth_data: Optional[dict] = None,
    ):
        self.user_id = user_id
        self.info = info
        self.internal_auth_data = internal_auth_data
        self._roles: Set[str] = set([])
        self._default_plan: Optional[str] = None

    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.user_id, self.info)

    def __str__(self):
        return self.user_id

    def __eq__(self, other):
        return type(self) is type(other) and all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

    def get_name(self):
        """Best effort name extraction"""
        if isinstance(self.info, dict):
            if "oidc_userinfo" in self.info:
                oidc_userinfo = self.info["oidc_userinfo"]
                if "name" in oidc_userinfo:
                    return oidc_userinfo["name"]
                if (
                    "voperson_verified_email" in oidc_userinfo
                    and len(oidc_userinfo["voperson_verified_email"]) > 0
                ):
                    return oidc_userinfo["voperson_verified_email"][0]
                if "email" in oidc_userinfo:
                    return oidc_userinfo["email"]
        # Fallback
        return self.user_id

    def add_role(self, role: str):
        self._roles.add(role)

    def add_roles(self, roles: Iterable[str]):
        self._roles.update(roles)

    def get_roles(self) -> Set[str]:
        return self._roles

    def set_default_plan(self, plan: str):
        self._default_plan = plan

    def get_default_plan(self) -> Union[str, None]:
        return self._default_plan


def user_id_b64_encode(user_id: str) -> str:
    """Encode a user id in way that is safe to use in urls"""
    return base64.urlsafe_b64encode(user_id.encode("utf8")).decode("ascii")


def user_id_b64_decode(encoded: str) -> str:
    """Decode a user id that was encoded with user_id_b64_encode"""
    return base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
