from typing import Union

import base64


class User:
    # TODO more fields
    def __init__(
        self, user_id: str, info: dict = None, internal_auth_data: dict = None
    ):
        self.user_id = user_id
        self.info = info
        self.internal_auth_data = internal_auth_data

    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.user_id, self.info)

    def __str__(self):
        return self.user_id

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

    def get_default_plan(self) -> Union[str, None]:
        # TODO: "default_plan" field? see https://github.com/Open-EO/openeo-api/issues/425
        if self.info:
            return self.info.get("default_plan")


def user_id_b64_encode(user_id: str) -> str:
    """Encode a user id in way that is safe to use in urls"""
    return base64.urlsafe_b64encode(user_id.encode("utf8")).decode("ascii")


def user_id_b64_decode(encoded: str) -> str:
    """Decode a user id that was encoded with user_id_b64_encode"""
    return base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
