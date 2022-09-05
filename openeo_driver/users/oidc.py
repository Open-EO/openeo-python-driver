from typing import List, NamedTuple

from openeo_driver.utils import extract_namedtuple_fields_from_dict


class OidcProvider(NamedTuple):
    """OIDC provider metadata"""
    id: str
    issuer: str
    title: str
    scopes: List[str] = ["openid"]
    description: str = None
    default_clients: List[dict] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'OidcProvider':
        d = extract_namedtuple_fields_from_dict(d, OidcProvider)
        return cls(**d)

    def prepare_for_json(self) -> dict:
        d = self._asdict()
        for omit_when_none in ["description", "default_clients"]:
            if d[omit_when_none] is None:
                d.pop(omit_when_none)
        return d

    @property
    def discovery_url(self):
        return self.issuer.rstrip("/") + '/.well-known/openid-configuration'

    def get_issuer(self) -> str:
        """Get normalized version of issuer (for comparison/mapping situations)"""
        return normalize_issuer_url(self.issuer)


def normalize_issuer_url(url: str) -> str:
    return url.rstrip("/").lower()
