from typing import List, NamedTuple, Optional, Tuple

from openeo_driver.utils import extract_namedtuple_fields_from_dict


class OidcProvider(NamedTuple):
    """OIDC provider metadata"""
    id: str
    issuer: str
    title: str
    scopes: List[str] = ["openid"]
    description: Optional[str] = None
    default_clients: Optional[List[dict]] = None

    # Optional: (client_id, client_secret) tuple to use with OIDC client credentials grant auth.
    service_account: Optional[Tuple[str, str]] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'OidcProvider':
        d = extract_namedtuple_fields_from_dict(d, OidcProvider)
        return cls(**d)

    def export_for_api(self) -> dict:
        """Export for publishing under `GET /credentials/oidc`"""
        data = {
            "id": self.id,
            "issuer": self.issuer,
            "title": self.title,
            "scopes": self.scopes,
        }
        if self.description:
            data["description"] = self.description
        if self.default_clients:
            data["default_clients"] = self.default_clients
        return data

    @property
    def discovery_url(self):
        return self.issuer.rstrip("/") + '/.well-known/openid-configuration'

    def get_issuer(self) -> str:
        """Get normalized version of issuer (for comparison/mapping situations)"""
        return normalize_issuer_url(self.issuer)


def normalize_issuer_url(url: str) -> str:
    return url.rstrip("/").lower()
