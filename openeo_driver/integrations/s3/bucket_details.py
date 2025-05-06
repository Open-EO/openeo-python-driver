from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from openeo_driver.config import get_backend_config


_BUCKET_TYPE_UNKNOWN = "UNKNOWN"
_BUCKET_TYPE_WORKSPACE = "WORKSPACE"
_REGION_UNKNOWN = "REGION_UNKNOWN"


@dataclass(frozen=True)
class BucketDetails:
    name: str
    region: str = _REGION_UNKNOWN
    type: str = _BUCKET_TYPE_UNKNOWN
    type_id: Optional[str] = None

    @classmethod
    def from_name(cls, bucket_name: str) -> BucketDetails:
        for ws_name, ws_details in get_backend_config().workspaces.items():
            #TODO: Move ObjectStorageWorkspace to openeo_driver?
            if hasattr(ws_details, 'bucket') and hasattr(ws_details, 'region'):
                if ws_details.bucket == bucket_name:
                    return cls(
                        name=bucket_name,
                        region=ws_details.region,
                        type=_BUCKET_TYPE_WORKSPACE,
                        type_id=ws_name
                    )

        return cls(
            name=bucket_name,
        )


def is_workspace_bucket(bucket_details: BucketDetails) -> bool:
    return bucket_details.type == _BUCKET_TYPE_WORKSPACE
