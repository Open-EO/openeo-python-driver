import collections
from typing import NamedTuple, Union, List, Optional


class SarBackscatterArgs(NamedTuple):
    """Arguments for the `sar_backscatter` process."""
    coefficient: Union[str, None] = "gamma0-terrain"
    elevation_model: Union[str, None] = None
    mask: bool = False
    contributing_area: bool = False
    local_incidence_angle: bool = False
    ellipsoid_incidence_angle: bool = False
    noise_removal: bool = True
    # Additional (non-standard) fine-tuning options
    options: dict = {}


class ResolutionMergeArgs(NamedTuple):
    """Arguments for the `resolution_merge` process."""
    method: str = None
    high_resolution_bands: List[str] = False
    low_resolution_bands: List[str] = False
    # Additional (non-standard) fine-tuning options
    options: dict = {}


# Simple type hint alias (for now) for a STAC Asset object (dictionary with at least a "href" item)
# https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md#asset-object
StacAsset = dict


def secretive_repr(*, hide_patterns: Optional[List[str]] = None):
    """
    Build secretive `__repr__`/`__str__` to hide secret fields
    in `typing.NamedTuple` (sub)classes

    Usage example:

        class Credentials(typing.NamedTuple):
            client_id: str
            client_secret: str
            __repr__ = __str__ = secretive_repr()
    """
    # Note: because of how namedtuple implementation details,
    # we can not leverage inheritance or mixins to easily inject
    # alternative `__repr__`/`__str__` implementations.
    # This is an attempt to still keep it simple
    # TODO: also allow usage as decorator?

    if not hide_patterns:
        hide_patterns = ["secret", "pass", "pwd"]

    def is_secret(field_name: str) -> bool:
        return any(p in field_name.lower() for p in hide_patterns)

    def __repr__(self: Union[NamedTuple]):
        fields = [
            (f, "***" if is_secret(f) else getattr(self, f)) for f in self._fields
        ]
        fields = [f"{f}={v!r}" for f, v in fields]
        return f"{self.__class__.__name__}({', '.join(fields)})"

    return __repr__
