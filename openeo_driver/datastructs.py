from typing import NamedTuple, Union, List


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
