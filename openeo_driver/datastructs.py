from typing import NamedTuple, Union, List


class SarBackscatterArgs(NamedTuple):
    """Arguments for the `sar_backscatter` process."""
    orthorectify: bool = True
    elevation_model: Union[str, None] = None
    rtc: bool = True
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