from typing import NamedTuple, Union, List


class SarBackscatterArgs(NamedTuple):
    """Arguments for the `sar_backscatter` process."""
    backscatter_coefficient: str = "gamma0"
    orthorectify: bool = False
    elevation_model: Union[str, None] = None
    # Additional (non-standard) fine-tuning options
    options: dict = {}

class ResolutionMergeArgs(NamedTuple):
    """Arguments for the `resolution_merge` process."""
    method: str = None
    high_resolution_bands: List[str] = False
    low_resolution_bands: List[str] = False
    # Additional (non-standard) fine-tuning options
    options: dict = {}