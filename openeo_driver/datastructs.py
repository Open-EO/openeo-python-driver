from typing import NamedTuple, Union


class SarBackscatterArgs(NamedTuple):
    """Arguments for the `sar_backscatter` process."""
    backscatter_coefficient: str = "gamma0"
    orthorectify: bool = False
    elevation_model: Union[str, None] = None
    # Additional (non-standard) fine-tuning options
    options: dict = {}
