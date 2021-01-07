from pathlib import Path
from typing import Union

from openeo_driver.utils import read_json

SPECS_ROOT = Path(__file__).parent


def read_spec(path: Union[str, Path]) -> dict:
    """Read specification JSON file (given by relative path)"""
    return read_json(SPECS_ROOT / path)
