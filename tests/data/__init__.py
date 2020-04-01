import json
from collections import Callable

from pathlib import Path


def get_path(filename: str) -> Path:
    """Get absolute pat to a test data file"""
    return Path(__file__).parent / filename


def load_json(filename: str, preprocess: Callable = None) -> dict:
    """Parse data from JSON file"""
    with get_path(filename).open("r") as f:
        data = f.read()
        if preprocess:
            data = preprocess(data)
        return json.loads(data)


def json_normalize(data: dict) -> dict:
    """
    Normalize python structures in nested dict to JSON compatible ones.
    For example: convert tuples to lists
    """
    return json.loads(json.dumps(data))
