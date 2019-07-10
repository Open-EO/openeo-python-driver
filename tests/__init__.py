import os
from pathlib import Path

os.environ["DRIVER_IMPLEMENTATION_PACKAGE"] = "dummy_impl"



def get_test_resource(relative_path):
    dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return str(dir / relative_path)


def load_json_resource(relative_path):
    import json
    with open(get_test_resource(relative_path), 'r+') as f:
        return json.load(f)
