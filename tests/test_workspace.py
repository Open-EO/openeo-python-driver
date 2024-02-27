import os
from pathlib import Path

import pytest

from openeo_driver.workspace import DiskWorkspace


@pytest.mark.parametrize("merge", [
    "subdirectory",
    "/subdirectory",
    "path/to/subdirectory",
    "/path/to/subdirectory",
])
def test_disk_workspace(tmp_path, merge):
    workspace = DiskWorkspace(root_directory=tmp_path)

    subdirectory = merge[1:] if merge.startswith("/") else merge
    target_directory = tmp_path / subdirectory

    input_file = Path(__file__)
    workspace.import_file(file=input_file, merge=merge)

    assert "test_workspace.py" in os.listdir(target_directory)
