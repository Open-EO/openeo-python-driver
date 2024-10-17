import pytest

from openeo_driver.workspace import DiskWorkspace


@pytest.mark.parametrize("merge", [
    "subdirectory",
    "/subdirectory",
    "path/to/subdirectory",
    "/path/to/subdirectory",
    ".",
])
def test_disk_workspace(tmp_path, merge):
    source_directory = tmp_path / "src"
    source_directory.mkdir()
    source_file = source_directory / "file"
    source_file.touch()

    subdirectory = merge[1:] if merge.startswith("/") else merge
    target_directory = tmp_path / subdirectory

    workspace = DiskWorkspace(root_directory=tmp_path)
    workspace.import_file(file=source_file, merge=merge)

    assert (target_directory / source_file.name).exists()
    assert (source_directory / source_file.name).exists()


@pytest.mark.parametrize("remove_original", [False, True])
def test_disk_workspace_remove_original(tmp_path, remove_original):
    source_directory = tmp_path / "src"
    source_directory.mkdir()
    source_file = source_directory / "file"
    source_file.touch()

    merge = "."
    target_directory = tmp_path / merge

    workspace = DiskWorkspace(root_directory=tmp_path)
    workspace.import_file(source_file, merge=merge, remove_original=remove_original)

    assert (target_directory / source_file.name).exists()
    assert (source_directory / source_file.name).exists() != remove_original
