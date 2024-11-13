import datetime as dt
from pathlib import Path

from pystac import Asset, Collection, Extent, Item, SpatialExtent, TemporalExtent
import pytest

from openeo_driver.workspace import DiskWorkspace


@pytest.mark.parametrize(
    "merge",
    [
        "subdirectory/collection.json",
        "/subdirectory/collection.json",
        "path/to/subdirectory/collection.json",
        "/path/to/subdirectory/collection.json",
        "collection.json",
    ],
)
def test_disk_workspace(tmp_path, merge):
    source_directory = tmp_path / "src"
    source_directory.mkdir()
    source_file = source_directory / "file"
    source_file.touch()

    subdirectory = Path(merge[1:] if merge.startswith("/") else merge).parent
    target_directory = tmp_path / subdirectory

    workspace = DiskWorkspace(root_directory=tmp_path)
    workspace.import_file(file=source_file, merge=merge)

    assert (target_directory / source_file.name).exists()
    assert source_file.exists()


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
    assert source_file.exists() != remove_original


def test_merge_from_disk_new(tmp_path):
    source_directory = tmp_path / "src"
    source_directory.mkdir()
    source_asset_file = source_directory / "asset.tif"
    source_asset_file.touch()

    new_stac_collection = _collection(source_asset_file)

    target = Path("path") / "to" / "collection.json"

    workspace = DiskWorkspace(root_directory=tmp_path)
    workspace.merge_files(stac_resource=new_stac_collection, target=target)

    workspace_dir = (workspace.root_directory / target).parent

    merged_stac_collection = Collection.from_file(str(workspace_dir / "collection.json"))
    assert merged_stac_collection.validate_all() == 1
    # TODO: check Collection

    assets = [
        asset for item in merged_stac_collection.get_items(recursive=True) for asset in item.get_assets().values()
    ]

    assert len(assets) == 1

    for asset in assets:
        asset.copy(str(tmp_path / Path(asset.href).name))  # downloads the asset file


def _collection(asset_file: Path) -> Collection:
    collection = Collection(
        id="somecollection",
        description="some description",
        extent=Extent(spatial=SpatialExtent([[-180, -90, 180, 90]]), temporal=TemporalExtent([[None, None]])),
    )

    item_id = asset_key = asset_file.name

    item = Item(id=item_id, geometry=None, bbox=None, datetime=dt.datetime.utcnow(), properties={})
    item.add_asset(key=asset_key, asset=Asset(href=str(asset_file)))

    collection.add_item(item)

    return collection


def skip_merge_into_existing():
    raise NotImplementedError
