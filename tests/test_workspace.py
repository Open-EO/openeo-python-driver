import datetime as dt
from pathlib import Path
from typing import List

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

    asset_file = source_directory / "asset.tif"
    asset_file.touch()

    new_stac_collection = _collection("collection", [asset_file])

    target = Path("path") / "to" / "collection.json"

    workspace = DiskWorkspace(root_directory=tmp_path)
    workspace.merge_files(stac_resource=new_stac_collection, target=target)

    workspace_dir = (workspace.root_directory / target).parent

    merged_stac_collection = Collection.from_file(str(workspace_dir / "collection.json"))
    assert merged_stac_collection.validate_all() == 1
    # TODO: check Collection

    assert _download_assets(merged_stac_collection, target_dir=tmp_path) == 1


def test_merge_from_disk_into_existing(tmp_path):
    source_directory = tmp_path / "src"
    source_directory.mkdir()

    asset_file1 = source_directory / "asset1.tif"
    asset_file1.touch()

    asset_file2 = source_directory / "asset2.tif"
    asset_file2.touch()

    existing_stac_collection = _collection(
        "existing_collection",
        [asset_file1],
        spatial_extent=SpatialExtent([[0, 50, 2, 52]]),
        temporal_extent=TemporalExtent([[dt.datetime.fromisoformat("2024-11-01T00:00:00+00:00"),
                                         dt.datetime.fromisoformat("2024-11-03T00:00:00+00:00")]])
    )
    new_stac_collection = _collection(
        "new_collection",
        [asset_file2],
        spatial_extent=SpatialExtent([[1, 51, 3, 53]]),
        temporal_extent=TemporalExtent([[dt.datetime.fromisoformat("2024-11-02T00:00:00+00:00"),
                                         dt.datetime.fromisoformat("2024-11-04T00:00:00+00:00")]])
    )

    target = Path("path") / "to" / "collection.json"

    workspace = DiskWorkspace(root_directory=tmp_path)
    workspace.merge_files(stac_resource=existing_stac_collection, target=target)
    workspace.merge_files(stac_resource=new_stac_collection, target=target)

    workspace_dir = (workspace.root_directory / target).parent

    merged_stac_collection = Collection.from_file(str(workspace_dir / "collection.json"))
    assert merged_stac_collection.validate_all() == 2

    assert _download_assets(merged_stac_collection, target_dir=tmp_path) == 2

    assert merged_stac_collection.extent.spatial.bboxes == [[0, 50, 3, 53]]
    assert merged_stac_collection.extent.temporal.intervals == [
        [dt.datetime.fromisoformat("2024-11-01T00:00:00+00:00"),
         dt.datetime.fromisoformat("2024-11-04T00:00:00+00:00")]
    ]


def _collection(collection_id: str,
                asset_files: List[Path],
                spatial_extent: SpatialExtent = SpatialExtent([[-180, -90, 180, 90]]),
                temporal_extent: TemporalExtent = TemporalExtent([[None, None]])) -> Collection:
    collection = Collection(
        id=collection_id,
        description=collection_id,
        extent=Extent(spatial=spatial_extent, temporal=temporal_extent),
    )

    for asset_file in asset_files:
        item_id = asset_key = asset_file.name

        item = Item(id=item_id, geometry=None, bbox=None, datetime=dt.datetime.utcnow(), properties={})
        item.add_asset(key=asset_key, asset=Asset(href=str(asset_file)))

        collection.add_item(item)

    return collection


def _download_assets(collection: Collection, target_dir: Path) -> int:
    assets = [asset for item in collection.get_items(recursive=True) for asset in item.get_assets().values()]

    for asset in assets:
        asset.copy(str(target_dir / Path(asset.href).name))  # downloads the asset file

    return len(assets)
