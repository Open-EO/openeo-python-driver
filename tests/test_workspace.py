import datetime as dt
import shutil
import tempfile
from pathlib import Path

from pystac import Asset, Collection, Extent, Item, SpatialExtent, TemporalExtent, CatalogType
import pytest
from pystac.layout import CustomLayoutStrategy

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
    workspace.import_file(common_path=source_directory, file=source_file, merge=merge)

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
    workspace.import_file(common_path=source_directory, file=source_file, merge=merge, remove_original=remove_original)

    assert (target_directory / source_file.name).exists()
    assert source_file.exists() != remove_original


def test_merge_from_disk_new(tmp_path):
    new_collection = _collection(
        root_path=tmp_path / "src" / "collection", collection_id="collection", asset_filename="asset.tif"
    )

    target = Path("path") / "to" / "collection.json"

    workspace = DiskWorkspace(root_directory=tmp_path)
    merged_collection = workspace.merge(stac_resource=new_collection, target=target)

    assert isinstance(merged_collection, Collection)
    asset_workspace_uris = {
        asset_key: asset.extra_fields["alternate"]["file"]
        for item in merged_collection.get_items()
        for asset_key, asset in item.get_assets().items()
    }
    assert asset_workspace_uris == {
        "asset.tif": f"file:{workspace.root_directory / 'path' / 'to' / 'collection.json_items' / 'asset.tif'}"
    }

    # load it again
    workspace_dir = (workspace.root_directory / target).parent
    exported_collection = Collection.from_file(str(workspace_dir / "collection.json"))
    assert exported_collection.validate_all() == 1
    assert _downloadable_assets(exported_collection) == 1

    assert exported_collection.extent.spatial.bboxes == [[-180, -90, 180, 90]]
    assert exported_collection.extent.temporal.intervals == [[None, None]]

    for item in exported_collection.get_items():
        for asset in item.get_assets().values():
            assert Path(item.get_self_href()).parent == Path(asset.get_absolute_href()).parent


def test_merge_from_disk_into_existing(tmp_path):
    existing_collection = _collection(
        root_path=tmp_path / "src" / "existing_collection",
        collection_id="existing_collection",
        asset_filename="asset1.tif",
        spatial_extent=SpatialExtent([[0, 50, 2, 52]]),
        temporal_extent=TemporalExtent([[dt.datetime.fromisoformat("2024-11-01T00:00:00+00:00"),
                                         dt.datetime.fromisoformat("2024-11-03T00:00:00+00:00")]])
    )
    new_collection = _collection(
        root_path=tmp_path / "src" / "new_collection",
        collection_id="new_collection",
        asset_filename="asset2.tif",
        spatial_extent=SpatialExtent([[1, 51, 3, 53]]),
        temporal_extent=TemporalExtent([[dt.datetime.fromisoformat("2024-11-02T00:00:00+00:00"),
                                         dt.datetime.fromisoformat("2024-11-04T00:00:00+00:00")]])
    )

    target = Path("path") / "to" / "collection.json"

    workspace = DiskWorkspace(root_directory=tmp_path)
    workspace.merge(stac_resource=existing_collection, target=target)
    merged_collection = workspace.merge(stac_resource=new_collection, target=target)

    assert isinstance(merged_collection, Collection)
    asset_workspace_uris = {
        asset_key: asset.extra_fields["alternate"]["file"]
        for item in merged_collection.get_items()
        for asset_key, asset in item.get_assets().items()
    }
    assert asset_workspace_uris == {
        "asset1.tif": f"file:{workspace.root_directory / 'path' / 'to' / 'collection.json_items' / 'asset1.tif'}",
        "asset2.tif": f"file:{workspace.root_directory / 'path' / 'to' / 'collection.json_items' / 'asset2.tif'}",
    }

    # load it again
    workspace_dir = (workspace.root_directory / target).parent
    exported_collection = Collection.from_file(str(workspace_dir / "collection.json"))
    assert exported_collection.validate_all() == 2
    assert _downloadable_assets(exported_collection) == 2

    assert exported_collection.extent.spatial.bboxes == [[0, 50, 3, 53]]
    assert exported_collection.extent.temporal.intervals == [
        [dt.datetime.fromisoformat("2024-11-01T00:00:00+00:00"),
         dt.datetime.fromisoformat("2024-11-04T00:00:00+00:00")]
    ]

    for item in exported_collection.get_items():
        for asset in item.get_assets().values():
            assert Path(item.get_self_href()).parent == Path(asset.get_absolute_href()).parent


def test_adjacent_collections_do_not_have_interfering_items_and_assets(tmp_path):
    workspace = DiskWorkspace(root_directory=tmp_path)

    collection1 = _collection(
        root_path=tmp_path / "src" / "collection1",
        collection_id="collection1",
        asset_filename="asset1.tif",
    )

    collection2 = _collection(
        root_path=tmp_path / "src" / "collection2",
        collection_id="collection2",
        asset_filename="asset1.tif",  # 2 distinct collections can have the same item IDs and assets
    )

    def asset_contents(collection_filename: str):
        assets = [
            asset
            for item in Collection.from_file(str(workspace.root_directory / collection_filename)).get_items()
            for asset in item.get_assets().values()
        ]

        assert len(assets) == 1

        with open(assets[0].get_absolute_href()) as f:
            return f.read()

    workspace.merge(collection1, target=Path("collection1.json"))
    assert asset_contents(collection_filename="collection1.json") == "collection1-asset1.tif-asset1.tif\n"

    # put collection2 next to collection1
    workspace.merge(collection2, target=Path("collection2.json"))
    assert asset_contents(collection_filename="collection2.json") == "collection2-asset1.tif-asset1.tif\n"

    # separate collection files
    assert (workspace.root_directory / "collection1.json").exists()
    assert (workspace.root_directory / "collection2.json").exists()

    # collection2 should not overwrite collection1's items/assets
    assert asset_contents(collection_filename="collection1.json") == "collection1-asset1.tif-asset1.tif\n"


def _collection(
    root_path: Path,
    collection_id: str,
    asset_filename: str,
    spatial_extent: SpatialExtent = SpatialExtent([[-180, -90, 180, 90]]),
    temporal_extent: TemporalExtent = TemporalExtent([[None, None]]),
) -> Collection:
    collection = Collection(
        id=collection_id,
        description=collection_id,
        extent=Extent(spatial_extent, temporal_extent),
    )

    item = Item(id=asset_filename, geometry=None, bbox=None, datetime=dt.datetime.utcnow(), properties={})

    asset_path = root_path / item.id / asset_filename
    asset = Asset(href=asset_path.name)  # relative to item

    item.add_asset(key=asset_filename, asset=asset)
    collection.add_item(item)

    collection.normalize_and_save(root_href=str(root_path), catalog_type=CatalogType.SELF_CONTAINED)

    with open(asset_path, "w") as f:
        f.write(f"{collection_id}-{item.id}-{asset_filename}\n")

    assert collection.validate_all() == 1
    assert _downloadable_assets(collection) == 1

    return collection


def _downloadable_assets(collection: Collection) -> int:
    assets = [asset for item in collection.get_items(recursive=True) for asset in item.get_assets().values()]

    for asset in assets:
        with tempfile.NamedTemporaryFile(mode="wb") as temp_file:
            shutil.copy(asset.get_absolute_href(), temp_file.name)  # "download" the asset without altering its href

    return len(assets)
