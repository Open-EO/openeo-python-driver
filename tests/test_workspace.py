import datetime as dt
import shutil
import tempfile
from pathlib import Path

from pystac import Asset, Collection, Extent, Item, SpatialExtent, TemporalExtent, CatalogType
import pytest
from pystac.layout import CustomLayoutStrategy

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
    new_collection = _collection(
        root_path=tmp_path / "src" / "collection", collection_id="collection", asset_filename="asset.tif"
    )

    target = Path("path") / "to" / "collection.json"

    workspace = DiskWorkspace(root_directory=tmp_path)
    merged_collection = workspace.merge_files(stac_resource=new_collection, target=target)

    assert isinstance(merged_collection, Collection)
    asset_workspace_uris = {
        asset_key: asset.extra_fields["alternate"]["file"]
        for item in merged_collection.get_items()
        for asset_key, asset in item.get_assets().items()
    }
    assert asset_workspace_uris == {
        "asset.tif": f"file:{workspace.root_directory / 'path' / 'to' / 'asset.tif' / 'asset.tif'}"
    }

    # load it again
    workspace_dir = (workspace.root_directory / target).parent
    exported_collection = Collection.from_file(str(workspace_dir / "collection.json"))
    assert exported_collection.validate_all() == 1
    assert _downloadable_assets(exported_collection) == 1

    # TODO: check Collection

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
    workspace.merge_files(stac_resource=existing_collection, target=target)
    merged_collection = workspace.merge_files(stac_resource=new_collection, target=target)

    assert isinstance(merged_collection, Collection)
    asset_workspace_uris = {
        asset_key: asset.extra_fields["alternate"]["file"]
        for item in merged_collection.get_items()
        for asset_key, asset in item.get_assets().items()
    }
    assert asset_workspace_uris == {
        "asset1.tif": f"file:{workspace.root_directory / 'path' / 'to' / 'asset1.tif' / 'asset1.tif'}",
        "asset2.tif": f"file:{workspace.root_directory / 'path' / 'to' / 'asset2.tif' / 'asset2.tif'}",
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

    collection.normalize_hrefs(root_href=str(root_path))
    collection.save(CatalogType.SELF_CONTAINED)

    with open(asset_path, "w") as f:
        f.write(f"{asset_filename}\n")

    assert collection.validate_all() == 1
    assert _downloadable_assets(collection) == 1

    return collection


def _downloadable_assets(collection: Collection) -> int:
    assets = [asset for item in collection.get_items(recursive=True) for asset in item.get_assets().values()]

    for asset in assets:
        with tempfile.NamedTemporaryFile(mode="wb") as temp_file:
            shutil.copy(asset.get_absolute_href(), temp_file.name)  # "download" the asset without altering its href

    return len(assets)


def test_create_and_export_collection(tmp_path):
    # tmp_path = Path("/tmp/test_create_and_export_collection")

    # write collection1
    collection1 = _collection(
        root_path=tmp_path / "src" / "collection1", collection_id="collection1", asset_filename="asset1.tif"
    )

    # export collection1
    exported_collection = collection1.full_copy()
    merge = tmp_path / "dst" / "merged-collection.json"

    def collection_func(_: Collection, parent_dir: str, is_root: bool) -> str:
        assert is_root
        return str(Path(parent_dir) / merge.name)

    layout_strategy = CustomLayoutStrategy(collection_func=collection_func)
    exported_collection.normalize_hrefs(root_href=str(merge.parent), strategy=layout_strategy)

    def replace_asset_href(asset_key: str, asset: Asset) -> Asset:
        asset.extra_fields["_original_absolute_href"] = asset.get_absolute_href()
        asset.href = asset_key  # asset key matches the asset filename, becomes the relative path
        return asset

    exported_collection = exported_collection.map_assets(replace_asset_href)

    exported_collection.save(CatalogType.SELF_CONTAINED)
    assert exported_collection.validate_all() == 1

    for item in exported_collection.get_items():
        for asset in item.get_assets().values():
            shutil.copy(asset.extra_fields["_original_absolute_href"], Path(item.get_self_href()).parent)

    # write collection2
    collection2 = _collection(
        root_path=tmp_path / "src" / "collection2", collection_id="collection2", asset_filename="asset2.tif"
    )

    # merge collection2 into existing
    existing_collection = Collection.from_file(str(merge))
    assert existing_collection.validate_all() == 1

    new_collection = collection2.full_copy()

    # "merge" some properties
    existing_collection.extent = new_collection.extent.clone()
    existing_collection.description = f"{existing_collection.description} + {new_collection.description}"

    # new_collection.make_all_asset_hrefs_absolute()
    new_collection = new_collection.map_assets(replace_asset_href)

    # add new items to existing
    for new_item in new_collection.get_items():
        new_item.clear_links()  # sever ties with previous collection
        existing_collection.add_item(new_item)

    existing_collection.normalize_hrefs(root_href=str(merge.parent), strategy=layout_strategy)
    existing_collection.save(CatalogType.SELF_CONTAINED)
    assert existing_collection.validate_all() == 2

    for item in new_collection.get_items():
        for asset in item.get_assets().values():
            shutil.copy(asset.extra_fields["_original_absolute_href"], Path(item.get_self_href()).parent)

    merged_collection = Collection.from_file(str(merge))
    assert merged_collection.validate_all() == 2
    assert merged_collection.id == "collection1"
    assert merged_collection.description == "collection1 + collection2"

    for item in merged_collection.get_items():
        for asset in item.get_assets().values():
            assert Path(item.get_self_href()).parent == Path(asset.get_absolute_href()).parent

    assert _downloadable_assets(merged_collection) == 2
