import abc
import logging
import os.path
import shutil
from pathlib import Path, PurePath
from typing import Optional, Union
from urllib.parse import urlparse

from openeo_driver.utils import remove_slash_prefix
from pystac import Asset, Collection, STACObject, SpatialExtent, TemporalExtent, Item, RelType
from pystac.catalog import CatalogType
from pystac.layout import HrefLayoutStrategy, CustomLayoutStrategy

_log = logging.getLogger(__name__)


class Workspace(abc.ABC):
    @abc.abstractmethod
    def import_file(self, common_path: Union[str, Path], file: Path, merge: str, remove_original: bool = False) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def import_object(
        self, common_path: Union[str, Path], s3_uri: str, merge: str, remove_original: bool = False
    ) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def merge(self, stac_resource: STACObject, target: PurePath, remove_original: bool = False) -> STACObject:
        """
        Merges a STAC resource, its children and their assets into this workspace at the given path,
        possibly removing the original assets.
        :param stac_resource: a STAC resource, typically a Collection
        :param target: a path identifier to a STAC resource to merge the given STAC resource into
        :param remove_original: remove the original assets?
        """
        raise NotImplementedError


class DiskWorkspace(Workspace):

    def __init__(self, root_directory: Path):
        self.root_directory = root_directory

    def import_file(self, common_path: Union[str, Path], file: Path, merge: str, remove_original: bool = False) -> str:
        subdirectory = remove_slash_prefix(os.path.normpath(merge))
        if file.is_absolute():
            file_relative = file.relative_to(common_path)
        else:
            file_relative = file
        file_absolute = Path(common_path) / file_relative
        target_directory = self.root_directory / subdirectory / file_relative.parent
        target_directory.relative_to(self.root_directory)  # assert target_directory is in root_directory

        target_directory.mkdir(parents=True, exist_ok=True)

        operation = shutil.move if remove_original else shutil.copy
        operation(str(file_absolute), str(target_directory))

        _log.debug(f"{'moved' if remove_original else 'copied'} {file_absolute} to {target_directory}")
        return f"file:{target_directory / file.name}"

    def import_object(self, common_path: str, s3_uri: str, merge: str, remove_original: bool = False):
        raise NotImplementedError(f"importing objects is not supported yet")

    def merge(self, stac_resource: STACObject, target: PurePath, remove_original: bool = False) -> STACObject:
        stac_resource = stac_resource.full_copy()

        target = self.root_directory / os.path.normpath(target).lstrip("/")
        target.relative_to(self.root_directory)  # assert target_directory is in root_directory

        file_operation = shutil.move if remove_original else shutil.copy

        if isinstance(stac_resource, Collection):
            new_collection = stac_resource

            existing_collection = None
            try:
                existing_collection = Collection.from_file(str(target))
            except FileNotFoundError:
                pass  # nothing to merge into

            def href_layout_strategy() -> HrefLayoutStrategy:
                def collection_func(_: Collection, parent_dir: str, is_root: bool) -> str:
                    if not is_root:
                        raise NotImplementedError("nested collections")
                    # make the collection file end up at $target, not at $target/collection.json
                    return str(Path(parent_dir) / target.name)

                def item_func(item: Item, parent_dir: str) -> str:
                    # item ID can be a relative_asset_path but does not have to be
                    unique_item_filename = item.id.replace("/", "_")
                    # prevent items/assets of 2 adjacent Collection documents from interfering with each other;
                    # unlike an object storage object, a Collection file cannot act as a parent "directory" as well
                    return f"{parent_dir}/{target.name}_items/{unique_item_filename}.json"

                return CustomLayoutStrategy(collection_func=collection_func, item_func=item_func)

            def replace_asset_href(asset: Asset, src_collection_path: Path) -> Asset:
                # pystac will handle STAC but not underlying assets; set asset hrefs up front
                asset_uri_parts = urlparse(asset.get_absolute_href())
                if asset_uri_parts.scheme not in ["", "file"]:  # TODO: convenient place; move elsewhere?
                    raise NotImplementedError(f"only importing files on disk is supported, found: {asset.href}")

                absolute_asset_path = Path(asset_uri_parts.path)
                # TODO: crummy way to export assets after STAC Collection has been written to disk with new asset hrefs;
                #  it ends up in the asset metadata on disk
                asset.extra_fields["_original_absolute_path"] = str(absolute_asset_path)
                relative_asset_path = absolute_asset_path.relative_to(src_collection_path.parent)
                asset.href = str(relative_asset_path)  # relative to item document
                return asset

            new_collection_path = Path(new_collection.get_self_href())

            if not existing_collection:
                new_collection.normalize_hrefs(root_href=str(target.parent), strategy=href_layout_strategy())
                new_collection = new_collection.map_assets(
                    lambda _, asset: replace_asset_href(asset, src_collection_path=new_collection_path)
                )
                new_collection.save(CatalogType.SELF_CONTAINED)

                for new_item in new_collection.get_items():
                    for asset in new_item.get_assets().values():
                        relative_asset_path = asset.href
                        asset_parent_dir = (
                            Path(new_collection.get_self_href()).parent / f"{target.name}_items" / relative_asset_path
                        ).parent
                        asset_parent_dir.mkdir(parents=True, exist_ok=True)  # asset might not end up next to item
                        file_operation(asset.extra_fields["_original_absolute_path"], str(asset_parent_dir))
            else:
                merged_collection = _merge_collection_metadata(existing_collection, new_collection)
                new_collection = new_collection.map_assets(
                    lambda _, asset: replace_asset_href(asset, src_collection_path=new_collection_path)
                )

                for new_item in new_collection.get_items():
                    new_item.clear_links()  # sever ties with previous collection
                    merged_collection.add_item(new_item, strategy=href_layout_strategy())

                merged_collection.normalize_hrefs(root_href=str(target.parent), strategy=href_layout_strategy())
                merged_collection.save(CatalogType.SELF_CONTAINED)

                for new_item in new_collection.get_items():
                    for asset in new_item.get_assets().values():
                        relative_asset_path = asset.href
                        asset_parent_dir = (
                            Path(merged_collection.get_self_href()).parent
                            / f"{target.name}_items"
                            / relative_asset_path
                        ).parent
                        asset_parent_dir.mkdir(parents=True, exist_ok=True)
                        file_operation(asset.extra_fields["_original_absolute_path"], str(asset_parent_dir))

            for item in new_collection.get_items():
                for asset in item.assets.values():
                    workspace_uri = f"file:{Path(item.get_self_href()).parent / Path(asset.href).name}"
                    asset.extra_fields["alternate"] = {"file": workspace_uri}

            return new_collection
        else:
            raise NotImplementedError(stac_resource)


def _merge_collection_metadata(existing_collection: Collection, new_collection: Collection) -> Collection:
    existing_collection.extent.spatial = _merge_spatial_extents(
        existing_collection.extent.spatial, new_collection.extent.spatial
    )

    existing_collection.extent.temporal = _merge_temporal_extents(
        existing_collection.extent.temporal, new_collection.extent.temporal
    )

    _merge_derived_from_links(existing_collection, new_collection)

    # TODO: retains existing collection ID and description; is this right?
    # TODO: merge additional metadata?

    return existing_collection


def _merge_spatial_extents(a: SpatialExtent, b: SpatialExtent) -> SpatialExtent:
    overall_bbox_a, *sub_bboxes_a = a.bboxes
    overall_bbox_b, *sub_bboxes_b = b.bboxes

    merged_overall_bbox = [
        min(overall_bbox_a[0], overall_bbox_b[0]),
        min(overall_bbox_a[1], overall_bbox_b[1]),
        max(overall_bbox_a[2], overall_bbox_b[2]),
        max(overall_bbox_a[3], overall_bbox_b[3])
    ]

    merged_sub_bboxes = sub_bboxes_a + sub_bboxes_b

    merged_spatial_extent = SpatialExtent([merged_overall_bbox])
    if merged_sub_bboxes:
        merged_spatial_extent.bboxes.append(merged_sub_bboxes)

    return merged_spatial_extent


def _merge_temporal_extents(a: TemporalExtent, b: TemporalExtent) -> TemporalExtent:
    overall_interval_a, *sub_intervals_a = a.intervals
    overall_interval_b, *sub_intervals_b = b.intervals

    def min_time(t1: Optional[str], t2: Optional[str]) -> Optional[str]:
        if t1 is None or t2 is None:
            return None

        return min(t1, t2)

    def max_time(t1: Optional[str], t2: Optional[str]) -> Optional[str]:
        if t1 is None or t2 is None:
            return None

        return max(t1, t2)

    merged_overall_interval = [
        min_time(overall_interval_a[0], overall_interval_b[0]),
        max_time(overall_interval_a[1], overall_interval_b[1])
    ]

    merged_sub_intervals = sub_intervals_a + sub_intervals_b

    merged_temporal_extent = TemporalExtent([merged_overall_interval])
    if merged_sub_intervals:
        merged_temporal_extent.intervals.append(merged_sub_intervals)

    return merged_temporal_extent


def _merge_derived_from_links(existing_collection: Collection, new_collection: Collection):
    for new_link in new_collection.get_links(rel=RelType.DERIVED_FROM):
        if not any(
            existing_link.href == new_link.href
            for existing_link in existing_collection.get_links(rel=RelType.DERIVED_FROM)
        ):
            existing_collection.add_link(new_link)
