import abc
import logging
import os.path
import shutil
from pathlib import Path, PurePath
from typing import Optional

import pystac
from pystac import Collection, STACObject, SpatialExtent, TemporalExtent
from pystac.catalog import CatalogType
from pystac.layout import TemplateLayoutStrategy, HrefLayoutStrategy, CustomLayoutStrategy

_log = logging.getLogger(__name__)


class Workspace(abc.ABC):
    @abc.abstractmethod
    def import_file(self, file: Path, merge: str, remove_original: bool = False) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def import_object(self, s3_uri: str, merge: str, remove_original: bool = False):
        raise NotImplementedError

    @abc.abstractmethod
    def merge_files(self, stac_resource: STACObject, target: PurePath, remove_original: bool = False) -> STACObject:
        # TODO: is a PurePath object fine as an abstraction?
        raise NotImplementedError


class DiskWorkspace(Workspace):

    def __init__(self, root_directory: Path):
        self.root_directory = root_directory

    def import_file(self, file: Path, merge: str, remove_original: bool = False) -> str:
        merge = os.path.normpath(merge)

        # merge points to a file with the STAC Collection; all STAC documents and assets end up in its parent directory
        subdirectory = Path(merge[1:] if merge.startswith("/") else merge).parent
        target_directory = self.root_directory / subdirectory
        target_directory.relative_to(self.root_directory)  # assert target_directory is in root_directory

        target_directory.mkdir(parents=True, exist_ok=True)

        operation = shutil.move if remove_original else shutil.copy
        operation(str(file), str(target_directory))

        _log.debug(f"{'moved' if remove_original else 'copied'} {file.absolute()} to {target_directory}")
        return f"file:{target_directory / file.name}"

    def import_object(self, s3_uri: str, merge: str, remove_original: bool = False):
        raise NotImplementedError(f"importing objects is not supported yet")

    def merge_files(self, stac_resource: STACObject, target: PurePath, remove_original: bool = False) -> STACObject:
        stac_resource = stac_resource.clone()

        target = os.path.normpath(target)
        target = Path(target[1:] if target.startswith("/") else target)
        target = self.root_directory / target
        target.relative_to(self.root_directory)  # assert target_directory is in root_directory

        target.parent.mkdir(parents=True, exist_ok=True)
        file_operation = shutil.move if remove_original else shutil.copy

        if isinstance(stac_resource, Collection):
            new_collection = stac_resource

            existing_collection = None
            try:
                existing_collection = pystac.Collection.from_file(str(target))
            except FileNotFoundError:
                pass  # nothing to merge into

            merged_collection = self._merge_collections(existing_collection, new_collection)

            def layout_strategy() -> HrefLayoutStrategy:
                def collection_file_at_merge(col: Collection, parent_dir: str, is_root: bool) -> str:
                    if not is_root:
                        raise ValueError("nested collections are not supported")
                    return str(target)

                return CustomLayoutStrategy(collection_func=collection_file_at_merge)

            # TODO: write to a tempdir, then copy/move everything to $merge?
            merged_collection.normalize_hrefs(root_href=str(target), strategy=layout_strategy())

            def with_href_relative_to_item(asset: pystac.Asset):
                # TODO: is crummy way to export assets after STAC Collection has been written to disk with new asset hrefs
                asset.extra_fields["_original_absolute_href"] = asset.get_absolute_href()
                filename = Path(asset.href).name
                asset.href = filename
                return asset

            # save STAC with proper asset hrefs
            merged_collection = merged_collection.map_assets(lambda _, asset: with_href_relative_to_item(asset))
            merged_collection.save(catalog_type=CatalogType.SELF_CONTAINED)

            # copy assets to workspace on disk
            for item in merged_collection.get_items(recursive=True):
                for asset in item.assets.values():
                    file_operation(asset.extra_fields["_original_absolute_href"], Path(item.get_self_href()).parent)
                    workspace_uri = f"file:{Path(item.get_self_href()).parent / Path(asset.href).name}"
                    asset.extra_fields["alternate"] = {"file": workspace_uri}
            return merged_collection
        else:
            raise NotImplementedError(stac_resource)

    def _merge_collections(self, existing_collection: Optional[Collection], new_collection: Collection) -> Collection:
        if existing_collection:
            existing_collection.extent.spatial = self._merge_spatial_extents(
                existing_collection.extent.spatial,
                new_collection.extent.spatial
            )

            existing_collection.extent.temporal = self._merge_temporal_extents(
                existing_collection.extent.temporal,
                new_collection.extent.temporal
            )

            for new_item in new_collection.get_items(recursive=True):
                if existing_collection.get_item(new_item.id, recursive=True):
                    raise ValueError(f"item {new_item.id} is already in collection {existing_collection.id}")

                existing_collection.add_item(new_item, strategy=TemplateLayoutStrategy(item_template="${id}.json"))
                return existing_collection

        return new_collection

    def _merge_spatial_extents(self, a: SpatialExtent, b: SpatialExtent) -> SpatialExtent:
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

    def _merge_temporal_extents(self, a: TemporalExtent, b: TemporalExtent) -> TemporalExtent:
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
