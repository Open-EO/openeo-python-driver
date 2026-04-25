"""
Load parameter extraction logic.

Extracted from openeo_driver/ProcessGraphDeserializer.py.
"""
import logging
import math
from typing import List, Optional, Union

from pyproj import CRS
from pyproj.exceptions import CRSError

from openeo_driver.backend import LoadParameters
from openeo_driver.dry_run import SourceId
from openeo_driver.errors import CollectionNotFoundException
from openeo_driver.processgraph.registry import ENV_MAX_BUFFER, ENV_SOURCE_CONSTRAINTS
from openeo_driver.util.geometry import BoundingBox, spatial_extent_union
from openeo_driver.utils import EvalEnv

_log = logging.getLogger(__name__)


def _collection_crs(collection_id: str, env: EvalEnv) -> Union[None, str, int]:
    """
    Get spatial reference system from of the data in openEO collection metadata
    (based on datacube STAC extension)
    """
    try:
        metadata = env.backend_implementation.catalog.get_collection_metadata(collection_id)
    except CollectionNotFoundException:
        return None
    crs = metadata.get("cube:dimensions", {}).get("x", {}).get("reference_system", None)
    return crs


def _collection_resolution(collection_id: str, env: EvalEnv) -> Optional[List[int]]:
    try:
        metadata = env.backend_implementation.catalog.get_collection_metadata(collection_id)
    except CollectionNotFoundException:
        return None
    x = metadata.get('cube:dimensions', {}).get('x', {})
    y = metadata.get('cube:dimensions', {}).get('y', {})
    if "step" in x and "step" in y:
        return [x['step'], y['step']]
    else:
        return None


def _align_extent(extent: dict, collection_id: str, env: EvalEnv, target_resolution=None) -> dict:
    try:
        metadata = env.backend_implementation.catalog.get_collection_metadata(collection_id)
    except CollectionNotFoundException:
        metadata = None

    # TODO #275 eliminate this VITO specific handling?
    if metadata is None or not metadata.get("_vito", {}).get("data_source", {}).get("realign", True):
        return extent

    crs = _collection_crs(collection_id=collection_id, env=env)
    collection_resolution = _collection_resolution(collection_id, env)
    isUTM = crs == "AUTO:42001" or "Auto42001" in str(crs)

    x = metadata.get('cube:dimensions', {}).get('x', {})
    y = metadata.get('cube:dimensions', {}).get('y', {})

    if target_resolution is None and collection_resolution is None:
        return extent

    if (
        crs == 4326
        and extent.get("crs", "") == "EPSG:4326"
        and "extent" in x
        and "extent" in y
        and (target_resolution is None or target_resolution == collection_resolution)
    ):
        target_resolution = collection_resolution

        def align(v, dimension, rounding, resolution):
            range = dimension.get('extent', [])
            if v < range[0]:
                v = range[0]
            elif v > range[1]:
                v = range[1]
            else:
                index = rounding((v - range[0]) / resolution)
                v = range[0] + index * resolution
            return v

        new_extent = {
            'west': align(extent['west'], x, math.floor, target_resolution[0]),
            'east': align(extent['east'], x, math.ceil, target_resolution[0]),
            'south': align(extent['south'], y, math.floor, target_resolution[1]),
            'north': align(extent['north'], y, math.ceil, target_resolution[1]),
            'crs': extent['crs']
        }
        _log.info(f"Realigned input extent {extent} into {new_extent}")
        return new_extent
    elif isUTM and collection_resolution:
        if collection_resolution[0] <= 20 and target_resolution[0] <= 20:
            bbox = BoundingBox.from_dict(extent, default_crs=4326)
            bbox_utm = bbox.reproject_to_best_utm()

            res = target_resolution if target_resolution[0] > 0 else collection_resolution

            new_extent = bbox_utm.round_to_resolution(res[0], res[1])
            _log.info(f"Realigned input extent {extent} into {new_extent}")
            return new_extent.as_dict()
        else:
            _log.info(f"Not realigning input extent {extent} because crs is UTM and resolution > 20m")
            return extent
    else:
        _log.info(f"Not realigning input extent {extent} (collection crs: {crs}, resolution: {collection_resolution})")
        return extent


def _extract_load_parameters(env: EvalEnv, source_id: SourceId) -> LoadParameters:
    """
    This is a side effect method that also removes source constraints from the list,
    which needs to happen in the right order!!
    """
    from openeo_driver import dry_run

    source_constraints = env[ENV_SOURCE_CONSTRAINTS]

    filtered_constraints = [c for c in source_constraints if c[0] == source_id]

    if len(filtered_constraints) == 0:
        raise Exception(
            f"Could not find source constraints for source {source_id}, "
            f"available constraints are: {set([id for id, _ in source_constraints])}"
        )

    if "global_extent" not in source_constraints[0][1]:
        global_extent = None

        for sid, constraint in source_constraints:
            if sid[0] == "load_collection":
                collection_id = sid[1][0]
            elif sid[0] == "load_stac":
                collection_id = "_load_stac_no_collection_id"
            else:
                collection_id = "_unknown_no_collection_id"

            extent = None
            if "spatial_extent" in constraint:
                extent = constraint["spatial_extent"]
            elif "weak_spatial_extent" in constraint:
                extent = constraint["weak_spatial_extent"]

            if extent is not None:
                collection_crs = _collection_crs(collection_id=collection_id, env=env)
                target_crs = constraint.get("resample", {}).get("target_crs", collection_crs) or collection_crs
                target_resolution = constraint.get("resample", {}).get("resolution", None) or _collection_resolution(
                    collection_id=collection_id, env=env
                )

                if "pixel_buffer" in constraint:
                    buffer = constraint["pixel_buffer"]["buffer_size"]

                    if (target_crs is not None) and target_resolution:
                        bbox = BoundingBox.from_dict(extent, default_crs=4326)
                        extent = bbox.reproject(target_crs).as_dict()

                        extent = {
                            "west": extent["west"] - target_resolution[0] * math.ceil(buffer[0]),
                            "east": extent["east"] + target_resolution[0] * math.ceil(buffer[0]),
                            "south": extent["south"] - target_resolution[1] * math.ceil(buffer[1]),
                            "north": extent["north"] + target_resolution[1] * math.ceil(buffer[1]),
                            "crs": extent["crs"]
                        }
                    else:
                        _log.warning("Not applying buffer to extent because the target CRS is not known.")

                load_collection_in_native_grid = "resample" not in constraint or target_crs == collection_crs
                if (not load_collection_in_native_grid) and collection_crs is not None and ("42001" in str(collection_crs)):
                    try:
                        load_collection_in_native_grid = "UTM zone" in CRS.from_user_input(target_crs).to_wkt()
                    except CRSError:
                        pass

                if load_collection_in_native_grid:
                    extent = _align_extent(
                        extent=extent, collection_id=collection_id, env=env, target_resolution=target_resolution
                    )

                global_extent = spatial_extent_union(global_extent, extent) if global_extent else extent

        for _, constraint in source_constraints:
            constraint["global_extent"] = global_extent

    process_types = filtered_constraints[0][1].get("process_types", None)
    if not process_types:
        process_types = set()
        for _, constraint in filtered_constraints:
            if "process_type" in constraint:
                process_types |= set(constraint["process_type"])
        for _, constraint in filtered_constraints:
            constraint["process_types"] = process_types

    max_buffer_cache = env[ENV_MAX_BUFFER]

    max_buffer = None
    if source_id not in max_buffer_cache:
        for _, constraint in filtered_constraints:
            buffer = constraint.get("pixel_buffer", {}).get("buffer_size", None)
            if buffer:
                if max_buffer is None:
                    max_buffer = buffer
                else:
                    max_buffer = [max(max_buffer[0], buffer[0]), max(max_buffer[1], buffer[1])]
        max_buffer_cache[source_id] = max_buffer
    else:
        max_buffer = max_buffer_cache[source_id]

    _, constraints = filtered_constraints.pop(0)
    source_constraints.remove((source_id, constraints))  # Side effect!

    params = LoadParameters()
    if temporal_extent := constraints.get("temporal_extent"):
        params.temporal_extent = temporal_extent
    labels_args = constraints.get("filter_labels", {})
    if "dimension" in labels_args and labels_args["dimension"] == "t":
        params.filter_temporal_labels = labels_args.get("condition")
    params.spatial_extent = constraints.get("spatial_extent", {})
    params.global_extent = constraints.get("global_extent", {})
    params.bands = constraints.get("bands", None)
    params.properties = constraints.get("properties", {})
    params.aggregate_spatial_geometries = constraints.get("aggregate_spatial", {}).get("geometries")
    if params.aggregate_spatial_geometries is None:
        params.aggregate_spatial_geometries = constraints.get("filter_spatial", {}).get("geometries")
    params.sar_backscatter = constraints.get("sar_backscatter", None)
    params.process_types = process_types
    params.custom_mask = constraints.get("custom_cloud_mask", {})
    params.data_mask = env.get("data_mask", None)
    if params.data_mask:
        _log.debug(f"extracted data_mask {params.data_mask}")
    params.target_crs = constraints.get("resample", {}).get("target_crs", None)
    params.target_resolution = constraints.get("resample", {}).get("resolution", None)
    params.resample_method = constraints.get("resample", {}).get("method", "near")
    params.pixel_buffer = max_buffer
    return params
