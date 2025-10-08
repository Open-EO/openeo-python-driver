"""

Dry-run evaluation of a process graph.

When evaluating a process graph we first do a "dry run" evaluation
of the process graph to detect various aspect of the input data
and processing (like temporal extent, bbox, bands, projection).
Knowing this in advance helps when doing the real evaluation
more efficiently.

The goal is to use as much of the real process graph processing mechanisms,
but pushing around dummy data cubes.

The architecture consists of these classes:
- DataTrace: starts from a `load_collection` (or other source) process and records what happens to this
    single data source (filter_temporal, filter_bbox, ...)
- DryRunDataTracer: observer that keeps track of all data traces during a dry run
- DryRunDataCube: dummy data cube that is passed around in processed

Their relationship is as follows:
- There is a single DryRunDataTracer for a dry-run, keeping track of all relevant operations on all sources
- A DryRunDataCube links to one or more DataTraces, describing the operations that happened
    on the sources that lead to the state of the DryRunDataCube. Often there is just one DataTrace
    in a DryRunDataCube, but when the DryRunDataCube is result of mask or merge_cubes operations,
    there will be multiple DataTraces.
    A DryRunDataCube also has a reference to the DryRunDataTracer in play, so that it can be informed
    when processes are applied to the DryRunDataCube.

When the dry-run phase is done, the DryRunDataTracer knows about all relevant operations
on each data source. It provides methods for example to extract source constraints (bbox/bands/date ranges)
which are used to bootstrap the EvalEnv that is used for the real process graph processing phase.
These source constraints can then be fetched from the EvalEnv at `load_collection` time.

"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy
import shapely.geometry.base
from openeo.metadata import (
    Band,
    BandDimension,
    CollectionMetadata,
    DimensionAlreadyExistsException,
    SpatialDimension,
    GeometryDimension,
    TemporalDimension,
    CubeMetadata,
)
from pyproj import CRS
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
import shapely.ops

from openeo.utils.normalize import normalize_resample_resolution
from openeo_driver import filter_properties
from openeo_driver.datacube import DriverDataCube, DriverVectorCube
from openeo_driver.datastructs import ResolutionMergeArgs, SarBackscatterArgs
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.errors import FeatureUnsupportedException, OpenEOApiException
from openeo_driver.util.geometry import (
    BoundingBox,
    GeometryBufferer,
    geojson_to_geometry, reproject_geometry,
)
import openeo_driver.stac.datacube
from openeo_driver.utils import EvalEnv, to_hashable

_log = logging.getLogger(__name__)

source_constraint_blockers = {
    "bands": [
        "sar_backscatter",
        "atmospheric_correction",
        "mask_scl_dilation",
        "resolution_merge",
        "custom_cloud_mask",
        "apply_neighborhood",
        "reduce_dimension",
        "merge_cubes",
    ],
    "spatial_extent": [],
    "temporal_extent": [],
    "resample": [
        "apply_kernel",
        "reduce_dimension",
        "apply",
        "apply_dimension",
        "resample_spatial",
        "apply_neighborhood",
        "reduce_dimension_binary",
    ],
}


class DataTraceBase:
    """Base class for data traces."""

    def __init__(self):
        self.children = []

    def __hash__(self):
        # Identity hash (e.g. memory address)
        return id(self)

    def get_source(self) -> "DataSource":
        raise NotImplementedError

    def get_arguments_by_operation(self, operation: str) -> List[Union[dict, tuple]]:
        """
         Return in parent->child order
        Args:
            operation:

        Returns:

        """
        return []

    def get_operation_closest_to_source(self, operations: Union[str, List[str]]) -> Union["DataTraceBase", None]:
        raise NotImplementedError

    def describe(self) -> str:
        return "_base"

    def add_child(self, child: "DataTrace"):
        self.children.append(child)

    def __repr__(self):
        return "<{c}#{i}({d})>".format(c=self.__class__.__name__, i=id(self), d=self.describe())


class DataSource(DataTraceBase):
    """Data source: a data (cube) generating process like `load_collection`, `load_disk_data`, ..."""

    __slots__ = ["_process", "_arguments"]

    def __init__(self, process: str = "load_collection", arguments: Union[dict, tuple] = ()):
        super().__init__()
        self._process = process
        self._arguments = arguments

    def get_source(self) -> "DataSource":
        return self

    def get_source_id(self) -> tuple:
        """Identifier for source (hashable tuple, to be used as dict key for example)."""
        return to_hashable((self._process, self._arguments))

    def get_operation_closest_to_source(self, operations: Union[str, List[str]]) -> Union["DataTraceBase", None]:
        if not isinstance(operations, list):
            operations = [operations]
        if self._process in operations:
            return self

    def __repr__(self):
        return "<{c}#{i}({p!r}, {a!r})>".format(
            c=self.__class__.__name__, i=id(self), p=self._process, a=self._arguments
        )

    def describe(self) -> str:
        return self._process

    @classmethod
    def load_collection(cls, collection_id, properties={}, bands=[], env=EvalEnv()) -> "DataSource":
        """Factory for a `load_collection` DataSource."""
        exact_property_matches = {
            property_name: filter_properties.extract_literal_match(condition, env)
            for property_name, condition in properties.items()
        }

        args = (
            (collection_id, exact_property_matches, bands)
            if len(bands) > 0
            else (collection_id, exact_property_matches)
        )
        return cls(process="load_collection", arguments=args)

    @classmethod
    def load_disk_data(cls, glob_pattern: str, format: str, options: dict) -> "DataSource":
        """Factory for a `load_disk_data` DataSource."""
        return cls(process="load_disk_data", arguments=(glob_pattern, format, options))

    @classmethod
    def load_result(cls, job_id: str) -> "DataSource":
        """Factory for a `load_result` DataSource."""
        return cls(process="load_result", arguments=(job_id,))

    @classmethod
    def load_stac(cls, url: str, properties={}, bands=[], env=EvalEnv()) -> "DataSource":
        """Factory for a `load_stac` DataSource."""
        exact_property_matches = {
            property_name: filter_properties.extract_literal_match(condition, env)
            for property_name, condition in properties.items()
        }

        return cls(process="load_stac", arguments=(url, exact_property_matches, bands))


class DataTrace(DataTraceBase):
    """
    Processed data: linked list of processes, ending at a data source node.

    Note: this is not the same as a data cube, as a data cube can be combination of multiple data
    traces (e.g. after mask or merge process).
    """

    __slots__ = ["parent", "_operation", "_arguments"]

    def __init__(self, parent: DataTraceBase, operation: str, arguments: Union[dict, tuple]):
        super().__init__()
        self.parent = parent
        parent.add_child(self)
        self._operation = operation
        self._arguments = arguments

    def get_source(self) -> DataSource:
        return self.parent if isinstance(self.parent, DataSource) else self.parent.get_source()

    def get_arguments_by_operation(self, operation: str) -> List[Union[dict, tuple]]:
        # Return in parent->child order
        res = self.parent.get_arguments_by_operation(operation)
        if self._operation == operation:
            res.append(self._arguments)
        return res

    def get_operation_closest_to_source(self, operations: Union[str, List[str]]) -> Union["DataTraceBase", None]:
        if not isinstance(operations, list):
            operations = [operations]
        # First look up in parent (because we want the one closest to source)
        parent_op = self.parent.get_operation_closest_to_source(operations)
        if parent_op:
            return parent_op
        elif self._operation in operations:
            return self

    def __repr__(self):
        return "<{c}#{i}(#{p}, {o}, {a})>".format(
            c=self.__class__.__name__, i=id(self), p=id(self.parent), o=self._operation, a=self._arguments
        )

    def describe(self) -> str:
        return self.parent.describe() + "<-" + self._operation


# Type hint for source constaints
# TODO make this a real class?
SourceConstraint = Tuple[Tuple[str, tuple], dict]


class DryRunDataTracer:
    """
    Observer that keeps track of data traces in various DryRunDataCubes
    """

    def __init__(self):
        self._traces: List[DataTraceBase] = []

    def __repr__(self):
        return "<{c} (traces: {n!r})>".format(c=self.__class__.__name__, n=self._traces)

    def add_trace(self, trace: DataTraceBase) -> DataTraceBase:
        """Keep track of given trace"""
        self._traces.append(trace)
        return trace

    def process_traces(self, traces: List[DataTraceBase], operation: str, arguments: dict) -> List[DataTraceBase]:
        """Process given traces with an operation (and keep track of the results)."""
        return [self.add_trace(DataTrace(parent=t, operation=operation, arguments=arguments)) for t in traces]

    def load_collection(
        self, collection_id: str, arguments: dict, metadata: dict = None, env: EvalEnv = EvalEnv()
    ) -> "DryRunDataCube":
        """Create a DryRunDataCube from a `load_collection` process."""
        metadata = CollectionMetadata(metadata=metadata)
        properties = {
            # TODO #275 avoid VITO/Terrascope specific handling here?
            **metadata.get("_vito", "properties", default={}),
            **arguments.get("properties", {}),
        }

        trace = DataSource.load_collection(
            collection_id=collection_id, properties=properties, bands=arguments.get("bands", []), env=env
        )
        self.add_trace(trace)

        cube = DryRunDataCube(traces=[trace], data_tracer=self, metadata=metadata)
        if "temporal_extent" in arguments:
            cube = cube.filter_temporal(*arguments["temporal_extent"])
        if "spatial_extent" in arguments:
            cube = cube.filter_bbox(**arguments["spatial_extent"])
        if "bands" in arguments:
            cube = cube.filter_bands(arguments["bands"])

        if properties:
            cube = cube.filter_properties(properties)

        return cube

    def load_disk_data(self, glob_pattern: str, format: str, options: dict) -> "DryRunDataCube":
        """Create a DryRunDataCube from a `load_disk_data` process."""
        trace = DataSource.load_disk_data(glob_pattern=glob_pattern, format=format, options=options)
        self.add_trace(trace)
        # Note: naive assumptions about the actual data cube dimensions here.
        metadata = CollectionMetadata(
            {},
            dimensions=[
                SpatialDimension(name="x", extent=[]),
                SpatialDimension(name="y", extent=[]),
                TemporalDimension(name="t", extent=[]),
                BandDimension(name="bands", bands=[Band("unknown")]),
            ],
        )
        return DryRunDataCube(traces=[trace], data_tracer=self, metadata=metadata)

    def load_result(self, job_id: str, arguments: dict) -> "DryRunDataCube":
        trace = DataSource.load_result(job_id=job_id)
        self.add_trace(trace)

        cube = DryRunDataCube(traces=[trace], data_tracer=self)
        if "temporal_extent" in arguments:
            cube = cube.filter_temporal(*arguments["temporal_extent"])
        if "spatial_extent" in arguments:
            cube = cube.filter_bbox(**arguments["spatial_extent"])
        if "bands" in arguments:
            cube = cube.filter_bands(arguments["bands"])

        return cube

    def load_stac(self, url: str, arguments: dict, env: EvalEnv = EvalEnv()) -> "DryRunDataCube":
        properties = arguments.get("properties", {})

        trace = DataSource.load_stac(url=url, properties=properties, bands=arguments.get("bands", []), env=env)
        self.add_trace(trace)

        try:
            metadata = openeo_driver.stac.datacube.stac_to_cube_metadata(stac_ref=url)
        except Exception as e:
            _log.warning(
                f"Dry-run load_stac: failed to parse cube metadata from {url!r} ({e!r}). Falling back on generic metadata."
            )
            metadata = CubeMetadata(
                dimensions=[
                    SpatialDimension(name="x", extent=[]),
                    SpatialDimension(name="y", extent=[]),
                    TemporalDimension(name="t", extent=[]),
                    BandDimension(name="bands", bands=[Band("unknown")]),
                ]
            )

        cube = DryRunDataCube(traces=[trace], data_tracer=self, metadata=metadata)
        if "temporal_extent" in arguments:
            cube = cube.filter_temporal(*arguments["temporal_extent"])
        if "spatial_extent" in arguments:
            cube = cube.filter_bbox(**arguments["spatial_extent"])
        if "bands" in arguments:
            cube = cube.filter_bands(arguments["bands"])
        if properties:
            cube = cube.filter_properties(properties)

        return cube

    def get_trace_leaves(self) -> List[DataTraceBase]:
        """
        Get all nodes in the tree of traces that are not parent of another trace.
        In openEO this could be for instance a save_result process that ends the workflow.
        """
        leaves = []

        def get_leaves(tree: DataTraceBase) -> List[DataTraceBase]:
            return (
                [tree] if len(tree.children) == 0 else [leaf for child in tree.children for leaf in get_leaves(child)]
            )

        leaves = [ leaf  for trace in self._traces for leaf in get_leaves(trace)]
        leaves = list(set(leaves))

        return leaves

    def get_metadata_links(self):
        result = {}
        for leaf in self.get_trace_leaves():
            source_id = leaf.get_source().get_source_id()
            result[source_id] = leaf.get_arguments_by_operation("log_metadata_link")
        return result

    def get_source_constraints(self, merge=True) -> List[SourceConstraint]:
        """
        Get the temporal/spatial constraints of all traced sources

        :param merge:
        :return: a list of constraints for sources in the same order that they appear in the process graph; the values
        consist of a source id (e.g. `("load_collection", "Sentinel2") and a dictionary with "temporal_extent",
        "spatial_extent", "bands" fields.
        """
        source_constraints = []
        for leaf in self.get_trace_leaves():
            constraints = {}
            pixel_buffer_op = leaf.get_operation_closest_to_source(["pixel_buffer"])
            if pixel_buffer_op:
                args = pixel_buffer_op.get_arguments_by_operation("pixel_buffer")
                if args:
                    buffer_size = args[0]["buffer_size"]
                    constraints["pixel_buffer"] = {"buffer_size": buffer_size}

            resampling_op = leaf.get_operation_closest_to_source(["resample_cube_spatial", "resample_spatial"])
            if resampling_op:
                resample_valid = True
                # the resampling parameters can be taken into account during load_collection,
                # under the condition that no operations occur in between that may be affected
                for op in [
                    "apply_kernel",
                    "reduce_dimension",
                    "apply",
                    "apply_dimension",
                    "apply_neighborhood",
                    "reduce_dimension_binary",
                    "mask",
                    "to_scl_dilation_mask",
                ]:
                    args = resampling_op.get_arguments_by_operation(op)
                    if args:
                        resample_valid = False
                        break
                if resample_valid:
                    args = resampling_op.get_arguments_by_operation("resample_cube_spatial")
                    if args:
                        target = args[0]["target"]
                        method = args[0]["method"]
                        metadata: CollectionMetadata = target.metadata
                        spatial_dim = metadata.spatial_dimensions[0]
                        # TODO: derive resolution from openeo:gsd instead (see openeo-geopyspark-driver)
                        resolutions = tuple(dim.step for dim in metadata.spatial_dimensions if dim.step is not None)
                        if len(resolutions) > 0 and spatial_dim.crs is not None:
                            constraints["resample"] = {
                                "target_crs": spatial_dim.crs,
                                "resolution": resolutions,
                                "method": method,
                            }
                    args = resampling_op.get_arguments_by_operation("resample_spatial")
                    if args:
                        resolution = normalize_resample_resolution(args[0]["resolution"])
                        projection = args[0]["projection"]
                        method = args[0].get("method", "near")
                        if method != "geocode":
                            constraints["resample"] = {"target_crs": projection, "resolution": resolution, "method": method}

            for op in [
                "temporal_extent",
                "spatial_extent",
                "weak_spatial_extent",
                "bands",
                "aggregate_spatial",
                "sar_backscatter",
                "process_type",
                "custom_cloud_mask",
                "properties",
                "filter_spatial",
                "filter_labels",
            ]:
                # 1 some processes can not be skipped when pushing filters down,
                # so find the subgraph that no longer contains these blockers
                leaf_without_blockers = leaf
                if op in source_constraint_blockers:
                    subgraph_without_blocking_processes = leaf.get_operation_closest_to_source(
                        source_constraint_blockers[op]
                    )
                    if subgraph_without_blocking_processes is not None:
                        leaf_without_blockers = subgraph_without_blocking_processes

                # 2 merge filtering arguments
                if leaf_without_blockers is not None:
                    args = leaf_without_blockers.get_arguments_by_operation(op)
                    if args:
                        if merge:
                            # Take first item (to reproduce original behavior)
                            # TODO: take temporal/spatial/categorical intersection instead?
                            #       see https://github.com/Open-EO/openeo-processes/issues/201
                            constraints[op] = args[0]
                        else:
                            constraints[op] = args

            if "weak_spatial_extent" in constraints:
                if "spatial_extent" not in constraints:
                    constraints["spatial_extent"] = constraints["weak_spatial_extent"]

            source_id = leaf.get_source().get_source_id()
            source_constraints.append((source_id, constraints))
        return source_constraints

    def get_geometries(
        self, operation="aggregate_spatial"
    ) -> List[Union[shapely.geometry.base.BaseGeometry, DelayedVector, DriverVectorCube]]:
        """Get geometries (polygons or DelayedVector), as used by aggregate_spatial"""
        geometries_by_id = {}
        for leaf in self.get_trace_leaves():
            for args in leaf.get_arguments_by_operation(operation):
                if "geometries" in args:
                    geometries = args["geometries"]
                    geometries_by_id[id(geometries)] = geometries
        # TODO: we just pass all (0 or more) geometries we encountered. Do something smarter when there are multiple?
        return list(geometries_by_id.values())

    def get_last_geometry(
        self, operation="aggregate_spatial"
    ) -> Union[shapely.geometry.base.BaseGeometry, DelayedVector, DriverVectorCube]:
        """Get geometries (polygons or DelayedVector), as used by aggregate_spatial"""

        for leaf in self.get_trace_leaves():
            args = leaf.get_arguments_by_operation(operation)
            args.reverse()
            for args in args:
                if "geometries" in args:
                    geometries = args["geometries"]
                    return geometries
        return None


class ProcessType(Enum):
    LOCAL = 1  # band math
    FOCAL_TIME = 2  # aggregate_temporal
    FOCAL_SPACE_TIME = 3  # apply_neighborhood
    GLOBAL_TIME = 4  # reduce_dimension
    UNKNOWN = 5
    FOCAL_SPACE = 6  # resampling, apply_kernel


class DryRunDataCube(DriverDataCube):
    """
    Data cube (mock/spy) to be used for a process graph dry-run,
    to detect data cube constraints (filter_bbox, filter_temporal, ...), resolution, tile layout,
    estimate memory/cpu usage, ...
    """

    def __init__(
        self, traces: List[DataTraceBase], data_tracer: DryRunDataTracer, metadata: Optional[CubeMetadata] = None
    ):
        super(DryRunDataCube, self).__init__(metadata=metadata)
        self._traces = traces or []
        self._data_tracer = data_tracer

    def _process(self, operation, arguments, metadata: CubeMetadata = None) -> "DryRunDataCube":
        """Helper to handle single-cube operations"""
        # New data cube with operation added to each trace
        traces = self._data_tracer.process_traces(traces=self._traces, operation=operation, arguments=arguments)
        # TODO: manipulate metadata properly?
        return DryRunDataCube(traces=traces, data_tracer=self._data_tracer, metadata=metadata or self.metadata)

    def _process_metadata(self, metadata: CollectionMetadata) -> "DryRunDataCube":
        """Just process metadata (leave traces as is)"""
        return DryRunDataCube(traces=self._traces, data_tracer=self._data_tracer, metadata=metadata)

    def filter_temporal(self, start: str, end: str) -> "DryRunDataCube":
        return self._process("temporal_extent", (start, end))

    def filter_bbox(
        self, west, south, east, north, crs=None, base=None, height=None, operation="spatial_extent"
    ) -> "DryRunDataCube":
        return self._process(
            operation, {"west": west, "south": south, "east": east, "north": north, "crs": (crs or "EPSG:4326")}
        )

    def filter_spatial(self, geometries):
        crs = None
        resolution = None
        if len(self.metadata.spatial_dimensions) > 0:
            spatial_dim = self.metadata.spatial_dimensions[0]
            crs = spatial_dim.crs
            resolution = spatial_dim.step
        geometries, bbox = self._normalize_geometry(geometries, target_crs=crs, target_resolution=resolution)
        cube = self.filter_bbox(**bbox, operation="weak_spatial_extent")
        return cube._process(operation="filter_spatial", arguments={"geometries": geometries})

    def filter_bands(self, bands) -> "DryRunDataCube":
        return self._process("bands", bands)

    def filter_properties(self, properties) -> "DryRunDataCube":
        return self._process("properties", properties)

    def save_result(self, filename: str, format: str, format_options: dict = None) -> str:
        # TODO: this method should be deprecated (limited to single asset) in favor of write_assets (supports multiple assets)
        return self._process("save_result", {"format": format, "options": format_options})

    def filter_labels(
        self, condition: dict, dimension: str, context: Optional[dict] = None, env: EvalEnv = None
    ) -> "DryRunDataCube":
        return self._process("filter_labels", arguments=dict(condition=condition, dimension=dimension, context=context))

    def mask(self, mask: "DryRunDataCube", replacement=None) -> "DryRunDataCube":
        # TODO: if mask cube has no temporal or bbox extent: copy from self?
        # TODO: or add reference to the self trace to the mask trace and vice versa?
        mask_resampled = mask._process("resample_cube_spatial", arguments={"target": self, "method": "near"})
        cube = self._process("mask", {"mask": mask_resampled})
        return DryRunDataCube(
            traces=cube._traces + mask_resampled._traces, data_tracer=cube._data_tracer, metadata=cube.metadata
        )

    def merge_cubes(self, other: "DryRunDataCube", overlap_resolver) -> "DryRunDataCube":
        return DryRunDataCube(
            traces=self._traces + other._traces,
            data_tracer=self._data_tracer,
            # TODO: properly merge (other) metadata?
            metadata=self.metadata,
        )._process("merge_cubes", arguments={})

    def mask_polygon(self, mask, replacement=None, inside: bool = False) -> "DriverDataCube":
        cube = self
        if not inside and replacement is None:
            mask, bbox = cube._normalize_geometry(mask)
            cube = self.filter_bbox(**bbox, operation="weak_spatial_extent")
        return cube._process(operation="mask_polygon", arguments={"mask": mask})

    def aggregate_spatial(
        self,
        geometries: Union[BaseGeometry, str, DriverVectorCube],
        reducer: dict,
        target_dimension: Optional[str] = None,
    ) -> "DryRunDataCube":
        if target_dimension:
            raise FeatureUnsupportedException(
                f"Argument `target_dimension` with value {target_dimension} not supported in `aggregate_spatial`"
            )
        # TODO #71 #114 EP-3981 normalize to vector cube instead of GeometryCollection
        geoms_is_empty = isinstance(geometries, DriverVectorCube) and len(geometries.get_geometries()) == 0
        cube = self
        if not geoms_is_empty:
            crs = None
            resolution = None
            if len(self.metadata.spatial_dimensions) > 0:
                spatial_dim = self.metadata.spatial_dimensions[0]
                resolution = spatial_dim.step
                crs = spatial_dim.crs
            geometries, bbox = self._normalize_geometry(geometries, target_crs=crs, target_resolution=resolution)
            cube = self.filter_bbox(**bbox, operation="weak_spatial_extent")
        return cube._process(operation="aggregate_spatial", arguments={"geometries": geometries})

    def _normalize_geometry(
        self, geometries, target_crs=None, target_resolution = None
    ) -> Tuple[Union[DriverVectorCube, DelayedVector, BaseGeometry], dict]:
        """
        Helper to preprocess geometries (as used in aggregate_spatial and mask_polygon)
        and extract bbox (e.g. for filter_bbox)

        :param geometries: geometries as BaseGeometry, GeoJSON dict, DelayedVector or DriverVectorCube
        :param target_crs: target CRS to reproject geometries to
        :param target_resolution: target resolution for geometries, in units of the target CRS, None by default which will use 10m
        """
        _log.debug(f"_normalize_geometry with {type(geometries)}")
        # TODO #71 #114 EP-3981 normalize to vector cube instead of GeometryCollection
        crs = "EPSG:4326"
        if isinstance(geometries, DriverVectorCube):
            # TODO: buffer distance of 10m assumes certain resolution (e.g. sentinel2 pixels)
            # TODO: use proper distance for collection resolution instead of using a default distance?
            # TODO: or eliminate need for buffering in the first place? https://github.com/Open-EO/openeo-python-driver/issues/148
            if target_crs is not None:
                is_utm = target_crs == "AUTO:42001" or "Auto42001" in str(target_crs)
                if is_utm:
                    target_crs = f"EPSG:{BoundingBox.from_wsen_tuple(geometries.get_bounding_box(),crs=geometries.get_crs()).best_utm()}"
                else:
                    target_crs = BoundingBox.normalize_crs(target_crs)
                points = []
                other = []
                for g in geometries.get_geometries():
                    if isinstance(g, Point):
                        points.append(g)
                    else:
                        other.append(g)
                other_hull = reproject_geometry(shapely.ops.unary_union(other),geometries.get_crs(),CRS.from_user_input(target_crs))

                if len(points) > 0:
                    point_hull = reproject_geometry(shapely.geometry.MultiPoint(points).envelope,geometries.get_crs(),CRS.from_user_input(target_crs))
                    buffered_points = None
                    if target_resolution == None:
                        loi_point = point_hull.representative_point()
                        bufferer = GeometryBufferer.from_meter_for_crs(
                            distance=10, crs=target_crs, loi=(loi_point.x,loi_point.y), loi_crs=target_crs
                        )
                        buffered_points = bufferer.buffer(point_hull)
                    else:
                        buffered_points = point_hull.buffer(distance=target_resolution)
                    other_hull = other_hull.union(buffered_points)
                bbox = other_hull.bounds
                crs = target_crs
            else:
                bbox = geometries.buffer_points(distance=10).get_bounding_box()
                crs = geometries.get_crs_str()
        elif isinstance(geometries, dict):
            return self._normalize_geometry(geojson_to_geometry(geometries),target_crs, target_resolution)
        elif isinstance(geometries, str):
            return self._normalize_geometry(DelayedVector(geometries),target_crs, target_resolution)
        elif isinstance(geometries, DelayedVector):
            bbox = geometries.bounds
        elif isinstance(geometries, shapely.geometry.base.BaseGeometry):
            # TODO: buffer distance of 10m assumes certain resolution (e.g. sentinel2 pixels)
            # TODO: use proper distance for collection resolution instead of using a default distance?
            # TODO: or eliminate need for buffering in the first place? https://github.com/Open-EO/openeo-python-driver/issues/148
            bufferer = GeometryBufferer.from_meter_for_crs(distance=10, crs="EPSG:4326")
            if isinstance(geometries, Point):
                geometries = bufferer.buffer(geometries)
            elif isinstance(geometries, GeometryCollection):
                # TODO #71 deprecate using GeometryCollection as feature collections
                geometries = GeometryCollection(
                    [bufferer.buffer(g) if isinstance(g, Point) else g for g in geometries.geoms]
                )
            bbox = geometries.bounds
        else:
            raise ValueError(geometries)
        bbox = dict(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3], crs=crs)
        return geometries, bbox

    def raster_to_vector(self):
        dimensions = [GeometryDimension(name=DriverVectorCube.DIM_GEOMETRY)]
        if self.metadata.has_temporal_dimension():
            dimensions.append(self.metadata.temporal_dimension)
        if self.metadata.has_band_dimension():
            dimensions.append(self.metadata.band_dimension)

        return self._process(
            operation="raster_to_vector", arguments={}, metadata=CollectionMetadata(metadata={}, dimensions=dimensions)
        )

    def run_udf(self):
        return self._process(operation="run_udf", arguments={})

    def resample_spatial(
        self,
        resolution: Union[float, Tuple[float, float]],
        projection: Union[int, str] = None,
        method: str = "near",
        align: str = "upper-left",
    ):
        return self._process(
            "resample_spatial",
            arguments={"resolution": resolution, "projection": projection, "method": method, "align": align},
            metadata=(self.metadata or CubeMetadata()).resample_spatial(resolution=resolution, projection=projection),
        )

    def resample_cube_spatial(self, target: "DryRunDataCube", method: str = "near") -> "DryRunDataCube":
        cube = self._process("process_type", [ProcessType.FOCAL_SPACE])
        cube = cube._process("resample_cube_spatial", arguments={"target": target, "method": method})
        if target.metadata:
            metadata = (self.metadata or CubeMetadata()).resample_cube_spatial(target=target.metadata)
        else:
            metadata = None
        return DryRunDataCube(
            traces=cube._traces + target._traces,
            data_tracer=self._data_tracer,
            metadata=metadata,
        )

    def reduce_dimension(
        self, reducer, *, dimension: str, context: Optional[dict] = None, env: EvalEnv
    ) -> "DryRunDataCube":
        dc = self
        if self.metadata.has_temporal_dimension() and self.metadata.temporal_dimension.name == dimension:
            # TODO: reduce is not necessarily global in call cases
            dc = self._process("process_type", [ProcessType.GLOBAL_TIME])

        return dc._process_metadata(self.metadata.reduce_dimension(dimension_name=dimension))._process(
            "reduce_dimension", arguments={}
        )

    def ndvi(self, nir: str = "nir", red: str = "red", target_band: Optional[str] = None) -> "DryRunDataCube":
        if target_band is None and self.metadata.has_band_dimension():
            return self._process_metadata(
                self.metadata.reduce_dimension(dimension_name=self.metadata.band_dimension.name)
            )
        elif target_band is not None and self.metadata.has_band_dimension():
            return self._process_metadata(
                self.metadata.append_band(Band(name=target_band, common_name=target_band, wavelength_um=None))
            )
        else:
            return self

    def chunk_polygon(
        # TODO #288: `chunks`: MultiPolygon should not be abused as collection of separate geometries.
        self,
        reducer,
        chunks: MultiPolygon,
        mask_value: float,
        env: EvalEnv,
        context: Optional[dict] = None,
    ) -> "DryRunDataCube":
        # TODO #229: rename/update `chunk_polygon` to `apply_polygon` (https://github.com/Open-EO/openeo-processes/pull/298)
        if isinstance(chunks, Polygon):
            polygons = [chunks]
        elif isinstance(chunks, MultiPolygon):
            polygons: List[Polygon] = chunks.geoms
        else:
            raise ValueError(f"Invalid type for `chunks`: {type(chunks)}")
        # TODO #71 #114 Deprecate/avoid usage of GeometryCollection
        geometries, bbox = self._normalize_geometry(GeometryCollection(polygons))
        cube = self.filter_bbox(**bbox, operation="weak_spatial_extent")
        return cube._process("chunk_polygon", arguments={"geometries": geometries})

    def add_dimension(self, name: str, label, type: str = "other") -> "DryRunDataCube":
        try:
            return self._process_metadata(self.metadata.add_dimension(name=name, label=label, type=type))
        except DimensionAlreadyExistsException:
            raise OpenEOApiException(
                code="DimensionExists", status_code=400, message=f"A dimension with name {name} already exists."
            )

    def drop_dimension(self, name: str) -> "DryRunDataCube":
        return self._process("drop_dimension", {"name": name}, metadata=self.metadata.drop_dimension(name=name))

    def sar_backscatter(self, args: SarBackscatterArgs) -> "DryRunDataCube":
        return self._process("sar_backscatter", args)

    def resolution_merge(self, args: ResolutionMergeArgs) -> "DryRunDataCube":
        return self._process("resolution_merge", args)


    def apply_kernel(self, kernel: numpy.ndarray, factor=1, border=0, replace_invalid=0) -> 'DriverDataCube':
        cube = self._process("process_type", [ProcessType.FOCAL_SPACE])
        cube = cube._process("pixel_buffer", arguments={"buffer_size": [x / 2.0 for x in kernel.shape]})
        return cube._process("apply_kernel", arguments={"kernel": kernel})

    def apply_dimension(
        self, process, *, dimension: str, target_dimension: Optional[str], context: Optional[dict] = None, env: EvalEnv
    ) -> "DriverDataCube":
        cube = self
        if self.metadata.has_temporal_dimension() and self.metadata.temporal_dimension.name == dimension:
            # TODO: reduce is not necessarily global in call cases
            cube = self._process("process_type", [ProcessType.GLOBAL_TIME])

        if target_dimension is not None:
            cube = cube._process_metadata(self.metadata.rename_dimension(source=dimension, target=target_dimension))

        return cube._process("apply_dimension", arguments={"dimension": dimension})

    def apply_tiles_spatiotemporal(self, process, context: Optional[dict] = None) -> "DriverDataCube":
        if self.metadata.has_temporal_dimension():
            return self._process("process_type", [ProcessType.GLOBAL_TIME])
        else:
            return self

    def apply(self, process: dict, *, context: Optional[dict] = None, env: EvalEnv) -> "DriverDataCube":
        cube = self._process("apply", {})
        return cube

    def apply_neighborhood(
        self,
        process,
        *,
        size: List[dict],
        overlap: Optional[List[dict]] = None,
        context: Optional[dict] = None,
        env: EvalEnv,
    ) -> "DriverDataCube":
        cube = self._process("apply_neighborhood", {})
        temporal_size = temporal_overlap = None
        size_dict = {e["dimension"]: e for e in size}
        overlap_dict = {e["dimension"]: e for e in (overlap or [])}
        if "x" in overlap_dict or "y" in overlap_dict:
            x_size = overlap_dict.get("x", {}).get("value", 0.0)
            y_size = overlap_dict.get("y", {}).get("value", 0.0)
            if (
                overlap_dict.get("x", {}).get("unit", "px") != "px"
                or overlap_dict.get("y", {}).get("unit", "px") != "px"
            ):
                raise OpenEOApiException(
                    f"apply_neighborhood: only pixel units are supported for overlap, but got: {overlap_dict.get('x',{}).get('unit','px')}"
                )
            cube = cube._process("pixel_buffer", arguments={"buffer_size": [float(x_size), float(y_size)]})
        if self.metadata.has_temporal_dimension():
            temporal_size = size_dict.get(self.metadata.temporal_dimension.name, None)
            temporal_overlap = overlap_dict.get(self.metadata.temporal_dimension.name, None)
            if temporal_size is None or temporal_size.get("value", None) is None:
                return cube._process("process_type", [ProcessType.GLOBAL_TIME])
        return cube

    def atmospheric_correction(
        self,
        method: Optional[str] = None,
        elevation_model: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> "DriverDataCube":
        # TODO #275 does this VITO reference belong here?
        method_link = "https://remotesensing.vito.be/case/icor"
        if method == "SMAC":
            method_link = "https://doi.org/10.1080/01431169408954055"

        aot_link = "https://atmosphere.copernicus.eu/catalogue#/product/urn:x-wmo:md:int.ecmwf::copernicus:cams:prod:fc:total-aod:pid094"
        # by default GLOBE DEM is used
        dem_doi = "https://doi.org/10.7289/V52R3PMS"
        # default APDA water vapour algorithm
        wvp_doi = "https://doi.org/10.1109/LGRS.2016.2635942"

        return (
            self._process("log_metadata_link", arguments={"rel": "atmospheric-scattering", "href": method_link})
            ._process("log_metadata_link", arguments={"rel": "related", "href": aot_link})
            ._process("log_metadata_link", arguments={"rel": "elevation-model", "href": dem_doi})
            ._process("log_metadata_link", arguments={"rel": "water-vapor", "href": wvp_doi})
        )

    def mask_scl_dilation(self, **kwargs) -> "DriverDataCube":
        return self._process("custom_cloud_mask", arguments={**{"method": "mask_scl_dilation"}, **kwargs})

    def to_scl_dilation_mask(
        self,
        erosion_kernel_size: int,
        mask1_values: List[int],
        mask2_values: List[int],
        kernel1_size: int,
        kernel2_size: int,
    ) -> DryRunDataCube:
        cube = self._process("process_type", [ProcessType.FOCAL_SPACE])
        size = kernel2_size
        cube = cube._process("pixel_buffer", arguments={"buffer_size": [size / 2.0, size / 2.0]})
        cube = cube._process("to_scl_dilation_mask", arguments={})
        return cube

    def mask_l1c(self) -> "DriverDataCube":
        return self._process("custom_cloud_mask", arguments={"method": "mask_l1c"})

    def _nop(self, *args, **kwargs) -> "DryRunDataCube":
        """No Operation: do nothing"""
        return self

    # TODO: some methods need metadata manipulation?

    apply_tiles = _nop

    reduce = _nop
    reduce_bands = _nop
    aggregate_temporal = _nop
    aggregate_temporal_period = _nop
    rename_labels = _nop
    rename_dimension = _nop
    water_vapor = _nop
    linear_scale_range = _nop
    dimension_labels = _nop
