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
import logging
from enum import Enum
from typing import List, Union, Tuple

import shapely.geometry.base

from openeo.metadata import CollectionMetadata
from openeo_driver import filter_properties
from openeo_driver.datacube import DriverDataCube
from openeo_driver.datastructs import SarBackscatterArgs, ResolutionMergeArgs
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.save_result import AggregatePolygonResult
from openeo_driver.utils import geojson_to_geometry, to_hashable, EvalEnv

_log = logging.getLogger(__name__)

source_constraint_blockers = {
    'bands': [
        'sar_backscatter', 'atmospheric_correction', 'mask_scl_dilation', 'resolution_merge', 'custom_cloud_mask'
    ],
    'spatial_extent': [],
    'temporal_extent': [],
    'resample': [
        "apply_kernel", "reduce_dimension", "apply", "apply_dimension", "resample_spatial", "apply_neighborhood",
        "reduce_dimension_binary"
    ]
}


class DataTraceBase:
    """Base class for data traces."""

    def __init__(self):
        self.children = []

    def __hash__(self):
        # Identity hash (e.g. memory address)
        return id(self)

    def get_source(self) -> 'DataSource':
        raise NotImplementedError

    def get_arguments_by_operation(self, operation: str) -> List[Union[dict, tuple]]:
        return []

    def get_operation_closest_to_source(self, operations: Union[str, List[str]]) -> Union["DataTraceBase", None]:
        raise NotImplementedError

    def describe(self) -> str:
        return "_base"

    def add_child(self, child: 'DataTrace'):
        self.children.append(child)


class DataSource(DataTraceBase):
    """Data source: a data (cube) generating process like `load_collection`, `load_disk_data`, ..."""
    __slots__ = ["_process", "_arguments"]

    def __init__(self, process: str = "load_collection", arguments: Union[dict, tuple] = ()):
        super().__init__()
        self._process = process
        self._arguments = arguments

    def get_source(self) -> 'DataSource':
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
        return '<{c}#{i}({p!r}, {a!r})>'.format(
            c=self.__class__.__name__, i=id(self), p=self._process, a=self._arguments
        )

    def describe(self) -> str:
        return self._process

    @classmethod
    def load_collection(cls, collection_id, properties={}) -> 'DataSource':
        """Factory for a `load_collection` DataSource."""
        exact_property_matches = {property_name: filter_properties.extract_literal_match(condition)
                                  for property_name, condition in properties.items()}

        return cls(process="load_collection", arguments=(collection_id, exact_property_matches))

    @classmethod
    def load_disk_data(cls, glob_pattern: str, format: str, options: dict) -> 'DataSource':
        """Factory for a `load_disk_data` DataSource."""
        return cls(process="load_disk_data", arguments=(glob_pattern, format, options))


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
        return '<{c}#{i}(#{p}, {o}, {a})>'.format(
            c=self.__class__.__name__, i=id(self), p=id(self.parent), o=self._operation, a=self._arguments
        )

    def describe(self) -> str:
        return self.parent.describe() + "<-" + self._operation


class DryRunDataTracer:
    """
    Observer that keeps track of data traces in various DryRunDataCubes
    """

    def __init__(self):
        self._traces: List[DataTraceBase] = []

    def add_trace(self, trace: DataTraceBase) -> DataTraceBase:
        """Keep track of given trace"""
        self._traces.append(trace)
        return trace

    def process_traces(self, traces: List[DataTraceBase], operation: str, arguments: dict) -> List[DataTraceBase]:
        """Process given traces with an operation (and keep track of the results)."""
        return [
            self.add_trace(DataTrace(parent=t, operation=operation, arguments=arguments))
            for t in traces
        ]

    def load_collection(self, collection_id: str, arguments: dict, metadata: dict = None) -> 'DryRunDataCube':
        """Create a DryRunDataCube from a `load_collection` process."""
        properties = {**CollectionMetadata(metadata).get("_vito", "properties", default={}),
                      **arguments.get("properties", {})}

        trace = DataSource.load_collection(collection_id=collection_id, properties=properties)
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

    def load_disk_data(self, glob_pattern: str, format: str, options: dict) -> 'DryRunDataCube':
        """Create a DryRunDataCube from a `load_disk_data` process."""
        trace = DataSource.load_disk_data(glob_pattern=glob_pattern, format=format, options=options)
        self.add_trace(trace)
        # TODO: metadata?
        return DryRunDataCube(traces=[trace], data_tracer=self)

    def get_trace_leaves(self) -> List[DataTraceBase]:
        """Get all nodes in the tree of traces that are not parent of another trace."""
        leaves = []

        def get_leaves(tree: DataTraceBase) -> List[DataTraceBase]:
            return ([tree] if len(tree.children) == 0
                    else [leaf for child in tree.children for leaf in get_leaves(child)])

        for trace in self._traces:
            for leaf in get_leaves(trace):
                if leaf not in leaves:
                    leaves.append(leaf)

        return leaves

    def get_metadata_links(self):
        result = {}
        for leaf in self.get_trace_leaves():
            source_id = leaf.get_source().get_source_id()
            result[source_id] = leaf.get_arguments_by_operation("log_metadata_link")
        return result

    def get_source_constraints(self, merge=True) -> List[Tuple[tuple, dict]]:
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
            resampling_op = leaf.get_operation_closest_to_source("resample_cube_spatial")
            if resampling_op:
                resample_valid = True
                # the resampling parameters can be taken into account during load_collection,
                # under the condition that no operations occur in between that may be affected
                for op in [
                    "apply_kernel", "reduce_dimension", "apply", "apply_dimension", "resample_spatial",
                    "apply_neighborhood", "reduce_dimension_binary"
                ]:
                    args = resampling_op.get_arguments_by_operation(op)
                    if args:
                        resample_valid = False
                        break
                if resample_valid:
                    args = resampling_op.get_arguments_by_operation("resample_cube_spatial")
                    if args:
                        target = args[0]["target"]
                        metadata: CollectionMetadata = target.metadata
                        spatial_dim = metadata.spatial_dimensions[0]
                        resolutions = [dim.step for dim in metadata.spatial_dimensions if dim.step is not None]
                        constraints["resample"] = {"target_crs": spatial_dim.crs, "resolution": resolutions}

            for op in [
                "temporal_extent", "spatial_extent", "_weak_spatial_extent", "bands", "aggregate_spatial",
                "sar_backscatter", "process_type", "custom_cloud_mask", "properties", "filter_spatial"
            ]:
                # 1 some processes can not be skipped when pushing filters down,
                # so find the subgraph that no longer contains these blockers
                if op in source_constraint_blockers:
                    subgraph_without_blocking_processes = leaf.get_operation_closest_to_source(
                        source_constraint_blockers[op]
                    )
                    if subgraph_without_blocking_processes is not None:
                        leaf = subgraph_without_blocking_processes

                # 2 merge filtering arguments
                args = leaf.get_arguments_by_operation(op)
                if args:
                    if merge:
                        # Take first item (to reproduce original behavior)
                        # TODO: take temporal/spatial/categorical intersection instead?
                        #       see https://github.com/Open-EO/openeo-processes/issues/201
                        constraints[op] = args[0]
                    else:
                        constraints[op] = args

            if "_weak_spatial_extent" in constraints:
                if "spatial_extent" not in constraints:
                    constraints["spatial_extent"] = constraints["_weak_spatial_extent"]
                del constraints["_weak_spatial_extent"]

            source_id = leaf.get_source().get_source_id()
            source_constraints.append((source_id, constraints))
        return source_constraints

    def get_geometries(
            self, operation="aggregate_spatial"
    ) -> List[Union[shapely.geometry.base.BaseGeometry, DelayedVector]]:
        """Get geometries (polygons or DelayedVector), as used by aggregate_spatial"""
        geometries_by_id = {}
        for leaf in self.get_trace_leaves():
            for args in leaf.get_arguments_by_operation(operation):
                if "geometries" in args:
                    geometries = args["geometries"]
                    geometries_by_id[id(geometries)] = geometries
        # TODO: we just pass all (0 or more) geometries we encountered. Do something smarter when there are multiple?
        return list(geometries_by_id.values())


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
            self,
            traces: List[DataTraceBase],
            data_tracer: DryRunDataTracer,
            metadata: CollectionMetadata = None
    ):
        super(DryRunDataCube, self).__init__(metadata=metadata)
        self._traces = traces or []
        self._data_tracer = data_tracer

    def _process(self, operation, arguments) -> 'DryRunDataCube':
        """Helper to handle single-cube operations"""
        # New data cube with operation added to each trace
        traces = self._data_tracer.process_traces(traces=self._traces, operation=operation, arguments=arguments)
        # TODO: manipulate metadata properly?
        return DryRunDataCube(traces=traces, data_tracer=self._data_tracer, metadata=self.metadata)

    def _process_metadata(self, metadata: CollectionMetadata) -> 'DryRunDataCube':
        """Just process metadata (leave traces as is)"""
        return DryRunDataCube(traces=self._traces, data_tracer=self._data_tracer, metadata=metadata)

    def filter_temporal(self, start: str, end: str) -> 'DryRunDataCube':
        return self._process("temporal_extent", (start, end))

    def filter_bbox(
            self, west, south, east, north, crs=None, base=None, height=None, operation="spatial_extent"
    ) -> 'DryRunDataCube':
        return self._process(operation, {"west": west, "south": south, "east": east, "north": north,
                                         "crs": (crs or "EPSG:4326")})

    def filter_spatial(self, geometries):
        geometries, bbox = self._normalize_geometry(geometries)
        cube = self.filter_bbox(**bbox, operation="_weak_spatial_extent")
        return cube._process(operation="filter_spatial", arguments={"geometries": geometries})

    def filter_bands(self, bands) -> 'DryRunDataCube':
        return self._process("bands", bands)

    def filter_properties(self, properties) -> 'DryRunDataCube':
        return self._process("properties", properties)

    def save_result(self, filename: str, format: str, format_options: dict = None) -> str:
        return self._process("save_result", {"format": format, "options": format_options})

    def mask(self, mask: 'DryRunDataCube', replacement=None) -> 'DryRunDataCube':
        # TODO: if mask cube has no temporal or bbox extent: copy from self?
        # TODO: or add reference to the self trace to the mask trace and vice versa?
        cube = self._process("mask", {"mask": mask})
        return DryRunDataCube(
            traces=cube._traces + mask._traces, data_tracer=cube._data_tracer,
            metadata=cube.metadata
        )

    def merge_cubes(self, other: 'DryRunDataCube', overlap_resolver) -> 'DryRunDataCube':
        return DryRunDataCube(
            traces=self._traces + other._traces, data_tracer=self._data_tracer,
            # TODO: properly merge (other) metadata?
            metadata=self.metadata
        )

    def mask_polygon(self, mask, replacement=None, inside: bool = False) -> 'DriverDataCube':
        cube = self
        if not inside and replacement is None:
            mask, bbox = cube._normalize_geometry(mask)
            cube = self.filter_bbox(**bbox, operation="_weak_spatial_extent")
        return cube._process(operation="mask_polygon", arguments={"mask": mask})

    def aggregate_spatial(
            self, geometries: Union[str, dict, DelayedVector, shapely.geometry.base.BaseGeometry],
            reducer, target_dimension: str = "result"
    ) -> AggregatePolygonResult:
        geometries, bbox = self._normalize_geometry(geometries)
        cube = self.filter_bbox(**bbox, operation="_weak_spatial_extent")
        cube._process(operation="aggregate_spatial", arguments={"geometries": geometries})
        return AggregatePolygonResult(timeseries={}, regions=geometries)

    def _normalize_geometry(self, geometries):
        """
        Helper to preprocess geometries (as used in aggregate_spatial and mask_polygon) and apply related filter_bbox
        """
        if isinstance(geometries, dict):
            geometries = geojson_to_geometry(geometries)
            bbox = geometries.bounds
        elif isinstance(geometries, str):
            geometries = DelayedVector(geometries)
            bbox = geometries.bounds
        elif isinstance(geometries, DelayedVector):
            bbox = geometries.bounds
        elif isinstance(geometries, shapely.geometry.base.BaseGeometry):
            bbox = geometries.bounds
        else:
            raise ValueError(geometries)
        bbox = dict(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3], crs="EPSG:4326")
        return geometries, bbox

    # TODO: this is a workaround until vectorcube is fully upgraded
    def raster_to_vector(self):
        return AggregatePolygonResult(timeseries={}, regions=None)

    def zonal_statistics(self, regions, func: str) -> AggregatePolygonResult:
        return self.aggregate_spatial(geometries=regions, reducer=func)

    def resample_cube_spatial(self, target: 'DryRunDataCube', method: str = 'near') -> 'DryRunDataCube':
        cube = self._process("process_type", [ProcessType.FOCAL_SPACE])
        cube = cube._process("resample_cube_spatial", arguments={"target": target, "method": method})
        return DryRunDataCube(
            traces=cube._traces + target._traces, data_tracer=self._data_tracer,
            # TODO: properly merge (other) metadata?
            metadata=self.metadata
        )

    def reduce_dimension(self, reducer, dimension: str, env: EvalEnv) -> 'DryRunDataCube':
        dc = self
        if self.metadata.has_temporal_dimension() and self.metadata.temporal_dimension.name == dimension:
            # TODO: reduce is not necessarily global in call cases
            dc = self._process("process_type", [ProcessType.GLOBAL_TIME])

        return dc._process_metadata(self.metadata.reduce_dimension(dimension_name=dimension))

    def add_dimension(self, name: str, label, type: str = "other") -> 'DryRunDataCube':
        return self._process_metadata(self.metadata.add_dimension(name=name, label=label, type=type))

    def sar_backscatter(self, args: SarBackscatterArgs) -> 'DryRunDataCube':
        return self._process("sar_backscatter", args)

    def resolution_merge(self, args: ResolutionMergeArgs) -> 'DryRunDataCube':
        return self._process("resolution_merge", args)

    def resample_spatial(
            self, resolution: Union[float, Tuple[float, float]], projection: Union[int, str] = None,
            method: str = 'near', align: str = 'upper-left'
    ):
        return self._process(
            "resample_spatial",
            arguments={"resolution": resolution, "projection": projection, "method": method, "align": align}
        )

    def apply_kernel(self, kernel: list, factor=1, border=0, replace_invalid=0) -> 'DriverDataCube':
        return self._process("apply_kernel", arguments={"kernel": kernel})

    def apply_dimension(self, process, dimension: str, target_dimension: str = None, context:dict = None, env: EvalEnv=None) -> 'DriverDataCube':
        cube = self
        if self.metadata.has_temporal_dimension() and self.metadata.temporal_dimension.name == dimension:
            # TODO: reduce is not necessarily global in call cases
            cube = self._process("process_type", [ProcessType.GLOBAL_TIME])

        if target_dimension is not None:
            cube = cube._process_metadata(self.metadata.rename_dimension(source=dimension, target=target_dimension))
        else:
            cube = self
        return cube._process("apply_dimension", arguments={"dimension": dimension})

    def apply_tiles_spatiotemporal(self, process, context={}) -> 'DriverDataCube':
        if (self.metadata.has_temporal_dimension()):
            return self._process("process_type", [ProcessType.GLOBAL_TIME])
        else:
            return self

    def apply_neighborhood(self, process, size: List[dict], overlap: List[dict], env: EvalEnv) -> 'DriverDataCube':
        temporal_size = temporal_overlap = None
        size_dict = {e['dimension']: e for e in size}
        overlap_dict = {e['dimension']: e for e in overlap}
        if self.metadata.has_temporal_dimension():
            temporal_size = size_dict.get(self.metadata.temporal_dimension.name, None)
            temporal_overlap = overlap_dict.get(self.metadata.temporal_dimension.name, None)
            if temporal_size is None or temporal_size.get('value', None) is None:
                return self._process("process_type", [ProcessType.GLOBAL_TIME])
        return self

    def atmospheric_correction(self, method: str = None, *args) -> 'DriverDataCube':
        method_link = "https://remotesensing.vito.be/case/icor"
        if method == "SMAC":
            method_link = "https://doi.org/10.1080/01431169408954055"

        aot_link = "https://atmosphere.copernicus.eu/catalogue#/product/urn:x-wmo:md:int.ecmwf::copernicus:cams:prod:fc:total-aod:pid094"
        # by default GLOBE DEM is used
        dem_doi = "https://doi.org/10.7289/V52R3PMS"
        # default APDA water vapour algorithm
        wvp_doi = "https://doi.org/10.1109/LGRS.2016.2635942"

        return self \
            ._process("log_metadata_link", arguments={"rel": "atmospheric-scattering", "href": method_link}) \
            ._process("log_metadata_link", arguments={"rel": "related", "href": aot_link}) \
            ._process("log_metadata_link", arguments={"rel": "elevation-model", "href": dem_doi}) \
            ._process("log_metadata_link", arguments={"rel": "water-vapor", "href": wvp_doi})

    def mask_scl_dilation(self) -> 'DriverDataCube':
        return self._process("custom_cloud_mask", arguments={"method": "mask_scl_dilation"})

    def _nop(self, *args, **kwargs) -> 'DryRunDataCube':
        """No Operation: do nothing"""
        return self

    # TODO: some methods need metadata manipulation?

    apply = _nop
    apply_tiles = _nop

    reduce = _nop
    reduce_bands = _nop
    aggregate_temporal = _nop
    rename_labels = _nop
    rename_dimension = _nop
    ndvi = _nop
    water_vapor = _nop
    linear_scale_range = _nop
