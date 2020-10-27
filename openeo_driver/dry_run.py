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
from typing import List, Union, Set, Dict

import shapely.geometry.base

from openeo.metadata import CollectionMetadata
from openeo_driver.datacube import DriverDataCube
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.save_result import AggregatePolygonResult
from openeo_driver.utils import geojson_to_geometry, to_hashable, bands_union, temporal_extent_union, \
    spatial_extent_union


class DataTraceBase:
    """Base class for data traces."""

    def __hash__(self):
        # Identity hash (e.g. memory address)
        return id(self)

    def get_source(self) -> 'DataSource':
        raise NotImplementedError

    def get_arguments_by_operation(self, operation: str) -> List[Union[dict, tuple]]:
        return []

    def describe(self) -> str:
        return "_base"


class DataSource(DataTraceBase):
    """Data source: a data (cube) generating process like `load_collection`, `load_disk_data`, ..."""
    __slots__ = ["_process", "_arguments"]

    def __init__(self, process: str = "load_collection", arguments: Union[dict, tuple] = ()):
        self._process = process
        self._arguments = arguments

    def get_source(self) -> 'DataSource':
        return self

    def get_source_id(self) -> tuple:
        """Identifier for source (hashable tuple, to be used as dict key for example)."""
        return to_hashable((self._process, self._arguments))

    def __repr__(self):
        return '<{c}#{i}({p!r}, {a!r})>'.format(
            c=self.__class__.__name__, i=id(self), p=self._process, a=self._arguments
        )

    def describe(self) -> str:
        return self._process

    @classmethod
    def load_collection(cls, collection_id) -> 'DataSource':
        """Factory for a `load_collection` DataSource."""
        return cls(process="load_collection", arguments=(collection_id,))

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
        self.parent = parent
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
        trace = DataSource.load_collection(collection_id=collection_id)
        self.add_trace(trace)
        cube = DryRunDataCube(traces=[trace], data_tracer=self, metadata=metadata)
        if "temporal_extent" in arguments:
            cube = cube.filter_temporal(*arguments["temporal_extent"])
        if "spatial_extent" in arguments:
            cube = cube.filter_bbox(**arguments["spatial_extent"])
        if "bands" in arguments:
            cube = cube.filter_bands(arguments["bands"])
        return cube

    def load_disk_data(self, glob_pattern: str, format: str, options: dict) -> 'DryRunDataCube':
        """Create a DryRunDataCube from a `load_disk_data` process."""
        trace = DataSource.load_disk_data(glob_pattern=glob_pattern, format=format, options=options)
        self.add_trace(trace)
        return DryRunDataCube(traces=[trace], data_tracer=self)

    def get_trace_leaves(self) -> Set[DataTraceBase]:
        """Get all nodes in the tree of traces that are not parent of another trace."""
        leaves = set(self._traces)
        for trace in self._traces:
            while isinstance(trace, DataTrace):
                leaves.discard(trace.parent)
                trace = trace.parent
        return leaves

    def get_source_constraints(self, merge=True) -> Dict[tuple, dict]:
        """
        Get the temporal/spatial constraints of all traced sources

        :param merge:
        :return: dictionary mapping source id (e.g. `("load_collection", "Sentinel2")
            to dictionary with "temporal_extent", "spatial_extent", "bands" fields.
        """
        source_constraints = {}
        for leaf in self.get_trace_leaves():
            constraints = {}
            for op in ["temporal_extent", "spatial_extent", "bands"]:
                args = leaf.get_arguments_by_operation(op)
                if args:
                    if merge:
                        # Take first item (to reproduce original behavior)
                        # TODO: take temporal/spatial/categorical intersection instead?
                        #       see https://github.com/Open-EO/openeo-processes/issues/201
                        constraints[op] = args[0]
                    else:
                        constraints[op] = args
            source_id = leaf.get_source().get_source_id()
            if merge:
                if source_id in source_constraints:
                    # Merge: take union where necessary
                    for field, value in constraints.items():
                        orig = source_constraints[source_id].get(field)
                        if orig:
                            if field == "bands":
                                source_constraints[source_id][field] = bands_union(orig, value)
                            elif field == "temporal_extent":
                                source_constraints[source_id][field] = temporal_extent_union(orig, value)
                            elif field == "spatial_extent":
                                source_constraints[source_id][field] = spatial_extent_union(orig, value)
                            else:
                                raise ValueError(field)
                        else:
                            source_constraints[source_id][field] = value
                else:
                    source_constraints[source_id] = constraints
            else:
                source_constraints[source_id] = source_constraints.get(source_id, []) + [constraints]
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
        # New data cube with operation added to each trace
        traces = self._data_tracer.process_traces(traces=self._traces, operation=operation, arguments=arguments)
        # TODO: manipulate metadata properly?
        return DryRunDataCube(traces=traces, data_tracer=self._data_tracer, metadata=self.metadata)

    def filter_temporal(self, start: str, end: str) -> 'DryRunDataCube':
        return self._process("temporal_extent", (start, end))

    def filter_bbox(self, west, south, east, north, crs=None, base=None, height=None) -> 'DryRunDataCube':
        return self._process("spatial_extent", {"west": west, "south": south, "east": east, "north": north, "crs": crs})

    def filter_bands(self, bands) -> 'DryRunDataCube':
        return self._process("bands", bands)

    def mask(self, mask: 'DryRunDataCube', replacement=None) -> 'DryRunDataCube':
        # TODO: if mask cube has no temporal or bbox extent: copy from self?
        # TODO: or add reference to the self trace to the mask trace and vice versa?
        return DryRunDataCube(traces=self._traces + mask._traces, data_tracer=self._data_tracer)

    def merge_cubes(self, other: 'DryRunDataCube', overlap_resolver) -> 'DryRunDataCube':
        return DryRunDataCube(traces=self._traces + other._traces, data_tracer=self._data_tracer)

    def aggregate_spatial(
            self, geometries: Union[str, dict, DelayedVector, shapely.geometry.base.BaseGeometry],
            reducer, target_dimension: str = "result"
    ) -> AggregatePolygonResult:
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
        cube = self.filter_bbox(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3], crs="EPSG:4326")
        cube._process(operation="aggregate_spatial", arguments={"geometries": geometries})
        return AggregatePolygonResult(timeseries={}, regions=geometries)

    def zonal_statistics(self, regions, func: str) -> AggregatePolygonResult:
        return self.aggregate_spatial(geometries=regions, reducer=func)

    def resample_cube_spatial(self, target: 'DryRunDataCube', method: str = 'near') -> 'DryRunDataCube':
        # TODO: EP3561 record resampling operation
        return self

    def _nop(self, *args, **kwargs) -> 'DryRunDataCube':
        """No Operation: do nothing"""
        return self

    # TODO: some methods need metadata manipulation?
    apply_kernel = _nop
    apply_neighborhood = _nop
    apply = _nop
    apply_tiles = _nop
    apply_tiles_spatiotemporal = _nop
    apply_dimension = _nop
    reduce = _nop
    reduce_dimension = _nop
    reduce_bands = _nop
    mask_polygon = _nop
    add_dimension = _nop
    aggregate_temporal = _nop
    rename_labels = _nop
    rename_dimension = _nop
    ndvi = _nop
