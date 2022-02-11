# TODO: rename this module to something in snake case? It doesn't even implement a ProcessGraphDeserializer class.

# pylint: disable=unused-argument
import datetime
import logging
import tempfile
import time
import warnings
import calendar
from pathlib import Path
from typing import Dict, Callable, List, Union, Tuple, Any

import geopandas as gpd
import numpy as np
import openeo_processes
import requests
from dateutil.relativedelta import relativedelta
from shapely.geometry import shape, mapping, MultiPolygon

import openeo.udf
from openeo.capabilities import ComparableVersion
from openeo.metadata import CollectionMetadata, MetadataException
from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo.util import load_json, rfc3339
from openeo_driver import dry_run
from openeo_driver.backend import UserDefinedProcessMetadata, LoadParameters, Processing, OpenEoBackendImplementation
from openeo_driver.datacube import DriverDataCube
from openeo_driver.datastructs import SarBackscatterArgs, ResolutionMergeArgs
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.dry_run import DryRunDataTracer, SourceConstraint
from openeo_driver.errors import ProcessParameterRequiredException, ProcessParameterInvalidException, \
    FeatureUnsupportedException, OpenEOApiException, ProcessGraphInvalidException
from openeo_driver.errors import ProcessUnsupportedException
from openeo_driver.processes import ProcessRegistry, ProcessSpec, DEFAULT_NAMESPACE
from openeo_driver.save_result import ImageCollectionResult, JSONResult, SaveResult, AggregatePolygonResult, NullResult
from openeo_driver.specs import SPECS_ROOT, read_spec
from openeo_driver.util.utm import auto_utm_epsg_for_geometry
from openeo_driver.utils import smart_bool, EvalEnv, geojson_to_geometry, spatial_extent_union, geojson_to_multipolygon

_log = logging.getLogger(__name__)

# Set up process registries (version dependent)
process_registry_040 = ProcessRegistry(spec_root=SPECS_ROOT / 'openeo-processes/0.4', argument_names=["args", "env"])
process_registry_100 = ProcessRegistry(spec_root=SPECS_ROOT / 'openeo-processes/1.x', argument_names=["args", "env"])

# Bootstrap with some mathematical/logical processes
process_registry_040.add_spec_by_name(
    'array_contains', 'array_element',
    'count', 'first', 'last', 'order', 'rearrange', 'sort',
    'between', 'eq', 'gt', 'gte', 'if', 'is_nan', 'is_nodata', 'is_valid', 'lt', 'lte', 'neq',
    'and', 'if', 'not', 'or', 'xor',
    'absolute', 'clip', 'divide', 'extrema', 'int', 'max', 'mean',
    'median', 'min', 'mod', 'multiply', 'power', 'product', 'quantiles', 'sd', 'sgn', 'sqrt',
    'subtract', 'sum', 'variance', 'e', 'pi', 'exp', 'ln', 'log',
    'ceil', 'floor', 'int', 'round',
    'arccos', 'arcosh', 'arcsin', 'arctan', 'arctan2', 'arsinh', 'artanh', 'cos', 'cosh', 'sin', 'sinh', 'tan', 'tanh',
    'count', 'first', 'last', 'max', 'mean', 'median', 'min', 'product', 'sd', 'sum', 'variance'
)


def _add_standard_processes(process_registry: ProcessRegistry, process_ids: List[str]):
    """
    Add standard processes as implemented by the openeo-processes-python project.
    """

    def wrap(process: Callable):
        """Adapter to connect the kwargs style of openeo-processes-python with args/EvalEnv"""

        def wrapped(args: dict, env: EvalEnv):
            return process(**args)

        return wrapped

    for pid in set(process_ids):
        if openeo_processes.has_process(pid):
            proc = openeo_processes.get_process(pid)
            wrapped = wrap(proc)
            spec = process_registry.load_predefined_spec(pid)
            process_registry.add_process(name=pid, function=wrapped, spec=spec)
        else:
            # TODO: this warning is triggered before logging is set up usually
            _log.warning("Adding process {p!r} without implementation".format(p=pid))
            process_registry.add_spec_by_name(pid)


_OPENEO_PROCESSES_PYTHON_WHITELIST = [
    'array_apply', 'array_contains', 'array_element', 'array_filter', 'array_find', 'array_labels',
    'count', 'first', 'last', 'order', 'rearrange', 'sort',
    'between', 'eq', 'gt', 'gte', 'if', 'is_nan', 'is_nodata', 'is_valid', 'lt', 'lte', 'neq',
    'all', 'and', 'any', 'if', 'not', 'or', 'xor',
    'absolute', 'add', 'clip', 'divide', 'extrema', 'int', 'max', 'mean',
    'median', 'min', 'mod', 'multiply', 'power', 'product', 'quantiles', 'sd', 'sgn', 'sqrt',
    'subtract', 'sum', 'variance', 'e', 'pi', 'exp', 'ln', 'log',
    'ceil', 'floor', 'int', 'round',
    'arccos', 'arcosh', 'arcsin', 'arctan', 'arctan2', 'arsinh', 'artanh', 'cos', 'cosh', 'sin', 'sinh', 'tan', 'tanh',
    'all', 'any', 'count', 'first', 'last', 'max', 'mean', 'median', 'min', 'product', 'sd', 'sum', 'variance'
]

_add_standard_processes(process_registry_100, _OPENEO_PROCESSES_PYTHON_WHITELIST)


# Type hint alias for a "process function":
# a Python function that implements some openEO process (as used in `apply_process`)
ProcessFunction = Callable[[dict, EvalEnv], Any]


def process(f: ProcessFunction) -> ProcessFunction:
    """Decorator for registering a process function in the process registries"""
    process_registry_040.add_function(f)
    process_registry_100.add_function(f)
    return f


# Decorator for registering deprecate/old process functions
deprecated_process = process_registry_040.add_deprecated


def non_standard_process(spec: ProcessSpec) -> Callable[[ProcessFunction], ProcessFunction]:
    """Decorator for registering non-standard process functions"""

    def decorator(f: ProcessFunction) -> ProcessFunction:
        process_registry_040.add_function(f=f, spec=spec.to_dict_040())
        process_registry_100.add_function(f=f, spec=spec.to_dict_100())
        return f

    return decorator


def custom_process(f: ProcessFunction):
    """Decorator for custom processes (e.g. in custom_processes.py)."""
    process_registry_040.add_hidden(f)
    process_registry_100.add_hidden(f)
    return f


def custom_process_from_process_graph(
        process_spec: Union[dict, Path],
        process_registry: ProcessRegistry = process_registry_100,
        namespace: str = DEFAULT_NAMESPACE
):
    """
    Register a custom process from a process spec containing a "process_graph" definition

    :param process_spec: process spec dict or path to a JSON file,
        containing keys like "id", "process_graph", "parameter"
    :param process_registry: process registry to register to
    """
    # TODO: option to hide process graph for (public) listing
    if isinstance(process_spec, Path):
        process_spec = load_json(process_spec)
    process_id = process_spec["id"]
    process_function = _process_function_from_process_graph(process_spec)
    process_registry.add_function(process_function, name=process_id, spec=process_spec, namespace=namespace)


def _process_function_from_process_graph(process_spec: dict) -> ProcessFunction:
    """
    Build a process function (to be used in `apply_process`) from a given process spec with process graph

    :param process_spec: process spec dict, containing keys like "id", "process_graph", "parameter"
    :return: process function
    """
    process_id = process_spec["id"]
    process_graph = process_spec["process_graph"]
    parameters = process_spec.get("parameters")

    def process_function(args: dict, env: EvalEnv):
        return _evaluate_process_graph_process(
            process_id=process_id, process_graph=process_graph, parameters=parameters,
            args=args, env=env
        )

    return process_function


def _register_fallback_implementations_by_process_graph(process_registry: ProcessRegistry = process_registry_100):
    """
    Register process functions for (yet undefined) processes that have
    a process graph based fallback implementation in their spec
    """
    for name in process_registry.list_predefined_specs():
        spec = process_registry.load_predefined_spec(name)
        if "process_graph" in spec and not process_registry.contains(name):
            _log.info(f"Registering fallback implementation of {name!r} by process graph ({process_registry})")
            custom_process_from_process_graph(process_spec=spec, process_registry=process_registry)


# Some (env) string constants to simplify code navigation
ENV_SOURCE_CONSTRAINTS = "source_constraints"
ENV_DRY_RUN_TRACER = "dry_run_tracer"


class SimpleProcessing(Processing):
    """
    Simple graph processing: just implement basic math/logic operators
    (based on openeo-processes-python implementation)
    """

    # For lazy loading of (global) process registry
    _registry_cache = {}

    def get_process_registry(self, api_version: Union[str, ComparableVersion]) -> ProcessRegistry:
        # Lazy load registry.
        assert ComparableVersion("1.0.0").or_higher(api_version)
        spec = 'openeo-processes/1.x'
        if spec not in self._registry_cache:
            registry = ProcessRegistry(spec_root=SPECS_ROOT / spec, argument_names=["args", "env"])
            _add_standard_processes(registry, _OPENEO_PROCESSES_PYTHON_WHITELIST)
            self._registry_cache[spec] = registry
        return self._registry_cache[spec]

    def get_basic_env(self, api_version=None) -> EvalEnv:
        return EvalEnv({
            "backend_implementation": OpenEoBackendImplementation(processing=self),
            "version": api_version or "1.0.0",  # TODO: get better default api version from somewhere?
        })

    def evaluate(self, process_graph: dict, env: EvalEnv = None):
        return evaluate(process_graph=process_graph, env=env or self.get_basic_env(), do_dry_run=False)


class ConcreteProcessing(Processing):
    """
    Concrete process graph processing: (most) processes have concrete Python implementation
    (manipulating `DriverDataCube` instances)
    """
    def get_process_registry(self, api_version: Union[str, ComparableVersion]) -> ProcessRegistry:
        if ComparableVersion("1.0.0").or_higher(api_version):
            return process_registry_100
        else:
            return process_registry_040

    def evaluate(self, process_graph: dict, env: EvalEnv = None):
        return evaluate(process_graph=process_graph, env=env)


def evaluate(
        process_graph: dict,
        env: EvalEnv,
        do_dry_run: Union[bool, DryRunDataTracer] = True
) -> Union[DriverDataCube, Any]:
    """
    Converts the json representation of a (part of a) process graph into the corresponding Python data cube.
    """

    if 'version' not in env:
        # TODO: make this a hard error, and stop defaulting to 0.4.0.
        warnings.warn("Blindly assuming 0.4.0")
        env = env.push({"version": "0.4.0"})

    top_level_node = ProcessGraphVisitor.dereference_from_node_arguments(process_graph)
    result_node = process_graph[top_level_node]

    if do_dry_run:
        dry_run_tracer = do_dry_run if isinstance(do_dry_run, DryRunDataTracer) else DryRunDataTracer()
        _log.info("Doing dry run")
        convert_node(result_node, env=env.push({ENV_DRY_RUN_TRACER: dry_run_tracer}))
        # TODO: work with a dedicated DryRunEvalEnv?
        source_constraints = dry_run_tracer.get_source_constraints()
        _log.info("Dry run extracted these source constraints: {s}".format(s=source_constraints))
        env = env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})

    return convert_node(result_node, env=env)


def convert_node(processGraph: Union[dict, list], env: EvalEnv = None):
    if isinstance(processGraph, dict):
        if 'process_id' in processGraph:
            return apply_process(
                process_id=processGraph['process_id'], args=processGraph.get('arguments', {}),
                namespace=processGraph.get("namespace", None), env=env
            )
        elif 'node' in processGraph:
            return convert_node(processGraph['node'], env=env)
        elif 'callback' in processGraph or 'process_graph' in processGraph:
            # a "process_graph" object is a new process graph, don't evaluate it in the parent graph
            return processGraph
        elif 'from_parameter' in processGraph:
            try:
                parameters = env.collect_parameters()
                return parameters[processGraph['from_parameter']]
            except KeyError:
                raise ProcessParameterRequiredException(process="n/a", parameter=processGraph['from_parameter'])
        elif 'from_argument' in processGraph:
            # 0.4-style argument referencer (equivalent with 1.0-style "from_parameter")
            argument_reference = processGraph.get('from_argument')
            # backwards compatibility for clients that still use 'dimension_data', can be removed when clients are upgraded
            if argument_reference == 'dimension_data':
                argument_reference = 'data'
            parameters = env.collect_parameters()
            return parameters.get(argument_reference)
        else:
            # TODO: Don't apply `convert_node` for some special cases (e.g. geojson objects)?
            return {k:convert_node(v, env=env) for k,v in processGraph.items()}
    elif isinstance(processGraph, list):
        return [convert_node(x, env=env) for x in processGraph]
    return processGraph


def extract_arg(args: dict, name: str, process_id='n/a'):
    """Get process argument by name."""
    # TODO: support optional default value for optional parameters?
    try:
        return args[name]
    except KeyError:
        # TODO: automate argument extraction directly from process spec instead of these exract_* functions?
        raise ProcessParameterRequiredException(process=process_id, parameter=name)


def extract_arg_list(args: dict, names: list):
    """Get process argument by list of (legacy/fallback/...) names."""
    for name in names:
        if name in args:
            return args[name]
    # TODO: find out process id for proper error message?
    raise ProcessParameterRequiredException(process='n/a', parameter=str(names))


def extract_deep(args: dict, *steps):
    """
    Walk recursively through a dictionary to get to a value.
    Also support trying multiple (legacy/fallback/...) keys at a certain level: specify step as a list of options
    """
    value = args
    for step in steps:
        keys = [step] if not isinstance(step, list) else step
        for key in keys:
            if key in value:
                value = value[key]
                break
        else:
            # TODO: find out process id for proper error message?
            raise ProcessParameterInvalidException(process="n/a", parameter=steps[0], reason=step)
    return value


def extract_args_subset(args: dict, keys: List[str], aliases: Dict[str, str] = None) -> dict:
    """
    Extract subset of given keys (where available) from given dictionary,
    possibly handling legacy aliases

    :param args: dictionary of arguments
    :param keys: keys to extract
    :param aliases: mapping of (legacy) alias to target key
    :return:
    """
    kwargs = {k: args[k] for k in keys if k in args}
    if aliases:
        for alias, key in aliases.items():
            if alias in args and key not in kwargs:
                kwargs[key] = args[alias]
    return kwargs


def extract_arg_enum(args: dict, name: str, enum_values: Union[set, list, tuple], process_id='n/a'):
    """Get process argument by name and check if it is proper enum value."""
    # TODO: support optional default value for optional parameters?
    value = extract_arg(args=args, name=name, process_id=process_id)
    if value not in enum_values:
        raise ProcessParameterInvalidException(
            parameter=name, process=process_id, reason=f"Invalid enum value {value!r}"
        )
    return value


def _extract_load_parameters(env: EvalEnv, source_id: tuple) -> LoadParameters:
    source_constraints: List[SourceConstraint] = env[ENV_SOURCE_CONSTRAINTS]
    global_extent = None
    process_types = set()
    for _, constraint in source_constraints:
        if "spatial_extent" in constraint:
            extent = constraint["spatial_extent"]
            global_extent = spatial_extent_union(global_extent, extent) if global_extent else extent
        if "process_type" in constraint:
            process_types |= set(constraint["process_type"])

    _, constraints = source_constraints.pop(0)
    params = LoadParameters()
    params.temporal_extent = constraints.get("temporal_extent", ["1970-01-01", "2070-01-01"])
    params.spatial_extent = constraints.get("spatial_extent", {})
    params.global_extent = global_extent
    params.bands = constraints.get("bands", None)
    params.properties = constraints.get("properties", {})
    params.aggregate_spatial_geometries = constraints.get("aggregate_spatial", {}).get("geometries")
    if params.aggregate_spatial_geometries == None:
        params.aggregate_spatial_geometries = constraints.get("filter_spatial", {}).get("geometries")
    params.sar_backscatter = constraints.get("sar_backscatter", None)
    params.process_types = process_types
    params.custom_mask = constraints.get("custom_cloud_mask", {})
    params.data_mask = env.get("data_mask",None)
    params.target_crs = constraints.get("resample", {}).get("target_crs",None)
    params.target_resolution = constraints.get("resample", {}).get("resolution", None)
    return params


@process
def load_collection(args: dict, env: EvalEnv) -> DriverDataCube:
    collection_id = extract_arg(args, 'id')

    # Sanitized arguments
    arguments = {}
    if args.get("temporal_extent"):
        arguments["temporal_extent"] = _extract_temporal_extent(
            args, field="temporal_extent", process_id="load_collection"
        )
    if args.get("spatial_extent"):
        arguments["spatial_extent"] = _extract_bbox_extent(
            args, field="spatial_extent", process_id="load_collection", handle_geojson=True
        )
        # TODO when spatial_extent is geojson: additional mask_polygon operation? https://github.com/Open-EO/openeo-python-driver/issues/49
    if args.get("bands"):
        arguments["bands"] = extract_arg(args, "bands", process_id="load_collection")
    if args.get("properties"):
        arguments["properties"] = extract_arg(args, 'properties', process_id="load_collection")

    if args.get("featureflags"):
        arguments["featureflags"] = extract_arg(args, 'featureflags', process_id="load_collection")

    metadata = env.backend_implementation.catalog.get_collection_metadata(collection_id)

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        return dry_run_tracer.load_collection(collection_id=collection_id, arguments=arguments, metadata=metadata)
    else:
        # Extract basic source constraints.
        properties = {**CollectionMetadata(metadata).get("_vito", "properties", default={}),
                      **arguments.get("properties", {})}

        source_id = dry_run.DataSource.load_collection(collection_id=collection_id,
                                                       properties=properties).get_source_id()
        load_params = _extract_load_parameters(env, source_id=source_id)
        # Override with explicit arguments
        load_params.update(arguments)
        return env.backend_implementation.catalog.load_collection(collection_id, load_params=load_params, env=env)


@non_standard_process(
    ProcessSpec(id='load_disk_data', description="Loads arbitrary from disk.")
        .param(name='format', description="the file format, e.g. 'GTiff'", schema={"type": "string"}, required=True)
        .param(name='glob_pattern', description="a glob pattern that matches the files to load from disk",
               schema={"type": "string"}, required=True)
        .param(name='options', description="options specific to the file format", schema={"type": "object"})
        .returns(description="the data as a data cube", schema={})
)
def load_disk_data(args: Dict, env: EvalEnv) -> DriverDataCube:
    # TODO: rename this to "load_uploaded_files" like in official openeo processes?
    kwargs = dict(
        glob_pattern=extract_arg(args, 'glob_pattern'),
        format=extract_arg(args, 'format'),
        options=args.get('options', {}),
    )
    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        return dry_run_tracer.load_disk_data(**kwargs)
    else:
        source_id = dry_run.DataSource.load_disk_data(**kwargs).get_source_id()
        load_params = _extract_load_parameters(env, source_id=source_id)
        return env.backend_implementation.load_disk_data(**kwargs, load_params=load_params, env=env)


@non_standard_process(
    ProcessSpec(id='vector_buffer', description="Add a buffer around a geometry.")
        .param(name='geometry', description="Input geometry (GeoJSON object) to add buffer to.",
               schema={"type": "object", "subtype": "geojson"}, required=True)
        .param(name='distance', description="The size of the buffer. Can be negative to subtract the buffer",
               schema={"type": "number"}, required=True)
        .param(name='unit', description="The unit in which the distance is measured.",
               schema={"type": "string", "enum": ["meter", "kilometer"]})
        .returns(description="Output geometry (GeoJSON object) with the added or subtracted buffer",
                 schema={"type": "object", "subtype": "geojson"})
)
def vector_buffer(args: Dict, env: EvalEnv) -> dict:
    geometry = extract_arg(args, 'geometry')
    distance = extract_arg(args, 'distance')
    unit = extract_arg(args, 'unit')
    input_crs = 'epsg:4326'
    buffer_resolution = 3

    if isinstance(geometry, str):
        geoms = list(DelayedVector(geometry).geometries)
    elif isinstance(geometry, dict):
        if geometry["type"] == "FeatureCollection":
            geoms = [shape(feat["geometry"]) for feat in geometry["features"]]
        elif geometry["type"] == "GeometryCollection":
            geoms = [shape(geom) for geom in geometry["geometries"]]
        else:
            geoms = [shape(geometry)]
    else:
        raise ProcessParameterInvalidException(
            parameter="geometry", process="vector_buffer", reason="The input geometry cannot be parsed"
        )
    geoms = gpd.GeoSeries(geoms, crs=input_crs)

    unit_scaling = {"meter": 1, "kilometer": 1000}
    if unit not in unit_scaling:
        raise ProcessParameterInvalidException(
            parameter="unit", process="vector_buffer",
            reason=f"Invalid unit {unit!r}. Should be one of {list(unit_scaling.keys())}."
        )
    distance = distance * unit_scaling[unit]

    epsg_utmzone = auto_utm_epsg_for_geometry(geoms.geometry[0])

    poly_buff_latlon = geoms.to_crs(epsg_utmzone).buffer(distance, resolution=buffer_resolution).to_crs(input_crs)
    return mapping(poly_buff_latlon[0]) if len(poly_buff_latlon) == 1 else mapping(poly_buff_latlon)


@process_registry_100.add_function
def apply_neighborhood(args: dict, env: EvalEnv) -> DriverDataCube:
    process = extract_deep(args, "process", "process_graph")
    size = extract_arg(args, 'size')
    overlap = extract_arg(args, 'overlap')
    # TODO: pass context?
    context = args.get('context', {})
    data_cube = extract_arg(args, 'data')
    return data_cube.apply_neighborhood(process=process, size=size, overlap=overlap, env=env)

@process
def apply_dimension(args: Dict, env: EvalEnv) -> DriverDataCube:
    process = extract_deep(args, 'process', "process_graph")
    dimension = extract_arg(args, 'dimension')
    target_dimension = args.get('target_dimension',None)
    data_cube = extract_arg(args, 'data')
    context = args.get('context',None)

    # do check_dimension here for error handling
    dimension, band_dim, temporal_dim = _check_dimension(cube=data_cube, dim=dimension, process="apply_dimension")

    transformed_collection = data_cube.apply_dimension(process, dimension,target_dimension=target_dimension,context=context)
    if target_dimension is not None and target_dimension not in transformed_collection.metadata.dimension_names():
        transformed_collection.rename_dimension(dimension, target_dimension)
    return transformed_collection


@process
def save_result(args: Dict, env: EvalEnv) -> SaveResult:
    format = extract_arg(args, 'format')
    options = args.get('options', {})
    data = extract_arg(args, 'data')

    if isinstance(data, SaveResult):
        data.set_format(format, options)
        return data
    elif isinstance(data, DriverDataCube):
        return ImageCollectionResult(data, format=format, options=options)
    elif isinstance(data, DelayedVector):
        geojsons = (mapping(geometry) for geometry in data.geometries)
        return JSONResult(geojsons)
    elif data is None:
        return data
    else:
        # Assume generic JSON result
        return JSONResult(data, format, options)


@process
def apply(args: dict, env: EvalEnv) -> DriverDataCube:
    """
    Applies a unary process (a local operation) to each value of the specified or all dimensions in the data cube.
    """
    if ComparableVersion("1.0.0").or_higher(env["version"]):
        apply_pg = extract_deep(args, "process", "process_graph")
        data_cube = extract_arg(args, 'data','apply')
        context = args.get('context',{})

        return data_cube.apply(apply_pg,context)
    else:
        return _evaluate_sub_process_graph(args, 'process', parent_process='apply', env=env)


@process_registry_040.add_function
def reduce(args: dict, env: EvalEnv) -> DriverDataCube:
    """
    https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#reduce
    """
    reduce_pg = extract_deep(args, "reducer", ["process_graph", "callback"])
    dimension = extract_arg(args, 'dimension')
    binary = smart_bool(args.get('binary', False))
    data_cube = extract_arg(args, 'data')

    # TODO: avoid special case handling for run_udf?
    dimension, band_dim, temporal_dim = _check_dimension(cube=data_cube, dim=dimension, process="reduce")
    if dimension == band_dim:
        if not binary and len(reduce_pg) == 1 and next(iter(reduce_pg.values())).get('process_id') == 'run_udf':
            return _evaluate_sub_process_graph(args, 'reducer', parent_process='reduce', env=env)
        visitor = env.backend_implementation.visit_process_graph(reduce_pg)
        return data_cube.reduce_bands(visitor)
    else:
        return _evaluate_sub_process_graph(args, 'reducer', parent_process='reduce', env=env)


@process_registry_100.add_function
def reduce_dimension(args: dict, env: EvalEnv) -> DriverDataCube:
    reduce_pg = extract_deep(args, "reducer", "process_graph")
    dimension = extract_arg(args, 'dimension')
    data_cube = extract_arg(args, 'data')

    # do check_dimension here for error handling
    dimension, band_dim, temporal_dim = _check_dimension(cube=data_cube, dim=dimension, process="reduce_dimension")
    return data_cube.reduce_dimension(reducer=reduce_pg, dimension=dimension, env=env)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/chunk_polygon.json"))
def chunk_polygon(args: dict, env: EvalEnv) -> DriverDataCube:
    import shapely
    reduce_pg = extract_deep(args, "process", "process_graph")
    chunks = extract_arg(args, 'chunks')
    mask_value = args.get('mask_value', None)
    data_cube = extract_arg(args, 'data')

    # Chunks parameter check.
    if isinstance(chunks, DelayedVector):
        polygons = list(chunks.geometries)
        for p in polygons:
            reason = "{m!s} is not a polygon.".format(m=p)
            raise ProcessParameterInvalidException(parameter='chunks', process='chunk_polygon', reason=reason)
        polygon = MultiPolygon(polygons)
    elif isinstance(chunks, shapely.geometry.base.BaseGeometry):
        polygon = MultiPolygon(chunks)
    elif isinstance(chunks, dict):
        polygon = geojson_to_multipolygon(chunks)
        if isinstance(polygon, shapely.geometry.Polygon):
            polygon = MultiPolygon([polygon])
    elif isinstance(chunks, str):
        # Delayed vector is not supported yet.
        reason = "Polygon of type string is not yet supported."
        raise ProcessParameterInvalidException(parameter='chunks', process='chunk_polygon', reason=reason)
    else:
        reason = "Polygon type is not supported."
        raise ProcessParameterInvalidException(parameter='chunks', process='chunk_polygon', reason=reason)
    if polygon.area == 0:
        reason = "Polygon {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
        raise ProcessParameterInvalidException(parameter='chunks', process='chunk_polygon', reason=reason)

    # Mask_value parameter check.
    if not isinstance(mask_value, float):
        reason = "mask_value parameter is not of type float. Actual type: {m!s}".format(m=type(mask_value))
        raise ProcessParameterInvalidException(parameter='mask_value', process='chunk_polygon', reason=reason)
    return data_cube.chunk_polygon(reducer=reduce_pg, chunks=polygon, mask_value=mask_value, env=env)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/fit_class_random_forest.json"))
def fit_class_random_forest(args: dict, env: EvalEnv) -> SaveResult:
    data_cube = extract_arg(args, 'data')
    predictors = extract_arg(args, 'predictors')
    target = extract_arg(args, 'target')
    training = int(extract_arg(args, 'training'))
    num_trees = int(extract_arg(args, 'num_trees'))
    mtry = args.get('mtry', None)
    return data_cube.fit_class_random_forest(predictors=predictors, target=target,
                                             training=training, num_trees=num_trees, mtry=mtry)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/predict_random_forest.json"))
def predict_random_forest(args: dict, env: EvalEnv) -> SaveResult:
    pass


@process
def add_dimension(args: dict, env: EvalEnv) -> DriverDataCube:
    data_cube = extract_arg(args, 'data')
    return data_cube.add_dimension(
        name=extract_arg(args, 'name'),
        label=extract_arg_list(args, ['label', 'value']),
        type=args.get("type", "other"),
    )


@process_registry_100.add_function
def drop_dimension(args: dict, env: EvalEnv) -> DriverDataCube:
    data_cube = extract_arg(args, 'data')
    return data_cube.drop_dimension(name=extract_arg(args, 'name'))


@process_registry_100.add_function
def dimension_labels(args: dict, env: EvalEnv) -> DriverDataCube:
    data_cube = extract_arg(args, 'data')
    return data_cube.dimension_labels(dimension=extract_arg(args, 'dimension'))

@process_registry_100.add_function
def rename_dimension(args: dict, env: EvalEnv) -> DriverDataCube:
    data_cube = extract_arg(args, 'data')
    return data_cube.rename_dimension(source=extract_arg(args, 'source'),target=extract_arg(args, 'target'))

@process_registry_100.add_function
def rename_labels(args: dict, env: EvalEnv) -> DriverDataCube:
    data_cube = extract_arg(args, 'data')
    return data_cube.rename_labels(
        dimension=extract_arg(args, 'dimension'),
        target=extract_arg(args, 'target'),
        source=args.get('source',[])
    )


def _check_dimension(cube: DriverDataCube, dim: str, process: str):
    """
    Helper to check/validate the requested and available dimensions of a cube.

    :return: tuple (requested dimension, name of band dimension, name of temporal dimension)
    """
    # Note: large part of this is support/adapting for old client
    # (pre https://github.com/Open-EO/openeo-python-client/issues/93)
    # TODO remove this legacy support when not necessary anymore
    metadata = cube.metadata
    try:
        band_dim = metadata.band_dimension.name
    except MetadataException:
        band_dim = None
    try:
        temporal_dim = metadata.temporal_dimension.name
    except MetadataException:
        temporal_dim = None

    if dim not in metadata.dimension_names():
        if dim in ["spectral_bands", "bands"] and band_dim:
            _log.warning("Probably old client requesting band dimension {d!r},"
                         " but actual band dimension name is {n!r}".format(d=dim, n=band_dim))
            dim = band_dim
        elif dim == "temporal" and temporal_dim:
            _log.warning("Probably old client requesting temporal dimension {d!r},"
                         " but actual temporal dimension name is {n!r}".format(d=dim, n=temporal_dim))
            dim = temporal_dim
        else:
            raise ProcessParameterInvalidException(
                parameter="dimension", process=process,
                reason="got {d!r}, but should be one of {n!r}".format(d=dim, n=metadata.dimension_names()))

    return dim, band_dim, temporal_dim


@process
def aggregate_temporal(args: dict, env: EvalEnv) -> DriverDataCube:
    data_cube = extract_arg(args, 'data')

    reduce_pg = extract_deep(args, "reducer", "process_graph")
    context = args.get('context', None)
    intervals = extract_arg(args, 'intervals')
    labels = args.get('labels', None)

    dimension = _get_time_dim_or_default(args, data_cube)
    return data_cube.aggregate_temporal(intervals=intervals,labels=labels,reducer=reduce_pg, dimension=dimension, context=context)


@process_registry_100.add_function
def aggregate_temporal_period(args: dict, env: EvalEnv) -> DriverDataCube:
    data_cube = extract_arg(args, 'data')

    reduce_pg = extract_deep(args, "reducer", "process_graph")

    context = args.get('context', None)
    period = extract_arg(args, 'period')

    dimension = _get_time_dim_or_default(args, data_cube, "aggregate_temporal_period")

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        intervals = []
    else:
        temporal_extent = data_cube.metadata.temporal_dimension.extent
        start = temporal_extent[0]
        end = temporal_extent[1]
        intervals = _period_to_intervals(start, end, period)

    return data_cube.aggregate_temporal(intervals=intervals, labels=None, reducer=reduce_pg, dimension=dimension,
                                        context=context)


def _period_to_intervals(start, end, period):
    from datetime import datetime, timedelta
    import pandas as pd
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    intervals = []
    if "week" == period:
        offset = timedelta(weeks=1)
        start_dates = pd.date_range(start - offset, end, freq='W', closed='left')
        end_dates = pd.date_range(start, end + offset, freq='W', closed='left')
        intervals = zip(start_dates, end_dates)
    elif "month" == period:
        offset = timedelta(weeks=4)
        start_dates = pd.date_range(start - offset, end, freq='MS', closed='left')
        end_dates = pd.date_range(start_dates[0] + timedelta(weeks=3), end + offset, freq='MS', closed='left')
        intervals = zip(start_dates, end_dates)
    elif "day" == period:
        offset = timedelta(days=1)
        start_dates = pd.date_range(start - offset, end, freq='D', closed='left')
        end_dates = pd.date_range(start, end + offset, freq='D', closed='left')
        intervals = zip(start_dates, end_dates)
    elif "dekad" == period:
        offset = timedelta(days=10)
        start_dates = pd.date_range(start - offset, end, freq='MS', closed='left')
        ten_days = pd.Timedelta(days=10)
        first_dekad_month = [(date,date+ten_days) for date in start_dates]
        second_dekad_month = [(date + ten_days, date + ten_days+ten_days ) for date in start_dates]
        end_month = [ date + pd.Timedelta(days=calendar.monthrange(date.year,date.month)[1]) for date in start_dates]

        third_dekad_month = list(zip([ date + ten_days+ten_days  for date in start_dates],end_month))
        intervals = (first_dekad_month + second_dekad_month + third_dekad_month)
        intervals.sort( key = lambda t:t[0])

    else:
        raise ProcessParameterInvalidException('period', 'aggregate_temporal_period',
                                               'No support for a period of type: ' + str(period))
    intervals = [i for i in intervals if i[0] < end ]
    return list(intervals)


def _get_time_dim_or_default(args, data_cube, process_id =  "aggregate_temporal"):
    dimension = args.get('dimension', None)
    if dimension is not None:
        dimension, _, _ = _check_dimension(cube=data_cube, dim=dimension, process=process_id)
    else:
        # default: there is a single temporal dimension
        try:
            dimension = data_cube.metadata.temporal_dimension.name
        except MetadataException:
            raise ProcessParameterInvalidException(
                parameter="dimension", process=process_id,
                reason="No dimension was set, and no temporal dimension could be found. Available dimensions: {n!r}".format(
                    n=data_cube.metadata.dimension_names()))
    # do check_dimension here for error handling
    dimension, band_dim, temporal_dim = _check_dimension(cube=data_cube, dim=dimension, process=process_id)
    return dimension


def _evaluate_sub_process_graph(args: dict, name: str, parent_process: str, env: EvalEnv) -> DriverDataCube:
    """
    Helper function to unwrap and evaluate a sub-process_graph

    :param args: arguments dictionary
    :param name: argument name of sub-process_graph
    :return:
    """
    pg = extract_deep(args, name, ["process_graph", "callback"])
    env = env.push(parameters=args, parent_process=parent_process)
    return evaluate(pg, env=env, do_dry_run=False)


@process_registry_040.add_function
def aggregate_polygon(args: dict, env: EvalEnv) -> DriverDataCube:
    return _evaluate_sub_process_graph(args, 'reducer', parent_process='aggregate_polygon', env=env)


@process_registry_100.add_function
def aggregate_spatial(args: dict, env: EvalEnv) -> DriverDataCube:
    reduce_pg = extract_deep(args, "reducer", "process_graph")
    if(len(reduce_pg)==1):
        return _evaluate_sub_process_graph(args, 'reducer', parent_process='aggregate_spatial', env=env)
    else:
        cube = extract_arg(args, 'data')
        target_dimension = args.get('target_dimension', None)

        geoms = extract_arg(args, 'geometries')
        if isinstance(geoms, dict):
            geoms = geojson_to_geometry(geoms)
        elif isinstance(geoms, DelayedVector):
            geoms = geoms.path
        else:
            raise ProcessParameterInvalidException(
                parameter="geometries", process="aggregate_spatial", reason=f"Invalid type: {type(geoms)} ({geoms!r})"
            )
        return cube.aggregate_spatial(geoms, reduce_pg, target_dimension=target_dimension)


@process_registry_040.add_function(name="mask")
def mask_04(args: dict, env: EvalEnv) -> DriverDataCube:
    mask = extract_arg(args, 'mask')
    replacement = args.get('replacement', None)
    cube = extract_arg(args, 'data')
    if isinstance(mask, DriverDataCube):
        image_collection = cube.mask(mask=mask, replacement=replacement)
    else:
        polygon = mask.geometries[0] if isinstance(mask, DelayedVector) else shape(mask)
        if polygon.area == 0:
            reason = "mask {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
            raise ProcessParameterInvalidException(parameter='mask', process='mask', reason=reason)
        image_collection = cube.mask_polygon(mask=polygon, replacement=replacement)
    return image_collection


@process_registry_100.add_function
def mask(args: dict, env: EvalEnv) -> DriverDataCube:
    cube = extract_arg(args, 'data')
    mask = extract_arg(args, 'mask')
    replacement = args.get('replacement', None)
    return cube.mask(mask=mask, replacement=replacement)


@process_registry_100.add_function
def mask_polygon(args: dict, env: EvalEnv) -> DriverDataCube:
    mask = extract_arg(args, 'mask')
    replacement = args.get('replacement', None)
    inside = args.get('inside', False)
    # TODO: avoid reading DelayedVector twice due to dry-run?
    # TODO: the `DelayedVector` case: aren't we ignoring geometries by doing `[0]`?
    polygon = list(mask.geometries)[0] if isinstance(mask, DelayedVector) else geojson_to_multipolygon(mask)
    if polygon.area == 0:
        reason = "mask {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
        raise ProcessParameterInvalidException(parameter='mask', process='mask', reason=reason)
    image_collection = extract_arg(args, 'data').mask_polygon(mask=polygon, replacement=replacement, inside=inside)
    return image_collection


def _extract_temporal_extent(args: dict, field="extent", process_id="filter_temporal") -> Tuple[str, str]:
    extent = extract_arg(args, name=field, process_id=process_id)
    if len(extent) != 2:
        raise ProcessParameterInvalidException(
            process=process_id, parameter=field, reason="should have length 2, but got {e!r}".format(e=extent)
        )

    start, end = extent[0], extent[1]

    if start is None and end is None:
        raise ProcessParameterInvalidException(
            process=process_id, parameter=field, reason="both start and end are null"
        )

    if (start is not None and end is not None
            and rfc3339.parse_date_or_datetime(end) < rfc3339.parse_date_or_datetime(start)):
        raise ProcessParameterInvalidException(
            process=process_id, parameter=field, reason="end '{e}' is before start '{s}'".format(e=end, s=start)
        )

    # TODO: convert to datetime? or at least normalize?
    return tuple(extent)


@process
def filter_temporal(args: dict, env: EvalEnv) -> DriverDataCube:
    cube = extract_arg(args, 'data')
    extent = _extract_temporal_extent(args, field="extent", process_id="filter_temporal")
    return cube.filter_temporal(start=extent[0], end=extent[1])


def _extract_bbox_extent(args: dict, field="extent", process_id="filter_bbox", handle_geojson=False) -> dict:
    extent = extract_arg(args, name=field, process_id=process_id)
    if handle_geojson and extent.get("type") in ["Polygon", "GeometryCollection", "Feature", "FeatureCollection"]:
        w, s, e, n = shape(extent).bounds
        d = {"west": w, "south": s, "east": e, "north": n, "crs": "EPSG:4326"}
    else:
        d = {
            k: extract_arg(extent, name=k, process_id=process_id)
            for k in ["west", "south", "east", "north"]
        }
        d["crs"] = extent.get("crs") or "EPSG:4326"
    return d


@process
def filter_bbox(args: Dict, env: EvalEnv) -> DriverDataCube:
    cube = extract_arg(args, 'data')
    spatial_extent = _extract_bbox_extent(args, "extent", process_id="filter_bbox")
    return cube.filter_bbox(**spatial_extent)

@process_registry_100.add_function
def filter_spatial(args: Dict, env: EvalEnv) -> DriverDataCube:
    cube = extract_arg(args, 'data')
    geometries = extract_arg(args,'geometries')
    return cube.filter_spatial(geometries)

@process
def filter_bands(args: Dict, env: EvalEnv) -> DriverDataCube:
    cube = extract_arg(args, 'data')
    bands = extract_arg(args, "bands", process_id="filter_bands")
    return cube.filter_bands(bands=bands)


# TODO deprecated process? also see https://github.com/Open-EO/openeo-python-client/issues/144
@deprecated_process
def zonal_statistics(args: Dict, env: EvalEnv) -> Dict:
    raise ProcessUnsupportedException("The zonal_statistics process has been deprecated, and can no longer be used, use aggregate_spatial instead.")


@process
def apply_kernel(args: Dict, env: EvalEnv) -> DriverDataCube:
    image_collection = extract_arg(args, 'data')
    kernel = np.asarray(extract_arg(args, 'kernel'))
    factor = args.get('factor', 1.0)
    border = args.get('border', 0)
    if border == "0":
        # R-client sends `0` border as a string
        border = 0
    replace_invalid = args.get('replace_invalid', 0)
    return image_collection.apply_kernel(kernel=kernel, factor=factor, border=border, replace_invalid=replace_invalid)


@process
def ndvi(args: dict, env: EvalEnv) -> DriverDataCube:
    image_collection = extract_arg(args, 'data')

    if ComparableVersion("1.0.0").or_higher(env["version"]):
        red = args.get("red")
        nir = args.get("nir")
        target_band = args.get("target_band")
        return image_collection.ndvi(nir=nir, red=red, target_band=target_band)
    else:
        name = args.get("name", "ndvi")
        return image_collection.ndvi(name=name)


@process
def resample_spatial(args: dict, env: EvalEnv) -> DriverDataCube:
    image_collection = extract_arg(args, 'data')
    resolution = args.get('resolution', 0)
    projection = args.get('projection', None)
    method = args.get('method', 'near')
    align = args.get('align', 'lower-left')
    return image_collection.resample_spatial(resolution=resolution, projection=projection, method=method, align=align)


@process
def resample_cube_spatial(args: dict, env: EvalEnv) -> DriverDataCube:
    image_collection = extract_arg(args, 'data')
    target_image_collection = extract_arg(args, 'target')
    method = args.get('method', 'near')
    return image_collection.resample_cube_spatial(target=target_image_collection, method=method)


@process
def merge_cubes(args: dict, env: EvalEnv) -> DriverDataCube:
    cube1 = extract_arg(args, 'cube1')
    cube2 = extract_arg(args, 'cube2')
    overlap_resolver = args.get('overlap_resolver')
    # TODO raise check if cubes overlap and raise exception if resolver is missing
    resolver_process = None
    if overlap_resolver:
        pg = extract_arg_list(overlap_resolver, ["process_graph", "callback"])
        if len(pg) != 1:
            raise ProcessParameterInvalidException(
                parameter='overlap_resolver', process='merge_cubes',
                reason='This backend only supports overlap resolvers with exactly one process for now.')
        resolver_process = next(iter(pg.values()))["process_id"]
    return cube1.merge_cubes(cube2, resolver_process)


@process
def run_udf(args: dict, env: EvalEnv):
    # TODO: note: this implements a non-standard usage of `run_udf`: processing "vector" cube (direct JSON or from aggregate_spatial, ...)
    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    data = extract_arg(args, 'data')
    udf, runtime = _get_udf(args, env=env)
    context = args.get('context',{})

    # TODO: this is simple heuristic about skipping `run_udf` in dry-run mode. Does this have to be more advanced?
    # TODO: would it be useful to let user hook into dry-run phase of run_udf (e.g. hint about result type/structure)?
    if dry_run_tracer and isinstance(data, AggregatePolygonResult):
        return JSONResult({})

    if isinstance(data, AggregatePolygonResult):
        pass
    if isinstance(data, (DelayedVector, dict)):
        if isinstance(data, dict):
            data = DelayedVector.from_json_dict(data)
        collection = openeo.udf.FeatureCollection(id='VectorCollection', data=data.as_geodataframe())
        data = openeo.udf.UdfData(
            proj={"EPSG": data.crs.to_epsg()}, feature_collection_list=[collection], user_context=context
        )
    elif isinstance(data, JSONResult):
        st = openeo.udf.StructuredData(description="Dictionary data", data=data.get_data(), type="dict")
        data = openeo.udf.UdfData(structured_data_list=[st], user_context=context)
    elif isinstance(data, list):
        data = openeo.udf.UdfData(
            structured_data_list=[openeo.udf.StructuredData(description="Data list", data=data, type="list")],
            user_context=context
        )
    else:
        raise ProcessParameterInvalidException(
            parameter='data', process='run_udf', reason=f"Invalid data type {type(data)!r}")

    result_data = openeo.udf.run_udf_code(udf, data)

    result_collections = result_data.get_feature_collection_list()
    if result_collections != None and len(result_collections) > 0:
        with tempfile.NamedTemporaryFile(suffix=".json.tmp", delete=False) as temp_file:
            result_collections[0].data.to_file(temp_file.name, driver='GeoJSON')
            return DelayedVector(temp_file.name)
    structured_result = result_data.get_structured_data_list()
    if structured_result != None and len(structured_result)>0:
        return JSONResult(structured_result[0].data)

    raise ProcessParameterInvalidException(
            parameter='udf', process='run_udf',
            reason='The provided UDF should return exactly either a feature collection or a structured result but got: %s .'%str(result_data) )


@process
def linear_scale_range(args: dict, env: EvalEnv) -> DriverDataCube:
    image_collection = extract_arg(args, 'x')

    inputMin = extract_arg(args, "inputMin")
    inputMax = extract_arg(args, "inputMax")
    outputMax = args.get("outputMax", 1.0)
    outputMin = args.get("outputMin", 0.0)

    return image_collection.linear_scale_range(inputMin, inputMax, outputMin, outputMax)


@process_registry_100.add_function
def constant(args: dict, env: EvalEnv):
    return args["x"]


@non_standard_process(
    ProcessSpec(id='histogram', description="A histogram groups data into bins and plots the number of members in each bin versus the bin number.")
    .param(name='data', description='An array of numbers', schema={
                "type": "array",
                "items": {
                    "type": [
                        "number",
                        "null"
                    ]
                }
            })
    .returns(description="A sequence of (bin, count) pairs", schema={
        "type": "object"
    })
)
def histogram(args, env: EvalEnv):
    # currently only available as a reducer passed to e.g. aggregate_polygon
    raise ProcessUnsupportedException('histogram')


def apply_process(process_id: str, args: dict, namespace: Union[str, None], env: EvalEnv):
    parent_process = env.get('parent_process')
    parameters = env.collect_parameters()

    if(process_id == "mask" and args.get("replacement",None) == None):
        mask_node = args.get("mask",None)
        #evaluate the mask
        the_mask = convert_node(mask_node,env=env)
        env = env.push(data_mask=the_mask)
        args = {"data": convert_node(args["data"], env=env), "mask":the_mask }
    else:
        # first we resolve child nodes and arguments in an arbitrary but deterministic order
        args = {name: convert_node(expr, env=env) for (name, expr) in sorted(args.items())}

    # when all arguments and dependencies are resolved, we can run the process
    if parent_process == "apply":
        # TODO EP-3404 this code path is for version <1.0.0, soon to be deprecated
        image_collection = extract_arg_list(args, ['x', 'data'])
        if process_id == "run_udf":
            udf, runtime = _get_udf(args, env=env)
            # TODO replace non-standard apply_tiles with standard "reduce_dimension" https://github.com/Open-EO/openeo-python-client/issues/140
            return image_collection.apply_tiles(udf, {}, runtime)
        else:
            # TODO : add support for `apply` with non-trivial child process graphs #EP-3404
            return image_collection.apply(process_id, args)
    elif parent_process in ["reduce", "reduce_dimension", "reduce_dimension_binary"]:
        # TODO EP-3285 this code path is for version <1.0.0, soon to be deprecated
        image_collection = extract_arg(args, 'data', process_id=process_id)
        dimension = extract_arg(parameters, 'dimension')
        binary = parameters.get('binary', False) or parent_process == "reduce_dimension_binary"
        dimension, band_dim, temporal_dim = _check_dimension(cube=image_collection, dim=dimension, process=parent_process)
        if 'run_udf' == process_id and not binary:
            udf, runtime = _get_udf(args, env=env)
            context = args.get("context", {})
            if dimension == temporal_dim:
                # EP-2760 a special case of reduce where only a single udf based callback is provided. The more generic case is not yet supported.
                return image_collection.apply_tiles_spatiotemporal(udf,context)
            elif dimension == band_dim:
                # TODO replace non-standard apply_tiles with standard "reduce_dimension" https://github.com/Open-EO/openeo-python-client/issues/140
                return image_collection.apply_tiles(udf,context,runtime)

        return image_collection.reduce(process_id, dimension)
    elif parent_process == 'apply_dimension':
        # TODO EP-3285 this code path is for version <1.0.0, soon to be deprecated
        image_collection = extract_arg(args, 'data', process_id=process_id)
        dimension = parameters.get('dimension', None) # By default, applies the the process on all pixel values (as apply does).
        target_dimension = parameters.get('target_dimension', None)
        dimension, band_dim, temporal_dim = _check_dimension(cube=image_collection, dim=dimension, process=parent_process)
        transformed_collection = None
        if process_id == "run_udf":
            udf, runtime = _get_udf(args, env=env)
            context = args.get("context",{})
            if dimension == temporal_dim:
                transformed_collection = image_collection.apply_tiles_spatiotemporal(udf,context)
            else:
                # TODO replace non-standard apply_tiles with standard "reduce_dimension" https://github.com/Open-EO/openeo-python-client/issues/140
                transformed_collection = image_collection.apply_tiles(udf,context,runtime)
        else:
            transformed_collection = image_collection.apply_dimension(process_id, dimension)
        if target_dimension is not None:
            transformed_collection.rename_dimension(dimension, target_dimension)
        return transformed_collection
    elif parent_process in ['aggregate_polygon', 'aggregate_spatial']:
        #TODO it should become possible to remove this code path
        image_collection = extract_arg(args, 'data', process_id=process_id)
        polygons = extract_arg_list(parameters, ['polygons', 'geometries'])

        if isinstance(polygons, dict):
            geometries = geojson_to_geometry(polygons)
            return image_collection.zonal_statistics(geometries, func=process_id)
        elif isinstance(polygons, DelayedVector):
            return image_collection.zonal_statistics(polygons.path, func=process_id)
        else:
            raise ProcessParameterInvalidException(
                parameter="geometries", process=parent_process, reason=f"Invalid type: {type(polygons)} ({polygons!r})"
            )

    elif parent_process == 'aggregate_temporal':
        # TODO EP-3285 this code path is for version <1.0.0, soon to be deprecated
        image_collection = extract_arg(args, 'data', process_id=process_id)
        intervals = extract_arg(parameters, 'intervals')
        labels = extract_arg(parameters, 'labels')
        dimension = parameters.get('dimension', None)
        if dimension is not None:
            dimension, _, _ = _check_dimension(cube=image_collection, dim=dimension, process=parent_process)
        else:
            #default: there is a single temporal dimension
            try:
                dimension = image_collection.metadata.temporal_dimension.name
            except MetadataException:
                raise ProcessParameterInvalidException(
                    parameter="dimension", process=process_id,
                    reason="No dimension was set, and no temporal dimension could be found. Available dimensions: {n!r}".format( n= image_collection.metadata.dimension_names()))
        return image_collection.aggregate_temporal(intervals, labels, process_id, dimension)

    if namespace and any(namespace.startswith(p) for p in ["http://", "https://"]):
        # TODO: HTTPS only by default and config to also allow HTTP (e.g. for localhost dev and testing)
        # TODO: security aspects: only allow for certain users, only allow whitelisted domains, ...?

        return evaluate_process_from_url(
            process_id=process_id, namespace=namespace, args=args, env=env
        )

    if namespace in ["user", None]:
        user = env.get("user")
        if user:
            # TODO: first check process registry with predefined processes because querying of user defined processes
            #   is more expensive IO-wise?
            # the DB-call can be cached if necessary, but how will a user be able to use a new pre-defined process of the same
            # name without renaming his UDP?
            udp = env.backend_implementation.user_defined_processes.get(user_id=user.user_id, process_id=process_id)
            if udp:
                if namespace is None:
                    _log.info("Using process {p!r} from namespace 'user'.".format(p=process_id))
                return evaluate_udp(process_id=process_id, udp=udp, args=args, env=env)

    # And finally: check registry of predefined (namespaced) processes
    if namespace is None:
        namespace = "backend"
        _log.info("Using process {p!r} from namespace 'backend'.".format(p=process_id))


    process_registry = env.backend_implementation.processing.get_process_registry(api_version=env["version"])
    process_function = process_registry.get_function(process_id, namespace=namespace)
    return process_function(args=args, env=env)


@non_standard_process(
    ProcessSpec("read_vector", description="Reads vector data from a file or a URL.")
        .param('filename', description="filename or http url of a vector file", schema={"type": "string"})
        .returns("TODO", schema={"type": "object", "subtype": "vector-cube"})
)
def read_vector(args: Dict, env: EvalEnv) -> DelayedVector:
    path = extract_arg(args, 'filename')
    return DelayedVector(path)


@non_standard_process(
    ProcessSpec("get_geometries", description="Reads vector data from a file or a URL or get geometries from a FeatureCollection")
        .param('filename', description="filename or http url of a vector file", schema={"type": "string"}, required=False)
        .param('feature_collection', description="feature collection", schema={"type": "object"}, required=False)
        .returns("TODO", schema={"type": "object", "subtype": "vector-cube"})
)
def get_geometries(args: Dict, env: EvalEnv) -> Union[DelayedVector, dict]:
    feature_collection = args.get('feature_collection', None)
    path = args.get('filename', None)
    if path is not None:
        return DelayedVector(path)
    else:
        return feature_collection


@non_standard_process(
    ProcessSpec("raster_to_vector", description="Converts this raster data cube into a vector data cube. The bounding polygon of homogenous areas of pixels is constructed.\n"
                                                "Only the first band is considered the others are ignored.")
        .param('data', description="A raster data cube.", schema={"type": "object", "subtype": "raster-cube"})
        .returns("vector-cube", schema={"type": "object", "subtype": "vector-cube"})
)
def raster_to_vector(args: Dict, env: EvalEnv):
    image_collection = extract_arg(args, 'data')
    return image_collection.raster_to_vector()


def _get_udf(args, env: EvalEnv):
    udf = extract_arg(args, "udf")
    runtime = extract_arg(args, "runtime")
    version = args.get("version", None)

    available_runtimes = env.backend_implementation.udf_runtimes.get_udf_runtimes()
    available_runtime_names = list(available_runtimes.keys())
    # Make lookup case insensitive
    available_runtimes.update({k.lower(): v for k, v in available_runtimes.items()})

    if not runtime or runtime.lower() not in available_runtimes:
        raise OpenEOApiException(
            status_code=400, code="InvalidRuntime",
            message=f"Unsupported UDF runtime {runtime!r}. Should be one of {available_runtime_names}"
        )
    available_versions = list(available_runtimes[runtime.lower()]["versions"].keys())
    if version and version not in available_versions:
        raise OpenEOApiException(
            status_code=400, code="InvalidVersion",
            message=f"Unsupported UDF runtime version {runtime} {version!r}. Should be one of {available_versions} or null"
        )

    return udf, runtime


def _evaluate_process_graph_process(
        process_id: str, process_graph: dict, parameters: List[dict], args: dict, env: EvalEnv
):
    """Evaluate a process specified as a process graph (e.g. user-defined process)"""
    args = args.copy()
    for param in parameters or []:
        name = param["name"]
        if name not in args:
            if "default" in param:
                args[name] = param["default"]
            else:
                raise ProcessParameterRequiredException(process=process_id, parameter=name)
    env = env.push(parameters=args)
    return evaluate(process_graph, env=env, do_dry_run=False)


def evaluate_udp(process_id: str, udp: UserDefinedProcessMetadata, args: dict, env: EvalEnv):
    return _evaluate_process_graph_process(
        process_id=process_id, process_graph=udp.process_graph, parameters=udp.parameters,
        args=args, env=env
    )


def evaluate_process_from_url(process_id: str, namespace: str, args: dict, env: EvalEnv):
    if namespace.endswith("/"):
        # Assume namespace is a folder possibly containing multiple processes
        candidates = [
            f"{namespace}{process_id}",
            f"{namespace}{process_id}.json",
        ]
    else:
        # Assume namespace is direct URL to process/UDP metadata
        candidates = [namespace]

    for candidate in candidates:
        # TODO: add request timeout, retry logic?
        res = requests.get(candidate)
        if res.status_code == 200:
            break
    else:
        raise ProcessUnsupportedException(process=process_id, namespace=namespace)

    try:
        spec = res.json()
        assert spec["id"] == process_id
        process_graph = spec["process_graph"]
        parameters = spec.get("parameters", [])
    except Exception:
        # TODO: log information about what is wrong, so user can debug issue properly
        raise ProcessGraphInvalidException()

    return _evaluate_process_graph_process(
        process_id=process_id, process_graph=process_graph, parameters=parameters, args=args, env=env
    )


@non_standard_process(
    ProcessSpec("sleep", description="Sleep for given amount of seconds (and just pass-through given data).")
        .param('data', description="Data to pass through.", schema={}, required=False)
        .param('seconds', description="Number of seconds to sleep.", schema={"type": "number"}, required=True)
        .returns("Original data", schema={})
)
def sleep(args: Dict, env: EvalEnv):
    data = extract_arg(args, "data")
    seconds = extract_arg(args, "seconds")
    _log.info("Sleeping {s} seconds".format(s=seconds))
    time.sleep(seconds)
    return data


@non_standard_process(
    # TODO: get spec directly from @process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/atmospheric_correction.json"))
    ProcessSpec(
        id='atmospheric_correction',
        description="Applies an atmospheric correction that converts top of atmosphere reflectance values into bottom of atmosphere/top of canopy reflectance values.",
        extra={
            "summary": "Apply atmospheric correction",
            "categories": ["cubes", "optical"],
            "experimental": True,
            "links": [
                {
                    "rel": "about",
                    "href": "https://bok.eo4geo.eu/IP1-7-1",
                    "title": "Atmospheric correction explained by EO4GEO body of knowledge."
                }
            ],
            "exceptions": {
                "DigitalElevationModelInvalid": {
                    "message": "The digital elevation model specified is either not a DEM or can't be used with the data cube given."
                }
            },
        }
    )
        .param('data', description="Data cube containing multi-spectral optical top of atmosphere reflectances to be corrected.", schema={"type": "object", "subtype": "raster-cube"})
        .param(name='method', description="The atmospheric correction method to use. To get reproducible results, you have to set a specific method.\n\nSet to `null` to allow the back-end to choose, which will improve portability, but reduce reproducibility as you *may* get different results if you run the processes multiple times.",                      schema={"type": "string"}, required=False)
        .param(name='elevation_model', description="The digital elevation model to use, leave empty to allow the back-end to make a suitable choice.", schema={"type": "string"}, required=False)
        .param(name='missionId', description="non-standard mission Id, currently defaults to sentinel2",                      schema={"type": "string"}, required=False)
        .param(name='sza',       description="non-standard if set, overrides sun zenith angle values [deg]",                  schema={"type": "number"}, required=False)
        .param(name='vza',       description="non-standard if set, overrides sensor zenith angle values [deg]",               schema={"type": "number"}, required=False)
        .param(name='raa',       description="non-standard if set, overrides rel. azimuth angle values [deg]",                schema={"type": "number"}, required=False)
        .param(name='gnd',       description="non-standard if set, overrides ground elevation [km]",                          schema={"type": "number"}, required=False)
        .param(name='aot',       description="non-standard if set, overrides aerosol optical thickness [], usually 0.1..0.2", schema={"type": "number"}, required=False)
        .param(name='cwv',       description="non-standard if set, overrides water vapor [], usually 0..7",                   schema={"type": "number"}, required=False)
        .param(name='appendDebugBands', description="non-standard if set to 1, saves debug bands",                            schema={"type": "number"}, required=False)
        .returns(description="the corrected data as a data cube", schema={"type": "object", "subtype": "raster-cube"})
)
def atmospheric_correction(args: Dict, env: EvalEnv) -> object:
    image_collection = extract_arg(args, 'data')
    method = args.get('method', None)
    elevation_model = args.get('elevation_model', None)
    missionId = args.get('missionId',None)
    sza = args.get('sza',None)
    vza = args.get('vza',None)
    raa = args.get('raa',None)
    gnd = args.get('gnd',None)
    aot = args.get('aot',None)
    cwv = args.get('cwv',None)
    appendDebugBands = args.get('appendDebugBands',None)
    return image_collection.atmospheric_correction(method,elevation_model, missionId, sza, vza, raa, gnd, aot, cwv, appendDebugBands)


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/sar_backscatter.json"))
def sar_backscatter(args: Dict, env: EvalEnv):
    cube: DriverDataCube = extract_arg(args, 'data')
    kwargs = extract_args_subset(
        args, keys=["coefficient", "elevation_model", "mask", "contributing_area", "local_incidence_angle",
                    "ellipsoid_incidence_angle", "noise_removal", "options"]
    )
    return cube.sar_backscatter(SarBackscatterArgs(**kwargs))


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/resolution_merge.json"))
def resolution_merge(args: Dict, env: EvalEnv):
    cube: DriverDataCube = extract_arg(args, 'data')
    kwargs = extract_args_subset(args, keys=["method", "high_resolution_bands", "low_resolution_bands", "options"])
    return cube.resolution_merge(ResolutionMergeArgs(**kwargs))


@non_standard_process(
    ProcessSpec("discard_result", description="Discards given data. Used for side-effecting purposes.")
        .param('data', description="Data to discard.", schema={}, required=False)
        .returns("Nothing", schema={})
)
def discard_result(args: Dict, env: EvalEnv):
    # TODO: keep a reference to the discarded result?
    return NullResult()


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/mask_scl_dilation.json"))
def mask_scl_dilation(args: Dict, env: EvalEnv):
    cube: DriverDataCube = extract_arg(args, 'data')
    if( "mask_scl_dilation" in dir(cube)):
        return cube.mask_scl_dilation()
    else:
        return cube

@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/mask_l1c.json"))
def mask_l1c(args: Dict, env: EvalEnv):
    cube: DriverDataCube = extract_arg(args, 'data')
    if( "mask_l1c" in dir(cube)):
        return cube.mask_l1c()
    else:
        return cube


custom_process_from_process_graph(read_spec("openeo-processes/1.x/proposals/ard_normalized_radar_backscatter.json"))

@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/array_append.json"))
def array_append(args: Dict, env: EvalEnv) -> str:
    pass

@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/array_interpolate_linear.json"))
def array_interpolate_linear(args: Dict, env: EvalEnv) -> str:
    pass

@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/date_shift.json"))
def date_shift(args: Dict, env: EvalEnv) -> str:
    date = rfc3339.parse_date_or_datetime(extract_arg(args, "date"))
    value = int(extract_arg(args, "value"))
    unit_values = {"year", "month", "week", "day", "hour", "minute", "second", "millisecond"}
    unit = extract_arg_enum(args, "unit", enum_values=unit_values, process_id="date_shift")
    if unit == "millisecond":
        raise FeatureUnsupportedException(message="Millisecond unit is not supported in date_shift")
    shifted = date + relativedelta(**{unit + "s": value})
    if type(date) is datetime.date and type(shifted) is datetime.datetime:
        shifted = shifted.date()
    return rfc3339.normalize(shifted)


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/array_concat.json"))
def array_concat(args: dict, env: EvalEnv) -> list:
    array1 = extract_arg(args, "array1")
    array2 = extract_arg(args, "array2")
    return list(array1) + list(array2)


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/array_create.json"))
def array_create(args: dict, env: EvalEnv) -> list:
    data = extract_arg(args, "data")
    repeat = args.get("repeat", 1)
    if not isinstance(repeat, int) or repeat < 1:
        raise ProcessParameterInvalidException(
            parameter="repeat", process="array_create",
            reason="The `repeat` parameter should be an integer of at least value 1."
        )
    return list(data) * repeat


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/load_result.json"))
def load_result(args: dict, env: EvalEnv) -> DriverDataCube:
    job_id = extract_arg(args, "id")
    user = env["user"]

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        return dry_run_tracer.load_result(job_id)
    else:
        source_id = dry_run.DataSource.load_result(job_id).get_source_id()
        load_params = _extract_load_parameters(env, source_id=source_id)

        return env.backend_implementation.load_result(job_id=job_id, user=user, load_params=load_params, env=env)


# Finally: register some fallback implementation if possible
_register_fallback_implementations_by_process_graph(process_registry_100)
