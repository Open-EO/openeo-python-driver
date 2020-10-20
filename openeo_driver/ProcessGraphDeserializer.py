# TODO: rename this module to something in snake case? It doesn't even implement a ProcessGraphDeserializer class.

# pylint: disable=unused-argument

import logging
import tempfile
import time
import warnings
from typing import Dict, Callable, List, Union

import numpy as np
import openeo_processes
import requests
from shapely.geometry import shape, mapping

from openeo.capabilities import ComparableVersion
from openeo.metadata import MetadataException
from openeo_driver import dry_run
from openeo_driver.backend import get_backend_implementation, UserDefinedProcessMetadata
from openeo_driver.datacube import DriverDataCube
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.errors import ProcessParameterRequiredException, ProcessParameterInvalidException
from openeo_driver.errors import ProcessUnsupportedException
from openeo_driver.processes import ProcessRegistry, ProcessSpec
from openeo_driver.save_result import ImageCollectionResult, JSONResult, SaveResult, AggregatePolygonResult
from openeo_driver.specs import SPECS_ROOT
from openeo_driver.utils import smart_bool, EvalEnv, geojson_to_geometry
from openeo_udf.api.feature_collection import FeatureCollection
from openeo_udf.api.structured_data import StructuredData
from openeo_udf.api.udf_data import UdfData

_log = logging.getLogger(__name__)

# Set up process registries (version dependent)
process_registry_040 = ProcessRegistry(spec_root=SPECS_ROOT / 'openeo-processes/0.4', argument_names=["args", "env"])
process_registry_100 = ProcessRegistry(spec_root=SPECS_ROOT / 'openeo-processes/1.0', argument_names=["args", "env"])

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

    def wrap(process):
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
            _log.warning("Adding process {p!r} without implementation".format(p=pid))
            process_registry.add_spec_by_name(pid)


_add_standard_processes(process_registry_100, [
    'array_apply', 'array_contains', 'array_element', 'array_filter', 'array_find', 'array_labels',
    'count', 'first', 'last', 'order', 'rearrange', 'sort',
    'between', 'eq', 'gt', 'gte', 'if', 'is_nan', 'is_nodata', 'is_valid', 'lt', 'lte', 'neq',
    'all', 'and', 'any', 'if', 'not', 'or', 'xor',
    'absolute', 'add', 'clip', 'divide', 'extrema', 'int', 'max', 'mean',
    'median', 'min', 'mod', 'multiply', 'normalized_difference', 'power', 'product', 'quantiles', 'sd', 'sgn', 'sqrt',
    'subtract', 'sum', 'variance', 'e', 'pi', 'exp', 'ln', 'log',
    'ceil', 'floor', 'int', 'round',
    'arccos', 'arcosh', 'arcsin', 'arctan', 'arctan2', 'arsinh', 'artanh', 'cos', 'cosh', 'sin', 'sinh', 'tan', 'tanh',
    'all', 'any', 'count', 'first', 'last', 'max', 'mean', 'median', 'min', 'product', 'sd', 'sum', 'variance'
])


def process(f: Callable) -> Callable:
    """Decorator for registering a process function in the process registries"""
    process_registry_040.add_function(f)
    process_registry_100.add_function(f)
    return f


# Decorator for registering deprecate/old process functions
deprecated_process = process_registry_040.add_deprecated


def non_standard_process(spec: ProcessSpec) -> Callable[[Callable], Callable]:
    """Decorator for registering non-standard process functions"""

    def decorator(f: Callable) -> Callable:
        process_registry_040.add_function(f=f, spec=spec.to_dict_040())
        process_registry_100.add_function(f=f, spec=spec.to_dict_100())
        return f

    return decorator


def custom_process(f):
    """Decorator for custom processes (e.g. in custom_processes.py)."""
    process_registry_040.add_hidden(f)
    process_registry_100.add_hidden(f)
    return f


def get_process_registry(api_version: ComparableVersion) -> ProcessRegistry:
    if api_version.at_least("1.0.0"):
        return process_registry_100
    else:
        return process_registry_040


backend_implementation = get_backend_implementation()

# Some (env) string constants to simplify code navigation
ENV_SOURCE_CONSTRAINTS = "source_constraints"
ENV_DRY_RUN_TRACER = "dry_run_tracer"


def evaluate(process_graph: dict, env: EvalEnv = None, do_dry_run=True) -> DriverDataCube:
    """
    Converts the json representation of a (part of a) process graph into the corresponding Python data cube.
    """
    if env is None:
        env = EvalEnv()

    if 'version' not in env:
        warnings.warn("Blindly assuming 0.4.0")
        env = env.push({"version": "0.4.0"})

    # TODO avoid local import
    from openeo.internal.process_graph_visitor import ProcessGraphVisitor
    preprocessed_process_graph = _expand_macros(process_graph)
    top_level_node = ProcessGraphVisitor.dereference_from_node_arguments(preprocessed_process_graph)
    result_node = preprocessed_process_graph[top_level_node]

    if do_dry_run:
        dry_run_tracer = dry_run.DryRunDataTracer()
        _log.info("Doing dry run")
        convert_node(result_node, env=env.push({ENV_DRY_RUN_TRACER: dry_run_tracer}))
        # TODO: work with a dedicated DryRunEvalEnv?
        source_constraints = dry_run_tracer.get_source_constraints()
        _log.info("Dry run extracted these source constraints: {s}".format(s=source_constraints))
        env = env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})

    return convert_node(result_node, env=env)


def _expand_macros(process_graph: dict) -> dict:
    """
    Expands macro nodes in a process graph by replacing them with other nodes, making sure their node identifiers don't
    clash with existing ones.

    :param process_graph:
    :return: a copy of the input process graph with the macros expanded
    """
    # TODO: can this system be combined with user defined processes (both kind of replace a single "virtual" node with a replacement process graph)

    def expand_macros_recursively(tree: dict) -> dict:
        def make_unique(node_identifier: str) -> str:
            return node_identifier if node_identifier not in tree else make_unique(node_identifier + '_')

        result = {}

        for key, value in tree.items():
            if isinstance(value, dict):
                if 'process_id' in value and value['process_id'] == 'normalized_difference':
                    normalized_difference_node = value
                    normalized_difference_arguments = normalized_difference_node['arguments']

                    subtract_key = make_unique(key + "_subtract")
                    add_key = make_unique(key + "_add")

                    # add "subtract" and "add"/"sum" processes
                    result[subtract_key] = {'process_id': 'subtract',
                                            'arguments': normalized_difference_arguments}
                    result[add_key] = {'process_id': 'sum' if 'data' in normalized_difference_arguments else 'add',
                                       'arguments': normalized_difference_arguments}

                    # replace "normalized_difference" with "divide" under the original key (it's being referenced)
                    result[key] = {
                        'process_id': 'divide',
                        'arguments': {
                            'x': {'from_node': subtract_key},
                            'y': {'from_node': add_key}
                        },
                        'result': normalized_difference_node.get('result', False)
                    }
                else:
                    result[key] = expand_macros_recursively(value)
            else:
                result[key] = value

        return result

    return expand_macros_recursively(process_graph)


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
                return env[processGraph['from_parameter']]
            except KeyError:
                raise ProcessParameterRequiredException(process="n/a", parameter=processGraph['from_parameter'])
        elif 'from_argument' in processGraph:
            # 0.4-style argument referencer (equivalent with 1.0-style "from_parameter")
            argument_reference = processGraph.get('from_argument')
            # backwards compatibility for clients that still use 'dimension_data', can be removed when clients are upgraded
            if argument_reference == 'dimension_data':
                argument_reference = 'data'
            return env.get(argument_reference)
        else:
            # TODO: Don't apply `convert_node` for some special cases (e.g. geojson objects)?
            return {k:convert_node(v, env=env) for k,v in processGraph.items()}
    elif isinstance(processGraph, list):
        return [convert_node(x, env=env) for x in processGraph]
    return processGraph


def extract_arg(args: dict, name: str, process_id='n/a'):
    """Get process argument by name."""
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


def _extract_viewing_parameters(env: EvalEnv, source_id: tuple):
    constraints = env[ENV_SOURCE_CONSTRAINTS][source_id]
    viewing_parameters = {}
    viewing_parameters["from"], viewing_parameters["to"] = constraints.get("temporal_extent", [None, None])
    spatial_extent = constraints.get("spatial_extent", {})
    # TODO: eliminate need for aliases (see openeo-geopyspark-driver)?
    for aliases in [["west", "left"], ["south", "bottom"], ["east", "right"], ["north", "top"], ["crs", "srs"]]:
        for param in aliases:
            viewing_parameters[param] = spatial_extent.get(aliases[0])
    viewing_parameters["bands"] = constraints.get("bands", None)
    viewing_parameters["properties"] = constraints.get("properties", None)
    for param in ["correlation_id", "require_bounds", "polygons", "pyramid_levels"]:
        # TODO: are all these params still properly working (e.g. these cached polygons)?
        if param in env:
            viewing_parameters[param] = env[param]
    return viewing_parameters


@process
def load_collection(args: dict, env: EvalEnv) -> DriverDataCube:
    collection_id = extract_arg(args, 'id')

    dry_run_tracer: dry_run.DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        arguments = {}
        if args.get("temporal_extent"):
            arguments["temporal_extent"] = _extract_temporal_extent(args, field="temporal_extent", process_id="load_collection")
        if args.get("spatial_extent"):
            # TODO: spatial_extent could also be a geojson object instead of bbox dict
            arguments["spatial_extent"] = _extract_bbox_extent(args, field="spatial_extent", process_id="load_collection")
        if args.get("bands"):
            arguments["bands"] = extract_arg(args, "bands", process_id="load_collection")
        if args.get('properties'):
            arguments["properties"] = extract_arg(args, 'properties', process_id="load_collection")
        metadata = backend_implementation.catalog.get_collection_metadata(collection_id)
        return dry_run_tracer.load_collection(collection_id=collection_id, arguments=arguments, metadata=metadata)
    else:
        source_id = dry_run.DataSource.load_collection(collection_id=collection_id).get_source_id()
        viewing_parameters= _extract_viewing_parameters(env, source_id=source_id)
        return backend_implementation.catalog.load_collection(collection_id, viewing_parameters=viewing_parameters)


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
    dry_run_tracer: dry_run.DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        return dry_run_tracer.load_disk_data(**kwargs)
    else:
        source_id = dry_run.DataSource.load_disk_data(**kwargs).get_source_id()
        viewing_parameters = _extract_viewing_parameters(env, source_id=source_id)
        return backend_implementation.load_disk_data(**kwargs, viewing_parameters=viewing_parameters)


@process_registry_100.add_function
def apply_neighborhood(args: dict, env: EvalEnv) -> DriverDataCube:
    process = extract_deep(args, "process", "process_graph")
    size = extract_arg(args, 'size')
    overlap = extract_arg(args, 'overlap')
    context = args.get( 'context',{})
    data_cube = extract_arg(args, 'data')
    return data_cube.apply_neighborhood(process,size,overlap)

@process
def apply_dimension(args: Dict, env: EvalEnv) -> DriverDataCube:
    return _evaluate_sub_process_graph(args, 'process', parent_process='apply_dimension', env=env)


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
        visitor = backend_implementation.visit_process_graph(reduce_pg)
        return data_cube.reduce_bands(visitor)
    else:
        return _evaluate_sub_process_graph(args, 'reducer', parent_process='reduce', env=env)


@process_registry_100.add_function
def reduce_dimension(args: dict, env: EvalEnv) -> DriverDataCube:
    reduce_pg = extract_deep(args, "reducer", "process_graph")
    dimension = extract_arg(args, 'dimension')
    data_cube = extract_arg(args, 'data')

    # TODO: avoid special case handling for run_udf?
    # do check_dimension here for error handling
    dimension, band_dim, temporal_dim = _check_dimension(cube=data_cube, dim=dimension, process="reduce_dimension")
    return data_cube.reduce_dimension(dimension, reduce_pg)


@process
def add_dimension(args: dict, env: EvalEnv) -> DriverDataCube:
    data_cube = extract_arg(args, 'data')
    return data_cube.add_dimension(
        name=extract_arg(args, 'name'),
        label=extract_arg_list(args, ['label', 'value']),
        type=extract_arg(args, 'type'),
    )


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
    return _evaluate_sub_process_graph(args, 'reducer', parent_process='aggregate_temporal', env=env)


def _evaluate_sub_process_graph(args: dict, name: str, parent_process: str, env: EvalEnv) -> DriverDataCube:
    """
    Helper function to unwrap and evaluate a sub-process_graph

    :param args: arguments dictionary
    :param name: argument name of sub-process_graph
    :return:
    """
    pg = extract_deep(args, name, ["process_graph", "callback"])
    # TODO: args are injected into env? looks strange EP-3509
    env = env.push(args, parent_process=parent_process)
    return evaluate(pg, env=env, do_dry_run=False)


@process_registry_040.add_function
def aggregate_polygon(args: dict, env: EvalEnv) -> DriverDataCube:
    return _evaluate_sub_process_graph(args, 'reducer', parent_process='aggregate_polygon', env=env)


@process_registry_100.add_function
def aggregate_spatial(args: dict, env: EvalEnv) -> DriverDataCube:
    return _evaluate_sub_process_graph(args, 'reducer', parent_process='aggregate_spatial', env=env)


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
    polygon = list(mask.geometries)[0] if isinstance(mask, DelayedVector) else geojson_to_geometry(mask)
    if polygon.area == 0:
        reason = "mask {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
        raise ProcessParameterInvalidException(parameter='mask', process='mask', reason=reason)
    image_collection = extract_arg(args, 'data').mask_polygon(mask=polygon, replacement=replacement, inside=inside)
    return image_collection


def _extract_temporal_extent(args: dict, field="extent", process_id="filter_temporal") -> list:
    extent = extract_arg(args, name=field, process_id=process_id)
    if len(extent) != 2:
        raise ProcessParameterInvalidException(
            process=process_id, parameter=field, reason="should have length 2, but got {e!r}".format(e=extent)
        )
    # TODO: convert to datetime? or at least normalize?
    return extent


@process
def filter_temporal(args: dict, env: EvalEnv) -> DriverDataCube:
    cube = extract_arg(args, 'data')
    extent = _extract_temporal_extent(args, field="extent", process_id="filter_temporal")
    return cube.filter_temporal(start=extent[0], end=extent[1])


def _extract_bbox_extent(args: dict, field="extent", process_id="filter_bbox") -> dict:
    extent = extract_arg(args, name=field, process_id=process_id)
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


@process
def filter_bands(args: Dict, env: EvalEnv) -> DriverDataCube:
    cube = extract_arg(args, 'data')
    bands = extract_arg(args, "bands", process_id="filter_bands")
    return cube.filter_bands(bands=bands)


# TODO deprecated process? also see https://github.com/Open-EO/openeo-python-client/issues/144
@deprecated_process
def zonal_statistics(args: Dict, env: EvalEnv) -> Dict:
    image_collection = extract_arg(args, 'data')
    geometry = extract_arg(args, 'regions')
    func = args.get('func', 'mean')

    # TODO: extract srs from geometry

    if func == 'mean' or func == 'avg':
        return image_collection.zonal_statistics(shape(geometry), func)
    else:
        raise AttributeError("func %s is not supported" % func)


@process
def apply_kernel(args: Dict, env: EvalEnv) -> DriverDataCube:
    image_collection = extract_arg(args, 'data')
    kernel = np.asarray(extract_arg(args, 'kernel'))
    factor = args.get('factor', 1.0)
    border = args.get('border',0)
    replace_invalid = args.get('replace_invalid',0)
    if border != 0:
        raise ProcessParameterInvalidException('border','apply_kernel','This backend does not support values other than 0 for the border parameter of apply_kernel. Please contact the developers if support is required.')
    if replace_invalid != 0:
        raise ProcessParameterInvalidException('replace_invalid','apply_kernel','This backend does not support values other than 0 for the replace_invalid parameter of apply_kernel. Please contact the developers if support is required.')
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
    data = extract_arg(args, 'data')

    if not isinstance(data, DelayedVector) and not isinstance(data,AggregatePolygonResult):
        if isinstance(data, dict):
            data = DelayedVector.from_json_dict(data)
        else:
            raise ProcessParameterInvalidException(
                parameter='data', process='run_udf',
                reason='The run_udf process can only be used on vector cubes or aggregated timeseries directly, or as part of a callback on a raster-cube! Tried to use: %s' % str(data) )

    from openeo_udf.api.run_code import run_user_code

    udf = _get_udf(args)
    context = args.get('context',{})

    if isinstance(data,DelayedVector):
        collection = FeatureCollection(id='VectorCollection', data=data.as_geodataframe())
        data = UdfData(proj={"EPSG":data.crs.to_epsg()}, feature_collection_list=[collection])
    elif isinstance(data,JSONResult):
        st = StructuredData(description="Dictionary data", data=data.get_data(), type="dict")
        data = UdfData(proj={},structured_data_list=[st])

    data.user_context=context

    result_data = run_user_code(udf, data)

    result_collections = result_data.get_feature_collection_list()
    if result_collections != None and len(result_collections) > 0:
        with tempfile.NamedTemporaryFile(suffix=".json.tmp", delete=False) as temp_file:
            result_collections[0].get_data().to_file(temp_file.name, driver='GeoJSON')
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


def apply_process(process_id: str, args: dict, namespace: str = None, env: EvalEnv = None):
    env = env or EvalEnv()
    parent_process = env.get('parent_process')

    # first we resolve child nodes and arguments
    args = {name: convert_node(expr, env=env) for (name, expr) in args.items()}

    # when all arguments and dependencies are resolved, we can run the process
    if parent_process == "apply":
        # TODO EP-3404 this code path is for version <1.0.0, soon to be deprecated
        image_collection = extract_arg_list(args, ['x', 'data'])
        if process_id == "run_udf":
            udf = _get_udf(args)
            # TODO replace non-standard apply_tiles with standard "reduce_dimension" https://github.com/Open-EO/openeo-python-client/issues/140
            return image_collection.apply_tiles(udf)
        else:
            # TODO : add support for `apply` with non-trivial child process graphs #EP-3404
            return image_collection.apply(process_id, args)
    elif parent_process in ["reduce", "reduce_dimension", "reduce_dimension_binary"]:
        # TODO EP-3285 this code path is for version <1.0.0, soon to be deprecated
        image_collection = extract_arg(args, 'data', process_id=process_id)
        dimension = extract_arg(env, 'dimension')
        binary = env.get('binary',False) or parent_process == "reduce_dimension_binary"
        dimension, band_dim, temporal_dim = _check_dimension(cube=image_collection, dim=dimension, process=parent_process)
        if 'run_udf' == process_id and not binary:
            udf = _get_udf(args)
            context = args.get("context", {})
            if dimension == temporal_dim:
                # EP-2760 a special case of reduce where only a single udf based callback is provided. The more generic case is not yet supported.
                return image_collection.apply_tiles_spatiotemporal(udf,context)
            elif dimension == band_dim:
                # TODO replace non-standard apply_tiles with standard "reduce_dimension" https://github.com/Open-EO/openeo-python-client/issues/140
                return image_collection.apply_tiles(udf,context)

        return image_collection.reduce(process_id, dimension)
    elif parent_process == 'apply_dimension':
        image_collection = extract_arg(args, 'data', process_id=process_id)
        dimension = env.get('dimension', None) # By default, applies the the process on all pixel values (as apply does).
        target_dimension = env.get('target_dimension', None)
        dimension, band_dim, temporal_dim = _check_dimension(cube=image_collection, dim=dimension, process=parent_process)
        transformed_collection = None
        if process_id == "run_udf":
            udf = _get_udf(args)
            context = args.get("context",{})
            if dimension == temporal_dim:
                transformed_collection = image_collection.apply_tiles_spatiotemporal(udf,context)
            else:
                # TODO replace non-standard apply_tiles with standard "reduce_dimension" https://github.com/Open-EO/openeo-python-client/issues/140
                transformed_collection = image_collection.apply_tiles(udf,context)
        else:
            transformed_collection = image_collection.apply_dimension(process_id, dimension)
        if target_dimension is not None:
            transformed_collection.rename_dimension(dimension, target_dimension)
        return transformed_collection
    elif parent_process in ['aggregate_polygon', 'aggregate_spatial']:
        image_collection = extract_arg(args, 'data', process_id=process_id)
        # TODO: binary and name are unused?
        binary = env.get('binary', False)
        name = env.get('name', 'result')
        polygons = extract_arg_list(env, ['polygons', 'geometries'])

        if isinstance(polygons, dict):
            geometries = geojson_to_geometry(polygons)
            return image_collection.zonal_statistics(geometries, func=process_id)
        elif isinstance(polygons, DelayedVector):
            return image_collection.zonal_statistics(polygons.path, func=process_id)
        else:
            raise ValueError(polygons)

    elif parent_process == 'aggregate_temporal':
        image_collection = extract_arg(args, 'data', process_id=process_id)
        intervals = extract_arg(env, 'intervals')
        labels = extract_arg(env, 'labels')
        dimension = env.get('dimension', None)
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
            udp = backend_implementation.user_defined_processes.get(user_id=user.user_id, process_id=process_id)
            if udp:
                if namespace is None:
                    _log.info("Using process {p!r} from namespace 'user'.".format(p=process_id))
                return evaluate_udp(process_id=process_id, udp=udp, args=args, env=env)

    if namespace in ["backend", None]:
        # And finally: check registry of predefined processes
        process_registry = get_process_registry(ComparableVersion(env["version"]))
        process_function = process_registry.get_function(process_id)
        if namespace is None:
            _log.info("Using process {p!r} from namespace 'backend'.".format(p=process_id))
        return process_function(args=args, env=env)

    # TODO: add namespace in error message? also see https://github.com/Open-EO/openeo-api/pull/328
    raise ProcessUnsupportedException(process=process_id)


@non_standard_process(
    ProcessSpec("read_vector", description="Reads vector data from a file or a URL.")
        .param('filename', description="filename or http url of a vector file", schema={"type": "string"})
        .returns("TODO", schema={"type": "object", "subtype": "vector-cube"})
)
def read_vector(args: Dict, env: EvalEnv) -> DelayedVector:
    path = extract_arg(args, 'filename')
    return DelayedVector(path)

@non_standard_process(
    ProcessSpec("raster_to_vector", description="Converts this raster data cube into a vector data cube. The bounding polygon of homogenous areas of pixels is constructed.\n"
                                                "Only the first band is considered the others are ignored.")
        .param('data', description="A raster data cube.", schema={"type": "object", "subtype": "raster-cube"})
        .returns("vector-cube", schema={"type": "object", "subtype": "vector-cube"})
)
def raster_to_vector(args: Dict, env: EvalEnv):
    image_collection = extract_arg(args, 'data')
    return image_collection.raster_to_vector()


def _get_udf(args):
    udf = extract_arg(args, "udf")
    runtime = extract_arg(args, "runtime")
    # TODO allow registration of supported runtimes, so we can be more generic
    if runtime != "Python":
        raise NotImplementedError("Unsupported runtime: " + runtime + " this backend only supports the Python runtime.")
    version = args.get("version", None)
    if version is not None and version != "3.5.1" and version != "latest":
        raise NotImplementedError("Unsupported Python version: " + version + "this backend only support version 3.5.1.")
    return udf


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
    env = env.push(args)
    return evaluate(process_graph, env=env, do_dry_run=False)


def evaluate_udp(process_id: str, udp: UserDefinedProcessMetadata, args: dict, env: EvalEnv):
    return _evaluate_process_graph_process(
        process_id=process_id, process_graph=udp.process_graph, parameters=udp.parameters,
        args=args, env=env
    )


def evaluate_process_from_url(process_id: str, namespace: str, args: dict, env: EvalEnv):
    if namespace.endswith('.json'):
        # TODO: if namespace URL is json file: handle it as collection of processes instead of a single process spec?
        url = namespace
    else:
        url = '{n}/{p}.json'.format(n=namespace.rstrip('/'), p=process_id)
    res = requests.get(url)
    if res.status_code != 200:
        raise ProcessUnsupportedException(process=process_id)
    spec = res.json()
    return _evaluate_process_graph_process(
        process_id=process_id, process_graph=spec["process_graph"], parameters=spec.get("parameters", []),
        args=args, env=env
    )


@non_standard_process(
    ProcessSpec("sleep", description="Sleep for given amount of seconds (and just pass-through given data.")
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
    ProcessSpec(id='apply_atmospheric_correction', description="iCor workflow test")
        .param('data', description="input data cube to be corrected", schema={"type": "object", "subtype": "raster-cube"})
        .returns(description="the corrected data as a data cube", schema={"type": "object", "subtype": "raster-cube"})
)
def apply_atmospheric_correction(args: Dict, env: EvalEnv) -> object:
    image_collection = extract_arg(args, 'data')
    return image_collection.apply_atmospheric_correction()
