# TODO: rename this module to something in snake case? It doesn't even implement a ProcessGraphDeserializer class.

import base64
import logging
import pickle
import warnings
from typing import Dict, Callable

import numpy as np
from openeo import ImageCollection
from openeo.capabilities import ComparableVersion
from openeo.metadata import MetadataException
from openeo_driver.backend import get_backend_implementation
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.errors import ProcessArgumentInvalidException, ProcessUnsupportedException, \
    ProcessArgumentRequiredException, ProcessParameterMissingException
from openeo_driver.processes import ProcessRegistry, ProcessSpec
from openeo_driver.save_result import ImageCollectionResult, JSONResult, SaveResult
from openeo_driver.specs import SPECS_ROOT
from openeo_driver.utils import smart_bool
from shapely.geometry import shape, mapping

_log = logging.getLogger(__name__)

# Set up process registries (version dependent)
process_registry_040 = ProcessRegistry(spec_root=SPECS_ROOT / 'openeo-processes/0.4')
process_registry_100 = ProcessRegistry(spec_root=SPECS_ROOT / 'openeo-processes/1.0')

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
process_registry_100.add_spec_by_name(
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
)


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

def evaluate(processGraph: dict, viewingParameters=None) -> ImageCollection:
    """
    Converts the json representation of a (part of a) process graph into the corresponding Python ImageCollection.
    :param processGraph:
    :param viewingParameters:
    :return:  an ImageCollection
    """
    if viewingParameters is None:
        warnings.warn("Blindly assuming 0.4.0")
        viewingParameters = {
            'version': '0.4.0'
        }
    # TODO avoid local import
    from openeo.internal.process_graph_visitor import ProcessGraphVisitor
    top_level_node = ProcessGraphVisitor.dereference_from_node_arguments(processGraph)
    return convert_node(processGraph[top_level_node], viewingParameters)


def convert_node(processGraph: dict, viewingParameters=None):
    if isinstance(processGraph, dict):
        if 'process_id' in processGraph:
            return apply_process(processGraph['process_id'], processGraph.get('arguments', {}), viewingParameters)
        elif 'node' in processGraph:
            return convert_node(processGraph['node'], viewingParameters)
        elif 'callback' in processGraph or 'process_graph' in processGraph:
            # a "process_graph" object is a new process graph, don't evaluate it in the parent graph
            return processGraph
        elif 'from_parameter' in processGraph:
            try:
                return viewingParameters[processGraph['from_parameter']]
            except KeyError:
                raise ProcessParameterMissingException(parameter=processGraph['from_parameter'])
        elif 'from_argument' in processGraph:
            # 0.4-style argument referencer (equivalent with 1.0-style "from_parameter")
            argument_reference = processGraph.get('from_argument')
            # backwards compatibility for clients that still use 'dimension_data', can be removed when clients are upgraded
            if argument_reference == 'dimension_data':
                argument_reference = 'data'
            return viewingParameters.get(argument_reference)
        else:
            # simply return nodes that do not require special handling, this is the case for geojson polygons
            return processGraph
    return processGraph


def extract_arg(args: dict, name: str):
    """Get process argument by name."""
    try:
        return args[name]
    except KeyError:
        # TODO: find out process id for proper error message?
        # TODO: automate argument extraction directly from process spec instead of these exract_* functions?
        raise ProcessArgumentRequiredException(process='n/a', argument=name)


def extract_arg_list(args: dict, names: list):
    """Get process argument by list of (legacy/fallback/...) names."""
    for name in names:
        if name in args:
            return args[name]
    # TODO: find out process id for proper error message?
    raise ProcessArgumentRequiredException(process='n/a', argument=str(names))


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
            raise ProcessArgumentRequiredException(process='n/a', argument=step)
    return value


# TODO deprecated process
@deprecated_process
def get_collection(args: Dict, viewingParameters) -> ImageCollection:
    name = extract_arg(args,'name')
    return backend_implementation.catalog.load_collection(name, viewingParameters)


@process
def load_collection(args: Dict, viewingParameters) -> ImageCollection:
    name = extract_arg(args,'id')
    if 'temporal_extent' in args and args['temporal_extent'] is not None:
        extent = args['temporal_extent']
        if len(extent) != 2:
            raise AttributeError("temporal_extent property should be an array of length 2, but got: " + str(extent))
        viewingParameters["from"] = extent[0]
        viewingParameters["to"] = extent[1]
    if "spatial_extent" in args and args['spatial_extent'] is not None:
        extent = args["spatial_extent"]
        viewingParameters["left"] = extract_arg(extent, "west")
        viewingParameters["right"] = extract_arg(extent, "east")
        viewingParameters["top"] = extract_arg(extent, "north")
        viewingParameters["bottom"] = extract_arg(extent, "south")
        viewingParameters["srs"] = extent.get("crs") or "EPSG:4326"
    if "bands" in args and args['bands'] is not None:
        viewingParameters["bands"] = extract_arg(args, "bands")
    if args.get('properties'):
        viewingParameters['properties'] = extract_arg(args, 'properties')

    return backend_implementation.catalog.load_collection(name, viewingParameters)


@non_standard_process(
    ProcessSpec(id='load_disk_data', description="Loads arbitrary from disk.")
        .param(name='format', description="the file format, e.g. 'GTiff'", schema={"type": "string"}, required=True)
        .param(name='glob_pattern', description="a glob pattern that matches the files to load from disk", schema={"type": "string"}, required=True)
        .param(name='options', description="options specific to the file format", schema={"type": "object"})
        .returns(description="the data as a data cube", schema={})
)
def load_disk_data(args: Dict, viewingParameters) -> object:
    format = extract_arg(args, 'format')
    glob_pattern = extract_arg(args, 'glob_pattern')
    options = args.get('options', {})

    return backend_implementation.load_disk_data(format, glob_pattern, options, viewingParameters)


# TODO deprecated process
@deprecated_process
def apply_pixel(args: Dict, viewingParameters) -> ImageCollection:
    """
    DEPRECATED
    :param args:
    :param viewingParameters:
    :return:
    """
    function = extract_arg(args,'function')
    bands = extract_arg(args,'bands')
    decoded_function = pickle.loads(base64.standard_b64decode(function))
    return extract_arg_list(args, ['data','imagery']).apply_pixel(bands, decoded_function)


@process
def apply_dimension(args: Dict, ctx: dict) -> ImageCollection:
    return _evaluate_sub_process_graph(args, 'process', parent_process='apply_dimension', version=ctx["version"])


@process
def save_result(args: Dict, viewingParameters) -> SaveResult:
    format = extract_arg(args,'format')
    options = args.get('options',{})
    data = extract_arg(args, 'data')

    if isinstance(data, SaveResult):
        data.set_format(format, data)
        return data
    if isinstance(data, ImageCollection):
        return ImageCollectionResult(data, format, {**viewingParameters, **options})
    elif isinstance(data, DelayedVector):
        geojsons = (mapping(geometry) for geometry in data.geometries)
        return JSONResult(geojsons)
    elif data is None:
        return data
    else:
        # Assume generic JSON result
        return JSONResult(data, format, options)


# TODO deprecated process
# TODO "apply_tiles" is also confusing: remove it. https://github.com/Open-EO/openeo-python-client/issues/140
@deprecated_process
def apply_tiles(args: Dict, viewingParameters) -> ImageCollection:
    function = extract_arg(args,'code')
    return extract_arg_list(args, ['data','imagery']).apply_tiles(function['source'])


@process
def apply(args: dict, ctx: dict)->ImageCollection:
    """
    Applies a unary process (a local operation) to each value of the specified or all dimensions in the data cube.

    :param args:
    :param viewingParameters:
    :return:
    """
    return _evaluate_sub_process_graph(args, 'process', parent_process='apply', version=ctx["version"])


@process_registry_040.add_function
def reduce(args: dict, ctx: dict) -> ImageCollection:
    """
    https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#reduce

    :param args:
    :param viewingParameters:
    :return:
    """
    reduce_pg = extract_deep(args, "reducer", ["process_graph", "callback"])
    dimension = extract_arg(args, 'dimension')
    binary = smart_bool(args.get('binary', False))
    data_cube = extract_arg_list(args, ['data', 'imagery'])

    # TODO: avoid special case handling for run_udf?
    dimension, band_dim, temporal_dim = _check_dimension(cube=data_cube, dim=dimension, process="reduce")
    if dimension == band_dim:
        if not binary and len(reduce_pg) == 1 and next(iter(reduce_pg.values())).get('process_id') == 'run_udf':
            return _evaluate_sub_process_graph(args, 'reducer', parent_process='reduce', version=ctx["version"])
        visitor = backend_implementation.visit_process_graph(reduce_pg)
        return data_cube.reduce_bands(visitor)
    else:
        return _evaluate_sub_process_graph(args, 'reducer', parent_process='reduce', version=ctx["version"])


@process_registry_100.add_function
def reduce_dimension(args: dict, ctx: dict) -> ImageCollection:
    reduce_pg = extract_deep(args, "reducer", "process_graph")
    dimension = extract_arg(args, 'dimension')
    data_cube = extract_arg(args, 'data')

    # TODO: avoid special case handling for run_udf?
    dimension, band_dim, temporal_dim = _check_dimension(cube=data_cube, dim=dimension, process="reduce_dimension")
    if dimension == band_dim:
        if len(reduce_pg) == 1 and next(iter(reduce_pg.values())).get('process_id') == 'run_udf':
            return _evaluate_sub_process_graph(args, 'reducer', parent_process='reduce_dimension',
                                               version=ctx["version"])
        visitor = backend_implementation.visit_process_graph(reduce_pg)
        return data_cube.reduce_bands(visitor)
    else:
        return _evaluate_sub_process_graph(args, 'reducer', parent_process='reduce_dimension', version=ctx["version"])


@process
def add_dimension(args: dict, ctx: dict):
    data_cube = extract_arg(args, 'data')
    return data_cube.add_dimension(
        name=extract_arg(args, 'name'),
        label=extract_arg_list(args, ['label', 'value']),
        type=extract_arg(args, 'type'),
    )


def _check_dimension(cube: ImageCollection, dim: str, process: str):
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
            raise ProcessArgumentInvalidException(
                argument="dimension", process=process,
                reason="got {d!r}, but should be one of {n!r}".format(d=dim, n=metadata.dimension_names()))

    return dim, band_dim, temporal_dim


@process
def aggregate_temporal(args: dict, ctx: dict) -> ImageCollection:
    """
    https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#reduce

    :param args:
    :param viewingParameters:
    :return:
    """
    return _evaluate_sub_process_graph(args, 'reducer', parent_process='aggregate_temporal', version=ctx["version"])


def _evaluate_sub_process_graph(args: dict, name: str, parent_process: str, version: str):
    """
    Helper function to unwrap and evaluate a sub-process_graph

    :param args: arguments dictionary
    :param name: argument name of sub-process_graph
    :return:
    """
    pg = extract_deep(args, name, ["process_graph", "callback"])
    # TODO: viewingParams are injected into args dict? looks strange
    args["parent_process"] = parent_process
    args["version"] = version
    return evaluate(pg, viewingParameters=args)


@process_registry_040.add_function
def aggregate_polygon(args: dict, ctx: dict) -> ImageCollection:
    return _evaluate_sub_process_graph(args, 'reducer', parent_process='aggregate_polygon', version=ctx["version"])


@process_registry_100.add_function
def aggregate_spatial(args: dict, ctx: dict) -> ImageCollection:
    return _evaluate_sub_process_graph(args, 'reducer', parent_process='aggregate_spatial', version=ctx["version"])


# TODO deprecated process
@deprecated_process
def reduce_by_time( args:Dict, viewingParameters)->ImageCollection:
    """
    Deprecated, use aggregate_temporal
    :param args:
    :param viewingParameters:
    :return:
    """
    function = extract_arg(args,'function')
    temporal_window = extract_arg(args,'temporal_window')
    decoded_function = pickle.loads(base64.standard_b64decode(function))
    return extract_arg(args, 'imagery').aggregate_time(temporal_window, decoded_function)


@process_registry_040.add_function
def mask(args: dict, viewingParameters) -> ImageCollection:
    mask = extract_arg(args, 'mask')
    replacement = args.get('replacement', None)
    cube = extract_arg_list(args, ['data', 'imagery'])
    if isinstance(mask, ImageCollection):
        image_collection = cube.mask(mask=mask, replacement=replacement)
    else:
        polygon = mask.geometries[0] if isinstance(mask, DelayedVector) else shape(mask)
        if polygon.area == 0:
            reason = "mask {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
            raise ProcessArgumentInvalidException(argument='mask', process='mask', reason=reason)
        image_collection = cube.mask_polygon(mask=polygon, replacement=replacement)
    return image_collection


@process_registry_100.add_function
def mask(args: dict, ctx: dict) -> ImageCollection:
    mask = extract_arg(args, 'mask')
    replacement = args.get('replacement', None)
    image_collection = extract_arg(args, 'data').mask(mask=mask, replacement=replacement)
    return image_collection


@process_registry_100.add_function
def mask_polygon(args: dict, ctx: dict) -> ImageCollection:
    mask = extract_arg(args, 'mask')
    replacement = args.get('replacement', None)
    inside = args.get('inside', False)
    polygon = mask.geometries[0] if isinstance(mask, DelayedVector) else shape(mask)
    if polygon.area == 0:
        reason = "mask {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
        raise ProcessArgumentInvalidException(argument='mask', process='mask', reason=reason)
    image_collection = extract_arg(args, 'data').mask_polygon(mask=polygon, replacement=replacement, inside=inside)
    return image_collection


# TODO deprecated process
@deprecated_process
def filter_daterange(args: Dict, viewingParameters) -> ImageCollection:
    #for now we take care of this filtering in 'viewingParameters'
    #from_date = extract_arg(args,'from')
    #to_date = extract_arg(args,'to')
    image_collection = extract_arg(args, 'imagery')
    return image_collection


@process
def filter_temporal(args: Dict, viewingParameters) -> ImageCollection:
    # Note: the temporal range is already extracted in `apply_process` and applied in `GeoPySparkLayerCatalog.load_collection` through the viewingParameters
    image_collection = extract_arg(args, 'data')
    return image_collection


@process
def filter_bbox(args: Dict, viewingParameters) -> ImageCollection:
    # Note: the bbox is already extracted in `apply_process` and applied in `GeoPySparkLayerCatalog.load_collection` through the viewingParameters
    image_collection = extract_arg_list(args, ['data','imagery'])
    return image_collection


@process
def filter_bands(args: Dict, viewingParameters) -> ImageCollection:
    # Note: the bands are already extracted in `apply_process` and applied in `GeoPySparkLayerCatalog.load_collection` through the viewingParameters
    image_collection = extract_arg_list(args, ['data','imagery'])
    return image_collection


# TODO deprecated process?
@deprecated_process
def zonal_statistics(args: Dict, viewingParameters) -> Dict:
    image_collection = extract_arg_list(args, ['data','imagery'])
    geometry = extract_arg(args, 'regions')
    func = args.get('func', 'mean')

    # TODO: extract srs from geometry

    if func == 'mean' or func == 'avg':
        return image_collection.zonal_statistics(shape(geometry), func)
    else:
        raise AttributeError("func %s is not supported" % func)


@process
def apply_kernel(args: Dict, viewingParameters) -> ImageCollection:
    image_collection = extract_arg(args, 'data')
    kernel = np.asarray(extract_arg(args, 'kernel'))
    factor = args.get('factor', 1.0)
    return image_collection.apply_kernel(kernel,factor)


@process
def ndvi(args: dict, viewingParameters: dict) -> ImageCollection:
    image_collection = extract_arg(args, 'data')
    name = args.get("name", "ndvi")
    return image_collection.ndvi(name=name)

@process
def resample_spatial(args: dict, viewingParameters: dict) -> ImageCollection:
    image_collection = extract_arg(args, 'data')
    resolution = args.get('resolution', 0)
    projection = args.get('projection', None)
    method = args.get('method', 'near')
    align = args.get('align', 'lower-left')
    return image_collection.resample_spatial(resolution=resolution,projection=projection,method=method,align=align)

@process
def merge_cubes(args:dict, viewingParameters:dict) -> ImageCollection:
    cube1 = extract_arg(args,'cube1')
    cube2 = extract_arg(args, 'cube2')
    overlap_resolver = args.get('overlap_resolver')
    #TODO raise check if cubes overlap and raise exception if resolver is missing
    resolver_process = None
    if overlap_resolver:
        pg = extract_arg_list(overlap_resolver, ["process_graph", "callback"])
        if len(pg) != 1:
            raise ProcessArgumentInvalidException(
                argument='overlap_resolver', process='merge_cubes',
                reason='This backend only supports overlap resolvers with exactly one process for now.')
        resolver_process = next(iter(pg.values()))["process_id"]
    return cube1.merge(cube2, resolver_process)


@process
def linear_scale_range(args:dict,viewingParameters:dict) -> ImageCollection:
    image_collection = extract_arg(args, 'x')

    inputMin = extract_arg(args, "inputMin")
    inputMax = extract_arg(args, "inputMax")
    outputMax = args.get("outputMax", 1.0)
    outputMin = args.get("outputMin", 0.0)

    return image_collection.linear_scale_range(inputMin,inputMax,outputMin,outputMax)


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
def histogram(_args, _viewingParameters) -> None:
    # currently only available as a reducer passed to e.g. aggregate_polygon
    raise ProcessUnsupportedException('histogram')


def apply_process(process_id: str, args: Dict, viewingParameters):
    parent_process = viewingParameters.get('parent_process')

    if 'filter_daterange' == process_id or 'filter_temporal' == process_id:
        """
        filter_daterange <= pre 0.3.x
        filter_temporal >= 0.4.x
        """
        # TODO `viewingParameters` is function argument, but written to/manipulated (used as some kind of state object)
        #       which is not obvious and confusing when debugging
        viewingParameters = viewingParameters or {}

        if 'extent' in args:
            #version >= 0.4
            extent = args['extent']
            if len(extent) != 2:
                raise AttributeError("extent property should be an array of length 2, but got: " + str(extent))
            viewingParameters["from"] = extent[0]
            viewingParameters["to"] = extent[1]
        else:
            viewingParameters["from"] = extract_arg(args,"from")
            viewingParameters["to"] = extract_arg(args,"to")
    elif 'filter_bbox' == process_id:
        viewingParameters = viewingParameters or {}
        if "left" in args:
            # <=0.3.x
            viewingParameters["left"] = extract_arg(args,"left")
            viewingParameters["right"] = extract_arg(args,"right")
            viewingParameters["top"] = extract_arg(args,"top")
            viewingParameters["bottom"] = extract_arg(args,"bottom")
            viewingParameters["srs"] = extract_arg(args,"srs")
        else:
            extent = args
            if "extent" in args:
                extent = args["extent"]
            # >=0.4.x
            viewingParameters["left"] = extract_arg(extent, "west")
            viewingParameters["right"] = extract_arg(extent, "east")
            viewingParameters["top"] = extract_arg(extent, "north")
            viewingParameters["bottom"] = extract_arg(extent, "south")
            viewingParameters["srs"] = extent.get("crs") or "EPSG:4326"
    elif process_id in ['zonal_statistics', 'aggregate_polygon', 'aggregate_spatial']:
        shapes = extract_arg_list(args, ['regions', 'polygons', 'geometries'])

        if viewingParameters.get("left") is None:
            if "type" in shapes:  # it's GeoJSON
                polygons = _as_geometry_collection(shapes) if shapes['type'] == 'FeatureCollection' else shapes
                viewingParameters["polygons"] = shape(polygons)
                bbox = viewingParameters["polygons"].bounds
            if "from_node" in shapes:  # it's a dereferenced from_node that contains a DelayedVector
                polygons = convert_node(shapes["node"], viewingParameters)
                viewingParameters["polygons"] = polygons.path
                bbox = polygons.bounds

            viewingParameters["left"] = bbox[0]
            viewingParameters["right"] = bbox[2]
            viewingParameters["bottom"] = bbox[1]
            viewingParameters["top"] = bbox[3]
            viewingParameters["srs"] = "EPSG:4326"

            args['polygons'] = polygons  # might as well cache the value instead of re-evaluating it further on

    elif 'filter_bands' == process_id:
        viewingParameters = viewingParameters or {}
        viewingParameters["bands"] = extract_arg(args, "bands")
    elif 'apply' == parent_process:
        if "data" in viewingParameters:
            # The `apply` process passes it's `data` parameter as `x` parameter to subprocess
            viewingParameters["x"] = viewingParameters["data"]

    #first we resolve child nodes and arguments
    args = {name: convert_node(expr, viewingParameters) for (name, expr) in args.items()}

    #when all arguments and dependencies are resolved, we can run the process
    if parent_process == "apply":
        image_collection = extract_arg_list(args, ['x', 'data', 'imagery'])
        if process_id == "run_udf":
            udf = _get_udf(args)
            return image_collection.apply_tiles(udf)
        else:
            # TODO : add support for `apply` with non-trivial child process graphs #EP-3404
           return image_collection.apply(process_id, args)
    elif parent_process in ["reduce", "reduce_dimension"]:
        image_collection = extract_arg_list(args, ['data', 'imagery'])
        dimension = extract_arg(viewingParameters, 'dimension')
        binary = viewingParameters.get('binary',False)
        dimension, band_dim, temporal_dim = _check_dimension(cube=image_collection, dim=dimension, process=parent_process)
        if 'run_udf' == process_id and not binary:
            if dimension == temporal_dim:
                udf = _get_udf(args)
                #EP-2760 a special case of reduce where only a single udf based callback is provided. The more generic case is not yet supported.
                return image_collection.apply_tiles_spatiotemporal(udf)
            elif dimension == band_dim:
                udf = _get_udf(args)
                return image_collection.apply_tiles(udf)

        return image_collection.reduce(process_id,dimension)
    elif parent_process == 'apply_dimension':
        image_collection = extract_arg(args, 'data')
        dimension = viewingParameters.get('dimension', None) # By default, applies the the process on all pixel values (as apply does).
        target_dimension = viewingParameters.get('target_dimension', None)
        dimension, band_dim, temporal_dim = _check_dimension(cube=image_collection, dim=dimension, process=parent_process)
        transformed_collection = None
        if process_id == "run_udf":
            udf = _get_udf(args)
            if dimension == temporal_dim:
                transformed_collection =  image_collection.apply_tiles_spatiotemporal(udf)
            else:
                transformed_collection = image_collection.apply_tiles(udf)
        else:
            transformed_collection = image_collection.apply_dimension(process_id,dimension)
        if target_dimension is not None:
            transformed_collection.rename_dimension(dimension,target_dimension)
        return transformed_collection
    elif parent_process in ['aggregate_polygon', 'aggregate_spatial']:
        image_collection = extract_arg_list(args, ['data', 'imagery'])
        binary = viewingParameters.get('binary',False)
        name = viewingParameters.get('name', 'result')
        polygons = extract_arg(viewingParameters, 'polygons')

        # can be either (inline) GeoJSON or something returned by read_vector
        is_geojson = isinstance(polygons, Dict)

        if is_geojson:
            geometries = shape(polygons)
            return image_collection.zonal_statistics(geometries, func=process_id)
        # TODO: rename to aggregate_polygon?
        return image_collection.zonal_statistics(polygons.path, func=process_id)

    elif parent_process == 'aggregate_temporal':
        image_collection = extract_arg_list(args, ['data', 'imagery'])
        intervals = extract_arg(viewingParameters, 'intervals')
        labels = extract_arg(viewingParameters, 'labels')
        dimension = viewingParameters.get('dimension', None)
        dimension, _, _ = _check_dimension(cube=image_collection, dim=dimension, process=parent_process)
        return image_collection.aggregate_temporal(intervals,labels,process_id,dimension)
    else:
        process_registry = get_process_registry(ComparableVersion(viewingParameters["version"]))
        process_function = process_registry.get_function(process_id)
        return process_function(args, viewingParameters)


@non_standard_process(
    ProcessSpec("read_vector", description="Reads vector data from a file or a URL.")
        .param('filename', description="filename or http url of a vector file", schema={"type": "string"})
        .returns("TODO", schema={"type": "object", "subtype": "vector-cube"})
)
def read_vector(args: Dict, viewingParameters) -> DelayedVector:
    path = extract_arg(args, 'filename')
    return DelayedVector(path)


def _get_udf(args):
    udf = extract_arg(args, "udf")
    runtime = extract_arg(args, "runtime")
    # TODO allow registration of supported runtimes, so we can be more generic
    if runtime != "Python":
        raise NotImplementedError("Unsupported runtime: " + runtime + " this backend only supports the Python runtime.")
    version = args.get("version", None)
    if version is not None and version != "3.5.1" and version != "latest" :
        raise NotImplementedError("Unsupported Python version: " + version + "this backend only support version 3.5.1.")
    return udf


def _as_geometry_collection(feature_collection: dict) -> dict:
    geometries = [feature['geometry'] for feature in feature_collection['features']]

    return {
        'type': 'GeometryCollection',
        'geometries': geometries
    }


