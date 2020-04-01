# TODO: rename this module to something in snake case? It doesn't even implement a ProcessGraphDeserializer class.

import base64
import importlib
import logging
import os
import pickle
from typing import Dict

import numpy as np
from shapely.geometry import shape, mapping

from openeo import ImageCollection
from openeo_driver.errors import ProcessArgumentInvalidException, ProcessUnsupportedException, OpenEOApiException
from openeo_driver.processes import ProcessRegistry, ProcessSpec
from openeo_driver.save_result import ImageCollectionResult, JSONResult, SaveResult
from openeo_driver.utils import smart_bool
from openeo_driver.delayed_vector import DelayedVector

_log = logging.getLogger(__name__)


# Set up process registry
process_registry = ProcessRegistry()

# Bootstrap with some mathematical/logical processes
for p in [
    'max', 'min', 'mean', 'variance', 'absolute', 'ln', 'ceil', 'floor', 'cos', 'sin', 'run_udf',
    'not', 'eq', 'lt', 'lte', 'gt', 'gte', 'or', 'and', 'divide', 'product', 'subtract', 'sum', 'median', 'sd', 'array_element'
]:
    process_registry.add_by_name(p)

# Decorator shortcut to easily register functions as processes
process = process_registry.add_function



def evaluate(processGraph: dict, viewingParameters=None) -> ImageCollection:
    """
    Converts the json representation of a (part of a) process graph into the corresponding Python ImageCollection.
    :param processGraph:
    :param viewingParameters:
    :return:  an ImageCollection
    """
    if viewingParameters is None:
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
            return viewingParameters[processGraph.get('from_parameter')]
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


class ProcessArgumentMissingException(OpenEOApiException):
    # TODO use correct exception subclass from errors.py. see #32 #31
    pass


def extract_arg(args: dict, name: str):
    """Get process argument by name."""
    try:
        return args[name]
    except KeyError:
        raise ProcessArgumentMissingException("Missing argument {n!r} in {args!r}".format(n=name, args=args))


def extract_arg_list(args: dict, names: list):
    """Get process argument by list of (legacy/fallback/...) names."""
    for name in names:
        if name in args:
            return args[name]
    raise ProcessArgumentMissingException("Missing argument (any of {n!r}) in {args!r}".format(n=names, args=args))


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
            raise ProcessArgumentMissingException("Missing argument at {s!r} in {a!r}".format(s=steps, a=args))
    return value


# TODO deprecated process
@process_registry.add_deprecated
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

    return backend_implementation.catalog.load_collection(name, viewingParameters)


@process_registry.add_function_with_spec(
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
@process_registry.add_deprecated
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
def apply_dimension(args: Dict, viewingParameters) -> ImageCollection:
    return _evaluate_sub_process_graph(args, 'process', 'apply_dimension')


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
@process_registry.add_deprecated
def apply_tiles(args: Dict, viewingParameters) -> ImageCollection:
    function = extract_arg(args,'code')
    return extract_arg_list(args, ['data','imagery']).apply_tiles(function['source'])


@process
def apply(args:Dict, viewingParameters)->ImageCollection:
    """
    Applies a unary process (a local operation) to each value of the specified or all dimensions in the data cube.

    :param args:
    :param viewingParameters:
    :return:
    """

    return _evaluate_sub_process_graph(args, 'process', 'apply')


@process
def reduce(args: dict, viewingParameters) -> ImageCollection:
    """
    https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#reduce

    :param args:
    :param viewingParameters:
    :return:
    """
    reduce_pg = extract_deep(args, "reducer", ["process_graph", "callback"])
    dimension = extract_arg(args,'dimension')
    binary = smart_bool(args.get('binary',False))

    data_cube = extract_arg_list(args, ['data', 'imagery'])
    # TODO band dimension name should not be hardcoded Open-EO/openeo-python-client#93
    if dimension == 'spectral_bands' or dimension == 'bands':
        if not binary and len(reduce_pg) == 1 and next(iter(reduce_pg.values())).get('process_id') == 'run_udf':
            #it would be better to avoid having special cases everywhere to support udf's
            return _evaluate_sub_process_graph(args, 'reducer', 'reduce')  # TODO #33: reduce -> reduce_dimension
        if create_process_visitor is not None:
            visitor = create_process_visitor().accept_process_graph(reduce_pg)
            return data_cube.reduce_bands(visitor)
        else:
            raise AttributeError('Reduce on bands is not supported by this backend.')
    else:
        return _evaluate_sub_process_graph(args, 'reducer', 'reduce')  # TODO #33: reduce -> reduce_dimension


@process
def aggregate_temporal(args: Dict, viewingParameters) -> ImageCollection:
    """
    https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#reduce

    :param args:
    :param viewingParameters:
    :return:
    """
    return _evaluate_sub_process_graph(args, 'reducer', 'aggregate_temporal')


def _evaluate_sub_process_graph(args, name: str, parent_process: str):
    """
    Helper function to unwrap and evaluate a sub-process_graph

    :param args: arguments dictionary
    :param name: argument name
    :return:
    """
    pg = extract_deep(args, name, ["process_graph", "callback"])
    # TODO: viewingParams are injected into args dict? looks strange
    args["parent_process"] = parent_process
    # TODO: why not taking original version? #33
    args["version"] = "0.4.0"
    return evaluate(pg, viewingParameters=args)


@process
def aggregate_polygon(args: Dict, viewingParameters) -> ImageCollection:
    """
    https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#reduce

    :param args:
    :param viewingParameters:
    :return:
    """
    return _evaluate_sub_process_graph(args, 'reducer', 'aggregate_polygon')  # TODO #33: aggregate_spatial


# TODO deprecated process
@process_registry.add_deprecated
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

@process
def mask(args: Dict, viewingParameters) -> ImageCollection:
    mask = extract_arg(args,'mask')
    replacement = args.get( 'replacement', None)
    if isinstance(mask, ImageCollection):
        image_collection = extract_arg_list(args, ['data', 'imagery']).mask(rastermask=mask, replacement=replacement)
    else:
        polygon = mask.geometries[0] if isinstance(mask, DelayedVector) else shape(mask)

        if polygon.area == 0:
            reason = "mask {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
            raise ProcessArgumentInvalidException(argument='mask', process='mask', reason=reason)

        image_collection = extract_arg_list(args, ['data', 'imagery']).mask(polygon=polygon, replacement=replacement)
    return image_collection


# TODO deprecated process
@process_registry.add_deprecated
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
@process_registry.add_deprecated
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


@process_registry.add_function_with_spec(
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
    elif 'zonal_statistics' == process_id or 'aggregate_polygon' == process_id:
        polygons = extract_arg_list(args, ['regions', 'polygons'])

        if viewingParameters.get("left") is None:
            if "type" in polygons:  # it's GeoJSON
                geometries = _as_geometry_collection(polygons) if polygons['type'] == 'FeatureCollection' else polygons
                bbox = shape(geometries).bounds
            if "from_node" in polygons:  # it's a dereferenced from_node that contains a DelayedVector
                geometries = convert_node(polygons["node"], viewingParameters)
                bbox = geometries.bounds

            viewingParameters["left"] = bbox[0]
            viewingParameters["right"] = bbox[2]
            viewingParameters["bottom"] = bbox[1]
            viewingParameters["top"] = bbox[3]
            viewingParameters["srs"] = "EPSG:4326"

            args['polygons'] = geometries  # might as well cache the value instead of re-evaluating it further on

    elif 'filter_bands' == process_id:
        viewingParameters = viewingParameters or {}
        viewingParameters["bands"] = extract_arg(args, "bands")

    #first we resolve child nodes and arguments
    args = {name: convert_node(expr, viewingParameters) for (name, expr) in args.items()}

    #when all arguments and dependencies are resolved, we can run the process
    if(viewingParameters.get("parent_process",None) == "apply"):
        image_collection = extract_arg_list(args, ['data', 'imagery'])
        if process_id == "run_udf":
            udf = _get_udf(args)

            return image_collection.apply_tiles(udf)
        else:
           return image_collection.apply(process_id,args)
    elif (viewingParameters.get('parent_process', None) == 'reduce'):
        image_collection = extract_arg_list(args, ['data', 'imagery'])
        dimension = extract_arg(viewingParameters, 'dimension')
        binary = viewingParameters.get('binary',False)
        if 'run_udf' == process_id and not binary:
            if 'temporal' == dimension:
                udf = _get_udf(args)
                #EP-2760 a special case of reduce where only a single udf based callback is provided. The more generic case is not yet supported.
                return image_collection.apply_tiles_spatiotemporal(udf)
            elif 'spectral_bands' == dimension or 'bands' == dimension:
                udf = _get_udf(args)
                return image_collection.apply_tiles(udf)

        return image_collection.reduce(process_id,dimension)
    elif (viewingParameters.get('parent_process', None) == 'apply_dimension'):
        image_collection = extract_arg(args, 'data')
        dimension = viewingParameters.get('dimension',None) # By default, applies the the process on all pixel values (as apply does).
        if process_id == "run_udf":
            udf = _get_udf(args)
            if 'temporal' == dimension:
                return image_collection.apply_tiles_spatiotemporal(udf)
            else:
                return image_collection.apply_tiles(udf)

        else:

            return image_collection.apply_dimension(process_id,dimension)

    elif (viewingParameters.get('parent_process', None) == 'aggregate_polygon'):
        image_collection = extract_arg_list(args, ['data', 'imagery'])
        binary = viewingParameters.get('binary',False)
        name = viewingParameters.get('name', 'result')
        polygons = extract_arg(viewingParameters, 'polygons')

        # can be either (inline) GeoJSON or something returned by read_vector
        is_geojson = isinstance(polygons, Dict)

        if is_geojson:
            geometries = shape(polygons)
            return image_collection.zonal_statistics(geometries, func=process_id)

        return image_collection.zonal_statistics(polygons.path, func=process_id)

    elif (viewingParameters.get('parent_process', None) == 'aggregate_temporal'):
        image_collection = extract_arg_list(args, ['data', 'imagery'])
        intervals = extract_arg(viewingParameters, 'intervals')
        labels = extract_arg(viewingParameters, 'labels')
        dimension = viewingParameters.get('dimension', None)
        return image_collection.aggregate_temporal(intervals,labels,process_id,dimension)
    else:
        process_function = process_registry.get_function(process_id)
        return process_function(args, viewingParameters)


@process_registry.add_function_with_spec(
    ProcessSpec("read_vector", description="Reads vector data from a file or a URL.")
        .param('filename', description="filename or http url of a vector file", schema={"type": "string"})
        .returns("TODO", schema={"type": "TODO"})
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



def getProcesses(substring: str = None):
    # TODO: move this also to OpenEoBackendImplementation ?
    return [spec for spec in process_registry._specs.values() if not substring or substring.lower() in spec['id']]


def getProcess(process_id: str):
    return process_registry._specs.get(process_id)


def _as_geometry_collection(feature_collection: dict) -> dict:
    geometries = [feature['geometry'] for feature in feature_collection['features']]

    return {
        'type': 'GeometryCollection',
        'geometries': geometries
    }



# TODO: avoid dumping all these functions at toplevel: wrap all this in a container with defined API and helpful type hinting
# TODO: move all this to views.py (where it will be used) or backend.py?
_driver_implementation_package = os.getenv('DRIVER_IMPLEMENTATION_PACKAGE', "dummy_impl")
_log.info('Using driver implementation package {d}'.format(d=_driver_implementation_package))
i = importlib.import_module(_driver_implementation_package)
try:
    create_process_visitor = i.create_process_visitor
except AttributeError as e:
    create_process_visitor = None
create_batch_job = i.create_batch_job
run_batch_job = i.run_batch_job
get_batch_job_info = i.get_batch_job_info
get_batch_jobs_info = i.get_batch_jobs_info
get_batch_job_result_filenames = i.get_batch_job_result_filenames
get_batch_job_result_output_dir = i.get_batch_job_result_output_dir
get_batch_job_log_entries = i.get_batch_job_log_entries
cancel_batch_job = i.cancel_batch_job

# TODO: this just-in-time import is to avoid circular dependency hell
from openeo_driver.backend import OpenEoBackendImplementation


def get_openeo_backend_implementation() -> OpenEoBackendImplementation:
    return i.get_openeo_backend_implementation()


backend_implementation = get_openeo_backend_implementation()

summarize_exception = i.summarize_exception if hasattr(i, 'summarize_exception') else lambda exception: exception
