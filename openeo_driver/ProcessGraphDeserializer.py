import base64
import importlib
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict
from urllib.parse import urlparse

import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import shape
from shapely.geometry.collection import GeometryCollection

from openeo import ImageCollection
from openeo_driver.processes import ProcessRegistry, ProcessSpec
from openeo_driver.save_result import ImageCollectionResult, JSONResult, SaveResult

_log = logging.getLogger(__name__)


# Set up process registry
process_registry = ProcessRegistry()

# Bootstrap with some mathematical/logical processes
for p in [
    'max', 'min', 'mean', 'variance', 'absolute', 'ln', 'ceil', 'floor', 'cos', 'sin', 'run_udf',
    'not', 'eq', 'lt', 'lte', 'gt', 'gte', 'or', 'and', 'divide', 'product', 'subtract', 'sum',
]:
    process_registry.add_by_name(p)

# Decorator shortcut to easily register functions as processes
process = process_registry.add_function


def getImageCollection(product_id:str, viewingParameters):
    raise Exception("Please provide getImageCollection method in your base package.")


def health_check():
    return "Default health check OK!"


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
        elif 'callback' in processGraph:
            # a callback object is a new process graph, don't evaluate it in the parent graph
            return processGraph
        elif 'from_argument' in processGraph:
            argument_reference = processGraph.get('from_argument')
            # backwards compatibility for clients that still use 'dimension_data', can be removed when clients are upgraded
            if argument_reference == 'dimension_data':
                argument_reference = 'data'
            return viewingParameters.get(argument_reference)
        else:
            # simply return nodes that do not require special handling, this is the case for geojson polygons
            return processGraph
    return processGraph


def extract_arg(args: Dict, name: str):
    try:
        return args[name]
    except KeyError:
        raise ValueError("Required argument {n!r} should not be null. Arguments: {a!r}".format(n=name, a=args))


def extract_arg_list(args:Dict,names:list):
    for name in names:
        value = args.get(name, None)
        if value is not None:
            return value
    raise AttributeError(
            "Required argument " +str(names) +" should not be null. Arguments: \n" + json.dumps(args,indent=1))


# TODO deprecated process
@process_registry.add_deprecated
def get_collection(args: Dict, viewingParameters) -> ImageCollection:
    name = extract_arg(args,'name')
    return getImageCollection(name,viewingParameters)


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

    return getImageCollection(name,viewingParameters)


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
    return _evaluate_callback_process(args, 'process', 'apply_dimension')


@process
def save_result(args: Dict, viewingParameters) -> SaveResult:
    format = extract_arg(args,'format')
    options = args.get('options',{})
    data = extract_arg(args, 'data')
    if isinstance(data, ImageCollection):
        return ImageCollectionResult(data,format,options)
    elif data is None:
        return data
    else:
        return JSONResult(data,format,options)


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

    return _evaluate_callback_process(args,'process','apply')


@process
def reduce(args: Dict, viewingParameters) -> ImageCollection:
    """
    https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#reduce

    :param args:
    :param viewingParameters:
    :return:
    """
    reducer = extract_arg(args,'reducer')
    callback = extract_arg(reducer,'callback')
    dimension = extract_arg(args,'dimension')
    binary = args.get('binary',False)
    if type(binary) == str:
        binary = binary.upper() == 'TRUE'
        args['binary'] = binary

    data_cube = extract_arg_list(args, ['data', 'imagery'])
    if dimension == 'spectral_bands':
        if not binary and len(callback)==1 and next(iter(callback.values())).get('process_id') == 'run_udf':
            #it would be better to avoid having special cases everywhere to support udf's
            return _evaluate_callback_process(args, 'reducer', 'reduce')
        if create_process_visitor is not None:
            visitor = create_process_visitor().accept_process_graph(callback)
            return data_cube.reduce_bands(visitor)
        else:
            raise AttributeError('Reduce on spectral_bands is not supported by this backend.')
    else:
        return _evaluate_callback_process(args, 'reducer','reduce')


@process
def aggregate_temporal(args: Dict, viewingParameters) -> ImageCollection:
    """
    https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#reduce

    :param args:
    :param viewingParameters:
    :return:
    """
    return _evaluate_callback_process(args,'reducer','aggregate_temporal')

def _evaluate_callback_process(args,callback_name:str,parent_process:str):
    """
    Helper function to unwrap and evaluate callback

    :param args:
    :param callback_name:
    :return:
    """

    callback_block = extract_arg(args,callback_name)
    callback = extract_arg(callback_block, 'callback')
    args["parent_process"] = parent_process
    args["version"] = "0.4.0"
    return evaluate(callback, args)


@process
def aggregate_polygon(args: Dict, viewingParameters) -> ImageCollection:
    """
    https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#reduce

    :param args:
    :param viewingParameters:
    :return:
    """
    return _evaluate_callback_process(args,'reducer','aggregate_polygon')


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


@process_registry.add_function_with_spec(
    ProcessSpec(id="min_time", description="Finds the minimum value of time series for all bands of the input dataset.")
        .param("data", "Raster data cube", schema=ProcessSpec.RASTERCUBE)
        .returns("Raster data cube", schema=ProcessSpec.RASTERCUBE)
)
def min_time(args: Dict, viewingParameters) -> ImageCollection:
    #TODO this function should invalidate any filter_daterange set in a parent node
    return extract_arg_list(args, ['data','imagery']).min_time()


@process_registry.add_function_with_spec(
    ProcessSpec(id="max_time", description="Finds the maximum value of time series for all bands of the input dataset.")
        .param("data", "Raster data cube", schema=ProcessSpec.RASTERCUBE)
        .returns("Raster data cube", schema=ProcessSpec.RASTERCUBE)
)
def max_time(args: Dict, viewingParameters) -> ImageCollection:
    #TODO this function should invalidate any filter_daterange set in a parent node
    return  extract_arg_list(args, ['data','imagery']).max_time()


# TODO deprecated process?
@process_registry.add_deprecated
def mask_polygon(args: Dict, viewingParameters) -> ImageCollection:
    geometry = extract_arg(args, 'mask_shape')
    srs_code = geometry.get("crs",{}).get("name","EPSG:4326")
    return  extract_arg_list(args, ['data','imagery']).mask_polygon(shape(geometry),srs_code)


@process
def mask(args: Dict, viewingParameters) -> ImageCollection:
    mask = extract_arg(args,'mask')
    replacement = args.get( 'replacement', None)
    if isinstance(mask, ImageCollection):
        image_collection = extract_arg_list(args, ['data', 'imagery']).mask(rastermask = mask, replacement=replacement)
    else:
        image_collection = extract_arg_list(args, ['data', 'imagery']).mask(polygon = shape(mask),replacement=replacement)
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
    # Note: the temporal range is already extracted in `apply_process` and applied in `getImageCollection` through the viewingParameters
    image_collection = extract_arg(args, 'data')
    return image_collection


@process
def filter_bbox(args: Dict, viewingParameters) -> ImageCollection:
    # Note: the bbox is already extracted in `apply_process` and applied in `getImageCollection` through the viewingParameters
    image_collection = extract_arg_list(args, ['data','imagery'])
    return image_collection


@process
def filter_bands(args: Dict, viewingParameters) -> ImageCollection:
    # Note: the bands are already extracted in `apply_process` and applied in `getImageCollection` through the viewingParameters
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
        polygons = extract_arg_list(args, ['regions','polygons'])

        if "type" in polygons:  # it's GeoJSON
            bbox = shape(polygons).bounds
            if(viewingParameters.get("left") is None ):
                viewingParameters["left"] = bbox[0]
                viewingParameters["right"] = bbox[2]
                viewingParameters["bottom"] = bbox[1]
                viewingParameters["top"] = bbox[3]
                viewingParameters["srs"] = "EPSG:4326"
    elif 'filter_bands' == process_id:
        viewingParameters = viewingParameters or {}
        viewingParameters["bands"] = extract_arg(args, "bands")

    #first we resolve child nodes and arguments
    args = {name: convert_node(expr, viewingParameters) for (name, expr) in args.items() }

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
            elif 'spectral_bands' == dimension:
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

        return image_collection.zonal_statistics(shape(polygons),process_id)
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
        .returns("Raster data cube", schema={"type": "TODO"})
)
def read_vector(args: Dict, viewingParameters):
    filename = extract_arg(args, 'filename')

    if filename.startswith("http"):
        if _is_shapefile(filename):
            local_shp_file = _download_shapefile(filename)
            shp = gpd.read_file(local_shp_file)
            geometry = GeometryCollection(shp.loc[:, 'geometry'].values.tolist())
        else:  # it's GeoJSON
            geojson = requests.get(filename).json()
            geometry = shape(geojson)
    else:  # it's a file on disk
        if filename.endswith(".shp"):
            shp = gpd.read_file(filename)
            geometry = GeometryCollection(shp.loc[:, 'geometry'].values.tolist())
        else:  # it's GeoJSON
            with open(filename, 'r') as f:
                geojson = json.load(f)
                geometry = shape(geojson)

    bbox = geometry.bounds

    if viewingParameters.get("left") is None:
        viewingParameters["left"] = bbox[0]
        viewingParameters["right"] = bbox[2]
        viewingParameters["bottom"] = bbox[1]
        viewingParameters["top"] = bbox[3]
        viewingParameters["srs"] = "EPSG:4326"

    return geometry  # FIXME: what is the API response of a read_vector-only process graph?


def _filename(url: str) -> str:
    return urlparse(url).path.split("/")[-1]


def _is_shapefile(url: str) -> bool:
    return _filename(url).endswith(".shp")


def _download_shapefile(shp_url: str) -> str:
    def expiring_download_directory():
        now = datetime.now()
        now_hourly_truncated = now - timedelta(minutes=now.minute, seconds=now.second, microseconds=now.microsecond)
        hourly_id = hash(shp_url + str(now_hourly_truncated))
        return "/mnt/ceph/Projects/OpenEO/download_%s" % hourly_id

    def save_as(src_url: str, dest_path: str):
        with open(dest_path, 'wb') as f:
            f.write(requests.get(src_url).content)

    download_directory = expiring_download_directory()
    shp_file = download_directory + "/" + _filename(shp_url)

    try:
        os.mkdir(download_directory)

        shx_file = shp_file.replace(".shp", ".shx")
        dbf_file = shp_file.replace(".shp", ".dbf")

        shx_url = shp_url.replace(".shp", ".shx")
        dbf_url = shp_url.replace(".shp", ".dbf")

        save_as(shp_url, shp_file)
        save_as(shx_url, shx_file)
        save_as(dbf_url, dbf_file)
    except FileExistsError:
        pass

    return shp_file


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





# TODO: avoid dumping all these functions at toplevel: wrap all this in a container with defined API and helpful type hinting
# TODO: move all this to views.py (where it will be used) or backend.py?
_driver_implementation_package = os.getenv('DRIVER_IMPLEMENTATION_PACKAGE', "dummy_impl")
_log.info('Using driver implementation package {d}'.format(d=_driver_implementation_package))
i = importlib.import_module(_driver_implementation_package)
getImageCollection = i.getImageCollection
get_layers = i.get_layers
get_layer = i.get_layer
try:
    create_process_visitor = i.create_process_visitor
except AttributeError as e:
    create_process_visitor = None
create_batch_job = i.create_batch_job
run_batch_job = i.run_batch_job
get_batch_job_info = i.get_batch_job_info
get_batch_job_result_filenames = i.get_batch_job_result_filenames
get_batch_job_result_output_dir = i.get_batch_job_result_output_dir
cancel_batch_job = i.cancel_batch_job

# TODO: this just-in-time import is to avoid circular dependency hell
from openeo_driver.backend import OpenEoBackendImplementation


def get_openeo_backend_implementation() -> OpenEoBackendImplementation:
    return i.get_openeo_backend_implementation()


backend_implementation = get_openeo_backend_implementation()

summarize_exception = i.summarize_exception if hasattr(i, 'summarize_exception') else lambda exception: exception

if i.health_check is not None:
    health_check = i.health_check

