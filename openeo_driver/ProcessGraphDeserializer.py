import base64
import importlib
import json
import os
import pickle
from typing import Dict, List
from shapely.geometry import shape

from openeo import ImageCollection
from .ProcessDetails import ProcessDetails
from distutils.version import LooseVersion


process_registry = {}


def process(description: str, args: List[ProcessDetails.Arg] = [], process_id: str = None):
    def add_to_registry(f):
        process_details = ProcessDetails(process_id or f.__name__, description, args)
        process_registry[process_details.process_id] = process_details
        return f

    return add_to_registry


def getImageCollection(product_id:str, viewingParameters):
    raise Exception("Please provide getImageCollection method in your base package.")

def health_check():
    return "Default health check OK!"

i = importlib.import_module(os.getenv('DRIVER_IMPLEMENTATION_PACKAGE', "dummy_impl"))
getImageCollection = i.getImageCollection
get_layers = i.get_layers
get_layer = i.get_layer
create_batch_job = i.create_batch_job
run_batch_job = i.run_batch_job
get_batch_job_info = i.get_batch_job_info
get_batch_job_result_filenames = i.get_batch_job_result_filenames
get_batch_job_result_output_dir = i.get_batch_job_result_output_dir

if i.health_check is not None:
    health_check = i.health_check


def evaluate_040(processGraph, viewingParameters = None):
    top_level_node = list_to_graph(processGraph)
    return convert_node(processGraph[top_level_node],viewingParameters)


def list_to_graph(processGraph):
    """
    Converts a list of process graph nodes into an actual graph, by resolving the references.
    :param processGraph:
    :return: a list containing the top level nodes in the DAG
    """
    result_node = None
    for node in processGraph:
        node_dict = processGraph.get(node)
        if(node_dict.get("result", False)):
            result_node = node
        arguments = node_dict.get("arguments",{})
        for a in arguments:
            arg = arguments[a]
            if type(arg) is dict and "from_node" in arg:
                from_node_id = arg["from_node"]
                from_node = processGraph.get(from_node_id,None)
                if(from_node is None):
                    raise ValueError("Node not found in process graph: " + from_node_id + ". Referenced by: " + node)
                arg["node"] = from_node

    if result_node is None:
        raise ValueError("The provided process graph does not contain a result node.")
    return result_node

def evaluate(processGraph, viewingParameters = None):
    """
    Converts the json representation of a (part of a) process graph into the corresponding Python ImageCollection.
    :param processGraph:
    :param viewingParameters:
    :return:  an ImageCollection
    """


    if viewingParameters is None:
        viewingParameters = {
            'version' : '0.3.1'
        }
    if LooseVersion(viewingParameters.get('version','0.3.1')) >= LooseVersion('0.4.0'):
        return evaluate_040(processGraph,viewingParameters)

    return convert_node(processGraph,viewingParameters)


def convert_node_03x(processGraph, viewingParameters = None):
    if type(processGraph) is dict:
        if 'product_id' in processGraph:
            return getImageCollection(processGraph['product_id'],viewingParameters)
        elif 'collection_id' in processGraph:
            return getImageCollection(processGraph['collection_id'],viewingParameters)
        elif 'process_graph' in processGraph:
            return convert_node(processGraph['process_graph'], viewingParameters)
        elif 'imagery' in processGraph:
            return convert_node(processGraph['imagery'], viewingParameters)
        elif 'process_id' in processGraph:
            return apply_process(processGraph['process_id'], processGraph['args'], viewingParameters)
    return processGraph

def convert_node_04x(processGraph, viewingParameters = None):
    if type(processGraph) is dict:
        if 'process_id' in processGraph:
            return apply_process(processGraph['process_id'], processGraph.get('arguments',{}), viewingParameters)
        elif 'node' in processGraph:
            return convert_node(processGraph['node'],viewingParameters)
        else:
            raise ValueError('Unsupported process graph node: \n' + json.dumps(processGraph,indent=1))
    return processGraph

def convert_node(dag, viewingParameters = None):
    """
    Depth first traversion and conversion of process graph
    :param dag:
    :param viewingParameters:
    :return:
    """
    if LooseVersion(viewingParameters.get('version', '0.3.1')) >= LooseVersion('0.4.0'):
        return convert_node_04x(dag, viewingParameters)
    else:
        return convert_node_03x(dag,viewingParameters)


def extract_arg(args:Dict,name:str):
    try:
        return args[name]
    except KeyError:
        raise AttributeError(
            "Required argument " +name +" should not be null. Arguments: \n" + json.dumps(args,indent=1))

@process(description="Load an data cube (image collection) based on it's name",
         args=[ProcessDetails.Arg('name', "The name of the collection to load."),])
def get_collection( args:Dict, viewingParameters)->ImageCollection:
    name = extract_arg(args,'name')
    return getImageCollection(name,viewingParameters)

@process(description="Apply a function to the given set of bands in this image collection.",
         args=[ProcessDetails.Arg('function', "A function that gets the value of one pixel (including all bands) as input and produces a single scalar or tuple output."),
               ProcessDetails.Arg('bands', "A set of bands.")])
def apply_pixel( args:Dict, viewingParameters)->ImageCollection:
    function = extract_arg(args,'function')
    bands = extract_arg(args,'bands')
    decoded_function = pickle.loads(base64.standard_b64decode(function))
    return extract_arg(args, 'imagery').apply_pixel(bands, decoded_function)

@process(description="Apply a function to the tiles of this image collection.",
         args=[ProcessDetails.Arg('function', "A function that gets a tile as input and produces a tile as output. The function should follow the openeo_udf specification.")])
def apply_tiles(args:Dict, viewingParameters)->ImageCollection:
    function = extract_arg(args,'code')
    return extract_arg(args, 'imagery').apply_tiles(function['source'])


@process(process_id="reduce_time",
         description="Applies a windowed reduction to a timeseries by applying a user defined function.",
         args=[ProcessDetails.Arg('function', "The function to apply to each time window."),
               ProcessDetails.Arg('temporal_window', "A time window.")])
def reduce_by_time( args:Dict, viewingParameters)->ImageCollection:
    function = extract_arg(args,'function')
    temporal_window = extract_arg(args,'temporal_window')
    decoded_function = pickle.loads(base64.standard_b64decode(function))
    return extract_arg(args, 'imagery').aggregate_time(temporal_window, decoded_function)


@process(description="Finds the minimum value of time series for all bands of the input dataset.")
def min_time(args:Dict,viewingParameters)->ImageCollection:
    #TODO this function should invalidate any filter_daterange set in a parent node
    return extract_arg(args, 'imagery').min_time()


@process(description="Finds the maximum value of time series for all bands of the input dataset.")
def max_time(args:Dict,viewingParameters)->ImageCollection:
    #TODO this function should invalidate any filter_daterange set in a parent node
    return  extract_arg(args, 'imagery').max_time()


@process(description="Mask the image collection using a polygon. All pixels outside the polygon should be set to the nodata value. "
                     "All pixels inside, or intersecting the polygon should retain their original value.",
         args=[ProcessDetails.Arg('mask_shape', "The shape to use as a mask")])
def mask(args:Dict,viewingParameters)->ImageCollection:
    geometry = extract_arg(args, 'mask_shape')
    srs_code = geometry.get("crs",{}).get("name","EPSG:4326")
    return  extract_arg(args, 'imagery').mask(shape(geometry),srs_code)

@process(description="Mask the image collection using another image collection. "
                     "The mask image collection will be regridded to match this image collection.",
         args=[ProcessDetails.Arg('mask', "The image collection to use as a mask")])
def mask_by_raster(args:Dict,viewingParameters)->ImageCollection:
    raise NotImplementedError("Not yet implemented")


@process(description="Specifies a date range filter to be applied on the ImageCollection.",
         args=[ProcessDetails.Arg('imagery', "The image collection to filter."),
               ProcessDetails.Arg('from', "Includes all data newer than the specified ISO 8601 date or date-time with simultaneous consideration of to."),
               ProcessDetails.Arg('to', "Includes all data older than the specified ISO 8601 date or date-time with simultaneous consideration of from.")])
def filter_daterange(args: Dict, viewingParameters)->ImageCollection:
    #for now we take care of this filtering in 'viewingParameters'
    #from_date = extract_arg(args,'from')
    #to_date = extract_arg(args,'to')
    image_collection = extract_arg(args, 'imagery')
    return image_collection

@process(description="Specifies a temporal filter to be applied on a data cube.",
         args=[ProcessDetails.Arg('data', "The data cube to filter."),
               ProcessDetails.Arg('from', "Includes all data newer than the specified ISO 8601 date or date-time with simultaneous consideration of to."),
               ProcessDetails.Arg('to', "Includes all data older than the specified ISO 8601 date or date-time with simultaneous consideration of from.")])
def filter_temporal(args: Dict, viewingParameters)->ImageCollection:
    #for now we take care of this filtering in 'viewingParameters'
    #from_date = extract_arg(args,'from')
    #to_date = extract_arg(args,'to')
    image_collection = extract_arg(args, 'data')
    return image_collection

@process(description="Specifies a bounding box to filter input image collections.",
         args=[ProcessDetails.Arg('imagery', "The image collection to filter."),
               ProcessDetails.Arg('left', "The left side of the bounding box."),
               ProcessDetails.Arg('right', "The right side of the bounding box."),
               ProcessDetails.Arg('top', "The top of the bounding box."),
               ProcessDetails.Arg('bottom', "The bottom of the bounding box."),
               ProcessDetails.Arg('srs', "The spatial reference system of the bounding box.")])
def filter_bbox(args:Dict,viewingParameters)->ImageCollection:

    left = extract_arg(args, "left")
    right = extract_arg(args, "right")
    top = extract_arg(args, "top")
    bottom = extract_arg(args, "bottom")
    srs = extract_arg(args, "srs")
    image_collection = extract_arg(args, 'imagery').bbox_filter(left,right,top,bottom,srs)
    return image_collection

@process(description="Computes zonal statistics over a given polygon",
         args=[ProcessDetails.Arg('imagery', "The image collection to compute statistics on."),
               ProcessDetails.Arg('regions', "The GeoJson Polygon defining the zone.")
               ])
def zonal_statistics(args: Dict, viewingParameters) -> Dict:
    image_collection = extract_arg(args, 'imagery')
    geometry = extract_arg(args, 'regions')
    func = args.get('func', 'mean')

    # TODO: extract srs from geometry

    if func == 'mean' or func == 'avg':
        return image_collection.zonal_statistics(shape(geometry), func)
    else:
        raise AttributeError("func %s is not supported" % func)


def apply_process(process_id: str, args: Dict, viewingParameters):

    if 'filter_daterange' == process_id or 'filter_temporal' == process_id:
        """
        filter_daterange <= pre 0.3.x
        filter_temporal >= 0.4.x
        """
        viewingParameters = viewingParameters or {}
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
            # >=0.4.x
            viewingParameters["left"] = extract_arg(args, "west")
            viewingParameters["right"] = extract_arg(args, "east")
            viewingParameters["top"] = extract_arg(args, "north")
            viewingParameters["bottom"] = extract_arg(args, "south")
            viewingParameters["srs"] = extract_arg(args, "crs")
    elif 'zonal_statistics' == process_id:
        geometry = extract_arg(args, 'regions')
        bbox = shape(geometry).bounds
        if(viewingParameters.get("left") is None ):
            viewingParameters["left"] = bbox[0]
            viewingParameters["right"] = bbox[2]
            viewingParameters["bottom"] = bbox[1]
            viewingParameters["top"] = bbox[3]
            viewingParameters["srs"] = "EPSG:4326"

    args = {name: convert_node(expr, viewingParameters) for (name, expr) in args.items() }

    print(globals().keys())
    process_function = globals()[process_id]
    if process_function is None:
        raise RuntimeError("No process found with name: "+process_id)
    return process_function(args, viewingParameters)


def getProcesses(substring: str = None):
    #def filter_details(process_details):
    #    return {k: v for k, v in process_details.items() if k in ['process_id', 'description']}

    return [process_details.serialize() for process_id, process_details in process_registry.items()
            if not substring or substring.lower() in process_id.lower()]


def getProcess(process_id: str):
    process_details = process_registry.get(process_id)
    return process_details.serialize() if process_details else None
