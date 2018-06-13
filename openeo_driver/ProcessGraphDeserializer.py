import base64
import importlib
import json
import os
import pickle
from typing import Dict, List
from shapely.geometry import shape

from openeo import ImageCollection
from .ProcessDetails import ProcessDetails

process_registry = {}


def process(description: str, args: List[ProcessDetails.Arg] = [], process_id: str = None):
    def add_to_registry(f):
        process_details = ProcessDetails(process_id or f.__name__, description, args)
        process_registry[process_details.process_id] = process_details
        return f

    return add_to_registry


def getImageCollection(product_id:str, viewingParameters):
    raise "Please provide getImageCollection method in your base package."

def health_check():
    return "Default health check OK!"

i = importlib.import_module(os.getenv('DRIVER_IMPLEMENTATION_PACKAGE', "openeogeotrellis"))
getImageCollection = i.getImageCollection
get_layers = i.get_layers

if i.health_check is not None:
    health_check = i.health_check


def evaluate(processGraph: Dict, viewingParameters = {}):
    if 'product_id' in processGraph:
        return getImageCollection(processGraph['product_id'],viewingParameters)
    elif 'collection_id' in processGraph:
        return getImageCollection(processGraph['collection_id'],viewingParameters)
    elif 'process_graph' in processGraph:
        return evaluate(processGraph['process_graph'], viewingParameters)
    elif 'process_id' in processGraph:
        return apply_process(processGraph['process_id'], processGraph['args'], viewingParameters)
    else:
        return processGraph


def extract_arg(args:Dict,name:str)->str:
    try:
        return args[name]
    except KeyError:
        raise AttributeError(
            "Required argument " +name +" should not be null in band_arithmetic. Arguments: \n" + json.dumps(args,indent=1))


@process(description="Apply a function to the given set of bands in this image collection.",
         args=[ProcessDetails.Arg('function', "A function that gets the value of one pixel (including all bands) as input and produces a single scalar or tuple output."),
               ProcessDetails.Arg('bands', "A set of bands.")])
def apply_pixel(input_collection:List[ImageCollection], args:Dict, viewingParameters)->ImageCollection:
    function = extract_arg(args,'function')
    bands = extract_arg(args,'bands')
    decoded_function = pickle.loads(base64.standard_b64decode(function))
    return input_collection[0].apply_pixel(bands, decoded_function)

def apply_tiles(input_collection:List[ImageCollection], args:Dict, viewingParameters)->ImageCollection:
    function = extract_arg(args,'code')
    return input_collection[0].apply_tiles(function['source'])


@process(process_id="reduce_time",
         description="Applies a windowed reduction to a timeseries by applying a user defined function.",
         args=[ProcessDetails.Arg('function', "The function to apply to each time window."),
               ProcessDetails.Arg('temporal_window', "A time window.")])
def reduce_by_time(input_collection:List[ImageCollection], args:Dict, viewingParameters)->ImageCollection:
    function = extract_arg(args,'function')
    temporal_window = extract_arg(args,'temporal_window')
    decoded_function = pickle.loads(base64.standard_b64decode(function))
    return input_collection[0].aggregate_time(temporal_window, decoded_function)


@process(description="Finds the minimum value of time series for all bands of the input dataset.")
def min_time(input_collection:List[ImageCollection],args:Dict,viewingParameters)->ImageCollection:
    #TODO this function should invalidate any filter_daterange set in a parent node
    return input_collection[0].min_time()


@process(description="Finds the maximum value of time series for all bands of the input dataset.")
def max_time(input_collection:List[ImageCollection],args:Dict,viewingParameters)->ImageCollection:
    #TODO this function should invalidate any filter_daterange set in a parent node
    return input_collection[0].max_time()


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

@process(description="Specifies a bounding box to filter input image collections.",
         args=[ProcessDetails.Arg('imagery', "The image collection to filter."),
               ProcessDetails.Arg('left', "The left side of the bounding box."),
               ProcessDetails.Arg('right', "The right side of the bounding box."),
               ProcessDetails.Arg('top', "The top of the bounding box."),
               ProcessDetails.Arg('bottom', "The bottom of the bounding box."),
               ProcessDetails.Arg('srs', "The spatial reference system of the bounding box.")])
def filter_bbox(input_collection:List[ImageCollection],args:Dict,viewingParameters)->ImageCollection:
    #for now we take care of this filtering in 'viewingParameters'
    #from_date = extract_arg(args,'from')
    #to_date = extract_arg(args,'to')
    return input_collection[0]


def zonal_statistics(args: Dict, viewingParameters) -> Dict:
    image_collection = extract_arg(args, 'imagery')
    geometry = extract_arg(args, 'geometry')
    func = args.get('func', 'avg')

    # TODO: extract srs from geometry

    if func == 'avg':
        return image_collection.polygonal_mean_timeseries(shape(geometry))
    else:
        raise AttributeError("func %s is not supported" % func)


def apply_process(process_id: str, args: Dict, viewingParameters):

    if 'filter_daterange' == process_id:
        viewingParameters = viewingParameters or {}
        viewingParameters["from"] = extract_arg(args,"from")
        viewingParameters["to"] = extract_arg(args,"to")
    elif 'filter_bbox' == process_id:
        viewingParameters = viewingParameters or {}
        viewingParameters["left"] = extract_arg(args,"left")
        viewingParameters["right"] = extract_arg(args,"right")
        viewingParameters["top"] = extract_arg(args,"top")
        viewingParameters["bottom"] = extract_arg(args,"bottom")
        viewingParameters["srs"] = extract_arg(args,"srs")

    args = {name: evaluate(expr, viewingParameters) for (name, expr) in args.items()}

    print(globals().keys())
    process_function = globals()[process_id]
    if process_function is None:
        raise RuntimeError("No process found with name: "+process_id)
    return process_function(args, viewingParameters)


def getProcesses(substring: str = None):
    def filter_details(process_details):
        return {k: v for k, v in process_details.items() if k in ['process_id', 'description']}

    return [filter_details(process_details.serialize()) for process_id, process_details in process_registry.items()
            if not substring or substring.lower() in process_id.lower()]


def getProcess(process_id: str):
    process_details = process_registry.get(process_id)
    return process_details.serialize() if process_details else None
