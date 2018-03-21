import base64
import importlib
import json
import os
import pickle
from typing import Dict, List

from openeo import ImageCollection

process_registry = []


def process(description: str, process_id: str = None):
    def add_to_registry(f):
        process_registry.append({
            "process_id": process_id or f.__name__,
            "description": description
        })
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


def graphToRdd(processGraph:Dict, viewingParameters)->ImageCollection:
    if 'product_id' in processGraph:
        return getImageCollection(processGraph['product_id'],viewingParameters)
    elif 'collection_id' in processGraph:
        return getImageCollection(processGraph['collection_id'],viewingParameters)
    elif 'process_graph' in processGraph:
        return graphToRdd(processGraph['process_graph'],viewingParameters)
    elif 'process_id' in processGraph:
        return getProcessImageCollection(processGraph['process_id'],processGraph['args'],viewingParameters)
    else:
        raise AttributeError("Process should contain either collection_id or process_id, but got: \n" + json.dumps(processGraph,indent=1))


def extract_arg(args:Dict,name:str)->str:
    try:
        return args[name]
    except KeyError:
        raise AttributeError(
            "Required argument " +name +" should not be null in band_arithmetic. Arguments: \n" + json.dumps(args,indent=1))


@process(description="Apply a function to the given set of bands in this image collection.")
def apply_pixel(input_collection:List[ImageCollection], args:Dict, viewingParameters)->ImageCollection:
    function = extract_arg(args,'function')
    bands = extract_arg(args,'bands')
    decoded_function = pickle.loads(base64.standard_b64decode(function))
    return input_collection[0].apply_pixel(bands, decoded_function)


@process(process_id="reduce_time", description="Applies a windowed reduction to a timeseries by applying a user defined function.")
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


@process(description="Specifies a date range filter to be applied on the ImageCollection.")
def filter_daterange(input_collection:List[ImageCollection],args:Dict,viewingParameters)->ImageCollection:
    #for now we take care of this filtering in 'viewingParameters'
    #from_date = extract_arg(args,'from')
    #to_date = extract_arg(args,'to')
    return input_collection[0]


@process(description="Specifies a bounding box to filter input image collections.")
def filter_bbox(input_collection:List[ImageCollection],args:Dict,viewingParameters)->ImageCollection:
    #for now we take care of this filtering in 'viewingParameters'
    #from_date = extract_arg(args,'from')
    #to_date = extract_arg(args,'to')
    return input_collection[0]

def getProcessImageCollection( process_id:str, args:Dict, viewingParameters)->ImageCollection:

    collections = extract_arg(args,'collections')
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

    child_collections = list(map(lambda c:graphToRdd(c,viewingParameters),collections))

    print(globals().keys())
    process_function = globals()[process_id]
    if process_function is None:
        raise RuntimeError("No process found with name: "+process_id)
    return process_function(child_collections,args,viewingParameters)


def getSupportedProcesses(substring: str = None):
    return [process for process in process_registry
            if substring.lower() in process["process_id"].lower()] if substring else process_registry
