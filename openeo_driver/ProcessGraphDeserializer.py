import base64
import importlib
import json
import os
import pickle
from typing import Dict, List

from openeo import ImageCollection


def getImageCollection(product_id:str, viewingParameters):
    raise "Please provide getImageCollection method in your base package."

def health_check():
    return "Default health check OK!"

i = importlib.import_module(os.getenv('DRIVER_IMPLEMENTATION_PACKAGE', "openeogeotrellis"))
getImageCollection = i.getImageCollection

if i.health_check is not None:
    health_check = i.health_check



def graphToRdd(processGraph:Dict, viewingParameters)->ImageCollection:
    if 'product_id' in processGraph:
        return getImageCollection(processGraph['product_id'],viewingParameters)
    elif 'process_id' in processGraph:
        return getProcessImageCollection(processGraph['process_id'],processGraph['args'],viewingParameters)
    else:
        raise AttributeError("Process should contain either product_id or process_id, but got: \n" + json.dumps(processGraph,indent=1))


def extract_arg(args:Dict,name:str)->str:
    try:
        return args[name]
    except KeyError:
        raise AttributeError(
            "Required argument " +name +" should not be null in band_arithmetic. Arguments: \n" + json.dumps(args,indent=1))


def apply_pixel(input_collection:List[ImageCollection], args:Dict, viewingParameters)->ImageCollection:
    function = extract_arg(args,'function')
    bands = extract_arg(args,'bands')
    decoded_function = pickle.loads(base64.standard_b64decode(function))
    return input_collection[0].apply_pixel(bands, decoded_function)


def reduce_by_time(input_collection:List[ImageCollection], args:Dict, viewingParameters)->ImageCollection:
    function = extract_arg(args,'function')
    temporal_window = extract_arg(args,'temporal_window')
    decoded_function = pickle.loads(base64.standard_b64decode(function))
    return input_collection[0].aggregate_time(temporal_window, decoded_function)


def getProcessImageCollection( process_id:str, args:Dict, viewingParameters)->ImageCollection:

    collections = extract_arg(args,'collections')
    child_collections = list(map(lambda c:graphToRdd(c,viewingParameters),collections))

    print(globals().keys())
    process_function = globals()[process_id]
    if process_function is None:
        raise RuntimeError("No process found with name: "+process_id)
    return process_function(child_collections,args,viewingParameters)
