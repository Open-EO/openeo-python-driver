import base64
import importlib
import json
import os
import pickle
from typing import Dict, List
from shapely.geometry import shape

from openeo import ImageCollection
from openeo_driver.save_result import ImageCollectionResult, JSONResult, SaveResult
from .ProcessDetails import ProcessDetails
from distutils.version import LooseVersion


process_registry = {}


def process(description: str, args: List[ProcessDetails.Arg] = [], process_id: str = None):
    def add_to_registry(f):
        process_details = ProcessDetails(process_id or f.__name__, description, args)
        process_registry[process_details.process_id] = process_details
        return f

    return add_to_registry

def register_extra_processes():
    descriptions = [
        'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/max.json',
        'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/min.json',
        'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/mean.json',
        'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/variance.json',
        'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/absolute.json',
        'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/ln.json',
        'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/ceil.json',
        'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/floor.json',
        'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/cos.json',
        'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/sin.json',
        'https://raw.githubusercontent.com/Open-EO/openeo-processes/master/run_udf.json'
    ]

    import requests
    def load_details(url:str):
        json_description = requests.get(url).json()
        parameters = json_description['parameters']
        args = [ProcessDetails.Arg(k,v['description'],v.get('required',False),v['schema']) for (k,v) in parameters.items()]
        return ProcessDetails(process_id=json_description['id'], description=json_description['description'],args=args,returns=json_description['returns'])
    details = [ load_details(d) for d in descriptions]
    process_registry.update({d.process_id:d for d in details})
register_extra_processes()


def getImageCollection(product_id:str, viewingParameters):
    raise Exception("Please provide getImageCollection method in your base package.")

def health_check():
    return "Default health check OK!"

i = importlib.import_module(os.getenv('DRIVER_IMPLEMENTATION_PACKAGE', "dummy_impl"))
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
get_secondary_services_info = i.get_secondary_services_info
get_secondary_service_info = i.get_secondary_service_info
summarize_exception = i.summarize_exception if hasattr(i, 'summarize_exception') else lambda exception: exception

if i.health_check is not None:
    health_check = i.health_check


def evaluate_040(processGraph, viewingParameters = None):
    if viewingParameters is None:
        viewingParameters = {
            'version' : '0.4.0'
        }
    from openeo.internal.process_graph_visitor import ProcessGraphVisitor
    top_level_node = ProcessGraphVisitor._list_to_graph(processGraph)
    return convert_node(processGraph[top_level_node],viewingParameters)



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
        return evaluate_040(processGraph.get('process_graph',processGraph),viewingParameters)

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
        elif 'callback' in processGraph:
            #a callback object is a new process graph, don't evaluate it in the parent graph
            return processGraph
        elif 'from_argument' in processGraph:
            argument_reference = processGraph.get('from_argument')
            #backwards compatibility for clients that still use 'dimension_data', can be removed when clients are upgraded
            if argument_reference == 'dimension_data':
                argument_reference = 'data'
            return viewingParameters.get(argument_reference)
        else:
            #simply return nodes that do not require special handling, this is the case for geojson polygons
            return processGraph
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
        args_clean = { k:str(v) for (k,v) in args.items()}
        raise AttributeError(
            "Required argument " +name +" should not be null. Arguments: \n" + json.dumps(args_clean,indent=1))

def extract_arg_list(args:Dict,names:list):
    for name in names:
        value = args.get(name, None)
        if value is not None:
            return value
    raise AttributeError(
            "Required argument " +str(names) +" should not be null. Arguments: \n" + json.dumps(args,indent=1))


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
    return extract_arg_list(args, ['data','imagery']).apply_pixel(bands, decoded_function)


@process(description="Save processed data to storage or export to http.",
         args=[ProcessDetails.Arg('data', "The data to save."),
               ProcessDetails.Arg('format', "The file format to save to. It must be one of the values that the server reports as supported output formats, which usually correspond to the short GDAL/OGR codes. This parameter is case insensitive."),
               ProcessDetails.Arg('options', "The file format options to be used to create the file(s). Must correspond to the options that the server reports as supported options for the chosen format. The option names and valid values usually correspond to the GDAL/OGR format options.")])
def save_result( args:Dict, viewingParameters)->SaveResult:
    format = extract_arg(args,'format')
    options = args.get('options',{})
    data = extract_arg(args, 'data')
    if isinstance(data, ImageCollection):
        return ImageCollectionResult(data,format,options)
    elif data is None:
        return data
    else:
        return JSONResult(data,format,options)

@process(description="Apply a function to the tiles of this image collection.",
         args=[ProcessDetails.Arg('function', "A function that gets a tile as input and produces a tile as output. The function should follow the openeo_udf specification.")])
def apply_tiles(args:Dict, viewingParameters)->ImageCollection:
    function = extract_arg(args,'code')
    return extract_arg_list(args, ['data','imagery']).apply_tiles(function['source'])

@process(description="Applies a unary process (a local operation) to each value of the specified or all dimensions in the data cube.",
         args=[ProcessDetails.Arg('process', "A process (callback) to be applied on each value. The specified process must be unary meaning that it must work on a single value."),
                ProcessDetails.Arg('dimensions', "The names of the dimensions to apply the process on. Defaults to an empty array so that all dimensions are used.")

               ])
def apply(args:Dict, viewingParameters)->ImageCollection:
    """
    Applies a unary process (a local operation) to each value of the specified or all dimensions in the data cube.

    :param args:
    :param viewingParameters:
    :return:
    """

    return _evaluate_callback_process(args,'process','apply')


@process(description="Applies a reducer to a data cube dimension by collapsing all the input values along the specified dimension into a single output value computed by the reducer.\nThe reducer must accept an array and return a single value (see parameter reducer). Nominal values are possible, but need to be mapped, e.g. band names to wavelengths, date strings to numeric timestamps since 1970 etc.",
         args=[ProcessDetails.Arg('reducer', "A reducer to be applied on the specified dimension. The reducer must be a callable process (or a set processes) that accepts an array and computes a single return value of the same type as the input values, for example median."),
                ProcessDetails.Arg('dimension', "The dimension over which to reduce.")
               ])
def reduce(args:Dict, viewingParameters)->ImageCollection:
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
        if create_process_visitor is not None:
            converted_process_graph = create_process_visitor().accept_process_graph(callback)
            return data_cube.reduce_bands(converted_process_graph)
        else:
            raise AttributeError('Reduce on spectral_bands is not supported by this backend.')
    else:
        return _evaluate_callback_process(args, 'reducer','reduce')


@process(description='Computes a temporal aggregation based on an array of date and/or time intervals.\nCalendar hierarchies such as year, month, week etc. must be transformed into specific intervals by the clients. For each interval, all data along the dimension will be passed through the reducer. The computed values will be projected to the labels, so the number of labels and the number of intervals need to be equal.\n If the dimension is not set, the data cube is expected to only have one temporal dimension.',
         args=[ProcessDetails.Arg('intervals','Temporal left-closed intervals so that the start time is contained, but not the end time.'),
               ProcessDetails.Arg('labels','Labels for the intervals. The number of labels and the number of groups need to be equal.'),
               ProcessDetails.Arg('reducer','A reducer to be applied on the specified dimension. The reducer must be a callable process (or a set processes) that accepts an array and computes a single return value of the same type as the input values, for example median.'),
               ProcessDetails.Arg('dimension','The temporal dimension for aggregation. All data along the dimension will be passed through the specified reducer. If the dimension is not set, the data cube is expected to only have one temporal dimension.')

               ])
def aggregate_temporal(args:Dict, viewingParameters)->ImageCollection:
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
    return evaluate_040(callback, args)


@process(description='Aggregates zonal statistics for one or multiple polygons over the spatial dimensions.\n\nThe process considers all pixels for which the point at the pixel center intersects with the corresponding polygon (as defined in the Simple Features standard by the OGC).\n\nThe data cube must have been reduced to only contain two raster dimensions and a third dimension the values are aggregated for, for example the temporal dimension to get a time series. Otherwise this process fails with the `TooManyDimensions` error.\n\nThe number of total and valid pixels is returned together with the calculated values.',
         args=[ProcessDetails.Arg('data','A data cube.'),
               ProcessDetails.Arg('polygons','One or more polygons to calculate zonal statistics for. Either specified as GeoJSON or vector data cube.\n\nFor GeoJSON this can be one of the following GeoJSON types:\n\n* A `Polygon` geometry,\n* a `GeometryCollection` containing Polygons,\n* a `Feature` with a `Polygon` geometry or\n* a `FeatureCollection` containing `Feature`s with a `Polygon` geometry.'),
               ProcessDetails.Arg('reducer','A reducer to be applied on all values of each geometry. The reducer must be a callable process (or a set of processes as process graph) such as ``mean()`` that accepts by default array as input. The process can also work on two values by setting the parameter `binary` to `true`.'),
               ProcessDetails.Arg('name','The property name (for GeoJSON) or the new dimension name (for vector cubes) to be used for storing the results. Defaults to `result`.'),
               ProcessDetails.Arg('binary','Specifies whether the process should pass two values to the reducer or a list of values (default).\n\nIf the process passes two values, the reducer must be both associative and commutative as the execution may be executed in parallel and therefore the order of execution is arbitrary.\n\nThis parameter is especially useful for UDFs passed as reducers. Back-ends may still optimize and parallelize processes that work on list of values.')

               ])
def aggregate_polygon(args:Dict, viewingParameters)->ImageCollection:
    """
    https://open-eo.github.io/openeo-api/v/0.4.0/processreference/#reduce

    :param args:
    :param viewingParameters:
    :return:
    """
    return _evaluate_callback_process(args,'reducer','aggregate_polygon')

@process(process_id="reduce_time",
         description="Applies a windowed reduction to a timeseries by applying a user defined function.\n Deprecated: use aggregate_temporal",
         args=[ProcessDetails.Arg('function', "The function to apply to each time window."),
               ProcessDetails.Arg('temporal_window', "A time window.")])
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


@process(description="Finds the minimum value of time series for all bands of the input dataset.", args=[ProcessDetails.Arg('data', "The raster data cube.")])
def min_time(args:Dict,viewingParameters)->ImageCollection:
    #TODO this function should invalidate any filter_daterange set in a parent node
    return extract_arg_list(args, ['data','imagery']).min_time()


@process(description="Finds the maximum value of time series for all bands of the input dataset.", args=[ProcessDetails.Arg('data', "The raster data cube.")])
def max_time(args:Dict,viewingParameters)->ImageCollection:
    #TODO this function should invalidate any filter_daterange set in a parent node
    return  extract_arg_list(args, ['data','imagery']).max_time()


@process(description="Mask the image collection using a polygon. All pixels outside the polygon should be set to the nodata value. "
                     "All pixels inside, or intersecting the polygon should retain their original value.",
         args=[ProcessDetails.Arg('mask_shape', "The shape to use as a mask")])
def mask_polygon(args:Dict,viewingParameters)->ImageCollection:
    geometry = extract_arg(args, 'mask_shape')
    srs_code = geometry.get("crs",{}).get("name","EPSG:4326")
    return  extract_arg_list(args, ['data','imagery']).mask_polygon(shape(geometry),srs_code)

@process(description="Applies a mask to a raster data cube. Therefore, compares the parallel elements of the raster data cubes specified for data and mask and replaces all elements in data that are non-zero (for numbers) or true (for boolean values) in mask. These elements are replaced with the value specified for replacement, which defaults to null (no data). No data values will be left untouched.",
         args=[ProcessDetails.Arg('mask', "The image collection to use as a mask"),
               ProcessDetails.Arg('replacement', "The value used to replace non-zero and true values with.")])
def mask(args:Dict,viewingParameters)->ImageCollection:
    mask = extract_arg(args,'mask')
    replacement = float(extract_arg(args, 'replacement'))
    image_collection = extract_arg_list(args, ['data', 'imagery']).mask(mask,replacement)
    return image_collection


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

    left = viewingParameters["left"]
    right = viewingParameters["right"]
    top = viewingParameters["top"]
    bottom = viewingParameters["bottom"]
    srs = viewingParameters["srs"]
    image_collection = extract_arg_list(args, ['data','imagery']).bbox_filter(left,right,top,bottom,srs)
    return image_collection

@process(description="Filters the bands in the data cube so that bands that don't match any of the criteria are dropped from the data cube. The data cube is expected to have only one spectral dimension.\n\nThe following criteria can be used to select bands:\n\n* `bands`: band name (e.g. `B01` or `B8A`)\n* `common_names`: common band names (e.g. `red` or `nir`)\n* `wavelengths`: ranges of wavelengths in micrometres (μm) (e.g. 0.5 - 0.6)\n\nTo keep algorithms interoperable it is recommended to prefer the common bands names or the wavelengths over collection and/or back-end specific band names.\n\nIf multiple criteria are specified, any of them must match and not all of them, i.e. they are combined with an OR-operation. If no criteria is specified, the `BandFilterParameterMissing` exception must be thrown.\n\n**Important:** The order of the specified array defines the order of the bands in the data cube, which can be important for subsequent processes.",
         args=[
               ProcessDetails.Arg("data", "A data cube with bands."),
               ProcessDetails.Arg("bands", "A list of band names.\nThe order of the specified array defines the order of the bands in the data cube!"),
              ])
def filter_bands(args:Dict,viewingParameters)->ImageCollection:
    bands = extract_arg(args, 'bands')
    image_collection = extract_arg_list(args, ['data','imagery']).band_filter(bands)
    return image_collection

@process(description="Computes zonal statistics over a given polygon",
         args=[ProcessDetails.Arg('imagery', "The image collection to compute statistics on."),
               ProcessDetails.Arg('regions', "The GeoJson Polygon defining the zone.")
               ])
def zonal_statistics(args: Dict, viewingParameters) -> Dict:
    image_collection = extract_arg_list(args, ['data','imagery'])
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
    elif 'zonal_statistics' == process_id or 'aggregate_polygon' == process_id:
        geometry = extract_arg_list(args, ['regions','polygons'])
        bbox = shape(geometry).bounds
        if(viewingParameters.get("left") is None ):
            viewingParameters["left"] = bbox[0]
            viewingParameters["right"] = bbox[2]
            viewingParameters["bottom"] = bbox[1]
            viewingParameters["top"] = bbox[3]
            viewingParameters["srs"] = "EPSG:4326"

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
        if 'run_udf' == process_id and 'temporal' == dimension and not binary:
            udf = _get_udf(args)
            #EP-2760 a special case of reduce where only a single udf based callback is provided. The more generic case is not yet supported.
            return image_collection.apply_tiles_spatiotemporal(udf)
        return image_collection.reduce(process_id,dimension)
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
        process_function = globals()[process_id]
        if process_function is None:
            raise RuntimeError("No process found with name: "+process_id)
        return process_function(args, viewingParameters)


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
    #def filter_details(process_details):
    #    return {k: v for k, v in process_details.items() if k in ['process_id', 'description']}

    return [process_details.serialize() for process_id, process_details in process_registry.items()
            if not substring or substring.lower() in process_id.lower()]


def getProcess(process_id: str):
    process_details = process_registry.get(process_id)
    return process_details.serialize() if process_details else None
