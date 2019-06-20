import os
import logging
from distutils.version import LooseVersion
from urllib.parse import unquote

from flask import request, url_for, jsonify, send_from_directory, abort, make_response,Blueprint,g, current_app
from werkzeug.exceptions import HTTPException, BadRequest

from openeo_driver import app
from openeo_driver.save_result import SaveResult
from .ProcessGraphDeserializer import (evaluate, health_check, get_layers, getProcesses, getProcess, get_layer,
                                       create_batch_job, run_batch_job, get_batch_job_info, cancel_batch_job,
                                       get_batch_job_result_filenames, get_batch_job_result_output_dir,
                                       get_secondary_services_info, get_secondary_service_info,
                                       summarize_exception)
from openeo import ImageCollection
from openeo.error_summary import ErrorSummary

SUPPORTED_VERSIONS = ['0.3.0','0.3.1', '0.4.0', '0.4.1']
DEFAULT_VERSION = '0.3.1'

openeo_bp = Blueprint('openeo', __name__)


@openeo_bp.url_defaults
def _add_version(endpoint, values):
    """Callback to automatically add "version" argument in `url_for` calls."""
    if 'version' not in values and app.url_map.is_endpoint_expecting(endpoint, 'version'):
        values['version'] = g.get('version', DEFAULT_VERSION)


@openeo_bp.url_value_preprocessor
def _pull_version(endpoint, values):
    """Get API version from request and store in global context"""
    g.version = values.pop('version', DEFAULT_VERSION)
    if g.version not in SUPPORTED_VERSIONS:
        error = HTTPException(response={
            "id": "550e8400-e29b-11d4-a716-446655440000",
            "code": 400,
            "message": "Unsupported version: " + g.version + ".  Supported versions: " + str(SUPPORTED_VERSIONS)
        })
        error.code = 400
        raise error


@app.errorhandler(HTTPException)
def handle_http_exceptions(error: HTTPException):
    return _error_response(error, error.code)


@app.errorhandler(Exception)
def handle_error(error: Exception):
    error = summarize_exception(error)

    if isinstance(error, ErrorSummary):
        return _error_response(error, 400, error.summary) if error.is_client_error \
            else _error_response(error, 500, error.summary)

    return _error_response(error, 500)


def _error_response(error: Exception, status_code: int, summary: str = None):
    error_json = {
        "message": summary if summary else str(error)
    }
    if type(error) is HTTPException and type(error.response) is dict:
        error_json = error.response
    import traceback
    if type(error) is ErrorSummary:
        exception = error.exception
    else:
        exception = error

    current_app.logger.error(exception, exc_info=True)

    response = jsonify(error_json)
    response.status_code = status_code
    return response

@openeo_bp.route('/' )
def index():
    return jsonify({
      "version": g.version,  # Deprecated pre-0.4.0 API version field
      "api_version": g.version,  # API version field since 0.4.0
      "backend_version": "0.0.1",  # TODO specify actual backend version
      # TODO: finetune title and description
      "title": "VITO Remote Sensing OpenEO API",
      "description": "This is the OpenEO API to the VITO Remote Sensing product catalog and services.",
      "endpoints": [
        {
          "path": "/collections",
          "methods": [
            "GET"
          ]
        },
        {
          "path": '/collections/{collection_id}',
          "methods": [
            "GET"
          ]
        },
        {
          "path": "/preview",
          "methods": [
              "POST"
          ]
        },
          {
              "path": "/result",
              "methods": [
                  "POST"
              ]
          },
        {
          "path": "/jobs",#url_for('.create_job'),
          "methods": [
            "GET",
            "POST"
          ]
        },
          {
              "path": "/processes",#url_for('.processes'),
              "methods": [
                  "GET"
              ]
          },
          {
              "path": "/udf_runtimes",
              "methods": [
                  "GET"
              ]
          },
          {
              "path": "/output_formats",
              "methods": [
                  "GET"
              ]
          },
          {
              "path": "/service_types",
              "methods": [
                  "GET"
              ]
          },
          {
              "path": "/services",
              "methods": [
                  "GET",
                  "POST"
              ]
          },
        {
          "path": "/jobs/{job_id}",#unquote(url_for('.get_job_info', job_id = '{job_id}')),
          "methods": [
            "GET",
            "DELETE",
            "PATCH"
          ]
        },
        {
          "path": "/jobs/{job_id}/results",
          "methods": [
              "GET",
              "DELETE",
              "POST"
          ]
        }
      ],
      "billing": {
        "currency": "EUR",
        "plans": [
          {
            "name": "free",
            "description": "Free plan. No limits!",
            "url": "http://openeo.org/plans/free-plan",
            "paid": False
          }
        ]
      }
    })

@openeo_bp.route('/health' )
def health():
    return health_check()

@openeo_bp.route('/capabilities')
def capabilities():
    return jsonify([
      "/data",
      "/execute",
      "/processes"
    ])


#deprecated:
@openeo_bp.route('/capabilities/output_formats' )
#OpenEO 0.3.0:
@openeo_bp.route('/output_formats' )
def output_formats():
    if LooseVersion(g.version) >= LooseVersion('0.4.0'):
        return jsonify({
            "GTiff": {"gis_data_types": ["raster"]},
            "GeoTiff": {"gis_data_types": ["raster"]},
        })
    else:
        return jsonify({
            "default": "GTiff",
            "formats": {
                "GTiff": {"gis_data_types": ["raster"]},
                "GeoTiff": {"gis_data_types": ["raster"]},
            }
        })


@openeo_bp.route('/udf_runtimes' )
def udf_runtimes():
    return jsonify({
      "Python": {
        "description": "Predefined Python runtime environment.",
        "default": "latest",
        "versions": {
            "3.5.1":{
                "libraries":{
                    "numpy":{
                        "version":"1.14.3"
                    },
                    "pandas": {
                        "version": "0.22.0"
                    },
                    "tensorflow":{
                        "version":"1.11.0"
                    }
                }

            }
        }
      }

    })



@openeo_bp.route('/timeseries' )
def timeseries():
    return 'OpenEO GeoPyspark backend. ' + url_for('.point')


@openeo_bp.route('/timeseries/point' , methods=['GET', 'POST'])
def point():
    if request.method == 'POST':
        print("Handling request: "+str(request))
        print("Post data: "+str(request.data))
        x = float(request.args.get('x', ''))
        y = float(request.args.get('y', ''))
        srs = request.args.get('srs', '')
        startdate = request.args.get('startdate', '')
        enddate = request.args.get('enddate', '')

        process_graph = request.get_json()
        image_collection = evaluate(process_graph)
        return jsonify(image_collection.timeseries(x, y, srs))
    else:
        return 'Usage: Query point timeseries using POST.'


@openeo_bp.route('/download' , methods=['GET', 'POST'])
def download():
    if request.method == 'POST':
        print("Handling request: "+str(request))
        print("Post data: "+str(request.data))
        outputformat = request.args.get('outputformat', 'geotiff')

        process_graph = request.get_json()
        image_collection = evaluate(process_graph)
        filename = image_collection.download(None,outputformat=outputformat)

        return send_from_directory(os.path.dirname(filename),os.path.basename(filename))
    else:
        return 'Usage: Download image using POST.'

@openeo_bp.route('/result' , methods=['GET', 'POST'])
def result():
    return execute()

@openeo_bp.route('/preview' , methods=['GET', 'POST'])
def preview():
    return execute()

@openeo_bp.route('/execute' , methods=['GET', 'POST'])
def execute():
    if request.method == 'POST':
        print("Handling request: "+str(request))
        print("Post data: "+str(request.data))

        post_data = request.get_json()

        result = evaluate(post_data,viewingParameters={'version':g.version})

        if isinstance(result, ImageCollection):
            format_options = post_data.get('output',{})
            filename = result.download(None, bbox="", time="", **format_options)
            return send_from_directory(os.path.dirname(filename), os.path.basename(filename))
        elif result is None:
            abort(500,"No result")
        elif isinstance(result, SaveResult):
            return result.create_flask_response()
        else:
            return jsonify(result)
    else:
        return 'Usage: Directly evaluate process graph using POST.'


@openeo_bp.route('/jobs', methods=['GET', 'POST'])
def create_job():
    if request.method == 'POST':
        print("Handling request: "+str(request))
        print("Post data: "+str(request.data))

        job_specification = request.get_json()

        if 'process_graph' not in job_specification:
            return abort(400)

        job_id = create_batch_job(g.version, job_specification)

        response = make_response("", 201)
        response.headers['Location'] = request.base_url + '/' + job_id

        return response
    else:
        return 'Usage: Create a new batch processing job using POST'


@openeo_bp.route('/jobs/<job_id>' , methods=['GET'])
def get_job_info(job_id):
    job_info = get_batch_job_info(job_id)
    return jsonify(job_info) if job_info else abort(404)


@openeo_bp.route('/jobs/<job_id>/results' , methods=['POST'])
def queue_job(job_id):
    print("Handling request: " + str(request))

    job_info = get_batch_job_info(job_id)

    if job_info:
        run_batch_job(job_id)
        return make_response("", 202)
    else:
        abort(404)


@openeo_bp.route('/jobs/<job_id>/results' , methods=['GET'])
def list_job_results(job_id):
    print("Handling request: " + str(request))

    filenames = get_batch_job_result_filenames(job_id)

    if filenames is not None:
        job_results = {
            "links": [{"href": request.base_url + "/" + filename} for filename in filenames]
        }

        return jsonify(job_results)
    else:
        return abort(404)


@openeo_bp.route('/jobs/<job_id>/results/<filename>' , methods=['GET'])
def get_job_result(job_id, filename):
    print("Handling request: " + str(request))

    output_dir = get_batch_job_result_output_dir(job_id)
    return send_from_directory(output_dir, filename)


@openeo_bp.route('/jobs/<job_id>/results', methods=['DELETE'])
def cancel_job(job_id):
    print("Handling request: " + str(request))

    job_info = get_batch_job_info(job_id)

    if job_info:
        cancel_batch_job(job_id)
        return make_response("", 204)
    else:
        abort(404)

#SERVICES API https://open-eo.github.io/openeo-api/v/0.3.0/apireference/#tag/Web-Service-Management


@openeo_bp.route('/service_types' )
def service_types():
    return jsonify({
  "WMTS": {
    "parameters": {
      "version": {
        "type": "string",
        "description": "The WMTS version to use.",
        "default": "1.0.0",
        "enum": [
          "1.0.0"
        ]
      }
    },
    "attributes": {
      "layers": {
        "type": "array",
        "description": "Array of layer names.",
        "example": [
          "roads",
          "countries",
          "water_bodies"
        ]
      }
    }
  }
})

@openeo_bp.route('/tile_service' , methods=['GET', 'POST'])
def tile_service():
    """
    This is deprecated, pre-0.3.0 API
    :return:
    """
    if request.method == 'POST':
        print("Handling request: "+str(request))
        print("Post data: "+str(request.data))
        process_graph = request.get_json()
        image_collection = evaluate(process_graph)
        return jsonify(image_collection.tiled_viewing_service())
    else:
        return 'Usage: Retrieve tile service endpoint.'

@openeo_bp.route('/services' , methods=['GET', 'POST'])
def services():
    """
    GET: Requests to this endpoint will list all running secondary web services submitted by a user with given id.
    POST: Calling this endpoint will create a secondary web service such as WMTS, TMS or WCS. The underlying data is processes on-demand, but a process graph may simply access results from a batch job. Computations should be performed in the sense that it is only evaluated for the requested spatial / temporal extent and resolution.

    Note: Costs incurred by shared secondary web services are usually paid by the owner, but this depends on the service type and whether it supports charging fees or not.
    https://open-eo.github.io/openeo-api/v/0.3.0/apireference/#tag/Secondary-Services-Management/paths/~1services/post

    :return:
    """
    if request.method == 'POST':
        print("Handling request: "+str(request))
        print("Post data: "+str(request.data))
        json_request = request.get_json()
        process_graph = json_request['process_graph']
        type = json_request['type']

        if 'tms' == type.lower():
            image_collection = evaluate(process_graph)
            return jsonify(image_collection.tiled_viewing_service(**json_request)),201, {'ContentType':'application/json','Location':url_for('.services')}
        if 'wmts' == type.lower():
            image_collection = evaluate(process_graph,viewingParameters={
                'version': g.version,
                'service_type':type
            })
            service = image_collection.tiled_viewing_service(**json_request)
            url = service['url']
            return "", 201, {'Location': url}
        else:
            raise BadRequest("Requested unsupported service type: " + type)
    elif request.method == 'GET':
        #TODO implement retrieval of user specific web services
        return jsonify(get_secondary_services_info())

    raise AssertionError(request.method)


@openeo_bp.route('/services/<service_id>', methods=['GET'])
def get_service_info(service_id):
    service_info = get_secondary_service_info(service_id)
    return jsonify(service_info) if service_info else abort(404)


@openeo_bp.route('/data' , methods=['GET'])
def data():
    """
    deprecated, use /collections
    :return:
    """
    return collections()

@openeo_bp.route('/data/<collection_id>' , methods=['GET'])
def collection(collection_id):
    """
    deprecated, use /collections
    :return:
    """
    return collection_by_id(collection_id)


@openeo_bp.route('/collections' , methods=['GET'])
def collections():
        layers = get_layers()
        return jsonify({
            'collections':layers,
            'links':[]
        })

@openeo_bp.route('/collections/<collection_id>' , methods=['GET'])
def collection_by_id(collection_id):
    try:
        layer = get_layer(collection_id)
    except ValueError as e:
        abort(404,"The requested collection: %s was not found." % collection_id)
    return jsonify(layer)


@openeo_bp.route('/processes' , methods=['GET'])
def processes():
    print("Handling request: " + str(request))

    substring = request.args.get('qname')

    return jsonify({
        'processes':getProcesses(substring),
        'links':[]
    })


@openeo_bp.route('/processes/<process_id>' , methods=['GET'])
def process(process_id):
    print("Handling request: " + str(request))

    process_details = getProcess(process_id)
    print(process_details)

    return jsonify(process_details) if process_details else abort(404)

app.register_blueprint(openeo_bp, url_prefix='/openeo')
app.register_blueprint(openeo_bp, url_prefix='/openeo/<version>')