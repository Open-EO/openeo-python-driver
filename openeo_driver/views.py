import os
import logging

from flask import request, url_for, jsonify, send_from_directory, abort, make_response
from werkzeug.exceptions import HTTPException

from openeo_driver import app
from .ProcessGraphDeserializer import (evaluate, health_check, get_layers, getProcesses, getProcess, get_layer,
                                       create_batch_job, run_batch_job, get_batch_job_info,
                                       get_batch_job_result_filenames, get_batch_job_result_output_dir)
from openeo import ImageCollection

ROOT = '/openeo'


@app.errorhandler(HTTPException)
def handle_http_exceptions(error: HTTPException):
    return _error_response(error, error.code)


@app.errorhandler(Exception)
def handle_invalid_usage(error: Exception):
    return _error_response(error, 500)


def _error_response(error: Exception, status_code: int):
    error_json = {
        "message":str(error)
    }
    import traceback
    print(traceback.format_exception(None,error,error.__traceback__))
    response = jsonify(error_json)
    response.status_code = status_code
    return response


@app.route('%s' % ROOT)
def index():
    return jsonify({
      "version": "0.3.0",
      "endpoints": [
        {
          "path": "%s/data"%ROOT,
          "methods": [
            "GET"
          ]
        },
        {
          "path": "%s/data/{data_id}"%ROOT,
          "methods": [
            "GET"
          ]
        },
        {
          "path": "%s/jobs"%ROOT,
          "methods": [
            "GET",
            "POST"
          ]
        },
          {
              "path": "%s/processes"%ROOT,
              "methods": [
                  "GET"
              ]
          },
        {
          "path": "%s/jobs/{job_id}"%ROOT,
          "methods": [
            "GET",
            "DELETE",
            "PATCH"
          ]
        }
      ],
      "billing": {
        "currency": "EUR",
        "plans": [
          {
            "name": "free",
            "description": "Free plan. No limits!",
            "url": "http://openeo.org/plans/free-plan"
          }
        ]
      }
    })

@app.route('%s/health' % ROOT)
def health():
    return health_check()

@app.route('%s/capabilities' % ROOT)
def capabilities():
    return jsonify([
      "/data",
      "/execute",
      "/processes"
    ])


#deprecated:
@app.route('%s/capabilities/output_formats' % ROOT)
#OpenEO 0.3.0:
@app.route('%s/output_formats' % ROOT)
def output_formats():
    return jsonify({
      "default": "GTiff",
      "formats": {
        "GTiff": {},
        "GeoTiff": {}
      }
    })



@app.route('%s/timeseries' % ROOT)
def timeseries():
    return 'OpenEO GeoPyspark backend. ' + url_for('point')


@app.route('%s/timeseries/point' % ROOT, methods=['GET', 'POST'])
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


@app.route('%s/download' % ROOT, methods=['GET', 'POST'])
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


@app.route('%s/execute' % ROOT, methods=['GET', 'POST'])
def execute():
    if request.method == 'POST':
        print("Handling request: "+str(request))
        print("Post data: "+str(request.data))

        post_data = request.get_json()

        result = evaluate(post_data['process_graph'])

        if isinstance(result, ImageCollection):
            filename = result.download(None, bbox="", time="", **post_data['output'])
            return send_from_directory(os.path.dirname(filename), os.path.basename(filename))
        else:
            return jsonify(result)
    else:
        return 'Usage: Directly evaluate process graph using POST.'


@app.route('%s/jobs' % ROOT, methods=['GET', 'POST'])
def create_job():
    if request.method == 'POST':
        print("Handling request: "+str(request))
        print("Post data: "+str(request.data))

        post_data = request.get_json()

        job_id = create_batch_job(post_data['process_graph'], post_data['output'])

        response = make_response("", 201)
        response.headers['Location'] = request.base_url + '/' + job_id

        return response
    else:
        return 'Usage: Create a new batch processing job using POST'


@app.route('%s/jobs/<job_id>' % ROOT, methods=['GET'])
def get_job_info(job_id):
    job_info = get_batch_job_info(job_id)
    return jsonify(job_info) if job_info else abort(404)


@app.route('%s/jobs/<job_id>/results' % ROOT, methods=['POST'])
def queue_job(job_id):
    print("Handling request: " + str(request))

    job_info = get_batch_job_info(job_id)

    if job_info:
        run_batch_job(job_id)
        return make_response("", 202)
    else:
        abort(404)


@app.route('%s/jobs/<job_id>/results' % ROOT, methods=['GET'])
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


@app.route('%s/jobs/<job_id>/results/<filename>' % ROOT, methods=['GET'])
def get_job_result(job_id, filename):
    print("Handling request: " + str(request))

    output_dir = get_batch_job_result_output_dir(job_id)
    return send_from_directory(output_dir, filename)

#SERVICES API https://open-eo.github.io/openeo-api/v/0.3.0/apireference/#tag/Web-Service-Management


@app.route('%s/service_types' % ROOT)
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

@app.route('%s/tile_service' % ROOT, methods=['GET', 'POST'])
def tile_service():
    if request.method == 'POST':
        print("Handling request: "+str(request))
        print("Post data: "+str(request.data))
        process_graph = request.get_json()
        image_collection = evaluate(process_graph)
        return jsonify(image_collection.tiled_viewing_service())
    else:
        return 'Usage: Retrieve tile service endpoint.'

@app.route('%s/services' % ROOT, methods=['GET', 'POST'])
def services():
    if request.method == 'POST':
        print("Handling request: "+str(request))
        print("Post data: "+str(request.data))
        json_request = request.get_json()
        process_graph = json_request['process_graph']
        type = json_request['type']

        if 'tms' == type.lower():
            image_collection = evaluate(process_graph)
            return jsonify(image_collection.tiled_viewing_service())
        if 'wmts' == type.lower():
            image_collection = evaluate(process_graph,viewingParameters={
                'service_type':type
            })
            return jsonify(image_collection.tiled_viewing_service(type=type))
        else:
            raise NotImplementedError("Requested unsupported service type: " + type)
    elif request.method == 'GET':
        #TODO implement retrieval of user specific web services
        return []
    else:
        abort(405)

@app.route('%s/data' % ROOT, methods=['GET'])
def data():
        print("Handling request: "+str(request))
        layers = get_layers()
        return jsonify(layers)

@app.route('%s/data/<collection_id>' % ROOT, methods=['GET'])
def collection(collection_id):
    print("Handling request: "+str(request))
    return jsonify(get_layer(collection_id))


@app.route('%s/processes' % ROOT, methods=['GET'])
def processes():
    print("Handling request: " + str(request))

    substring = request.args.get('qname')

    return jsonify(getProcesses(substring))


@app.route('%s/processes/<process_id>' % ROOT, methods=['GET'])
def process(process_id):
    print("Handling request: " + str(request))

    process_details = getProcess(process_id)
    print(process_details)

    return jsonify(process_details) if process_details else abort(404)
