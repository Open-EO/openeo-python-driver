import os
import logging

from flask import request, url_for, jsonify, send_from_directory, abort

from openeo_driver import app
from .ProcessGraphDeserializer import evaluate, health_check, get_layers, getProcesses, getProcess
from openeo import ImageCollection

ROOT = '/openeo'

@app.errorhandler(Exception)
def handle_invalid_usage(error:Exception):
    error_json = {
        "message":str(error)
    }
    import traceback
    print(traceback.format_exception(None,error,error.__traceback__))
    response = jsonify(error_json)
    response.status_code = 500
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


@app.route('%s/output_formats' % ROOT)
def output_formats():
    return jsonify({
      "default": "GTiff",
      "formats": {
        "GTiff": {},
        "GeoTiff": {}
      }
    })

@app.route('%s/service_types' % ROOT)
def service_types():
    return jsonify([
      "tms"
    ])


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
        if 'tms' == type:
            image_collection = evaluate(process_graph)
            return jsonify(image_collection.tiled_viewing_service())
        else:
            raise NotImplementedError("Requested unsupported service type: " + type)
    else:
        return 'Usage: Retrieve tile service endpoint.'

@app.route('%s/data' % ROOT, methods=['GET'])
def data():
        print("Handling request: "+str(request))
        layers = get_layers()
        return jsonify(layers)


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
