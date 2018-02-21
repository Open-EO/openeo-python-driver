import os

from flask import request, url_for, jsonify, send_from_directory

from openeo_driver import app
from .ProcessGraphDeserializer import graphToRdd, health_check,get_layers

ROOT = '/openeo'


@app.route('%s' % ROOT)
def index():
    return 'OpenEO GeoPyspark backend. ' + url_for('timeseries')

@app.route('%s/health' % ROOT)
def health():
    return health_check()

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
        image_collection = graphToRdd(process_graph, {})
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
        image_collection = graphToRdd(process_graph, None)
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
        image_collection = graphToRdd(post_data['process_graph'], None)
        filename = image_collection.download(None,bbox="",time="",**post_data['output'])

        return send_from_directory(os.path.dirname(filename),os.path.basename(filename))
    else:
        return 'Usage: Directly evaluate process graph using POST.'


@app.route('%s/tile_service' % ROOT, methods=['GET', 'POST'])
def tile_service():
    if request.method == 'POST':
        print("Handling request: "+str(request))
        print("Post data: "+str(request.data))
        process_graph = request.get_json()
        image_collection = graphToRdd(process_graph, None)
        return jsonify(image_collection.tiled_viewing_service())
    else:
        return 'Usage: Retrieve tile service endpoint.'

@app.route('%s/data' % ROOT, methods=['GET'])
def data():
        print("Handling request: "+str(request))
        layers = get_layers()
        return jsonify(layers)
