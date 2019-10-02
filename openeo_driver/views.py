import base64
import hashlib
import logging
import os
from distutils.version import LooseVersion

import requests
from flask import Flask, request, url_for, jsonify, send_from_directory, abort, make_response, Blueprint, g, \
    current_app, redirect
from werkzeug.exceptions import HTTPException, NotFound

from openeo import ImageCollection
from openeo.error_summary import ErrorSummary
from openeo.rest.auth.auth import BearerAuth
from openeo_driver.ProcessGraphDeserializer import (
    evaluate, getProcesses, getProcess,
    create_batch_job, run_batch_job, get_batch_job_info, cancel_batch_job,
    get_batch_job_result_filenames, get_batch_job_result_output_dir,
    backend_implementation,
    summarize_exception)
from openeo_driver.errors import OpenEOApiException
from openeo_driver.save_result import SaveResult
from openeo_driver.utils import replace_nan_values

SUPPORTED_VERSIONS = [
    '0.3.0',
    '0.3.1',
    '0.4.0',
    '0.4.1',
    '0.4.2',
]
DEFAULT_VERSION = '0.3.1'


app = Flask(__name__)
app.config['APPLICATION_ROOT'] = '/openeo'
# TODO: get this OpenID config url from a real config
app.config['OPENID_CONNECT_CONFIG_URL'] = "https://sso-dev.vgt.vito.be/auth/realms/terrascope/.well-known/openid-configuration"

openeo_bp = Blueprint('openeo', __name__)

_log = logging.getLogger('openeo.driver')

@openeo_bp.url_defaults
def _add_version(endpoint, values):
    """Callback to automatically add "version" argument in `url_for` calls."""
    if 'version' not in values and current_app.url_map.is_endpoint_expecting(endpoint, 'version'):
        values['version'] = g.get('version', DEFAULT_VERSION)


@openeo_bp.url_value_preprocessor
def _pull_version(endpoint, values):
    """Get API version from request and store in global context"""
    g.version = values.pop('version', DEFAULT_VERSION)
    if g.version not in SUPPORTED_VERSIONS:
        # TODO replace with OpenEOApiException?
        error = HTTPException(response={
            "id": "550e8400-e29b-11d4-a716-446655440000",
            "code": 400,
            "message": "Unsupported version: " + g.version + ".  Supported versions: " + str(SUPPORTED_VERSIONS)
        })
        error.code = 400
        raise error


@app.errorhandler(HTTPException)
def handle_http_exceptions(error: HTTPException):
    # Convert to OpenEOApiException based handling
    return handle_openeoapi_exception(OpenEOApiException(
        message=str(error),
        code="NotFound" if isinstance(error, NotFound) else "Internal",
        status_code=error.code
    ))


@app.errorhandler(OpenEOApiException)
def handle_openeoapi_exception(error: OpenEOApiException):
    error_dict = error.to_dict()
    _log.error(str(error_dict), exc_info=True)
    return jsonify(error_dict), error.status_code


@app.errorhandler(Exception)
def handle_error(error: Exception):
    # TODO: convert to OpenEOApiException based handling
    error = summarize_exception(error)

    if isinstance(error, ErrorSummary):
        return _error_response(error, 400, error.summary) if error.is_client_error \
            else _error_response(error, 500, error.summary)

    return _error_response(error, 500)


def _error_response(error: Exception, status_code: int, summary: str = None):
    # TODO: convert to OpenEOApiException based handling
    error_json = {
        "message": summary if summary else str(error)
    }
    if type(error) is HTTPException and type(error.response) is dict:
        error_json = error.response
    if type(error) is ErrorSummary:
        exception = error.exception
    else:
        exception = error

    current_app.logger.error(exception, exc_info=True)

    response = jsonify(error_json)
    response.status_code = status_code
    return response


def response_204_no_content():
    return make_response('', 204, {"Content-Type": "application/json"})


@openeo_bp.route('/' )
def index():
    app_config = current_app.config
    return jsonify({
      "version": g.version,  # Deprecated pre-0.4.0 API version field
      "api_version": g.version,  # API version field since 0.4.0
      "backend_version": app_config.get('OPENEO_BACKEND_VERSION', '0.0.1'),
      "title": app_config.get('OPENEO_TITLE', 'OpenEO API'),
      "description": app_config.get('OPENEO_DESCRIPTION', 'OpenEO API'),
      # TODO only list endpoints that are actually supported by the backend.
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
        },
          {"path": "/credentials/basic", "methods": ["GET"]},
          {"path": "/credentials/oidc", "methods": ["GET"]},
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


@openeo_bp.route('/health')
def health():
    return jsonify({
        "health": backend_implementation.health_check()
    })


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


@openeo_bp.route("/credentials/basic", methods=["GET"])
def credentials_basic():
    auth_type, auth_code = request.headers["Authorization"].split(' ')
    if auth_type != 'Basic':
        raise OpenEOApiException(message="Authentication method not supported", code="AuthenticationSchemeInvalid",
                                 status_code=403)
    username, password = base64.b64decode(auth_code.encode('ascii')).decode('utf-8').split(':')
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    _log.info("Handling basic auth for user {u!r} with sha256(pwd)={p})".format(u=username, p=password_hash))
    # TODO: do real password check
    if password != username + '123':
        raise OpenEOApiException(message="Credentials are not correct", code="CredentialsInvalid", status_code=403)
    # TODO: generate real access token and link to user in some key value store
    access_token = "basic:" + base64.urlsafe_b64encode(username.encode('utf-8')).decode('ascii')
    return jsonify({"access_token": access_token, "user_id": username})


@openeo_bp.route("/credentials/oidc", methods=["GET"])
def credentials_oidc():
    return redirect(current_app.config["OPENID_CONNECT_CONFIG_URL"])


@openeo_bp.route("/me", methods=["GET"])
def me():
    # Get bearer token
    auth_type, auth_code = request.headers["Authorization"].split(' ')
    if auth_type != 'Bearer':
        raise OpenEOApiException(message="Unauthorized", code="AuthenticationRequired", status_code=401)
    if auth_code.startswith("basic:"):
        return jsonify({"user": "TODO"})
    else:
        # Assume token is OpenID Connect access token: get user info from `userinfo` endpoint
        userinfo_url = requests.get(current_app.config["OPENID_CONNECT_CONFIG_URL"]).json()["userinfo_endpoint"]
        r = requests.get(userinfo_url, auth=BearerAuth(bearer=auth_code))
        return jsonify(r.json())


@openeo_bp.route('/timeseries' )
def timeseries():
    return 'OpenEO GeoPyspark backend. ' + url_for('.point')


@openeo_bp.route('/timeseries/point', methods=['POST'])
def point():
    print("Handling request: " + str(request))
    print("Post data: " + str(request.data))
    x = float(request.args.get('x', ''))
    y = float(request.args.get('y', ''))
    srs = request.args.get('srs', None)
    process_graph = request.get_json()['process_graph']
    image_collection = evaluate(process_graph, viewingParameters={'version': g.version})
    return jsonify(image_collection.timeseries(x, y, srs))


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


@openeo_bp.route('/result' , methods=['POST'])
def result():
    return execute()


@openeo_bp.route('/preview' , methods=['GET', 'POST'])
def preview():
    # TODO: is this an old endpoint/shortcut or a custom extension of the API?
    return execute()


@openeo_bp.route('/execute', methods=['POST'])
def execute():
    print("Handling request: " + str(request))
    print("Post data: " + str(request.data))

    post_data = request.get_json()
    process_graph = post_data['process_graph']
    result = evaluate(process_graph, viewingParameters={'version': g.version})

    if isinstance(result, ImageCollection):
        format_options = post_data.get('output', {})
        filename = result.download(None, bbox="", time="", **format_options)
        return send_from_directory(os.path.dirname(filename), os.path.basename(filename))
    elif result is None:
        abort(500, "Process graph evaluation gave no result")
    elif isinstance(result, SaveResult):
        return result.create_flask_response()
    else:
        return jsonify(replace_nan_values(result))


@openeo_bp.route('/jobs', methods=['GET', 'POST'])
def create_job():
    # TODO required authenticated user
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
        # TODO "GET Requests to this endpoint will list all batch jobs submitted by a user"
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


@openeo_bp.route('/service_types', methods=['GET'])
def service_types():
    return jsonify(backend_implementation.secondary_services.service_types())


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


@openeo_bp.route('/services', methods=['POST'])
def services_post():
    """
    Create a secondary web service such as WMTS, TMS or WCS. The underlying data is processes on-demand, but a process graph may simply access results from a batch job. Computations should be performed in the sense that it is only evaluated for the requested spatial / temporal extent and resolution.

    Note: Costs incurred by shared secondary web services are usually paid by the owner, but this depends on the service type and whether it supports charging fees or not.
    https://open-eo.github.io/openeo-api/v/0.3.0/apireference/#tag/Secondary-Services-Management/paths/~1services/post

    :return:
    """
    # TODO require authenticated user
    print("Handling request: " + str(request))
    print("Post data: " + str(request.data))
    data = request.get_json()
    # TODO avoid passing api version this hackish way?
    data['api_version'] = g.version
    url, identifier = backend_implementation.secondary_services.create_service(data)

    return make_response('', 201, {
        'Content-Type': 'application/json',
        'Location': url,
        'OpenEO-Identifier': identifier,
    })


@openeo_bp.route('/services', methods=['GET'])
def services_get():
    """List all running secondary web services for authenticated user"""
    # TODO Require authentication
    return jsonify(backend_implementation.secondary_services.list_services())


@openeo_bp.route('/services/<service_id>', methods=['GET'])
def get_service_info(service_id):
    # TODO Require authentication
    return jsonify(backend_implementation.secondary_services.service_info(service_id))


@openeo_bp.route('/services/<service_id>', methods=['PATCH'])
def service_patch(service_id):
    # TODO Require authentication
    data = request.get_json()
    # TODO sanitize/check data?
    backend_implementation.secondary_services.update_service(service_id, data=data)
    return response_204_no_content()


@openeo_bp.route('/services/<service_id>', methods=['DELETE'])
def service_delete(service_id):
    # TODO Require authentication
    backend_implementation.secondary_services.remove_service(service_id)
    return response_204_no_content()


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


@openeo_bp.route('/collections', methods=['GET'])
def collections():
    return jsonify({
        'collections': backend_implementation.catalog.get_all_metadata(),
        'links': []
    })


@openeo_bp.route('/collections/<collection_id>', methods=['GET'])
def collection_by_id(collection_id):
    return jsonify(backend_implementation.catalog.get_collection_metadata(collection_id))


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


# Note: /.well-known/openeo should be available directly under domain, without version prefix.
@app.route('/.well-known/openeo', methods=['GET'])
def well_known_openeo():
    return jsonify({
        'versions': [
            {
                "url": url_for('openeo.index', version=version, _external=True),
                "api_version": version,
                # TODO: flag some versions as not available for production?
                "production": True,
            }
            for version in SUPPORTED_VERSIONS
        ]
    })
