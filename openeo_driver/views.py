import copy
import functools
import logging
import os
import re
import uuid
from collections import namedtuple, defaultdict
from typing import Callable, Tuple, List

import flask
import flask_cors
import numpy as np
from flask import Flask, request, url_for, jsonify, send_from_directory, abort, make_response, Blueprint, g, current_app
from werkzeug.exceptions import HTTPException, NotFound
from werkzeug.middleware.proxy_fix import ProxyFix

from openeo.capabilities import ComparableVersion
from openeo.util import dict_no_none, deep_get, Rfc3339
from openeo_driver import urlsigning
from openeo_driver.backend import ServiceMetadata, BatchJobMetadata, UserDefinedProcessMetadata, \
    ErrorSummary, OpenEoBackendImplementation, BatchJobs
from openeo_driver.datacube import DriverDataCube
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.errors import OpenEOApiException, ProcessGraphMissingException, ServiceNotFoundException, \
    FilePathInvalidException, ProcessGraphNotFoundException, FeatureUnsupportedException, ProcessUnsupportedException, \
    JobNotFinishedException
from openeo_driver.save_result import SaveResult, get_temp_file
from openeo_driver.users import HttpAuthHandler, User, user_id_b64_encode, user_id_b64_decode
from openeo_driver.utils import replace_nan_values, EvalEnv, smart_bool

_log = logging.getLogger(__name__)

ApiVersionInfo = namedtuple("ApiVersionInfo", ["version", "supported", "wellknown", "production"])

# TODO: move this version info listing and default version configurable too?
# Available OpenEO API versions: map of URL version component to API version info
API_VERSIONS = {
    "0.4.0": ApiVersionInfo(version="0.4.0", supported=True, wellknown=False, production=True),
    "0.4.1": ApiVersionInfo(version="0.4.1", supported=True, wellknown=False, production=True),
    "0.4.2": ApiVersionInfo(version="0.4.2", supported=True, wellknown=False, production=True),
    "0.4": ApiVersionInfo(version="0.4.2", supported=True, wellknown=True, production=True),
    "1.0.0": ApiVersionInfo(version="1.0.0", supported=True, wellknown=False, production=True),
    "1.0": ApiVersionInfo(version="1.0.0", supported=True, wellknown=True, production=True),
    "1.0.1": ApiVersionInfo(version="1.0.1", supported=True, wellknown=False, production=True),
    "1.1.0": ApiVersionInfo(version="1.1.0", supported=True, wellknown=False, production=False),
}
DEFAULT_VERSION = '1.0.0'

_log.info("API Versions: {v}".format(v=API_VERSIONS))
_log.info("Default API Version: {v}".format(v=DEFAULT_VERSION))


class OpenEoApiApp(Flask):

    def __init__(self, import_name):
        super().__init__(import_name=import_name)

        # Make sure app handles reverse proxy aspects (e.g. HTTPS) correctly.
        self.wsgi_app = ProxyFix(self.wsgi_app)

        # Setup up general CORS headers (for all HTTP methods)
        flask_cors.CORS(
            self,
            origins="*",
            send_wildcard=True,
            supports_credentials=False,
            allow_headers=["Content-Type", "Authorization"],
            expose_headers=["Location", "OpenEO-Identifier", "OpenEO-Costs", "Link"]
        )

    def make_default_options_response(self):
        # Customization of OPTIONS response
        rv = super().make_default_options_response()
        rv.status_code = 204
        rv.content_type = "application/json"
        return rv


def build_app(
        backend_implementation: OpenEoBackendImplementation,
        error_handling=True,
        import_name=__name__,
) -> OpenEoApiApp:
    """
    Build Flask app serving the endpoints that are implemented in given backend implementation

    After building the flask app you can configure it with standard flask configuration tools
    (https://flask.palletsprojects.com/en/2.0.x/config/).

    Some example patterns used in various places:

        # Build app
        app = build_app(backend_implementation=backend_implementation)

        # Directly set config values
        app.config['TESTING'] = True
        app.config['SERVER_NAME'] = 'oeo.net'

        # Load config values from a module (upper case variables)
        from openeogeotrellis.deploy import flask_config
        app.config.from_object(flask_config)

        # Load from a dictionary/mapping
        app.config.from_mapping(
            OPENEO_TITLE="Local GeoPySpark",
            OPENEO_DESCRIPTION="Local openEO API using GeoPySpark driver",
        )

    :param backend_implementation:
    :param import_name:
    :return:
    """
    app = OpenEoApiApp(import_name=import_name)

    @app.url_defaults
    def _add_version(endpoint, values):
        """Blueprint.url_defaults handler to automatically add "version" argument in `url_for` calls."""
        if 'version' not in values and current_app.url_map.is_endpoint_expecting(endpoint, 'version'):
            values['version'] = g.get('request_version', DEFAULT_VERSION)

    @app.url_value_preprocessor
    def _pull_version(endpoint, values):
        """Get API version from request and store in global context"""
        version = (values or {}).pop('version', DEFAULT_VERSION)
        if not (version in API_VERSIONS and API_VERSIONS[version].supported):
            raise OpenEOApiException(
                status_code=501,
                code="UnsupportedApiVersion",
                message="Unsupported version component in URL: {v!r}.  Available versions: {s!r}".format(
                    v=version, s=[k for k, v in API_VERSIONS.items() if v.supported]
                )
            )
        g.request_version = version
        g.api_version = API_VERSIONS[version].version

    @app.before_request
    def _before_request():
        # Log some info about request
        data = request.data
        if len(data) > 1000:
            data = repr(data[:1000] + b'...') + ' ({b} bytes)'.format(b=len(data))
        else:
            data = repr(data)
        _log.info("Handling {method} {url} with data {data}".format(
            method=request.method, url=request.url, data=data
        ))

    if error_handling:
        register_error_handlers(app=app, backend_implementation=backend_implementation)

    # Note: /.well-known/openeo should be available directly under domain, without version prefix.
    @app.route('/.well-known/openeo', methods=['GET'])
    def well_known_openeo():
        return jsonify({
            'versions': [
                {
                    "url": url_for('openeo.index', version=k, _external=True),
                    "api_version": v.version,
                    "production": v.production,
                }
                for k, v in API_VERSIONS.items()
                if v.wellknown
            ]
        })

    auth = HttpAuthHandler(oidc_providers=backend_implementation.oidc_providers())
    api_reg = EndpointRegistry()
    bp = Blueprint("openeo", import_name=__name__)

    register_views_general(
        blueprint=bp, backend_implementation=backend_implementation, api_endpoint=api_reg, auth_handler=auth
    )
    register_views_auth(
        blueprint=bp, backend_implementation=backend_implementation, api_endpoint=api_reg, auth_handler=auth
    )

    if backend_implementation.catalog:
        register_views_catalog(
            blueprint=bp, backend_implementation=backend_implementation, api_endpoint=api_reg, auth_handler=auth
        )

    if backend_implementation.processing:
        register_views_processing(
            blueprint=bp, backend_implementation=backend_implementation, api_endpoint=api_reg, auth_handler=auth
        )

    if backend_implementation.batch_jobs:
        register_views_batch_jobs(
            blueprint=bp, backend_implementation=backend_implementation, api_endpoint=api_reg, auth_handler=auth
        )

    if backend_implementation.secondary_services:
        register_views_secondary_services(
            blueprint=bp, backend_implementation=backend_implementation, api_endpoint=api_reg, auth_handler=auth
        )

    if backend_implementation.user_defined_processes:
        register_views_udp(
            blueprint=bp, backend_implementation=backend_implementation, api_endpoint=api_reg, auth_handler=auth
        )

    if backend_implementation.user_files:
        register_views_user_files(
            blueprint=bp, backend_implementation=backend_implementation, api_endpoint=api_reg, auth_handler=auth
        )

    app.register_blueprint(bp, url_prefix='/openeo')  # TODO: do we still need this?
    app.register_blueprint(bp, url_prefix='/openeo/<version>')

    # Build endpoint metadata dictionary
    # TODO: find another way to pass this to `index()`
    global _openeo_endpoint_metadata
    _openeo_endpoint_metadata = api_reg.get_path_metadata(bp)

    # Load default config.
    app.config.from_object("openeo_driver.config.flask_defaults")

    return app


def requested_api_version() -> ComparableVersion:
    """Get the currently requested API version as a ComparableVersion object"""
    return ComparableVersion(g.api_version)


def register_error_handlers(app: flask.Flask, backend_implementation: OpenEoBackendImplementation):
    """Register error handlers to the app"""
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
        error = backend_implementation.summarize_exception(error)

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


EndpointMetadata = namedtuple("EndpointMetadata", ["hidden", "for_version"])


class EndpointRegistry:
    """
    Registry of OpenEO API endpoints, to be used as decorator with flask view functions.

    Allows setting some additional metadata and automatic generation of
    the OpenEO API endpoints listing in the "capabilities" endpoint.
    """

    def __init__(self):
        self._endpoints = {}

    def add_endpoint(self, view_func: Callable, hidden=False, version: Callable = None):
        """Register endpoint metadata"""
        self._endpoints[view_func.__name__] = EndpointMetadata(hidden=hidden, for_version=version)
        return view_func

    def __call__(self, view_func: Callable = None, *, hidden=False, version: Callable = None):
        if view_func is None:
            # Decorator with arguments: return wrapper to call with decorated function.
            return functools.partial(self.add_endpoint, hidden=hidden, version=version)
        else:
            # Argument-less decorator call: we already have the function to wrap, use default options.
            return self.add_endpoint(view_func)

    def get_path_metadata(self, blueprint: Blueprint) -> List[Tuple[str, set, EndpointMetadata]]:
        """
        Join registered blueprint routes with endpoint metadata
        and get a listing of (path, methods, metadata) tuples
        :return:
        """
        app = Flask("dummy")
        app.register_blueprint(blueprint)
        metadata = []
        for rule in app.url_map.iter_rules():
            if rule.endpoint.startswith(blueprint.name + '.'):
                name = rule.endpoint.split('.', 1)[1]
                if name in self._endpoints:
                    metadata.append((rule.rule, rule.methods.difference({'HEAD', 'OPTIONS'}), self._endpoints[name]))
        return metadata

    @staticmethod
    def get_capabilities_endpoints(metadata: List[Tuple[str, set, EndpointMetadata]], api_version) -> List[dict]:
        """
        Extract "capabilities" endpoint listing from metadata list
        """
        endpoint_methods = defaultdict(set)
        for path, methods, info in metadata:
            if not info.hidden and (info.for_version is None or info.for_version(api_version)):
                endpoint_methods[path].update(methods)
        endpoints = [
            {"path": p.replace('<', '{').replace('>', '}'), "methods": list(ms)}
            for p, ms in endpoint_methods.items()
        ]
        return endpoints


def register_views_general(
        blueprint: Blueprint, backend_implementation: OpenEoBackendImplementation, api_endpoint: EndpointRegistry,
        auth_handler: HttpAuthHandler
):
    @blueprint.route('/')
    def index():
        app_config = current_app.config

        api_version = requested_api_version().to_string()
        title = app_config.get('OPENEO_TITLE', 'OpenEO API')
        service_id = app_config.get('OPENEO_SERVICE_ID', re.sub(r"\s+", "", title.lower() + '-' + api_version))
        # TODO only list endpoints that are actually supported by the backend.
        endpoints = EndpointRegistry.get_capabilities_endpoints(_openeo_endpoint_metadata, api_version=api_version)
        deploy_metadata = app_config.get('OPENEO_BACKEND_DEPLOY_METADATA') or {}

        capabilities = {
            "version": api_version,  # Deprecated pre-0.4.0 API version field
            "api_version": api_version,  # API version field since 0.4.0
            "backend_version": app_config.get('OPENEO_BACKEND_VERSION', '0.0.1'),
            "stac_version": "0.9.0",
            "id": service_id,
            "title": title,
            "description": app_config.get('OPENEO_DESCRIPTION', 'OpenEO API'),
            "production": API_VERSIONS[g.request_version].production,
            "endpoints": endpoints,
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
            },
            "_backend_deploy_metadata": deploy_metadata,
            "links": [
                {
                    "rel": "version-history",
                    "href": url_for("well_known_openeo", _external=True),
                    "type": "application/json",
                },
                {
                    "rel": "data",
                    "href": url_for("openeo.collections", _external=True),
                    "type": "application/json",
                },
                {
                    "rel": "conformance",
                    "href": url_for("openeo.conformance", _external=True),
                    "type": "application/json",
                },
                # TODO #60: also add terms-of-service, privacy policy
            ]
        }

        return jsonify(capabilities)

    @blueprint.route('/conformance')
    def conformance():
        return jsonify({"conformsTo": [
            # TODO: expand/manage conformance classes?
            "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core"
        ]})

    @blueprint.route('/health')
    def health():
        return jsonify({
            "health": backend_implementation.health_check()
        })

    @api_endpoint(version=ComparableVersion("1.0.0").accept_lower)
    @blueprint.route('/output_formats')
    def output_formats():
        # TODO deprecated endpoint, remove it when v0.4 API support is not necessary anymore
        return jsonify(backend_implementation.file_formats()["output"])

    @api_endpoint(version=ComparableVersion("1.0.0").or_higher)
    @blueprint.route('/file_formats')
    def file_formats():
        return jsonify(backend_implementation.file_formats())

    @api_endpoint
    @blueprint.route('/udf_runtimes')
    def udf_runtimes():
        # TODO: move this to OpenEoBackendImplementation?
        runtimes = {
            "Python": {
                "title": "Python 3.5",
                "description": "Predefined Python runtime environment.",
                "type": "language",
                "default": "latest",
                "versions": {
                    # TODO: get these versions from somewhere instead of hardcoding them?
                    "3.5.1": {
                        "libraries": {
                            "numpy": {"version": "1.14.3"},
                            "pandas": {"version": "0.22.0"},
                            "tensorflow": {"version": "1.11.0"},
                        }
                    }
                }
            }
        }
        return jsonify(runtimes)

    @blueprint.route('/.well-known/openeo')
    def versioned_well_known_openeo():
        # Clients might request this for version discovery. Avoid polluting (error) logs by explicitly handling this.
        error = OpenEOApiException(status_code=404, code="NotFound", message="Not a well-known openEO URI")
        return make_response(jsonify(error.to_dict()), error.status_code)

    @blueprint.route('/_debug/echo', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'])
    def debug_echo():
        return jsonify({
            "url": request.url,
            "path": request.path,
            "method": request.method,
            "headers": dict(request.headers.items()),
            "args": request.args,
            "data": repr(request.get_data()),
            "remote_addr": request.remote_addr,
            "environ": {k: request.environ.get(k) for k in ["HTTP_USER_AGENT", "SERVER_PROTOCOL", "wsgi.url_scheme"]}
        })

    @blueprint.route('/_debug/error', methods=["GET", "POST"])
    def debug_error():
        raise Exception("Computer says no.")


def register_views_auth(
        blueprint: Blueprint, backend_implementation: OpenEoBackendImplementation, api_endpoint: EndpointRegistry,
        auth_handler: HttpAuthHandler
):
    @api_endpoint
    @blueprint.route("/credentials/basic", methods=["GET"])
    @auth_handler.requires_http_basic_auth
    def credentials_basic():
        access_token, user_id = auth_handler.authenticate_basic(request)
        resp = {"access_token": access_token}
        if requested_api_version().below("1.0.0"):
            resp["user_id"] = user_id
        return jsonify(resp)

    @api_endpoint
    @blueprint.route("/credentials/oidc", methods=["GET"])
    @auth_handler.public
    def credentials_oidc():
        providers = backend_implementation.oidc_providers()
        if requested_api_version().at_least("1.0.0"):
            return jsonify({
                "providers": [p.prepare_for_json() for p in providers]
            })
        else:
            return flask.redirect(providers[0].issuer + '/.well-known/openid-configuration', code=303)

    @api_endpoint
    @blueprint.route("/me", methods=["GET"])
    @auth_handler.requires_bearer_auth
    def me(user: User):
        return jsonify({
            "user_id": user.user_id,
            "info": user.info
            # TODO more fields
        })


def _extract_process_graph(post_data: dict) -> dict:
    """
    Extract process graph dictionary from POST data

    see https://github.com/Open-EO/openeo-api/pull/262
    """
    try:
        if requested_api_version().at_least("1.0.0"):
            pg = post_data["process"]["process_graph"]
        else:
            # API v0.4 style
            pg = post_data['process_graph']
        assert isinstance(pg, dict)
    except (KeyError, TypeError, AssertionError) as e:
        raise ProcessGraphMissingException
    return pg


def register_views_processing(
        blueprint: Blueprint, backend_implementation: OpenEoBackendImplementation, api_endpoint: EndpointRegistry,
        auth_handler: HttpAuthHandler
):
    @blueprint.route('/timeseries')
    def timeseries():
        # TODO: deprecated? do we still need this endpoint? #35
        return 'OpenEO GeoPyspark backend. ' + url_for('.point')

    @blueprint.route('/timeseries/point', methods=['POST'])
    def point():
        # TODO: deprecated? do we still need this endpoint? #35
        x = float(request.args.get('x', ''))
        y = float(request.args.get('y', ''))
        srs = request.args.get('srs', None)
        process_graph = _extract_process_graph(request.json)
        image_collection = backend_implementation.processing.evaluate(
            process_graph=process_graph,
            env=EvalEnv({"backend_implementation": backend_implementation, 'version': g.api_version})
        )
        return jsonify(image_collection.timeseries(x, y, srs))

    @blueprint.route('/download', methods=['GET', 'POST'])
    def download():
        # TODO: deprecated?
        if request.method == 'POST':
            outputformat = request.args.get('outputformat', 'geotiff')

            process_graph = request.get_json()
            image_collection = backend_implementation.processing.evaluate(
                process_graph=process_graph,
                env=EvalEnv({"backend_implementation": backend_implementation, 'version': g.api_version})
            )
            # TODO Unify with execute?

            filename = image_collection.save_result(filename=get_temp_file(), format=outputformat, format_options={})
            return send_from_directory(os.path.dirname(filename), os.path.basename(filename))
        else:
            return 'Usage: Download image using POST.'

    @api_endpoint
    @blueprint.route('/result', methods=['POST'])
    @auth_handler.requires_bearer_auth
    def result(user: User):
        post_data = request.get_json()
        process_graph = _extract_process_graph(post_data)

        result = backend_implementation.processing.evaluate(process_graph=process_graph, env=EvalEnv({
            "backend_implementation": backend_implementation,
            'version': g.api_version,
            'pyramid_levels': 'highest',
            'user': user,
            'require_bounds': True,
            'correlation_id': str(uuid.uuid4())
        }))

        # TODO unify all this output handling within SaveResult logic?
        if isinstance(result, DriverDataCube):
            format_options = post_data.get('output', {})
            filename = result.save_result(filename=get_temp_file(), format="GTiff", format_options=format_options)
            return send_from_directory(os.path.dirname(filename), os.path.basename(filename))
        elif result is None:
            abort(500, "Process graph evaluation gave no result")
        elif isinstance(result, SaveResult):
            return result.create_flask_response()
        elif isinstance(result, DelayedVector):
            from shapely.geometry import mapping
            geojsons = (mapping(geometry) for geometry in result.geometries)
            return jsonify(list(geojsons))
        elif isinstance(result, flask.Response):
            return result
        elif isinstance(result, np.ndarray):
            return jsonify(result.tolist())
        elif isinstance(result, np.generic):
            # Convert numpy datatype to native Python datatype first
            return jsonify(result.item())
        else:
            return jsonify(replace_nan_values(result))

    @blueprint.route('/execute', methods=['POST'])
    @auth_handler.requires_bearer_auth
    def execute(user: User):
        # TODO:  This is not an official endpoint, does this "/execute" still have to be exposed as route?
        _log.warning(f"Request to non-standard `/execute` endpoint by {user.user_id}")
        return result(user=user)

    @api_endpoint
    @blueprint.route('/processes', methods=['GET'])
    def processes():
        # TODO: this `qname` feature is non-standard. Is this necessary for some reason?
        substring = request.args.get('qname')
        process_registry = backend_implementation.processing.get_process_registry(api_version=requested_api_version())
        processes = process_registry.get_specs(substring)
        return jsonify({'processes': processes, 'links': []})

    @api_endpoint
    @blueprint.route('/processes/<namespace>', methods=['GET'])
    def processes_from_namespace(namespace):
        # TODO: this endpoint is in draft at the moment
        #       see https://github.com/Open-EO/openeo-api/issues/310, https://github.com/Open-EO/openeo-api/pull/348
        # TODO: convention for user namespace? use '@' instead of "u:"
        # TODO: unify with `/processes` endpoint?
        full = smart_bool(request.args.get("full", False))
        if namespace.startswith("u:"):
            user_id = namespace.partition("u:")[-1]
            user_udps = [p for p in backend_implementation.user_defined_processes.get_for_user(user_id) if p.public]
            processes = [_jsonable_udp_metadata(udp, full=full, user=User(user_id=user_id)) for udp in user_udps]
        elif ":" not in namespace:
            process_registry = backend_implementation.processing.get_process_registry(
                api_version=requested_api_version()
            )
            processes = process_registry.get_specs(namespace=namespace)
            if not full:
                # Strip some fields
                processes = [
                    {k: v for k, v in p.items() if k not in ["process_graph"]}
                    for p in processes
                ]
        else:
            raise OpenEOApiException("Could not handle namespace {n!r}".format(n=namespace))
        # TODO: pagination links?
        return jsonify({'processes': processes, 'links': []})

    @api_endpoint
    @blueprint.route('/processes/<namespace>/<process_id>', methods=['GET'])
    def processes_details(namespace, process_id):
        # TODO: this endpoint is in draft at the moment
        #       see https://github.com/Open-EO/openeo-api/issues/310, https://github.com/Open-EO/openeo-api/pull/348
        if namespace.startswith("u:"):
            user_id = namespace.partition("u:")[-1]
            udp = backend_implementation.user_defined_processes.get(user_id=user_id, process_id=process_id)
            if not udp:
                raise ProcessUnsupportedException(process=process_id, namespace=namespace)
            process = _jsonable_udp_metadata(udp, full=True, user=User(user_id=user_id))
        elif ":" not in namespace:
            process_registry = backend_implementation.processing.get_process_registry(
                api_version=requested_api_version()
            )
            process = process_registry.get_spec(name=process_id, namespace=namespace)
        else:
            raise OpenEOApiException("Could not handle namespace {n!r}".format(n=namespace))
        return jsonify(process)


def _jsonable_batch_job_metadata(metadata: BatchJobMetadata, full=True) -> dict:
    """API-version-aware conversion of batch job metadata to jsonable dict"""
    d = metadata.prepare_for_json()
    # Fields to export
    fields = ['id', 'title', 'description', 'status', 'created', 'updated', 'plan', 'costs', 'budget']
    if full:
        fields.extend([
            'process', 'progress', 'duration_seconds', 'duration_human_readable',
            'memory_time_megabyte_seconds', 'memory_time_human_readable',
            'cpu_time_seconds', 'cpu_time_human_readable',
        ])
    d = {k: v for (k, v) in d.items() if k in fields}

    if requested_api_version().below("1.0.0"):
        d["process_graph"] = d.pop("process", {}).get("process_graph")
        d["submitted"] = d.pop("created", None)
        # TODO wider status checking coverage?
        if d["status"] == "created":
            d["status"] = "submitted"

    return dict_no_none(**d)


def _properties_from_job_info(job_info: BatchJobMetadata) -> dict:
    to_datetime = Rfc3339(propagate_none=True).datetime

    properties = dict_no_none(**{
        "title": job_info.title,
        "description": job_info.description,
        "created": to_datetime(job_info.created),
        "updated": to_datetime(job_info.updated),
        "card4l:specification": "SR",
        "card4l:specification_version": "5.0",
        # TODO: eliminate hard coded VITO/Spark/Geotrellis references. See https://github.com/Open-EO/openeo-python-driver/issues/74
        "processing:facility": 'VITO - SPARK',
        "processing:software": 'openeo-geotrellis-' + current_app.config.get('OPENEO_BACKEND_VERSION', '0.0.1')
    })
    properties["datetime"] = None

    start_datetime = to_datetime(job_info.start_datetime)
    end_datetime = to_datetime(job_info.end_datetime)

    if start_datetime == end_datetime:
        properties["datetime"] = start_datetime
    else:
        if start_datetime:
            properties["start_datetime"] = start_datetime
        if end_datetime:
            properties["end_datetime"] = end_datetime

    if job_info.instruments:
        properties['instruments'] = job_info.instruments

    if job_info.epsg:
        properties['proj:epsg'] = job_info.epsg

    properties['card4l:processing_chain'] = job_info.process

    return properties


def register_views_batch_jobs(
        blueprint: Blueprint, backend_implementation: OpenEoBackendImplementation, api_endpoint: EndpointRegistry,
        auth_handler: HttpAuthHandler
):
    @api_endpoint
    @blueprint.route('/jobs', methods=['POST'])
    @auth_handler.requires_bearer_auth
    def create_job(user: User):
        # TODO: wrap this job specification in a 1.0-style ProcessGrahpWithMetadata?
        post_data = request.get_json()
        process = {"process_graph": _extract_process_graph(post_data)}
        job_options = post_data.get("job_options")
        job_info = backend_implementation.batch_jobs.create_job(
            user_id=user.user_id,
            process=process,
            api_version=g.api_version,
            metadata=dict_no_none(title=post_data.get("title"), description=post_data.get("description")),
            job_options=job_options,
        )
        job_id = job_info.id
        response = make_response("", 201)
        response.headers['Location'] = url_for('.get_job_info', job_id=job_id)
        response.headers['OpenEO-Identifier'] = str(job_id)
        return response

    @api_endpoint
    @blueprint.route('/jobs', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def list_jobs(user: User):
        return jsonify({
            "jobs": [
                _jsonable_batch_job_metadata(m, full=False)
                for m in backend_implementation.batch_jobs.get_user_jobs(user.user_id)
            ],
            "links": [],
        })

    @api_endpoint
    @blueprint.route('/jobs/<job_id>', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def get_job_info(job_id, user: User):
        job_info = backend_implementation.batch_jobs.get_job_info(job_id, user)
        return jsonify(_jsonable_batch_job_metadata(job_info))

    @api_endpoint(hidden=True)
    @blueprint.route('/jobs/<job_id>', methods=['DELETE'])
    @auth_handler.requires_bearer_auth
    def delete_job(job_id, user: User):
        backend_implementation.batch_jobs.delete_job(job_id=job_id, user_id=user.user_id)
        return response_204_no_content()

    @api_endpoint(hidden=True)
    @blueprint.route('/jobs/<job_id>', methods=['PATCH'])
    @auth_handler.requires_bearer_auth
    def modify_job(job_id, user: User):
        # TODO
        raise FeatureUnsupportedException()

    @api_endpoint
    @blueprint.route('/jobs/<job_id>/results', methods=['POST'])
    @auth_handler.requires_bearer_auth
    def queue_job(job_id, user: User):
        backend_implementation.batch_jobs.start_job(job_id=job_id, user=user)
        return make_response("", 202)

    @api_endpoint
    @blueprint.route('/jobs/<job_id>/results', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def list_job_results(job_id, user: User):
        job_info = backend_implementation.batch_jobs.get_job_info(job_id, user)
        if job_info.status != "finished":
            raise JobNotFinishedException()

        results = backend_implementation.batch_jobs.get_results(job_id=job_id, user_id=user.user_id)

        if smart_bool(current_app.config.get('SIGNED_URL')):
            signer = urlsigning.Signer.from_config(current_app.config)
        else:
            signer = None

        def download_url(filename) -> str:
            if signer:
                expires = signer.get_expires()
                secure_key = signer.sign_job_asset(
                    job_id=job_id, user_id=user.user_id, filename=filename, expires=expires
                )
                user_base64 = user_id_b64_encode(user.user_id)
                return url_for(
                    '.download_job_result_signed',
                    job_id=job_id, user_base64=user_base64, filename=filename, expires=expires, secure_key=secure_key,
                    _external=True
                )
            else:
                return url_for('.download_job_result', job_id=job_id, filename=filename, _external=True)

        def asset_object(filename: str, asset_metadata: dict) -> dict:
            bands = asset_metadata.get("bands")
            nodata = asset_metadata.get("nodata")

            return dict_no_none(**{
                "title": asset_metadata.get("title", filename),  # there has to be title
                "href": asset_metadata.get(BatchJobs.ASSET_PUBLIC_HREF) or download_url(filename),
                "type": asset_metadata.get("media_type"),
                "eo:bands": [dict_no_none(**{"name": band.name, "center_wavelength": band.wavelength_um})
                             for band in bands] if bands else None,
                "file:nodata": [nodata],
                "roles": asset_metadata.get("roles", ["data"])
            })

        if requested_api_version().at_least("1.0.0"):
            links = job_info.links
            if links == None:
                links = []
            links.append({
                "rel": "self",
                "href": url_for('.list_job_results', job_id=job_id, _external=True),
                "type": "application/json"
            })
            links.append({
                "rel": "card4l-document",
                "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                "type": "application/pdf"
            })
            result = {
                "stac_version": "0.9.0",
                "id": job_info.id,
                "type": "Feature",
                "properties": _properties_from_job_info(job_info),
                "assets": {
                    filename: asset_object(filename, asset_metadata) for filename, asset_metadata in results.items()
                },
                "links": links
            }

            geometry = job_info.geometry
            result["geometry"] = geometry
            if geometry:
                result["bbox"] = job_info.bbox

            result["stac_extensions"] = [
                "processing", "card4l-eo", "https://stac-extensions.github.io/file/v1.0.0/schema.json"
            ]

            if any("eo:bands" in asset_object for asset_object in result["assets"].values()):
                result["stac_extensions"].append("eo")

            if "proj:epsg" in result["properties"]:
                result["stac_extensions"].append("projection")
        else:
            result = {
                "links": [
                    {"href": download_url(filename)} for filename in results.keys()
                ]
            }

        # TODO "OpenEO-Costs" header?
        return jsonify(result)

    @api_endpoint
    @blueprint.route('/jobs/<job_id>/results/<filename>', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def download_job_result(job_id, filename, user: User):
        results = backend_implementation.batch_jobs.get_results(job_id=job_id, user_id=user.user_id)
        if filename not in results.keys():
            raise FilePathInvalidException(str(filename) + ' not in ' + str(list(results.keys())))
        output_dir = results[filename]["output_dir"]
        return send_from_directory(output_dir, filename, mimetype=results[filename].get("media_type"))

    @api_endpoint
    @blueprint.route('/jobs/<job_id>/results/<user_base64>/<secure_key>/<filename>', methods=['GET'])
    def download_job_result_signed(job_id, user_base64, secure_key, filename):
        expires = request.args.get('expires')
        signer = urlsigning.Signer.from_config(current_app.config)
        user_id = user_id_b64_decode(user_base64)
        signer.verify_job_asset(
            signature=secure_key,
            job_id=job_id, user_id=user_id, filename=filename, expires=expires
        )

        results = backend_implementation.batch_jobs.get_results(job_id=job_id, user_id=user_id)
        if filename not in results.keys():
            raise FilePathInvalidException(str(filename) + ' not in ' + str(list(results.keys())))
        output_dir = results[filename]["output_dir"]
        return send_from_directory(output_dir, filename, mimetype=results[filename].get("media_type"))


    @api_endpoint
    @blueprint.route('/jobs/<job_id>/logs', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def get_job_logs(job_id, user: User):
        offset = request.args.get('offset')
        # TODO: implement paging support: `limit`, next/prev/first/last `links`, ...
        return jsonify({
            "logs": backend_implementation.batch_jobs.get_log_entries(
                job_id=job_id, user_id=user.user_id, offset=offset
            ),
            "links": [],
        })

    @api_endpoint
    @blueprint.route('/jobs/<job_id>/results', methods=['DELETE'])
    @auth_handler.requires_bearer_auth
    def cancel_job(job_id, user: User):
        backend_implementation.batch_jobs.cancel_job(job_id=job_id, user_id=user.user_id)
        return make_response("", 204)

    @api_endpoint(hidden=True)
    @blueprint.route('/jobs/<job_id>/estimate', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def job_estimate(job_id, user: User):
        # TODO: implement cost estimation?
        raise FeatureUnsupportedException()


def _jsonable_service_metadata(metadata: ServiceMetadata, full=True) -> dict:
    """API-version-aware conversion of service metadata to jsonable dict"""
    d = metadata.prepare_for_json()
    if not full:
        d.pop("process")
        d.pop("attributes")
    if requested_api_version().below("1.0.0"):
        d["process_graph"] = d.pop("process", {}).get("process_graph")
        d["parameters"] = d.pop("configuration", None) or ({} if full else None)
        d["submitted"] = d.pop("created", None)
    return dict_no_none(**d)


def register_views_secondary_services(
        blueprint: Blueprint, backend_implementation: OpenEoBackendImplementation, api_endpoint: EndpointRegistry,
        auth_handler: HttpAuthHandler
):
    @api_endpoint
    @blueprint.route('/service_types', methods=['GET'])
    def service_types():
        service_types = backend_implementation.secondary_services.service_types()
        expected_fields = {"configuration", "process_parameters"}
        assert all(expected_fields.issubset(st.keys()) for st in service_types.values())
        return jsonify(service_types)

    @api_endpoint
    @blueprint.route('/services', methods=['POST'])
    @auth_handler.requires_bearer_auth
    def services_post(user: User):
        """
        Create a secondary web service such as WMTS, TMS or WCS. The underlying data is processes on-demand, but a process graph may simply access results from a batch job. Computations should be performed in the sense that it is only evaluated for the requested spatial / temporal extent and resolution.

        Note: Costs incurred by shared secondary web services are usually paid by the owner, but this depends on the service type and whether it supports charging fees or not.

        :return:
        """
        post_data = request.get_json()
        service_id = backend_implementation.secondary_services.create_service(
            user_id=user.user_id,
            process_graph=_extract_process_graph(post_data),
            service_type=post_data["type"],
            api_version=g.api_version,
            configuration=post_data.get("configuration", {})
        )

        return make_response('', 201, {
            'Content-Type': 'application/json',
            'Location': url_for('.get_service_info', service_id=service_id),
            'OpenEO-Identifier': service_id
        })

    @api_endpoint
    @blueprint.route('/services', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def services_get(user: User):
        """List all running secondary web services for authenticated user"""
        return jsonify({
            "services": [
                _jsonable_service_metadata(m, full=False)
                for m in backend_implementation.secondary_services.list_services(user_id=user.user_id)
            ],
            "links": [],
        })

    @api_endpoint
    @blueprint.route('/services/<service_id>', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def get_service_info(service_id, user: User):
        try:
            metadata = backend_implementation.secondary_services.service_info(
                user_id=user.user_id, service_id=service_id
            )
        except Exception:
            raise ServiceNotFoundException(service_id)
        return jsonify(_jsonable_service_metadata(metadata, full=True))

    @api_endpoint(hidden=True)
    @blueprint.route('/services/<service_id>', methods=['PATCH'])
    @auth_handler.requires_bearer_auth
    def service_patch(service_id, user: User):
        process_graph = _extract_process_graph(request.get_json())
        backend_implementation.secondary_services.update_service(user_id=user.user_id, service_id=service_id,
                                                                 process_graph=process_graph)
        return response_204_no_content()

    @api_endpoint
    @blueprint.route('/services/<service_id>', methods=['DELETE'])
    @auth_handler.requires_bearer_auth
    def service_delete(service_id, user: User):
        backend_implementation.secondary_services.remove_service(user_id=user.user_id, service_id=service_id)
        return response_204_no_content()

    @api_endpoint
    @blueprint.route('/services/<service_id>/logs', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def service_logs(service_id, user: User):
        offset = request.args.get('offset', 0)
        logs = backend_implementation.secondary_services.get_log_entries(
            service_id=service_id, user_id=user.user_id, offset=offset
        )
        return jsonify({"logs": logs, "links": []})


def register_views_udp(
        blueprint: Blueprint, backend_implementation: OpenEoBackendImplementation, api_endpoint: EndpointRegistry,
        auth_handler: HttpAuthHandler
):
    _process_id_regex = re.compile(r"^\w+$")

    def _check_valid_process_graph_id(process_id: str):
        if not _process_id_regex.match(process_id):
            # TODO: official error code? See https://github.com/Open-EO/openeo-api/issues/402
            raise OpenEOApiException(
                code="InvalidId", status_code=400,
                message=f"Invalid process identifier {process_id!r}, must match {_process_id_regex.pattern!r}."
            )

    @api_endpoint(hidden=True)
    @blueprint.route('/validation', methods=["POST"])
    def udp_validate():
        # TODO
        raise FeatureUnsupportedException()

    @api_endpoint
    @blueprint.route('/process_graphs/<process_graph_id>', methods=['PUT'])
    @auth_handler.requires_bearer_auth
    def udp_store(process_graph_id: str, user: User):
        _check_valid_process_graph_id(process_id=process_graph_id)
        spec: dict = request.get_json()
        spec["id"] = process_graph_id
        if "process_graph" not in spec:
            raise ProcessGraphMissingException()
        backend_implementation.user_defined_processes.save(
            user_id=user.user_id,
            process_id=process_graph_id,
            spec=spec
        )

        return make_response("", 200)

    @api_endpoint
    @blueprint.route('/process_graphs/<process_graph_id>', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def udp_get(process_graph_id: str, user: User):
        _check_valid_process_graph_id(process_id=process_graph_id)
        udp = backend_implementation.user_defined_processes.get(user_id=user.user_id, process_id=process_graph_id)
        if udp:
            return _jsonable_udp_metadata(udp, full=True , user=user)

        raise ProcessGraphNotFoundException(process_graph_id)

    @api_endpoint
    @blueprint.route('/process_graphs', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def udp_list_for_user(user: User):
        user_udps = backend_implementation.user_defined_processes.get_for_user(user.user_id)
        return {
            'processes': [_jsonable_udp_metadata(udp, full=False) for udp in user_udps],
            # TODO: pagination links?
            "links": [],
        }

    @api_endpoint
    @blueprint.route('/process_graphs/<process_graph_id>', methods=['DELETE'])
    @auth_handler.requires_bearer_auth
    def udp_delete(process_graph_id: str, user: User):
        _check_valid_process_graph_id(process_id=process_graph_id)
        backend_implementation.user_defined_processes.delete(user_id=user.user_id, process_id=process_graph_id)
        return response_204_no_content()


def _jsonable_udp_metadata(metadata: UserDefinedProcessMetadata, full=True, user: User = None) -> dict:
    """API-version-aware conversion of UDP metadata to jsonable dict"""
    d = metadata.prepare_for_json()
    if not full:
        # API recommends to limit response size by omitting larger/optional fields
        d = {k: v for k, v in d.items() if k in ["id", "summary", "description", "parameters", "returns"]}
    elif metadata.public and user:
        namespace = "u:" + user.user_id
        d["links"] = (d.get("links") or []) + [
            {
                # TODO: standardized link rel type? See https://github.com/Open-EO/openeo-api/issues/405
                "rel": "public",
                # TODO: use signed url?
                "href": url_for(".processes_details", namespace=namespace, process_id=metadata.id, _external=True),
                "title": f"Public URL for user-defined process {metadata.id!r}"
            }
        ]

    return dict_no_none(**d)


def _normalize_collection_metadata(metadata: dict, api_version: ComparableVersion, full=False) -> dict:
    """
    Make sure the given collection metadata roughly complies to desired version of OpenEO spec.
    """
    # Make copy and remove all "private" fields
    metadata = copy.deepcopy(metadata)
    metadata = {k: v for (k, v) in metadata.items() if not k.startswith('_')}
    default_bbox = [0, 0, 0, 0]
    default_temporal_interval = [None, None]

    # Metadata should at least contain an id.
    if "id" not in metadata:
        _log.error("Collection metadata should have 'id' field: {m!r}".format(m=metadata))
        raise KeyError("id")
    collection_id = metadata["id"]

    # Version dependent metadata conversions
    cube_dims_100 = deep_get(metadata, "cube:dimensions", default=None)
    cube_dims_040 = deep_get(metadata, "properties", "cube:dimensions", default=None)
    eo_bands_100 = deep_get(metadata, "summaries", "eo:bands", default=None)
    eo_bands_040 = deep_get(metadata, "properties", "eo:bands", default=None)
    extent_spatial_100 = deep_get(metadata, "extent", "spatial", "bbox", default=None)
    extent_spatial_040 = deep_get(metadata, "extent", "spatial", default=None)
    extent_temporal_100 = deep_get(metadata, "extent", "temporal", "interval", default=None)
    extent_temporal_040 = deep_get(metadata, "extent", "temporal", default=None)
    rfc3339_dt = Rfc3339(propagate_none=True).datetime
    if api_version.below("1.0.0"):
        if full and not cube_dims_040 and cube_dims_100:
            metadata.setdefault("properties", {})
            metadata["properties"]["cube:dimensions"] = cube_dims_100
        if full and not eo_bands_040 and eo_bands_100:
            metadata.setdefault("properties", {})
            metadata["properties"]["eo:bands"] = eo_bands_100
        if extent_spatial_100:
            metadata["extent"]["spatial"] = extent_spatial_100[0]
        if extent_temporal_100 or extent_spatial_040:
            metadata["extent"]["temporal"] = [rfc3339_dt(v) for v in (extent_temporal_100[0] or extent_temporal_040)]
    else:
        if full and not cube_dims_100 and cube_dims_040:
            _log.warning("Collection metadata 'cube:dimensions' in API 0.4 style instead of 1.0 style")
            metadata["cube:dimensions"] = cube_dims_040
        if full and not eo_bands_100 and eo_bands_040:
            _log.warning("Collection metadata 'eo:bands' in API 0.4 style instead of 1.0 style")
            metadata.setdefault("summaries", {})
            metadata["summaries"]["eo:bands"] = eo_bands_040
        if not extent_spatial_100 and extent_spatial_040:
            _log.warning("Collection metadata 'extent': 'spatial' in API 0.4 style instead of 1.0 style")
            metadata["extent"]["spatial"] = {}
            metadata["extent"]["spatial"]["bbox"] = [extent_spatial_040]
        if not extent_temporal_100 and extent_temporal_040:
            _log.warning("Collection metadata 'extent': 'temporal' in API 0.4 style instead of 1.0 style")
            metadata["extent"]["temporal"] = {}
            metadata["extent"]["temporal"]["interval"] = [[rfc3339_dt(v) for v in extent_temporal_040]]
        elif extent_temporal_100:
            metadata["extent"]["temporal"]["interval"] = [[rfc3339_dt(v) for v in r] for r in extent_temporal_100]
        if full:
            bbox = deep_get(metadata, "extent", "spatial", "bbox", default=[default_bbox])[0]
            interval = deep_get(metadata, "extent", "temporal", "interval", default=[default_temporal_interval])[0]
            for _, dim in metadata.get("cube:dimensions", {}).items():
                if "extent" not in dim:
                    if dim.get("type") == "spatial" and dim.get("axis") == "x":
                        dim["extent"] = bbox[0::2]
                    elif dim.get("type") == "spatial" and dim.get("axis") == "y":
                        dim["extent"] = bbox[1::2]
                    elif dim.get("type") == "temporal":
                        dim["extent"] = interval

    # Make sure some required fields are set.
    metadata.setdefault("stac_version", "0.9.0" if api_version.at_least("1.0.0") else "0.6.2")
    metadata.setdefault("stac_extensions", ["datacube"])
    metadata.setdefault("links", [])
    metadata.setdefault("description", collection_id)
    metadata.setdefault("license", "proprietary")
    # Warn about missing fields where simple defaults are not feasible.
    fallbacks = {
        "extent": {"spatial": {"bbox": [default_bbox]}, "temporal": {"interval": [default_temporal_interval]}},
    } if api_version.at_least("1.0.0") else {
        "extent": {"spatial": default_bbox, "temporal": default_temporal_interval},
    }
    if full:
        if api_version.at_least("1.0.0"):
            fallbacks["cube:dimensions"] = {}
            fallbacks["summaries"] = {}
        else:
            fallbacks["properties"] = {}
            fallbacks["other_properties"] = {}

    for key, value in fallbacks.items():
        if key not in metadata:
            _log.warning("Collection {c!r} metadata does not have field {k!r}.".format(c=collection_id, k=key))
            metadata[key] = value

    if not full:
        basic_keys = [
            "stac_version", "stac_extensions", "id", "title", "description", "keywords", "version",
            "deprecated", "license", "providers", "extent", "links"
        ]
        metadata = {k: v for k, v in metadata.items() if k in basic_keys}

    return metadata


def register_views_catalog(
        blueprint: Blueprint, backend_implementation: OpenEoBackendImplementation, api_endpoint: EndpointRegistry,
        auth_handler: HttpAuthHandler
):
    @api_endpoint
    @blueprint.route('/collections', methods=['GET'])
    def collections():
        metadata = [
            _normalize_collection_metadata(metadata=m, api_version=requested_api_version(), full=False)
            for m in backend_implementation.catalog.get_all_metadata()
        ]
        return jsonify({
            'collections': metadata,
            'links': []
        })

    @api_endpoint
    @blueprint.route('/collections/<collection_id>', methods=['GET'])
    def collection_by_id(collection_id):
        metadata = backend_implementation.catalog.get_collection_metadata(collection_id=collection_id)
        metadata = _normalize_collection_metadata(metadata=metadata, api_version=requested_api_version(), full=True)
        return jsonify(metadata)


def register_views_user_files(
        blueprint: Blueprint, backend_implementation: OpenEoBackendImplementation, api_endpoint: EndpointRegistry,
        auth_handler: HttpAuthHandler
):
    @api_endpoint
    @blueprint.route('/files', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def files_list_for_user(user: User):
        # TODO EP-3538
        raise FeatureUnsupportedException()

    @api_endpoint
    @blueprint.route('/files/<path>', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def fildes_download(path, user: User):
        # TODO EP-3538
        raise FeatureUnsupportedException()

    @api_endpoint
    @blueprint.route('/files/<path>', methods=['PUT'])
    @auth_handler.requires_bearer_auth
    def files_upload(path, user: User):
        # TODO EP-3538
        raise FeatureUnsupportedException()

    @api_endpoint
    @blueprint.route('/files/<path>', methods=['DELETE'])
    @auth_handler.requires_bearer_auth
    def files_delete(path, user: User):
        # TODO EP-3538
        raise FeatureUnsupportedException()
