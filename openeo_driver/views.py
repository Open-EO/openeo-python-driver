import copy
import functools
import json
import logging
import pathlib
import re
import uuid
from collections import namedtuple, defaultdict
from typing import Callable, Tuple, List

import flask
import flask_cors
import numpy as np
from flask import Flask, request, url_for, jsonify, send_from_directory, abort, make_response, Blueprint, g, \
    current_app, redirect
from werkzeug.exceptions import HTTPException, NotFound
from werkzeug.middleware.proxy_fix import ProxyFix

from openeo.capabilities import ComparableVersion
from openeo.util import dict_no_none, deep_get, Rfc3339

from openeo_driver import urlsigning
from openeo_driver.backend import ServiceMetadata, BatchJobMetadata, UserDefinedProcessMetadata, \
    ErrorSummary, OpenEoBackendImplementation, BatchJobs, is_not_implemented
from openeo_driver.datacube import DriverMlModel
from openeo_driver.errors import OpenEOApiException, ProcessGraphMissingException, ServiceNotFoundException, \
    FilePathInvalidException, ProcessGraphNotFoundException, FeatureUnsupportedException, ProcessUnsupportedException, \
    JobNotFinishedException, ProcessGraphInvalidException, InternalException
from openeo_driver.save_result import SaveResult, to_save_result
from openeo_driver.users import User, user_id_b64_encode, user_id_b64_decode
from openeo_driver.users.auth import HttpAuthHandler
from openeo_driver.util.logging import FlaskRequestCorrelationIdLogging
from openeo_driver.utils import EvalEnv, smart_bool

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
    "1.1": ApiVersionInfo(version="1.1.0", supported=True, wellknown=True, production=True),
}
DEFAULT_VERSION = '1.1.0'

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
            allow_headers=["Content-Type", "Authorization", "Range"],
            expose_headers=["Location", "OpenEO-Identifier", "OpenEO-Costs", "Link","Accept-Ranges","Content-Range", "Content-Encoding"]
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

        FlaskRequestCorrelationIdLogging.before_request()

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

    @app.route('/', methods=['GET'])
    def redirect_root():
        return redirect(url_for('openeo.index'))

    auth = HttpAuthHandler(
        oidc_providers=backend_implementation.oidc_providers(),
        user_access_validation=backend_implementation.user_access_validation
    )
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

    app.register_blueprint(bp, url_prefix='/openeo', name="openeo_old")  # TODO: do we still need this?
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
    # Dedicated log channel for unhandled exceptions (unhandled by the view functions internally)
    _log = logging.getLogger(f"{__name__}.error")

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
        _log.error(repr(error), exc_info=True)
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
            "stac_extensions": [
                "https://stac-extensions.github.io/processing/v1.0.0/schema.json",
            ],
            "version": api_version,  # Deprecated pre-0.4.0 API version field
            "api_version": api_version,  # API version field since 0.4.0
            "backend_version": app_config.get('OPENEO_BACKEND_VERSION', '0.0.1'),
            "stac_version": "0.9.0",
            "id": service_id,
            "title": title,
            "description": app_config.get('OPENEO_DESCRIPTION', 'OpenEO API'),
            "production": API_VERSIONS[g.request_version].production,
            "endpoints": endpoints,
            "billing": backend_implementation.capabilities_billing(),
            # TODO: deprecate custom _backend_deploy_metadata
            "_backend_deploy_metadata": deploy_metadata,
            "processing:software": deploy_metadata.get("versions", {}),
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
                {
                    "rel": "terms-of-service",
                    "href": "https://openeo.cloud/aup",
                    "type": "text/html",
                },
                {
                    "rel": "privacy-policy",
                    "href": "https://terrascope.be/en/privacy-policy",
                    "type": "text/html",
                }
            ]
        }

        capabilities = backend_implementation.postprocess_capabilities(capabilities)

        return jsonify(capabilities)

    @blueprint.route('/conformance')
    def conformance():
        return jsonify({"conformsTo": [
            # TODO: expand/manage conformance classes?
            "http://www.opengis.net/spec/ogcapi-features-1/1.0/conf/core"
        ]})

    @blueprint.route('/health')
    def health():
        response = backend_implementation.health_check()
        if isinstance(response, str):
            # Legacy style
            response = jsonify({"health": response})
        elif not isinstance(response, flask.Response):
            response = jsonify(response)
        return response

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
        runtimes = backend_implementation.udf_runtimes.get_udf_runtimes()
        return jsonify(runtimes)

    @blueprint.route('/.well-known/openeo')
    def versioned_well_known_openeo():
        # Clients might request this for version discovery. Avoid polluting (error) logs by explicitly handling this.
        error = OpenEOApiException(status_code=404, code="NotFound", message="Not a well-known openEO URI")
        return make_response(jsonify(error.to_dict()), error.status_code)

    @blueprint.route('/CHANGELOG', methods=['GET'])
    def changelog():
        changelog = backend_implementation.changelog()
        if isinstance(changelog, pathlib.Path) and changelog.exists():
            return flask.send_file(changelog, mimetype="text/plain")
        elif isinstance(changelog, str):
            return make_response((changelog, {"Content-Type": "text/plain"}))
        else:
            return changelog

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

    @blueprint.route('/_debug/error/api', methods=["GET", "POST"])
    @blueprint.route('/_debug/error/api/<int:status>/<code>', methods=["GET", "POST"])
    def debug_error_api(status: int = None, code: str = None):
        raise OpenEOApiException(message="Computer says no.", code=code, status_code=status)

    @blueprint.route('/_debug/error/http', methods=["GET", "POST"])
    @blueprint.route('/_debug/error/http/<int:status>', methods=["GET", "POST"])
    def debug_error_http(status: int = 500):
        abort(status, "Computer says no.")


def register_views_auth(
        blueprint: Blueprint, backend_implementation: OpenEoBackendImplementation, api_endpoint: EndpointRegistry,
        auth_handler: HttpAuthHandler
):

    if backend_implementation.enable_basic_auth:
        @api_endpoint
        @blueprint.route("/credentials/basic", methods=["GET"])
        @auth_handler.requires_http_basic_auth
        def credentials_basic():
            access_token, user_id = auth_handler.authenticate_basic(request)
            resp = {"access_token": access_token}
            if requested_api_version().below("1.0.0"):
                resp["user_id"] = user_id
            return jsonify(resp)

    if backend_implementation.enable_oidc_auth:
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
        return jsonify(dict_no_none({
            "user_id": user.user_id,
            "name": user.get_name(),
            "info": user.info,
            # TODO: "default_plan" field? see https://github.com/Open-EO/openeo-api/issues/425
            "default_plan": user.get_default_plan(),
            # TODO more fields
        }))


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
    except (KeyError, TypeError) as e:
        raise ProcessGraphMissingException
    if not isinstance(pg, dict):
        # TODO: more validity checks for (flat) process graph?
        raise ProcessGraphInvalidException
    return pg


def register_views_processing(
        blueprint: Blueprint, backend_implementation: OpenEoBackendImplementation, api_endpoint: EndpointRegistry,
        auth_handler: HttpAuthHandler
):

    @api_endpoint(hidden=is_not_implemented(backend_implementation.processing.validate))
    @blueprint.route('/validation', methods=["POST"])
    def validation():
        post_data = request.get_json()
        try:
            process_graph = post_data["process_graph"]
        except (KeyError, TypeError) as e:
            raise ProcessGraphMissingException
        env = EvalEnv({
            "backend_implementation": backend_implementation,
            "version": g.api_version,
            "user": None
        })
        errors = backend_implementation.processing.validate(process_graph=process_graph, env=env)
        return jsonify({"errors": errors})

    @api_endpoint
    @blueprint.route('/result', methods=['POST'])
    @auth_handler.requires_bearer_auth
    def result(user: User):
        post_data = request.get_json()
        process_graph = _extract_process_graph(post_data)

        env = EvalEnv({
            "backend_implementation": backend_implementation,
            'version': g.api_version,
            'pyramid_levels': 'highest',
            'user': user,
            'require_bounds': True,
            'correlation_id': str(uuid.uuid4()),
        })
        result = backend_implementation.processing.evaluate(process_graph=process_graph, env=env)

        if result is None:
            # TODO: is it still necessary to handle `None` as an error condition?
            raise InternalException(message="Process graph evaluation gave no result")
        elif isinstance(result, flask.Response):
            # TODO: handle flask.Response in `to_save_result` too?
            return result
        else:
            if not isinstance(result, SaveResult):
                # Implicit save result (using default/best effort format and options)
                result = to_save_result(data=result)
            return result.create_flask_response()

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
    stac_item_media_type = "application/geo+json"

    @api_endpoint
    @blueprint.route('/jobs', methods=['POST'])
    @auth_handler.requires_bearer_auth
    def create_job(user: User):
        # TODO: wrap this job specification in a 1.0-style ProcessGrahpWithMetadata?
        post_data = request.get_json()
        # TODO: preserve original non-process_graph process fields too
        process = {"process_graph": _extract_process_graph(post_data)}
        # TODO: this "job_options" is not part of official API. See https://github.com/Open-EO/openeo-api/issues/276
        job_options = post_data.get("job_options")
        job_info = backend_implementation.batch_jobs.create_job(
            user_id=user.user_id,
            process=process,
            api_version=g.api_version,
            metadata={k: post_data[k] for k in ["title", "description", "plan", "budget"] if k in post_data},
            job_options=job_options,
        )
        job_id = job_info.id
        response = make_response("", 201)
        response.headers['Location'] = url_for('.get_job_info', job_id=job_id, _external=True)
        response.headers['OpenEO-Identifier'] = str(job_id)
        return response

    @api_endpoint
    @blueprint.route('/jobs', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def list_jobs(user: User):
        # TODO: support for `limit` param and paging links?

        listing = backend_implementation.batch_jobs.get_user_jobs(user.user_id)
        if isinstance(listing, list):
            jobs = listing
            links = []
            extra = {}
        elif isinstance(listing, dict):
            jobs = listing["jobs"]
            links = listing.get("links", [])
            # TODO: this "extra" whitelist is from experimental
            #       "federation extension API" https://github.com/Open-EO/openeo-api/pull/419
            extra = {k: listing[k] for k in ["federation:missing"] if k in listing}
        else:
            raise InternalException(f"Invalid user jobs listing {type(listing)}")

        resp = dict(
            jobs=[
                m.to_api_dict(full=False, api_version=requested_api_version())
                for m in jobs
            ],
            links=links,
            **extra
        )
        return jsonify(resp)

    @api_endpoint
    @blueprint.route('/jobs/<job_id>', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def get_job_info(job_id, user: User):
        job_info: BatchJobMetadata = backend_implementation.batch_jobs.get_job_info(job_id, user.user_id)
        return jsonify(job_info.to_api_dict(full=True, api_version=requested_api_version()))

    @api_endpoint()
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
        """Add a batch job to the procsessing queue."""
        job_info = backend_implementation.batch_jobs.get_job_info(job_id, user.user_id)
        if job_info.status in {"created", "canceled"}:
            backend_implementation.batch_jobs.start_job(job_id=job_id, user=user)
        return make_response("", 202)

    def _job_result_download_url(job_id, user_id, filename) -> str:
        if smart_bool(current_app.config.get('SIGNED_URL')):
            signer = urlsigning.Signer.from_config(current_app.config)
        else:
            signer = None

        if signer:
            expires = signer.get_expires()
            secure_key = signer.sign_job_asset(
                job_id=job_id, user_id=user_id, filename=filename, expires=expires
            )
            user_base64 = user_id_b64_encode(user_id)
            return url_for(
                '.download_job_result_signed',
                job_id=job_id, user_base64=user_base64, filename=filename, expires=expires, secure_key=secure_key,
                _external=True
            )
        else:
            return url_for('.download_job_result', job_id=job_id, filename=filename, _external=True)

    @api_endpoint
    @blueprint.route('/jobs/<job_id>/results', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def list_job_results(job_id, user: User):
        return _list_job_results(job_id, user.user_id)

    @api_endpoint
    @blueprint.route('/jobs/<job_id>/results/<user_base64>/<secure_key>', methods=['GET'])
    def list_job_results_signed(job_id, user_base64, secure_key):
        expires = request.args.get('expires')
        signer = urlsigning.Signer.from_config(current_app.config)
        user_id = user_id_b64_decode(user_base64)
        signer.verify_job_results(signature=secure_key, job_id=job_id, user_id=user_id, expires=expires)
        return _list_job_results(job_id, user_id)

    def _list_job_results(job_id, user_id):
        job_info = backend_implementation.batch_jobs.get_job_info(job_id, user_id)
        if job_info.status != "finished":
            raise JobNotFinishedException()

        results = backend_implementation.batch_jobs.get_results(job_id=job_id, user_id=user_id)

        if requested_api_version().at_least("1.0.0"):
            def job_results_canonical_url() -> str:
                if not smart_bool(current_app.config.get('SIGNED_URL')):
                    return url_for('.list_job_results', job_id=job_id, _external=True)

                signer = urlsigning.Signer.from_config(current_app.config)

                expires = signer.get_expires()
                secure_key = signer.sign_job_results(
                    job_id=job_id, user_id=user_id, expires=expires
                )
                user_base64 = user_id_b64_encode(user_id)
                return url_for(
                    '.list_job_results_signed',
                    job_id=job_id, user_base64=user_base64, expires=expires, secure_key=secure_key, _external=True
                )

            links = job_info.links
            if links is None:
                links = []

            links.extend([{
                "rel": "self",
                "href": url_for('.list_job_results', job_id=job_id, _external=True),  # MUST be absolute
                "type": "application/json"
            }, {
                "rel": "canonical",
                "href": job_results_canonical_url(),
                "type": "application/json"
            }, {
                "rel": "card4l-document",
                "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
                "type": "application/pdf"
            }])

            assets = {filename: _asset_object(job_id, user_id, filename, asset_metadata)
                      for filename, asset_metadata in results.items() if asset_metadata.get('asset', True)}

            if requested_api_version().at_least("1.1.0"):
                to_datetime = Rfc3339(propagate_none=True).datetime
                ml_model_metadata = None

                def job_result_item_url(item_id) -> str:
                    if not smart_bool(current_app.config.get('SIGNED_URL')):
                        return url_for('.get_job_result_item', job_id=job_id, item_id=item_id, _external=True)

                    signer = urlsigning.Signer.from_config(current_app.config)

                    expires = signer.get_expires()
                    secure_key = signer.sign_job_item(
                        job_id=job_id, user_id=user_id, item_id=item_id, expires=expires
                    )
                    user_base64 = user_id_b64_encode(user_id)
                    return url_for(
                        '.get_job_result_item_signed',
                        job_id=job_id, user_base64=user_base64, secure_key=secure_key, item_id=item_id, expires=expires,
                        _external=True)

                for filename, metadata in results.items():
                    if "data" in metadata.get("roles", []) and "geotiff" in metadata.get("type", ""):
                        links.append({
                            "rel": "item",
                            "href": job_result_item_url(item_id=filename),
                            "type": stac_item_media_type
                        })
                    elif metadata.get('ml_model_metadata', False):
                        # TODO: Currently we only support one ml_model per batch job.
                        ml_model_metadata = metadata
                        links.append({
                                "rel": "item",
                                "href": job_result_item_url(item_id=filename),
                                "type": "application/json"
                        })

                result = dict_no_none(**{
                    "type": "Collection",
                    "stac_version": "1.0.0",
                    "stac_extensions": ["eo", "file"],
                    "id": job_id,
                    "title": job_info.title,
                    "description": job_info.description or f"Results for batch job {job_id}",
                    "license": "proprietary",  # TODO?
                    "extent": {
                        "spatial": {
                            "bbox": [job_info.bbox]
                        },
                        "temporal": {
                            "interval": [[to_datetime(job_info.start_datetime), to_datetime(job_info.end_datetime)]]
                        }
                    },
                    "links": links,
                    "assets": assets
                })
                if ml_model_metadata is not None:
                    result["stac_extensions"].extend(ml_model_metadata.get("stac_extensions", []))
                    if "summaries" not in result.keys():
                        result["summaries"] = {}
                    if "properties" in ml_model_metadata.keys():
                        ml_model_properties = ml_model_metadata["properties"]
                        learning_approach = ml_model_properties.get("ml-model:learning_approach", None)
                        prediction_type = ml_model_properties.get("ml-model:prediction_type", None)
                        architecture = ml_model_properties.get("ml-model:architecture", None)
                        result["summaries"].update({
                            "ml-model:learning_approach": [learning_approach] if learning_approach is not None else [],
                            "ml-model:prediction_type": [prediction_type] if prediction_type is not None else [],
                            "ml-model:architecture": [architecture] if architecture is not None else [],
                        })
            else:
                result = {
                    "type": "Feature",
                    "stac_version": "0.9.0",
                    "id": job_info.id,
                    "properties": _properties_from_job_info(job_info),
                    "assets": assets,
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
                    {"href": _job_result_download_url(job_id, user_id, filename)} for filename in results.keys()
                ]
            }

        # TODO "OpenEO-Costs" header?
        return jsonify(result)

    def _download_job_result(job_id: str, filename: str, user_id: str) -> flask.Response:
        results = backend_implementation.batch_jobs.get_results(job_id=job_id, user_id=user_id)
        if filename not in results.keys():
            raise FilePathInvalidException(f"{filename!r} not in {list(results.keys())}")
        result = results[filename]
        if "output_dir" in result:
            resp = send_from_directory(result["output_dir"], filename, mimetype=result.get("type"))
            resp.headers['Accept-Ranges'] = 'bytes'
            return resp
        elif "json_response" in result:
            return jsonify(result["json_response"])
        else:
            _log.error(f"Unsupported job result: {result!r}")
            raise InternalException("Unsupported job result")

    @api_endpoint
    @blueprint.route('/jobs/<job_id>/results/items/<user_base64>/<secure_key>/<item_id>', methods=['GET'])
    def get_job_result_item_signed(job_id, user_base64, secure_key, item_id):
        expires = request.args.get('expires')
        signer = urlsigning.Signer.from_config(current_app.config)
        user_id = user_id_b64_decode(user_base64)
        signer.verify_job_item(signature=secure_key, job_id=job_id, user_id=user_id, item_id=item_id, expires=expires)
        return _get_job_result_item(job_id, item_id, user_id)

    @api_endpoint(version=ComparableVersion("1.1.0").or_higher)
    @blueprint.route('/jobs/<job_id>/results/items/<item_id>', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def get_job_result_item(job_id: str, item_id: str, user: User) -> flask.Response:
        return _get_job_result_item(job_id, item_id, user.user_id)

    def _get_job_result_item(job_id, item_id, user_id):
        if item_id == DriverMlModel.METADATA_FILE_NAME:
            return _download_ml_model_metadata(job_id, item_id, user_id)

        results = backend_implementation.batch_jobs.get_results(job_id, user_id)

        assets_for_item_id = {
            asset_filename: metadata for asset_filename, metadata in results.items()
            if asset_filename.startswith(item_id)
        }

        if len(assets_for_item_id) != 1:
            raise AssertionError(f"expected exactly 1 asset with file name {item_id}")

        asset_filename, metadata = next(iter(assets_for_item_id.items()))

        geometry = metadata.get("geometry")
        bbox = metadata.get("bbox")

        properties = {"datetime": metadata.get("datetime")}
        if properties["datetime"] is None:
            job_info = backend_implementation.batch_jobs.get_job_info(job_id, user_id)
            to_datetime = Rfc3339(propagate_none=True).datetime

            start_datetime = to_datetime(job_info.start_datetime)
            end_datetime = to_datetime(job_info.end_datetime)

            if start_datetime == end_datetime:
                properties["datetime"] = start_datetime
            else:
                if start_datetime:
                    properties["start_datetime"] = start_datetime
                if end_datetime:
                    properties["end_datetime"] = end_datetime

        stac_item = {
            "type": "Feature",
            "stac_version": "0.9.0",
            "stac_extensions": ["eo", "file"],
            "id": item_id,
            "geometry": geometry,
            "bbox": bbox,
            "properties": properties,
            "links": [{
                "rel": "self",
                # MUST be absolute
                "href": url_for('.get_job_result_item', job_id=job_id, item_id=item_id, _external=True),
                "type": stac_item_media_type
            }, {
                "rel": "collection",
                "href": url_for('.list_job_results', job_id=job_id, _external=True),  # SHOULD be absolute
                "type": "application/json"
            }],
            "assets": {
                asset_filename: _asset_object(job_id, user_id, asset_filename, metadata)
            },
            "collection": job_id
        }

        resp = jsonify(stac_item)
        resp.mimetype = stac_item_media_type
        return resp

    def _download_ml_model_metadata(job_id: str, file_name: str, user_id) -> flask.Response:
        results = backend_implementation.batch_jobs.get_results(job_id, user_id)
        ml_model_metadata: dict = results.get(file_name, None)
        if ml_model_metadata is None:
            raise FilePathInvalidException(f"{file_name!r} not in {list(results.keys())}")
        assets = deep_get(ml_model_metadata, "assets", default={})
        for asset in assets.values():
            if not asset["href"].startswith("http"):
                asset_file_name = pathlib.Path(asset["href"]).name
                asset["href"] = url_for('.download_job_result', job_id=job_id, item_id=asset_file_name, _external=True)
        stac_item = {
            "stac_version": ml_model_metadata.get("stac_version", "0.9.0"),
            "stac_extensions": ml_model_metadata.get("stac_extensions", []),
            "type": "Feature",
            "id": ml_model_metadata.get("id"),
            "collection": job_id,
            "bbox": ml_model_metadata.get("bbox", []),
            "geometry": ml_model_metadata.get("geometry", {}),
            'properties': ml_model_metadata.get("properties", {}),
            'links': ml_model_metadata.get("links", []),
            'assets': ml_model_metadata.get("assets", {})
        }
        resp = jsonify(stac_item)
        resp.mimetype = stac_item_media_type
        return resp

    def _asset_object(job_id, user_id, filename: str, asset_metadata: dict) -> dict:
        result_dict = dict_no_none({
            "title": asset_metadata.get("title", filename),
            "href": asset_metadata.get(BatchJobs.ASSET_PUBLIC_HREF) or _job_result_download_url(job_id, user_id, filename),
            "type": asset_metadata.get("type", asset_metadata.get("media_type","application/octet-stream")),
            "roles": asset_metadata.get("roles", ["data"])
        })
        if filename.endswith(".model"):
            # Machine learning models.
            return result_dict
        bands = asset_metadata.get("bands")
        nodata = asset_metadata.get("nodata")

        result_dict.update(dict_no_none(**{
            "eo:bands": [dict_no_none(**{"name": band.name, "center_wavelength": band.wavelength_um})
                         for band in bands] if bands else None,
            "file:nodata": ["nan" if nodata!=None and np.isnan(nodata) else nodata],
        }))
        return result_dict

    @api_endpoint
    @blueprint.route('/jobs/<job_id>/results/assets/<filename>', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def download_job_result(job_id, filename, user: User):
        return _download_job_result(job_id=job_id, filename=filename, user_id=user.user_id)

    @api_endpoint
    @blueprint.route('/jobs/<job_id>/results/assets/<user_base64>/<secure_key>/<filename>', methods=['GET'])
    def download_job_result_signed(job_id, user_base64, secure_key, filename):
        expires = request.args.get('expires')
        signer = urlsigning.Signer.from_config(current_app.config)
        user_id = user_id_b64_decode(user_base64)
        signer.verify_job_asset(
            signature=secure_key,
            job_id=job_id, user_id=user_id, filename=filename, expires=expires
        )
        return _download_job_result(job_id=job_id, filename=filename, user_id=user_id)

    @api_endpoint
    @blueprint.route('/jobs/<job_id>/logs', methods=['GET'])
    @auth_handler.requires_bearer_auth
    def get_job_logs(job_id, user: User):
        offset = request.args.get('offset')
        # TODO: implement paging support: `limit`, next/prev/first/last `links`, ...
        logs = backend_implementation.batch_jobs.get_log_entries(job_id=job_id, user_id=user.user_id, offset=offset)

        def generate():
            yield """{"logs":["""

            sep = ""
            try:
                for log in logs:
                    yield sep + json.dumps(log)
                    sep = ","

            except Exception as e:
                # TODO: because of chunked response, we can not update already sent 200 HTTP status code.
                #       We could however wait sending the status based on successfully getting first log item.
                _log.error(f"Log collection for job {job_id} failed", exc_info=True)
                log = {
                    "id": "-1", "code": "Internal", "level": "error",
                    "message": f"Log collection failed: {e!r}"
                }
                yield sep + json.dumps(log)

            yield """],"links":[]}"""

        return current_app.response_class(generate(), mimetype="application/json")

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
                "rel": "canonical",
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

    @blueprint.route('/collections/<collection_id>/items', methods=['GET'])
    def collection_items(collection_id):
        # Note: This is not an openEO API endpoint, but a STAC API endpoint
        # https://stacspec.org/STAC-api.html#operation/getFeatures
        params = {k: request.args.get(k) for k in ["limit", "bbox", "datetime"] if k in request.args}
        res = backend_implementation.catalog.get_collection_items(collection_id=collection_id, parameters=params)
        if isinstance(res, dict):
            return jsonify(res)
        elif isinstance(res, flask.Response):
            return res
        else:
            raise ValueError(f"Invalid get_collection_items result: {type(res)}")


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
