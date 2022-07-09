"""

Base structure of a backend implementation (e.g. Geotrellis based)
to be exposed with HTTP REST frontend.

It is organised in microservice-like parts following https://open-eo.github.io/openeo-api/apireference/
to allow composability, isolation and better reuse.

Also see https://github.com/Open-EO/openeo-python-driver/issues/8
"""

import abc
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Union, NamedTuple, Dict, Optional, Callable, Iterable

import flask

import openeo_driver.util.view_helpers
from openeo.capabilities import ComparableVersion
from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo.util import rfc3339, dict_no_none
from openeo_driver.datacube import DriverDataCube, DriverMlModel
from openeo_driver.datastructs import SarBackscatterArgs
from openeo_driver.dry_run import SourceConstraint
from openeo_driver.errors import CollectionNotFoundException, ServiceUnsupportedException, FeatureUnsupportedException
from openeo_driver.processes import ProcessRegistry
from openeo_driver.users import User
from openeo_driver.users.oidc import OidcProvider
from openeo_driver.utils import read_json, dict_item, EvalEnv, extract_namedtuple_fields_from_dict, get_package_versions

logger = logging.getLogger(__name__)


class MicroService:
    """
    Base class for a backend "microservice"
    (grouped subset of backend functionality)

    https://openeo.org/documentation/1.0/developers/arch.html#microservices
    """


def not_implemented(f: Callable):
    """Decorator for functions/methods that are not implemented"""
    f._not_implemented = True
    return f


def is_not_implemented(f: Callable):
    """Checker for functions/methods that are not implemented"""
    return hasattr(f, "_not_implemented") and f._not_implemented


class ServiceMetadata(NamedTuple):
    """
    Container for service metadata
    """
    # TODO: move this to openeo-python-client?
    # TODO: also add user metadata?

    # Required fields (no default)
    id: str
    process: dict  # TODO: also encapsulate this "process graph with metadata" struct (instead of free-form dict)?
    url: str
    type: str
    enabled: bool
    attributes: dict
    configuration: dict

    # Optional fields (with default)
    title: str = None
    description: str = None
    created: datetime = None
    plan: str = None
    costs: float = None
    budget: float = None

    def prepare_for_json(self) -> dict:
        """Prepare metadata for JSON serialization"""
        d = self._asdict()  # pylint: disable=no-member
        d["created"] = rfc3339.datetime(self.created) if self.created else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ServiceMetadata':
        """Load ServiceMetadata from dict (e.g. parsed JSON dump)."""
        d = extract_namedtuple_fields_from_dict(d, ServiceMetadata)
        created = d.get("created")
        if isinstance(created, str):
            d["created"] = rfc3339.parse_datetime(created)
        return cls(**d)


class SecondaryServices(MicroService):
    """
    Base contract/implementation for Secondary Services "microservice"
    https://openeo.org/documentation/1.0/developers/api/reference.html#tag/Secondary-Services
    """

    def service_types(self) -> dict:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-service-types"""
        return {}

    def list_services(self, user_id: str) -> List[ServiceMetadata]:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-services"""
        return []

    def service_info(self, user_id: str, service_id: str) -> ServiceMetadata:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/describe-service"""
        raise NotImplementedError()

    def create_service(self, user_id: str, process_graph: dict, service_type: str, api_version: str,
                       configuration: dict) -> str:
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/create-service
        :return: (location, openeo_identifier)
        """
        if service_type.lower() not in set(st.lower() for st in self.service_types()):
            raise ServiceUnsupportedException(service_type)

        return self._create_service(user_id, process_graph, service_type, api_version, configuration)

    def _create_service(self, user_id: str, process_graph: dict, service_type: str, api_version: str,
                       configuration: dict) -> str:
        raise NotImplementedError()

    def update_service(self, user_id: str, service_id: str, process_graph: dict) -> None:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/update-service"""
        raise NotImplementedError()

    def remove_service(self, user_id: str, service_id: str) -> None:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/delete-service"""
        raise NotImplementedError()

    def get_log_entries(self, service_id: str, user_id: str, offset: str) -> List[dict]:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/debug-service"""
        # TODO require auth/user handle?
        return []


class LoadParameters(dict):
    """Container for load_collection related parameters and optimization hints"""
    # Some attributes pointing to dict items for more explicit tracing where parameters are set and read.
    temporal_extent = dict_item(default=(None, None))
    spatial_extent = dict_item(default={})
    global_extent = dict_item(default={})
    bands = dict_item(default=None)
    properties = dict_item(default={})
    # TODO: rename this to filter_spatial_geometries (because it is used for load_collection-time filtering)?
    aggregate_spatial_geometries = dict_item(default=None)
    sar_backscatter: Union[SarBackscatterArgs, None] = dict_item(default=None)
    process_types = dict_item(default=set())
    custom_mask = dict_item(default={})
    data_mask = dict_item(default={})
    target_crs = dict_item(default=None)
    target_resolution = dict_item(default=None)
    backend_provider = dict_item(default=None)

    def copy(self) -> "LoadParameters":
        return LoadParameters(super().copy())

    def __hash__(self) -> int:
        return hash(json.dumps(self, sort_keys=True))



class AbstractCollectionCatalog(MicroService, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_all_metadata(self) -> List[dict]:
        """Basic metadata for all collections"""
        ...

    @abc.abstractmethod
    def get_collection_metadata(self, collection_id: str) -> dict:
        """Full metadata for a specific collections"""
        ...

    @abc.abstractmethod
    def load_collection(self, collection_id: str, load_params: LoadParameters, env: EvalEnv) -> DriverDataCube:
        """Load a collection as a DriverDataCube"""
        ...

    def get_collection_items(self, collection_id: str, parameters: dict) -> Union[dict, flask.Response]:
        """
        Optional STAC API endpoint `GET /collections/{collectionId}/items`
        """
        raise FeatureUnsupportedException


class CollectionCatalog(AbstractCollectionCatalog):
    """
    Basic implementation of a catalog of collections/EO data, based on predefined list of collections
    """

    def __init__(self, all_metadata: List[dict]):
        self._catalog = {layer["id"]: dict(layer) for layer in all_metadata}

    @classmethod
    def from_json_file(cls, filename: Union[str, Path] = "layercatalog.json"):
        """Factory to read catalog from a JSON file"""
        return cls(read_json(filename))

    def get_all_metadata(self) -> List[dict]:
        """
        Basic metadata for all datasets
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-collections
        """
        return list(self._catalog.values())

    def get_collection_metadata(self, collection_id: str) -> dict:
        """
        Full metadata for a specific dataset
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/describe-collection
        """
        try:
            return self._catalog[collection_id]
        except KeyError:
            raise CollectionNotFoundException(collection_id)

    def load_collection(self, collection_id: str, load_params: LoadParameters, env: EvalEnv) -> DriverDataCube:
        raise NotImplementedError


class CollectionIncompleteMetadataWarning(UserWarning):
    pass


class BatchJobMetadata(NamedTuple):
    """
    Container for batch job metadata
    """
    # TODO: move this to openeo-python-client?
    # TODO: also add user metadata?

    # Required fields (no default)
    id: str
    status: str  # created/queued/running/canceled/finished/error
    created: datetime

    # Optional fields (with default)
    process: dict = None  # TODO: also encapsulate this "process graph with metadata" struct (instead of free-form dict)?
    job_options: dict = None
    title: str = None
    description: str = None
    progress: float = None
    updated: datetime = None
    plan: str = None
    costs: float = None
    budget: float = None
    started: datetime = None
    finished: datetime = None
    duration_: timedelta = None
    memory_time_megabyte: timedelta = None
    cpu_time: timedelta = None
    geometry: dict = None
    bbox: List[float] = None
    start_datetime: datetime = None
    end_datetime: datetime = None
    instruments: List[str] = None
    epsg: int = None
    links: List[Dict] = None
    usage: Dict = None

    @property
    def duration(self) -> Union[timedelta, None]:
        """Returns the job duration if possible, else None."""
        if self.duration_:
            return self.duration_
        elif self.finished and self.started:
            return self.finished - self.started

    @classmethod
    def from_api_dict(cls, d: dict) -> 'BatchJobMetadata':
        """Populate from an openEO API compatible dictionary."""
        kwargs = extract_namedtuple_fields_from_dict(
            d, BatchJobMetadata, convert_datetime=True, convert_timedelta=True
        )

        usage = d.get("usage")
        if usage:
            if usage.get("cpu"):
                # TODO: support other units too
                assert usage["cpu"]["unit"] == "cpu-seconds"
                kwargs["cpu_time"] = timedelta(seconds=usage["cpu"]["value"])
            if usage.get("memory"):
                assert usage["memory"]["unit"] == "mb-seconds"
                kwargs["memory_time_megabyte"] = timedelta(seconds=usage["memory"]["value"])
            if usage.get("duration"):
                assert usage["duration"]["unit"] == "seconds"
                kwargs["duration_"] = timedelta(seconds=usage["duration"]["value"])

        return cls(**kwargs)

    def to_api_dict(self, full=True, api_version: ComparableVersion = None) -> dict:
        """
        API-version-aware conversion of batch job metadata to jsonable openEO API compatible dict.
        see https://openeo.org/documentation/1.0/developers/api/reference.html#operation/describe-job
        """
        # Basic/full fields to export
        fields = ["id", "title", "description", "status", "progress", "created", "updated", "plan", "costs", "budget"]
        if full:
            fields.extend(["process"])
        result = {f: getattr(self, f) for f in fields}

        # Additional cleaning and massaging.
        result["created"] = rfc3339.datetime(self.created) if self.created else None
        result["updated"] = rfc3339.datetime(self.updated) if self.updated else None

        if full:
            usage = self.usage or {}
            if self.cpu_time:
                usage["cpu"] = {"value": int(round(self.cpu_time.total_seconds())), "unit": "cpu-seconds"}
            if self.duration:
                usage["duration"] = {"value": int(round(self.duration.total_seconds())), "unit": "seconds"}
            if self.memory_time_megabyte:
                usage["memory"] = {"value": int(round(self.memory_time_megabyte.total_seconds())), "unit": "mb-seconds"}
            if usage:
                result["usage"] = usage

        if api_version and api_version.below("1.0.0"):
            result["process_graph"] = result.pop("process", {}).get("process_graph")
            result["submitted"] = result.pop("created", None)
            # TODO wider status checking coverage?
            if result["status"] == "created":
                result["status"] = "submitted"

        return dict_no_none(result)


class BatchJobs(MicroService):
    """
    Base contract/implementation for Batch Jobs "microservice"
    https://openeo.org/documentation/1.0/developers/api/reference.html#operation/stop-job
    """

    ASSET_PUBLIC_HREF = "public_href"

    def __init__(self):
        # TODO this "proxy user" feature is YARN/Spark/VITO specific. Move it to oppeno-geopyspark-driver?
        self._get_proxy_user: Callable[[User], Optional[str]] = lambda user: None

    def create_job(
            self, user_id: str, process: dict, api_version: str,
            metadata: dict, job_options: dict = None
    ) -> BatchJobMetadata:
        # TODO: why return a full BatchJobMetadata? only job id is used
        raise NotImplementedError

    def get_job_info(self, job_id: str, user_id: str) -> BatchJobMetadata:
        """
        Get details about a batch job
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/describe-job
        Should raise `JobNotFoundException` on invalid job/user id
        """
        raise NotImplementedError

    def get_user_jobs(self, user_id: str) -> Union[List[BatchJobMetadata], dict]:
        """
        Get details about all batch jobs of a user
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-jobs
        """
        raise NotImplementedError

    def start_job(self, job_id: str, user: User):
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/start-job
        """
        raise NotImplementedError

    def get_results(self, job_id: str, user_id: str) -> Dict[str, dict]:
        """
        Return result files as (filename, metadata) mapping: `filename` is the part that
        the user can see (in download url), `metadata` contains internal (root) dir where
        output is stored.

        related:
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-results
        """
        raise NotImplementedError

    def get_log_entries(self, job_id: str, user_id: str, offset: Optional[str] = None) -> Iterable[dict]:
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/debug-job
        """
        raise NotImplementedError

    def cancel_job(self, job_id: str, user_id: str):
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/stop-job
        """
        raise NotImplementedError

    def delete_job(self, job_id: str, user_id: str):
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/delete-job
        """
        raise NotImplementedError

    # TODO this "proxy user" feature is YARN/Spark/VITO specific. Move it to oppeno-geopyspark-driver?
    def get_proxy_user(self, user: User) -> Optional[str]:
        return self._get_proxy_user(user)

    def set_proxy_user_getter(self, getter: Callable[[User], Optional[str]]):
        self._get_proxy_user = getter


class UserDefinedProcessMetadata(NamedTuple):
    """
    Container for user-defined process metadata.
    """
    process_graph: dict
    id: str
    parameters: List[dict] = None
    returns: dict = None
    summary: str = None
    description: str = None
    links: list = None
    public: bool = False  # Note: experimental non-standard flag

    @classmethod
    def from_dict(cls, d: dict) -> 'UserDefinedProcessMetadata':
        d = extract_namedtuple_fields_from_dict(d, UserDefinedProcessMetadata)
        return cls(**d)

    def prepare_for_json(self) -> dict:
        return self._asdict()  # pylint: disable=no-member


class UserDefinedProcesses(MicroService):
    """
    Base contract/implementation for User-Defined Processes "microservice"
    https://openeo.org/documentation/1.0/developers/api/reference.html#tag/User-Defined-Processes
    """

    def get(self, user_id: str, process_id: str) -> Union[UserDefinedProcessMetadata, None]:
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/describe-custom-process
        """
        raise NotImplementedError

    def get_for_user(self, user_id: str) -> List[UserDefinedProcessMetadata]:
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-custom-processes
        """
        raise NotImplementedError

    def save(self, user_id: str, process_id: str, spec: dict) -> None:
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/store-custom-process
        """
        raise NotImplementedError

    def delete(self, user_id: str, process_id: str) -> None:
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/delete-custom-process
        """
        raise NotImplementedError


class Processing(MicroService):
    """
    Base contract/implementation for processing: available processes and process graph evaluation
    """

    def get_process_registry(self, api_version: Union[str, ComparableVersion]) -> ProcessRegistry:
        raise NotImplementedError

    def evaluate(self, process_graph: dict, env: EvalEnv = None):
        """Evaluate given process graph (flat dict format)."""
        raise NotImplementedError

    @not_implemented
    def validate(self, process_graph: dict, env: EvalEnv = None) -> List[dict]:
        """
        Process graph validation

        :return: List (or generator) of validation error dicts (having at least a "code" and "message" field)
        """
        raise NotImplementedError


class ErrorSummary(Exception):
    # TODO: this is specific for openeo-geopyspark-driver: can we avoid defining it in openeo-python-driver?
    #       For example by allowing to inject custom error handlers
    def __init__(self, exception: Exception, is_client_error: bool, summary: str = None):
        self.exception = exception
        self.is_client_error = is_client_error
        self.summary = summary or str(exception)

    def __str__(self):
        return str({
            'exception': "%s: %s" % (type(self.exception).__name__, self.exception),
            'is_client_error': self.is_client_error,
            'summary': self.summary
        })


class UdfRuntimes(MicroService):
    # Python libraries to list
    python_libraries = [
        "openeo",
        "openeo_driver",
        "numpy",
        "scipy",
        "pandas",
        "xarray",
        "geopandas",
        "netCDF4",
        "shapely",
        "pyproj",
        "rasterio",
    ]

    def __init__(self):
        pass

    def get_python_versions(self):
        major, minor, patch = (str(v) for v in sys.version_info[:3])
        aliases = [
            f"{major}",
            f"{major}.{minor}",
            f"{major}.{minor}.{patch}"
        ]
        default_version = major
        return major, aliases, default_version

    def _get_python_udf_runtime_metadata(self):
        major, aliases, default_version = self.get_python_versions()
        # TODO: get actual library version (instead of version of current environment).
        libraries = {
            p: {"version": v.split(" ", 1)[-1]}
            for p, v in get_package_versions(self.python_libraries, na_value=None).items()
            if v
        }

        return {
            "title": f"Python {major}",
            "description": f"Python {major} runtime environment.",
            "type": "language",
            "default": default_version,
            "versions": {
                v: {"libraries": libraries}
                for v in aliases
            }
        }

    def get_udf_runtimes(self) -> dict:
        # TODO add caching of this result
        return {
            # TODO: toggle these runtimes through dependency injection or config?
            "Python": self._get_python_udf_runtime_metadata(),
            "Python-Jep": self._get_python_udf_runtime_metadata(),
        }


class OpenEoBackendImplementation:
    """
    Simple container of all openEo "microservices"
    """
    enable_basic_auth = True
    enable_oidc_auth = True

    def __init__(
            self,
            secondary_services: Optional[SecondaryServices] = None,
            catalog: Optional[AbstractCollectionCatalog] = None,
            batch_jobs: Optional[BatchJobs] = None,
            user_defined_processes: Optional[UserDefinedProcesses] = None,
            processing: Optional[Processing] = None,
    ):
        self.secondary_services = secondary_services
        self.catalog = catalog
        self.batch_jobs = batch_jobs
        self.user_defined_processes = user_defined_processes
        self.user_files = None  # TODO: implement user file storage microservice
        self.processing = processing
        self.udf_runtimes = UdfRuntimes()

        # Overridable cache control header injecting decorator for static, public view functions
        self.cache_control = openeo_driver.util.view_helpers.cache_control(
            max_age=timedelta(minutes=15), public=True,
        )

    def health_check(self, options: Optional[dict] = None) -> Union[str, dict, flask.Response]:
        return "OK"

    def oidc_providers(self) -> List[OidcProvider]:
        return []

    def file_formats(self) -> dict:
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-file-types
        """
        return {"input": {}, "output": {}}

    def load_disk_data(
            self, format: str, glob_pattern: str, options: dict, load_params: LoadParameters, env: EvalEnv
    ) -> DriverDataCube:
        # TODO: move this to catalog "microservice"
        # TODO: rename this to "load_uploaded_files" like in official openeo processes
        raise NotImplementedError

    def load_result(self, job_id: str, user_id: str, load_params: LoadParameters, env: EvalEnv) -> DriverDataCube:
        raise NotImplementedError

    def load_ml_model(self, job_id: str) -> DriverMlModel:
        raise NotImplementedError

    def visit_process_graph(self, process_graph: dict) -> ProcessGraphVisitor:
        """Create a process graph visitor and accept given process graph"""
        return ProcessGraphVisitor().accept_process_graph(process_graph)

    def summarize_exception(self, error: Exception) -> Union[ErrorSummary, Exception]:
        return error

    # TODO this "proxy user" feature is YARN/Spark/VITO specific. Move it to oppeno-geopyspark-driver?
    def set_preferred_username_getter(self, getter: Callable[[User], Optional[str]]):
        self.batch_jobs.set_proxy_user_getter(getter)

    def user_access_validation(self, user: User, request: flask.Request) -> User:
        """Additional user access validation based on flask request."""
        return user

    def capabilities_billing(self) -> dict:
        """Capabilities doc: field 'billing'"""
        return {
            "currency": None,
        }

    def postprocess_capabilities(self, capabilities: dict) -> dict:
        """Postprocess the capabilities document"""
        return capabilities

    def changelog(self) -> Union[str, Path]:
        return "No changelog"
