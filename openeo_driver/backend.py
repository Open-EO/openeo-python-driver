"""

Base structure of a backend implementation (e.g. Geotrellis based)
to be exposed with HTTP REST frontend.

It is organised in microservice-like parts following https://open-eo.github.io/openeo-api/apireference/
to allow composability, isolation and better reuse.

Also see https://github.com/Open-EO/openeo-python-driver/issues/8
"""

import importlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Union, NamedTuple, Dict

from openeo import ImageCollection
from openeo.error_summary import ErrorSummary
from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo.util import rfc3339
from openeo_driver.errors import CollectionNotFoundException, ServiceUnsupportedException
from openeo_driver.utils import read_json

logger = logging.getLogger(__name__)


class MicroService:
    """
    Base class for a backend "microservice"
    (grouped subset of backend functionality)

    https://openeo.org/documentation/1.0/developers/arch.html#microservices
    """


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

    # Optional fields (with default)
    title: str = None
    description: str = None
    configuration: dict = None
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
        created = d.get("created")
        if isinstance(created, str):
            d = d.copy()
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

    def list_services(self) -> List[ServiceMetadata]:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-services"""
        return []

    def service_info(self, service_id: str) -> ServiceMetadata:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/describe-service"""
        raise NotImplementedError()

    def create_service(self, process_graph: dict, service_type: str, api_version: str, post_data: dict) -> ServiceMetadata:
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/create-service
        :return: (location, openeo_identifier)
        """
        from openeo_driver.ProcessGraphDeserializer import evaluate
        # TODO require auth/user handle?
        if service_type.lower() not in set(st.lower() for st in self.service_types()):
            raise ServiceUnsupportedException(service_type)

        image_collection = evaluate(process_graph, viewingParameters={'version': api_version, 'pyramid_levels': 'all'})
        service_metadata = image_collection.tiled_viewing_service(
            service_type=service_type,
            process_graph=process_graph,
            post_data=post_data
        )
        return service_metadata

    def update_service(self, service_id: str, process_graph: dict) -> None:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/update-service"""
        # TODO require auth/user handle?
        raise NotImplementedError()

    def remove_service(self, service_id: str) -> None:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/delete-service"""
        # TODO require auth/user handle?
        raise NotImplementedError()

    def get_log_entries(self, service_id: str, user_id: str, offset: str) -> List[dict]:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/debug-service"""
        # TODO require auth/user handle?
        return []


class CollectionCatalog(MicroService):
    """
    Basic implementation of a catalog of collections/EO data
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

    def _get(self, collection_id: str) -> dict:
        try:
            return self._catalog[collection_id]
        except KeyError:
            raise CollectionNotFoundException(collection_id)

    def get_collection_metadata(self, collection_id: str) -> dict:
        """
        Full metadata for a specific dataset
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/describe-collection
        """
        return self._get(collection_id=collection_id)

    def load_collection(self, collection_id: str, viewing_parameters: dict) -> ImageCollection:
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
    process: dict  # TODO: also encapsulate this "process graph with metadata" struct (instead of free-form dict)?
    status: str
    created: datetime

    # Optional fields (with default)
    job_options: dict = None
    title: str = None
    description: str = None
    progress: float = None
    updated: datetime = None
    plan = None
    costs = None
    budget = None
    started: datetime = None
    finished: datetime = None
    memory_time_megabyte: timedelta = None
    cpu_time: timedelta = None

    @property
    def duration(self) -> timedelta:
        """Returns the job duration if possible, else None."""
        return self.finished - self.started if self.started and self.finished else None

    def prepare_for_json(self) -> dict:
        """Prepare metadata for JSON serialization"""
        d = self._asdict()  # pylint: disable=no-member
        d["created"] = rfc3339.datetime(self.created) if self.created else None
        d["updated"] = rfc3339.datetime(self.updated) if self.updated else None
        d["duration_seconds"] = int(round(self.duration.total_seconds())) if self.duration else None
        d["duration_human_readable"] = str(self.duration) if self.duration else None
        d["memory_time_megabyte_seconds"] = int(round(self.memory_time_megabyte.total_seconds())) if self.memory_time_megabyte else None
        d["memory_time_human_readable"] = "{s:.0f} MB-seconds".format(s=self.memory_time_megabyte.total_seconds()) if self.memory_time_megabyte else None
        d["cpu_time_seconds"] = int(round(self.cpu_time.total_seconds())) if self.cpu_time else None
        d["cpu_time_human_readable"] = "{s:.0f} cpu-seconds".format(s=self.cpu_time.total_seconds()) if self.cpu_time else None
        return d


class BatchJobs(MicroService):
    """
    Base contract/implementation for Batch Jobs "microservice"
    https://openeo.org/documentation/1.0/developers/api/reference.html#operation/stop-job
    """

    def create_job(self, user_id: str, process: dict, api_version: str, job_options: dict = None) -> BatchJobMetadata:
        raise NotImplementedError

    def get_job_info(self, job_id: str, user_id: str) -> BatchJobMetadata:
        """
        Get details about a batch job
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/describe-job
        Should raise `JobNotFoundException` on invalid job/user id
        """
        raise NotImplementedError

    def get_user_jobs(self, user_id: str) -> List[BatchJobMetadata]:
        """
        Get details about all batch jobs of a user
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-jobs
        """
        raise NotImplementedError

    def start_job(self, job_id: str, user_id: str):
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/start-job
        """
        raise NotImplementedError

    def get_results(self, job_id: str, user_id: str) -> Dict[str, str]:
        """
        Return result files as (filename, output_dir) mapping: `filename` is the part that
        the user can see (in download url), `output_dir` is internal (root) dir where
        output is stored.

        related:
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-results
        """
        # TODO: #EP-3281 not only return asset path, but also media type, description, ...
        raise NotImplementedError

    def get_log_entries(self, job_id: str, user_id: str, offset: str) -> List[dict]:
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/debug-job
        """
        raise NotImplementedError

    def cancel_job(self, job_id: str, user_id: str):
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/stop-job
        """
        raise NotImplementedError


class OidcProvider(NamedTuple):
    """OIDC provider metadata"""
    id: str
    issuer: str
    scopes: List[str]
    title: str
    description: str = None


class UserDefinedProcessMetadata(NamedTuple):
    """
    Container for user-defined process metadata.
    """
    id: str
    process_graph: dict
    parameters: List[dict] = None
    public: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> 'UserDefinedProcessMetadata':
        return UserDefinedProcessMetadata(
            id=d['id'],
            process_graph=d['process_graph'],
            parameters=d.get('parameters'),
            public=d.get('public', False)
        )

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


class OpenEoBackendImplementation:
    """
    Simple container of all openEo "microservices"
    """

    def __init__(
            self,
            secondary_services: SecondaryServices,
            catalog: CollectionCatalog,
            batch_jobs: BatchJobs,
            user_defined_processes: UserDefinedProcesses
    ):
        self.secondary_services = secondary_services
        self.catalog = catalog
        self.batch_jobs = batch_jobs
        self.user_defined_processes = user_defined_processes

    def health_check(self) -> str:
        return "OK"

    def oidc_providers(self) -> List[OidcProvider]:
        return []

    def file_formats(self) -> dict:
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-file-types
        """
        return {"input": {}, "output": {}}

    def load_disk_data(self, format: str, glob_pattern: str, options: dict, viewing_parameters: dict) -> object:
        # TODO: move this to catalog "microservice"
        raise NotImplementedError

    def visit_process_graph(self, process_graph: dict) -> ProcessGraphVisitor:
        """Create a process graph visitor and accept given process graph"""
        return ProcessGraphVisitor().accept_process_graph(process_graph)

    def summarize_exception(self, error: Exception) -> Union[ErrorSummary, Exception]:
        return error


_backend_implementation = None


def get_backend_implementation() -> OpenEoBackendImplementation:
    global _backend_implementation  # pylint: disable=global-statement
    if _backend_implementation is None:
        # TODO: #36 avoid non-standard importing through env var DRIVER_IMPLEMENTATION_PACKAGE
        _driver_implementation_package = os.getenv('DRIVER_IMPLEMENTATION_PACKAGE', "openeo_driver.dummy.dummy_backend")
        logger.info('Using driver implementation package {d}'.format(d=_driver_implementation_package))
        module = importlib.import_module(_driver_implementation_package)
        _backend_implementation = module.get_openeo_backend_implementation()
    return _backend_implementation
