"""

Base structure of a backend implementation (e.g. Geotrellis based)
to be exposed with HTTP REST frontend.

It is organised in microservice-like parts following https://open-eo.github.io/openeo-api/apireference/
to allow composability, isolation and better reuse.

Also see https://github.com/Open-EO/openeo-python-driver/issues/8
"""

import copy
import importlib
import os
import warnings
import logging
from pathlib import Path
from typing import List, Tuple, Union, NamedTuple, Dict
from datetime import datetime

from openeo import ImageCollection
from openeo.error_summary import ErrorSummary
from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo.util import date_to_rfc3339
from openeo_driver.errors import OpenEOApiException, CollectionNotFoundException
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
        d = self._asdict()
        d["created"] = date_to_rfc3339(self.created) if self.created else None
        return d

    @classmethod
    def from_dict(cls, d:dict) -> 'ServiceMetadata':
        """Load ServiceMetadata from dict (e.g. parsed JSON dump)."""
        created = d.get("created")
        if isinstance(created, str):
            d = d.copy()
            d["created"] = datetime.strptime(created, '%Y-%m-%dT%H:%M:%SZ')
        return cls(**d)


class SecondaryServices(MicroService):
    """
    Base contract/implementation for Secondary Services "microservice"
    https://openeo.org/documentation/1.0/developers/api/reference.html#tag/Secondary-Services
    """

    def service_types(self) -> dict:
        """https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1service_types/get"""
        return {}

    def list_services(self) -> List[ServiceMetadata]:
        """https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1services/get"""
        return []

    def service_info(self, service_id: str) -> ServiceMetadata:
        """https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1services~1{service_id}/get"""
        raise NotImplementedError()

    def create_service(self, process_graph: dict, service_type: str, api_version: str, post_data: dict) -> ServiceMetadata:
        """
        https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1services/post
        :return: (location, openeo_identifier)
        """
        from openeo_driver.ProcessGraphDeserializer import evaluate
        # TODO require auth/user handle?
        if service_type.lower() not in set(st.lower() for st in self.service_types()):
            raise OpenEOApiException(
                message="Secondary service type {t!r} is not supported.".format(t=service_type),
                code="ServiceUnsupported", status_code=400
            )

        image_collection = evaluate(process_graph, viewingParameters={'version': api_version})
        service_metadata = image_collection.tiled_viewing_service(
            service_type=service_type,
            process_graph=process_graph,
            post_data=post_data
        )
        return service_metadata

    def update_service(self, service_id: str, process_graph: dict) -> None:
        """https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1services~1{service_id}/patch"""
        # TODO require auth/user handle?
        raise NotImplementedError()

    def remove_service(self, service_id: str) -> None:
        """https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1services~1{service_id}/delete"""
        # TODO require auth/user handle?
        raise NotImplementedError()

    # TODO https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1subscription/get


class CollectionCatalog(MicroService):
    """
    Basic implementation of a catalog of collections/EO data
    """

    _stac_version = "0.7.0"

    def __init__(self, all_metadata: List[dict]):
        self._catalog = {layer["id"]: dict(layer) for layer in all_metadata}

    @classmethod
    def from_json_file(cls, filename: Union[str, Path] = "layercatalog.json", *args, **kwargs):
        """Factory to read catalog from a JSON file"""
        return cls(read_json(filename), *args, **kwargs)

    def assert_collection_id(self, collection_id):
        if collection_id not in self._catalog:
            raise CollectionNotFoundException(collection_id)

    def get_all_metadata(self) -> List[dict]:
        """
        Basic metadata for all datasets
        https://open-eo.github.io/openeo-api/apireference/#tag/EO-Data-Discovery/paths/~1collections/get
        :return:
        """
        return [self._normalize_layer_metadata(m) for m in self._catalog.values()]

    def get_collection_metadata(self, collection_id) -> dict:
        """
        Full metadata for a specific dataset
        https://open-eo.github.io/openeo-api/apireference/#tag/EO-Data-Discovery/paths/~1collections~1{collection_id}/get
        :param collection_id:
        :return:
        """

        self.assert_collection_id(collection_id)
        return self._normalize_layer_metadata(self._catalog[collection_id])

    def load_collection(self, collection_id: str, viewing_parameters: dict) -> ImageCollection:
        raise NotImplementedError

    def _normalize_layer_metadata(self, metadata: dict) -> dict:
        """
        Make sure the given layer metadata roughly complies to OpenEO spec.
        """
        metadata = copy.deepcopy(metadata)

        # Metadata should at least contain an id.
        collection_id = metadata["id"]

        # Make sure required fields are set.
        metadata.setdefault("stac_version", self._stac_version)
        metadata.setdefault("links", [])
        metadata.setdefault("other_properties", {})
        # Warn about missing fields where sensible defaults are not feasible
        fallbacks = {
            "description": "Description of {c} (#TODO)".format(c=collection_id),
            "license": "proprietary",
            "extent": {"spatial": [0, 0, 0, 0], "temporal": [None, None]},
            "properties": {"cube:dimensions": {}},
        }
        for key, value in fallbacks.items():
            if key not in metadata:
                warnings.warn(
                    "Collection {c} is missing required metadata field {k!r}.".format(c=collection_id, k=key),
                    category=CollectionIncompleteMetadataWarning
                )
                metadata[key] = value

        return metadata


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
    title: str = None
    description: str = None
    progress: float = None
    updated: datetime = None
    plan = None
    costs = None
    budget = None

    def prepare_for_json(self) -> dict:
        """Prepare metadata for JSON serialization"""
        d = self._asdict()
        d["created"] = date_to_rfc3339(self.created) if self.created else None
        d["updated"] = date_to_rfc3339(self.updated) if self.updated else None
        return d


class BatchJobs(MicroService):
    """
    Base contract/implementation for Batch Jobs "microservice"
    https://openeo.org/documentation/1.0/developers/api/reference.html#operation/stop-job
    """

    def create_job(self, user_id: str, job_specification: dict, api_version: str) -> BatchJobMetadata:
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

    def cancel_job(self, job_id:str, user_id:str):
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/stop-job
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
    ):
        self.secondary_services = secondary_services
        self.catalog = catalog
        self.batch_jobs = batch_jobs

    def health_check(self) -> str:
        return "OK"

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
    global _backend_implementation
    if _backend_implementation is None:
        # TODO: #36 avoid non-standard importing through env var DRIVER_IMPLEMENTATION_PACKAGE
        _driver_implementation_package = os.getenv('DRIVER_IMPLEMENTATION_PACKAGE', "dummy_impl")
        logger.info('Using driver implementation package {d}'.format(d=_driver_implementation_package))
        module = importlib.import_module(_driver_implementation_package)
        _backend_implementation = module.get_openeo_backend_implementation()
    return _backend_implementation
