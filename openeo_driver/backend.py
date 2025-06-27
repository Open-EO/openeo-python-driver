"""

Base structure of a backend implementation (e.g. Geotrellis based)
to be exposed with HTTP REST frontend.

It is organised in microservice-like parts following https://open-eo.github.io/openeo-api/apireference/
to allow composability, isolation and better reuse.

Also see https://github.com/Open-EO/openeo-python-driver/issues/8
"""
from __future__ import annotations
import abc
import json
import logging
import dataclasses
import inspect
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Union, NamedTuple, Dict, Optional, Callable, Iterable

import flask

import openeo_driver.util.view_helpers
from openeo.utils.version import ComparableVersion
from openeo.internal.process_graph_visitor import ProcessGraphVisitor
import openeo.udf
from openeo.util import rfc3339, dict_no_none
from openeo_driver.config import OpenEoBackendConfig, get_backend_config
from openeo_driver.datacube import DriverDataCube, DriverMlModel, DriverVectorCube
from openeo_driver.datastructs import SarBackscatterArgs
from openeo_driver.dry_run import SourceConstraint
from openeo_driver.errors import CollectionNotFoundException, ServiceUnsupportedException, FeatureUnsupportedException
from openeo_driver.constants import JOB_STATUS, DEFAULT_LOG_LEVEL_RETRIEVAL
from openeo_driver.processes import ProcessRegistry
from openeo_driver.users import User
from openeo_driver.users.oidc import OidcProvider
from openeo_driver.util.date_math import simple_job_progress_estimation
from openeo_driver.util.logging import just_log_exceptions
from openeo_driver.utils import read_json, dict_item, EvalEnv, extract_namedtuple_fields_from_dict, \
    get_package_versions

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


def is_not_implemented(f: Union[Callable, None]):
    """Checker for functions/methods that are not implemented"""
    return f is None or (hasattr(f, "_not_implemented") and f._not_implemented)


class ServiceMetadata(NamedTuple):
    """
    Container for service metadata
    """
    # TODO: move this to openeo-python-client?
    # TODO: also add user metadata?

    # Required fields (no default)
    id: str
    url: str
    type: str
    process: dict = None  # TODO: also encapsulate this "process graph with metadata" struct (instead of free-form dict)?
    enabled: bool = True  # TODO: required or with default True? https://github.com/Open-EO/openeo-api/issues/473
    attributes: dict = None
    configuration: dict = None

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

    def get_log_entries(
        self,
        service_id: str,
        *,
        user_id: str,
        offset: Union[str, None] = None,
        limit: Union[int, None] = None,
        level: str = DEFAULT_LOG_LEVEL_RETRIEVAL,
    ) -> List[dict]:
        """https://openeo.org/documentation/1.0/developers/api/reference.html#operation/debug-service"""
        # TODO require auth/user handle?
        return []


class LoadParameters(dict):
    """
    Dictionary based container for load_collection related parameters and optimization hints.
    """
    # TODO: these objects are part of the cache key for load_collection's lru_cache so mutating them will cause both
    #       unwanted cache hits and unwanted cache misses; can we make them immutable like EvalEnv? #140

    # Note: these `dict_item`s provide attribute-style access to dictionary items
    # which simplifies discovery where certain parameters are written/read
    # (and allows setting defaults centrally).
    # TODO: use a more standard solution like dataclassses from stdlib or attrs?
    temporal_extent = dict_item(default=(None, None))
    spatial_extent = dict_item(default={})
    global_extent = dict_item(default={})
    filter_temporal_labels = dict_item(default=None)
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
    resample_method = dict_item(default="near")

    """
    A buffer provided in the units of the target CRS. If target CRS is not provided, then it is assumed to be the native CRS
    of the collection.

    This buffer is applied to AOI when constructing the datacube, allowing operations that require neighbouring pixels
    to be implemented correctly. Examples are apply_kernel and apply_neighborhood, but also certain resampling operations
    could be affected by this.

    The buffer has to be considered in the global extent!
    """
    pixel_buffer = dict_item(default=None)

    def copy(self) -> "LoadParameters":
        return LoadParameters(super().copy())

    def __hash__(self) -> int:
        return 0  # poorly hashable but load_collection's lru_cache is small anyway

    def _copy_for_eq(self):
        copy = super().copy()
        #load_collection with or without data mask should return an 'equivalent' result
        copy["data_mask"] = {}
        return copy

    def __eq__(self, o: object) -> bool:
        if not isinstance(o,LoadParameters):
            return False
        if self.sar_backscatter != o.sar_backscatter:
            logger.info(f"load_collection: sar_backscatter not equal { self.sar_backscatter } != { o.sar_backscatter }")
            return False
        copy1 = self._copy_for_eq()
        copy2 = o._copy_for_eq()
        copy1["sar_backscatter"] = None
        copy2["sar_backscatter"] = None
        return copy1 == copy2


class CollectionsListing:
    # TODO #377 pagination support
    def __init__(self, collections: List[dict]):
        """
        :param collections: list of collection metadata dictionaries
        """
        self.collections = collections

    def filter_by_id(self, exclusion_list: Iterable[str]) -> CollectionsListing:
        return CollectionsListing(collections=[c for c in self.collections if c["id"] not in exclusion_list])

    def to_response_dict(self, normalize: Callable[[dict], dict]) -> dict:
        collections = [normalize(c) for c in self.collections]
        return {
            "collections": collections,
            "links": [],
        }

class AbstractCollectionCatalog(MicroService, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_all_metadata(self) -> List[dict]:
        """Basic metadata for all collections"""
        # TODO: deprecate/remove this in favor of get_collections_listing?
        ...

    def get_collections_listing(self) -> CollectionsListing:
        # TODO make this an abstract method once `get_all_metadata` has been eliminated
        return CollectionsListing(collections=self.get_all_metadata())

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

    process: Optional[dict] = None  # TODO: better encapsulation of this "process graph with metadata" structure?
    job_options: Optional[dict] = None
    title: Optional[str] = None
    description: Optional[str] = None
    # Progress as percent value: 0.0 (just started) - 100.0 (fully completed)
    progress: Optional[float] = None
    updated: Optional[datetime] = None
    plan: Optional[str] = None
    costs: Optional[float] = None
    budget: Optional[float] = None
    started: Optional[datetime] = None
    finished: Optional[datetime] = None
    # TODO #191 Deprecated in favor of general "usage" field
    duration_: Optional[timedelta] = None
    # TODO #191 Deprecated in favor of general "usage" field
    memory_time_megabyte: Optional[timedelta] = None
    # TODO #191 Deprecated in favor of general "usage" field
    cpu_time: Optional[timedelta] = None
    # TODO #190 most fields below are not batch job metadata, but batch job *result* metadata:
    #      move these to BatchJobResultMetadata for better separation of concerns
    geometry: Optional[dict] = None
    bbox: Optional[List[float]] = None
    # TODO: #190 start_datetime is actually not metadata of batch job itself, but of the result (assets)
    start_datetime: Optional[datetime] = None
    # TODO: #190 end_datetime is actually not metadata of batch job itself, but of the result (assets)
    end_datetime: Optional[datetime] = None
    instruments: Optional[List[str]] = None
    epsg: Optional[int] = None
    # TODO: #190 openEO API associates `links` with the job *result* metadata, not the job itself
    links: Optional[List[Dict]] = None
    usage: Optional[Dict] = None
    # TODO #190 the STAC projection extension fields "proj:..." are not batch job metadata, but batch job *result* metadata:
    proj_shape: Optional[List[int]] = None
    proj_bbox: Optional[List[int]] = None

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
            # TODO #191 All these fields are deprecated in favor of general "usage" field
            if usage.get("cpu"):
                if usage["cpu"]["unit"] == "cpu-seconds":
                    kwargs["cpu_time"] = timedelta(seconds=usage["cpu"]["value"])
            if usage.get("memory"):
                if usage["memory"]["unit"] == "mb-seconds":
                    kwargs["memory_time_megabyte"] = timedelta(seconds=usage["memory"]["value"])
            if usage.get("duration"):
                if usage["duration"]["unit"] == "seconds":
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

        progress = self.progress
        if (
            self.status == JOB_STATUS.RUNNING
            and self.started
            and progress is None
            and get_backend_config().simple_job_progress_estimation
        ):
            # TODO: is there a cleaner place to inject this fallback progress estimation?
            with just_log_exceptions(log=logger, name="simple_job_progress_estimation"):
                progress = 100 * simple_job_progress_estimation(
                    started=self.started, average_run_time=get_backend_config().simple_job_progress_estimation
                )
        # Clamp "progress" for certain "status" values according to the spec.
        progress = {"created": 0, "queued": 0, "finished": 100}.get(self.status, progress)
        result["progress"] = progress

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
            if result["status"] == JOB_STATUS.CREATED:
                result["status"] = "submitted"

        return dict_no_none(result)


@dataclasses.dataclass(frozen=True)
class BatchJobResultMetadata:
    # Basic dataclass based wrapper for batch job result metadata (allows cleaner code navigation and discovery)
    assets: Dict[str, dict] = dataclasses.field(default_factory=dict)
    links: List[dict] = dataclasses.field(default_factory=list)
    providers: List[dict] = dataclasses.field(default_factory=list)
    # TODO: more fields


class JobListing:
    """
    Basic container/interface for user job listings,
    (e.g. including pagination)
    """

    # TODO #332 just implement as frozen dataclass, or does that conflict with later need to subclass?
    __slots__ = ["_jobs", "_next_parameters"]

    def __init__(self, jobs: List[BatchJobMetadata], next_parameters: Optional[dict] = None):
        """

        :param jobs: list of job metadata constructs
        :param next_parameters: dictionary of URL parameters to add to `GET /jobs` URL to link to next page
        """
        self._jobs = jobs
        self._next_parameters = next_parameters

    def to_response_dict(self, build_url: Callable[[dict], str], api_version: ComparableVersion = None) -> dict:
        """
        Produce `GET /jobs` response data, to be JSONified.

        :param build_url: function to generate a paginated" URL from given pagination related parameters,
            e.g. `lambda params: flask.url_for(".list_jobs", **params, _external=True)`
        """
        links = []
        if self._next_parameters:
            links.append({"rel": "next", "href": build_url(self._next_parameters)})

        return {
            "jobs": [m.to_api_dict(full=False, api_version=api_version) for m in self._jobs],
            "links": links,
        }

    def __len__(self) -> int:
        return len(self._jobs)


class BatchJobs(MicroService):
    """
    Base contract/implementation for Batch Jobs "microservice"
    https://openeo.org/documentation/1.0/developers/api/reference.html#operation/stop-job
    """

    ASSET_PUBLIC_HREF = "public_href"

    def __init__(self):
        # TODO #275 this "proxy_user" feature is YARN (VITO) specific. Move it to oppeno-geopyspark-driver?
        self._get_proxy_user: Callable[[User], Optional[str]] = lambda user: None

    def create_job(
        self,
        *,
        user: User,
        process: dict,
        api_version: str,
        metadata: dict,
        job_options: Optional[dict] = None,
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

    def get_user_jobs(
        self,
        user_id: str,
        limit: Optional[int] = None,
        request_parameters: Optional[dict] = None,
        # TODO #332 settle on returning just `JobListing` and eliminate other options/code paths.
    ) -> Union[List[BatchJobMetadata], dict, JobListing]:
        """
        Get details about all batch jobs of a user
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-jobs

        :param limit: limit parameter in request (as defined by openEO API spec)
        :param request_parameters: all request parameters.
            Note that backend implementation has freedom here
            how to leverage these to implement pagination
        """
        raise NotImplementedError

    def start_job(self, job_id: str, user: User):
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/start-job
        """
        raise NotImplementedError

    def get_result_metadata(self, job_id: str, user_id: str) -> BatchJobResultMetadata:
        """
        Get job result metadata

        https://openeo.org/documentation/1.0/developers/api/reference.html#tag/Batch-Jobs/operation/list-results
        """
        # Default implementation, based on existing components
        return BatchJobResultMetadata(
            assets=self.get_result_assets(job_id=job_id, user_id=user_id),
            links=[],
            providers=self._get_providers(job_id=job_id, user_id=user_id),
        )

    def _get_providers(self, job_id: str, user_id: str):
        """Creates a standard value for the STAC providers object.

        Helper method for default implementation of `get_result_metadata`.

        For an example implementation, see:
           openeo_driver.dummy.dummy_backend.DummyBatchJobs._get_providers

        """
        config: OpenEoBackendConfig = get_backend_config()
        job: BatchJobMetadata = self.get_job_info(job_id=job_id, user_id=user_id)
        return [
            {
                "name": config.capabilities_title,
                "description": config.capabilities_description,
                "roles": ["processor"],
                "processing:facility": config.processing_facility,
                "processing:software": {config.processing_software: config.capabilities_backend_version},
                "processing:expression": {"format": "openeo", "expression": job.process},
            }
        ]

    def get_result_assets(self, job_id: str, user_id: str) -> Dict[str, dict]:
        """
        Return result assets as (filename, metadata) mapping: `filename` is the part that
        the user can see (in download url), `metadata` contains internal (root) dir where
        output is stored.

        related:
        https://openeo.org/documentation/1.0/developers/api/reference.html#tag/Batch-Jobs/operation/list-results
        """
        # Default implementation, based on legacy API
        return self.get_results(job_id=job_id, user_id=user_id)

    def get_results(self, job_id: str, user_id: str) -> Dict[str, dict]:
        """
        Deprecated: use/implement `get_result_assets` instead.
        """
        # TODO: eliminate this method in favor of `get_result_assets`
        raise NotImplementedError

    def get_log_entries(
        self,
        job_id: str,
        *,
        user_id: str,
        offset: Union[str, None] = None,
        limit: Union[str, None] = None,
        level: str = DEFAULT_LOG_LEVEL_RETRIEVAL,
    ) -> Iterable[dict]:
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

    def get_proxy_user(self, user: User) -> Optional[str]:
        # TODO #275 this "proxy_user" feature is YARN (VITO) specific. Move it to oppeno-geopyspark-driver?
        return self._get_proxy_user(user)

    def set_proxy_user_getter(self, getter: Callable[[User], Optional[str]]):
        # TODO #275 this "proxy_user" feature is YARN (VITO) specific. Move it to oppeno-geopyspark-driver?
        self._get_proxy_user = getter


class UserDefinedProcessMetadata(NamedTuple):
    """
    Container for user-defined process metadata.
    """

    # TODO: generalize this to generic openEO process definitions
    #       (not only "user-defined", but also remote/public process definitions)

    id: str
    # Note: "process_graph" is optional for multiple UDP listings (`GET /process_graphs`),
    # but required for full, single UDP metadata requests (`GET /process_graphs/{process_graph_id}`)
    process_graph: Optional[dict] = None
    parameters: List[dict] = None
    returns: Optional[dict] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    links: Optional[list] = None
    public: bool = False  # Note: experimental non-standard flag

    @classmethod
    def from_dict(cls, d: dict) -> UserDefinedProcessMetadata:
        d = extract_namedtuple_fields_from_dict(d, UserDefinedProcessMetadata)
        return cls(**d)

    def prepare_for_json(self) -> dict:
        return self._asdict()  # pylint: disable=no-member

    def add_link(self, *, rel: str, href: str, title: Optional[str] = None):
        """Create new UserDefinedProcessMetadata with added link"""
        kwargs = self._asdict()
        kwargs["links"] = (kwargs["links"] or []) + [dict_no_none(rel=rel, href=href, title=title)]
        return UserDefinedProcessMetadata(**kwargs)

    def to_api_dict(self, *, full: bool = True) -> dict:
        """Produce openEO API-style UDP dict construct (to be jsonified)."""
        d = self.prepare_for_json()
        if not full:
            # API recommends to limit response size by omitting larger/optional fields
            d = {k: v for k, v in d.items() if k in ["id", "summary", "description", "parameters", "returns"]}
        if "public" in d and not d["public"]:
            # Don't include non-standard "public" field when false to stay closer to standard API
            del d["public"]

        return dict_no_none(**d)


class UserDefinedProcessesListing:
    # TODO #377 pagination support
    def __init__(self, udps: List[UserDefinedProcessMetadata]):
        self.udps = udps

    def to_response_dict(self, full: bool = True) -> dict:
        return {
            "processes": [udp.to_api_dict(full=full) for udp in self.udps],
            "links": [],
        }


class UserDefinedProcesses(MicroService):
    """
    Base contract/implementation for User-Defined Processes "microservice"
    https://openeo.org/documentation/1.0/developers/api/reference.html#tag/User-Defined-Processes
    """

    def get(self, user_id: str, process_id: str) -> Union[UserDefinedProcessMetadata, None]:
        """
        Fetch metadata of given UDP. Return None if not found.
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/describe-custom-process
        """
        raise NotImplementedError

    def get_for_user(self, user_id: str) -> List[UserDefinedProcessMetadata]:
        """
        List user's UDPs
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-custom-processes
        """
        # TODO: eliminate this legacy API method
        raise NotImplementedError

    def list_for_user(self, user_id: str) -> UserDefinedProcessesListing:
        # TODO: replace this adapter implementation with `raise NotImplementedError`
        #       once `get_for_user` can be eliminated
        logger.warning("Falling back on deprecated `get_for_user`. Backend should implement `list_for_user` instead.")
        udps = self.get_for_user(user_id=user_id)
        return UserDefinedProcessesListing(udps=udps)

    def save(self, user_id: str, process_id: str, spec: dict) -> None:
        """
        Store new UDP
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/store-custom-process
        """
        raise NotImplementedError

    def delete(self, user_id: str, process_id: str) -> None:
        """
        Remove UDP
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

    def run_udf(self, udf: str, data: openeo.udf.UdfData) -> openeo.udf.UdfData:
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
    # TODO: move listing of non-generic libs to openeo-geopyspark-driver
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
        "tensorflow",
        "pytorch",
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

    # Overridable vector cube implementation
    vector_cube_cls = DriverVectorCube

    DEFAULT_CONFORMANCE_CLASSES = [
        # general openEO conformance class
        "https://api.openeo.org/1.2.0",
        # Support the "remote process definition" extension (originally known as the "remote-udp" extension)
        "https://api.openeo.org/extensions/remote-process-definition/0.1.0",
        # Processing Parameters extension
        "https://api.openeo.org/extensions/processing-parameters/0.1.0",
        # STAC API conformance classes
        # "https://api.stacspec.org/v1.0.0/core",  # TODO #363 can we claim this conformance class already?
        "https://api.stacspec.org/v1.0.0/collections",
    ]

    def __init__(
        self,
        *,
        secondary_services: Optional[SecondaryServices] = None,
        catalog: Optional[AbstractCollectionCatalog] = None,
        batch_jobs: Optional[BatchJobs] = None,
        user_defined_processes: Optional[UserDefinedProcesses] = None,
        processing: Optional[Processing] = None,
        config: Optional[OpenEoBackendConfig] = None,
        conformance_classes: Optional[List[str]] = None,
    ):
        self.config: OpenEoBackendConfig = config or get_backend_config()
        self.secondary_services = secondary_services
        self.catalog = catalog
        self.batch_jobs = batch_jobs
        self.user_defined_processes = user_defined_processes
        self.user_files = None  # TODO: implement user file storage microservice
        self.processing = processing
        self.udf_runtimes = UdfRuntimes()
        self._conformance_classes = conformance_classes or self.DEFAULT_CONFORMANCE_CLASSES

        # Overridable cache control header injecting decorator for static, public view functions
        self.cache_control = openeo_driver.util.view_helpers.cache_control(
            max_age=timedelta(minutes=15), public=True,
        )

    def health_check(self, options: Optional[dict] = None) -> Union[str, dict, flask.Response]:
        return "OK"

    def oidc_providers(self) -> List[OidcProvider]:
        return self.config.oidc_providers

    def conformance_classes(self) -> List[str]:
        return self._conformance_classes

    def file_formats(self) -> dict:
        """
        https://openeo.org/documentation/1.0/developers/api/reference.html#operation/list-file-types
        """
        return {"input": {}, "output": {}}

    def processing_parameters(self) -> list:
        """
        List the available processing parameters in the format defined by openEO processing parameters extension.

        https://github.com/Open-EO/openeo-api/blob/draft/extensions/processing-parameters/openapi.yaml

        Example output:
        ```
        [{
          "name": "memory"
          "description": "Maximum amount of memory that will be allocated for processing, in gigabytes."
          "optional": True
          "default": 32
          "schema":{
            "type": "integer"
            "minimum": 1
          }
        }]
        ```
        """
        return []

    def load_disk_data(
            self, format: str, glob_pattern: str, options: dict, load_params: LoadParameters, env: EvalEnv
    ) -> DriverDataCube:
        # TODO: move this to catalog "microservice"
        # TODO: rename this to "load_uploaded_files" like in official openeo processes
        raise NotImplementedError

    def load_result(self, job_id: str, user_id: Optional[str], load_params: LoadParameters,
                    env: EvalEnv) -> DriverDataCube:
        raise NotImplementedError

    def load_stac(self, url: str, load_params: LoadParameters, env: EvalEnv) -> DriverDataCube:
        raise NotImplementedError

    def load_ml_model(self, job_id: str) -> DriverMlModel:
        raise NotImplementedError

    def vector_to_raster(self, input_vector_cube: DriverVectorCube, target: DriverDataCube) -> DriverDataCube:
        raise NotImplementedError

    def visit_process_graph(self, process_graph: dict) -> ProcessGraphVisitor:
        """Create a process graph visitor and accept given process graph"""
        return ProcessGraphVisitor().accept_process_graph(process_graph)

    def summarize_exception(self, error: Exception) -> Union[ErrorSummary, Exception]:
        return error

    def set_preferred_username_getter(self, getter: Callable[[User], Optional[str]]):
        # TODO #275 this "proxy_user" feature is YARN/Spark/VITO specific. Move it to oppeno-geopyspark-driver?
        # TODO this method talks about "preferred_username" but is currently used to propagate
        #      the YARN proxy_user (which must be unique and does align very well with a user chosen preferred username)
        #      In other cases (eu-cdse/openeo-cdse-infra#56) we might generate the YARN proxy_user
        #      in an even more constrained way, completely separate from any preferred username.
        #      In short: this method's name might set wrong expectations.
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

    def changelog(self) -> Union[str, Path, flask.Response]:
        return "No changelog"

    def set_request_id(self, request_id: str):
        pass

    def after_request(self, request_id: str):
        pass

    def request_costs(
        self,
        *,
        user: User,
        job_options: Union[dict, None] = None,
        request_id: str,
        success: bool,
    ) -> Optional[float]:
        """
        Report resource usage of (current) synchronous processing request and get associated cost.

        :param user: user object
        :param job_options: job options (if any provided in the sync processing request)
        """
        return None


def function_has_argument(function: Callable, argument: str) -> bool:
    """Does function support given argument?"""
    signature = inspect.signature(function)
    return argument in signature.parameters
