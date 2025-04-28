from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import os
import pprint
import shlex
import textwrap
import time
import typing
import urllib.parse
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Union

import requests
import reretry
from deprecated.classic import deprecated
from openeo.util import TimingLogger, repr_truncate, rfc3339, url_join

import openeo_driver._version
from openeo_driver.backend import BatchJobMetadata, JobListing
from openeo_driver.config import get_backend_config
from openeo_driver.constants import JOB_STATUS
from openeo_driver.errors import InternalException, JobNotFoundException
from openeo_driver.util.auth import ClientCredentials, ClientCredentialsAccessTokenHelper
from openeo_driver.util.logging import ExtraLoggingFilter
from openeo_driver.utils import generate_unique_id

_log = logging.getLogger(__name__)



class PARTIAL_JOB_STATUS:
    RUNNING = "running"
    CANCELED = "canceled"
    FINISHED = "finished"
    ERROR = "error"

    _job_status_to_partial_job_status_mapping = {
        JOB_STATUS.CREATED: RUNNING,
        JOB_STATUS.QUEUED: RUNNING,
        JOB_STATUS.RUNNING: RUNNING,
        JOB_STATUS.CANCELED: CANCELED,
        JOB_STATUS.FINISHED: FINISHED,
        JOB_STATUS.ERROR: ERROR
    }

    @staticmethod
    def for_job_status(job_status: str) -> str:
        try:
            return PARTIAL_JOB_STATUS._job_status_to_partial_job_status_mapping[job_status]
        except KeyError:
            raise AssertionError(f"unexpected job status: {job_status!r}")


class DEPENDENCY_STATUS:
    """
    Container of dependency status constants
    """

    # TODO #153: this is specific for SentinelHub batch process tracking in openeo-geopyspark-driver.
    #   Can this be moved there?

    AWAITING = "awaiting"
    AVAILABLE = "available"
    AWAITING_RETRY = "awaiting_retry"
    ERROR = "error"


# Just a type alias for now
JobDict = Dict[str, Any]


class JobRegistryInterface:
    """Base interface for job registries"""

    @staticmethod
    def generate_job_id() -> str:
        return generate_unique_id(prefix="j")

    def create_job(
        self,
        *,
        process: dict,
        user_id: str,
        job_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        parent_id: Optional[str] = None,
        api_version: Optional[str] = None,
        job_options: Optional[dict] = None,
    ) -> JobDict:
        raise NotImplementedError

    def get_job(self, job_id: str, *, user_id: Optional[str] = None) -> JobDict:
        raise NotImplementedError

    def delete_job(self, job_id: str, *, user_id: Optional[str] = None) -> None:
        raise NotImplementedError

    def set_status(
        self,
        job_id: str,
        *,
        user_id: Optional[str] = None,
        status: str,
        updated: Optional[str] = None,
        started: Optional[str] = None,
        finished: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    def set_dependencies(
        self, job_id: str, *, user_id: Optional[str] = None, dependencies: List[Dict[str, str]]
    ) -> None:
        raise NotImplementedError

    def remove_dependencies(self, job_id: str, *, user_id: Optional[str] = None) -> None:
        raise NotImplementedError

    def set_dependency_status(self, job_id: str, *, user_id: Optional[str] = None, dependency_status: str) -> None:
        raise NotImplementedError

    def set_dependency_usage(self, job_id: str, *, user_id: Optional[str] = None, dependency_usage: Decimal) -> None:
        raise NotImplementedError

    def set_proxy_user(self, job_id: str, *, user_id: Optional[str] = None, proxy_user: str) -> None:
        # TODO #275 this "proxy_user" is a pretty implementation (YARN/VITO) specific field. Generalize this in some way?
        raise NotImplementedError

    def set_application_id(self, job_id: str, *, user_id: Optional[str] = None, application_id: str) -> None:
        raise NotImplementedError

    # TODO: improve name?
    def set_results_metadata(
        self,
        job_id: str,
        *,
        user_id: Optional[str] = None,
        costs: Optional[float],
        usage: dict,
        results_metadata: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def list_user_jobs(
        self,
        user_id: str,
        *,
        fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        request_parameters: Optional[dict] = None,
        # TODO #332 settle on returning just `JobListing` and eliminate other options/code paths.
    ) -> Union[JobListing, List[JobDict]]:
        """
        List all jobs of a user

        :param user_id: user id of user to list jobs for
        :param fields: job metadata fields that should be included in result

        :return: list of job metadata dictionaries
            The "process" field should not be included
        """
        raise NotImplementedError

    def list_active_jobs(
        self,
        *,
        fields: Optional[List[str]] = None,
        max_age: Optional[int] = None,
        max_updated_ago: Optional[int] = None,
        require_application_id: bool = False,
    ) -> List[JobDict]:
        """
        List active jobs (created, queued, running)

        :param fields: job metadata fields that should be included in result
        :param max_age: (optional) only return jobs created at most `max_age` days ago
        :param max_updated_ago: (optional) only return jobs `updated` at most `max_updated_ago` days ago
        :param require_application_id: whether to only return jobs with an application_id
        """
        # TODO: option for job metadata fields that should be included in result
        raise NotImplementedError


def ejr_job_info_to_metadata(job_info: JobDict, full: bool = True) -> BatchJobMetadata:
    """Convert job info dict (from JobRegistryInterface) to BatchJobMetadata"""
    # TODO: make this a classmethod in a more appropriate place, e.g. JobRegistryInterface?

    def map_safe(prop: str, f):
        value = job_info.get(prop)
        return f(value) if value else None

    def get_results_metadata(result_metadata_prop: str):
        return job_info.get("results_metadata", {}).get(result_metadata_prop)

    def map_results_metadata_safe(result_metadata_prop: str, f):
        value = get_results_metadata(result_metadata_prop)
        return f(value) if value is not None else None

    return BatchJobMetadata(
        id=job_info["job_id"],
        status=job_info["status"],
        created=map_safe("created", rfc3339.parse_datetime),
        process=job_info.get("process") if full else None,
        job_options=job_info.get("job_options") if full else None,
        title=job_info.get("title"),
        description=job_info.get("description"),
        updated=map_safe("updated", rfc3339.parse_datetime),
        started=map_safe("started", rfc3339.parse_datetime),
        finished=map_safe("finished", rfc3339.parse_datetime),
        memory_time_megabyte=map_safe(
            "memory_time_megabyte_seconds", lambda seconds: datetime.timedelta(seconds=seconds)
        ),
        cpu_time=map_safe("cpu_time_seconds", lambda seconds: datetime.timedelta(seconds=seconds)),
        geometry=get_results_metadata("geometry"),
        bbox=get_results_metadata("bbox"),
        start_datetime=map_results_metadata_safe("start_datetime", rfc3339.parse_datetime),
        end_datetime=map_results_metadata_safe("end_datetime", rfc3339.parse_datetime),
        instruments=get_results_metadata("instruments"),
        epsg=get_results_metadata("epsg"),
        links=get_results_metadata("links"),
        usage=job_info.get("usage"),
        costs=job_info.get("costs"),
        proj_shape=get_results_metadata("proj:shape"),
        proj_bbox=get_results_metadata("proj:bbox"),
    )


class EjrError(Exception):
    """Elastic Job Registry error (base class)."""

    pass


class EjrApiError(EjrError):
    """Generic error when trying/doing an EJR API request."""

    pass


class EjrApiResponseError(EjrApiError):
    """Error deducted from EJR API response"""
    def __init__(self, msg: str, status_code: Optional[int]):
        super().__init__(msg)
        self.status_code = status_code

    @classmethod
    def from_response(cls, response: requests.Response) -> EjrApiResponseError:
        request = response.request
        return cls(
            msg=f"Error communicating with batch job system, consider trying again. Details: {response.status_code} {response.reason!r} on `{request.method} {request.url!r}`: {response.text}",
            status_code=response.status_code,
        )


def get_ejr_credentials_from_env(
    env: Optional[typing.Mapping] = None, *, strict: bool = True
) -> Union[ClientCredentials, None]:
    # TODO only really used in openeo-geopyspark-driver atm
    # TODO Generalize this functionality (map env vars to NamedTuple) in some way?
    env = env or os.environ

    if os.environ.get("OPENEO_EJR_OIDC_CLIENT_CREDENTIALS"):
        creds = ClientCredentials.from_credentials_string(
            os.environ["OPENEO_EJR_OIDC_CLIENT_CREDENTIALS"], strict=strict
        )
        if creds:
            _log.debug("Got EJR credentials from env var (compact style)")
            return creds

    # TODO: deprecate this legacy code path
    env_var_mapping = {
        "oidc_issuer": "OPENEO_EJR_OIDC_ISSUER",
        "client_id": "OPENEO_EJR_OIDC_CLIENT_ID",
        "client_secret": "OPENEO_EJR_OIDC_CLIENT_SECRET",
    }
    try:
        kwargs = {a: env[e] for a, e in env_var_mapping.items()}
    except KeyError:
        if strict:
            missing = set(env_var_mapping.values()).difference(env.keys())
            raise EjrError(f"Failed building {ClientCredentials.__name__} from env: missing {missing!r}") from None
    else:
        _log.debug("Got EJR credentials from env vars (legacy style)")
        return ClientCredentials(**kwargs)


@dataclasses.dataclass(frozen=True)
class _PaginatedSearchResult:
    jobs: List[JobDict]
    next_page: Union[int, None]


class ElasticJobRegistry(JobRegistryInterface):
    """
    (Base)class to manage storage of batch job metadata
    using the Job Registry Elastic API (openeo-job-tracker-elastic-api)
    """

    _REQUEST_TIMEOUT = 20

    # Request parameter used for page in pagination (e.g. of user job listings)
    PAGINATION_URL_PARAM = "page"

    _log = logging.getLogger(f"{__name__}.elastic")

    def __init__(
        self,
        api_url: str,
        backend_id: Optional[str] = None,
        *,
        session: Optional[requests.Session] = None,
        _debug_show_curl: bool = False,
    ):
        if not api_url:
            raise ValueError(api_url)

        self._log.debug(f"Creating ElasticJobRegistry with {backend_id=} and {api_url=}")
        self._backend_id: Optional[str] = backend_id
        self._api_url = api_url
        self._access_token_helper = ClientCredentialsAccessTokenHelper(session=session)

        if session:
            self._session = session
        else:
            self._session = requests.Session()
            self._set_user_agent()

        self._debug_show_curl = _debug_show_curl

    def _set_user_agent(self):
        user_agent = f"openeo_driver-{openeo_driver._version.__version__}/{self.__class__.__name__}"
        if self._backend_id:
            user_agent += f"/{self._backend_id}"
        self._session.headers["User-Agent"] = user_agent

    @property
    def backend_id(self) -> str:
        assert self._backend_id
        return self._backend_id

    def setup_auth_oidc_client_credentials(self, credentials: ClientCredentials) -> None:
        """Set up OIDC client credentials authentication."""
        # TODO: just move this to __init__ instead of requiring this additional setup method?
        self._access_token_helper.setup_credentials(credentials)

    def _do_request(
        self,
        method: str,
        path: str,
        *,
        json: Union[dict, list, None] = None,
        params: Optional[dict] = None,
        use_auth: bool = True,
        expected_status: int = 200,
        log_response_errors: bool = True,
        retry: bool = False,
    ) -> Union[dict, list, None]:
        """Do an HTTP request to Elastic Job Tracker service."""
        with TimingLogger(logger=self._log.debug, title=f"EJR Request `{method} {path}`"):
            headers = {}
            if use_auth:
                access_token = self._access_token_helper.get_access_token()
                headers["Authorization"] = f"Bearer {access_token}"

            url = url_join(self._api_url, path)
            self._log.debug(f"Doing EJR request `{method} {url}` {params=} {headers.keys()=}")
            if self._debug_show_curl:
                curl_command = self._as_curl(method=method, url=url, params=params, data=json, headers=headers)
                self._log.debug(f"Equivalent curl command: {curl_command}")
            try:
                do_request = lambda: self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=headers,
                    timeout=self._REQUEST_TIMEOUT,
                )
                if retry:
                    response = reretry.retry_call(
                        do_request,
                        exceptions=requests.exceptions.RequestException,
                        logger=self._log,
                        **get_backend_config().ejr_retry_settings,
                    )
                else:
                    response = do_request()
            except Exception as e:
                self._log.exception(f"Failed to do EJR API request `{method} {url}`: {e!r}")
                raise EjrApiError(f"Failed to do EJR API request `{method} {url}`") from e
            self._log.debug(f"EJR response on `{method} {path}`: {response.status_code!r}")
            if expected_status and response.status_code != expected_status:
                exc = EjrApiResponseError.from_response(response=response)
                if log_response_errors:
                    self._log.error(str(exc))
                raise exc
            else:
                response.raise_for_status()

            if response.content:
                return response.json()

    def _as_curl(self, method: str, url: str, *, params: Optional[dict] = None, data: dict, headers: dict):
        cmd = ["curl", "-i", "-X", method.upper()]
        cmd += ["-H", "Content-Type: application/json"]
        for k, v in headers.items():
            cmd += ["-H", f"{k}: {v}"]
        cmd += ["--data", json.dumps(data, separators=(",", ":"))]
        if params:
            url += "?" + urllib.parse.urlencode(params)
        cmd += [url]
        return " ".join(shlex.quote(c) for c in cmd)

    def health_check(self, use_auth: bool = True, log: bool = True) -> dict:
        response = self._do_request("GET", "/health", use_auth=use_auth)
        if log:
            self._log.info(f"EJR health check {response}")
        return response

    def create_job(
        self,
        *,
        process: dict,
        user_id: str,
        job_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        parent_id: Optional[str] = None,
        api_version: Optional[str] = None,
        job_options: Optional[dict] = None,
    ) -> JobDict:
        """
        Store/Initialize a new job
        """
        if not job_id:
            job_id = self.generate_job_id()
        created = rfc3339.now_utc()
        job_data = {
            # Essential identifiers
            "backend_id": self.backend_id,
            "user_id": user_id,
            "job_id": job_id,
            # Essential job creation fields (as defined by openEO API)
            "process": process,
            "title": title,
            "description": description,
            # TODO: fields plan, budget?
            # Initialize essential status fields (as defined by openEO API)
            "status": JOB_STATUS.CREATED,
            "created": created,
            "updated": created,
            # TODO: costs and usage?
            # job options: important, but not standardized (yet?)
            "job_options": job_options,
            # various internal/optional housekeeping fields
            "parent_id": parent_id,
            "application_id": None,
            "api_version": api_version,
            # TODO: additional technical metadata, see https://github.com/Open-EO/openeo-api/issues/472
        }
        with ExtraLoggingFilter.with_extra_logging(job_id=job_id):
            self._log.info(f"EJR creating {job_id=} {created=}")
            result = self._do_request("POST", "/jobs", json=job_data, expected_status=201)
            return result

    def get_job(self, job_id: str, *, user_id: Optional[str] = None) -> JobDict:
        return self._get_job(job_id=job_id, user_id=user_id)

    def _get_job(self, job_id: str, *, user_id: Optional[str] = None, fields: Optional[List[str]] = None) -> JobDict:
        with ExtraLoggingFilter.with_extra_logging(job_id=job_id, user_id=user_id):
            self._log.debug(f"EJR get job data {job_id=} {user_id=}")

            filters = [
                {"term": {"backend_id": self.backend_id}},
                {"term": {"job_id": job_id}},
            ]
            if user_id is not None:
                filters.append({"term": {"user_id": user_id}})

            query = {
                "bool": {
                    "filter": filters
                }
            }

            # Return full document, by default
            jobs = self._search(query=query, fields=fields or ["*"])
            if len(jobs) == 1:
                job = jobs[0]
                assert job["job_id"] == job_id, f"{job['job_id']=} != {job_id=}"
                assert user_id is None or job["user_id"] == user_id, f"{job['user_id']=} != {user_id=}"
                return job
            elif len(jobs) == 0:
                self._log.warning(f"Found no jobs for {job_id=} {user_id=}")
                raise JobNotFoundException(job_id=job_id)
            else:
                summary = [{k: j.get(k) for k in ["user_id", "created"]} for j in jobs]
                self._log.error(
                    f"Found multiple ({len(jobs)}) jobs for {job_id=} {user_id=}: {repr_truncate(summary, width=200)}"
                )
                raise InternalException(message=f"Found {len(jobs)} jobs for {job_id=} {user_id=}")

    def delete_job(self, job_id: str, *, user_id: Optional[str] = None) -> None:
        with ExtraLoggingFilter.with_extra_logging(job_id=job_id, user_id=user_id):
            try:
                self._get_job(job_id=job_id, user_id=user_id, fields=["job_id"])  # assert own job
                self._do_request(method="DELETE", path=f"/jobs/{job_id}", log_response_errors=False)
                self._log.info(f"EJR deleted {job_id=}")
            except EjrApiResponseError as e:
                if e.status_code == 404:
                    raise JobNotFoundException(job_id=job_id) from e
                raise e
            self._verify_job_existence(job_id=job_id, user_id=user_id, exists=False)

    def _verify_job_existence(self, job_id: str, user_id: Optional[str] = None, exists: bool = True,
                              backoffs: Sequence[float] = (0, 0.1, 1.0, 5.0)):
        """
        Verify that EJR committed the job creation/deletion
        :param job_id: job id
        :param exists: whether the job should exist (after creation) or not exist (after deletion)
        :return:
        """
        if not backoffs:
            return
        for backoff in backoffs:
            self._log.debug(f"_verify_job_existence {job_id=} {user_id=} {exists=} {backoff=}")
            time.sleep(backoff)
            try:
                self._get_job(job_id=job_id, user_id=user_id, fields=["job_id"])
                if exists:
                    return
            except JobNotFoundException:
                if not exists:
                    return
            except Exception as e:
                # TODO: fail hard instead of just logging?
                self._log.exception(f"Unexpected error while verifying {job_id=} {user_id=} {exists=}: {e=}")
                return
        # TODO: fail hard instead of just logging?
        self._log.error(f"Verification of {job_id=} {user_id=} {exists=} unsure after {len(backoffs)} attempts")

    def set_status(
        self,
        job_id: str,
        *,
        user_id: Optional[str] = None,
        status: str,
        updated: Optional[str] = None,
        started: Optional[str] = None,
        finished: Optional[str] = None,
    ) -> None:
        data = {
            "status": status,
            "updated": rfc3339.datetime(updated) if updated else rfc3339.now_utc(),
        }
        if started:
            data["started"] = rfc3339.datetime(started)
        if finished:
            data["finished"] = rfc3339.datetime(finished)
        self._update(job_id=job_id, data=data)

    def _update(self, job_id: str, data: dict) -> JobDict:
        """Generic update method"""
        with ExtraLoggingFilter.with_extra_logging(job_id=job_id):
            self._log.info(f"EJR update {job_id=} {data=}")
            return self._do_request("PATCH", f"/jobs/{job_id}", json=data)

    def set_dependencies(
        self, job_id: str, *, user_id: Optional[str] = None, dependencies: List[Dict[str, str]]
    ) -> None:
        self._update(job_id=job_id, data={"dependencies": dependencies})

    def remove_dependencies(self, job_id: str, *, user_id: Optional[str] = None) -> None:
        self._update(job_id=job_id, data={"dependencies": None, "dependency_status": None})

    def set_dependency_status(self, job_id: str, *, user_id: Optional[str] = None, dependency_status: str) -> None:
        self._update(job_id=job_id, data={"dependency_status": dependency_status})

    def set_dependency_usage(self, job_id: str, *, user_id: Optional[str] = None, dependency_usage: Decimal) -> None:
        self._update(job_id=job_id, data={"dependency_usage": str(dependency_usage)})

    def set_proxy_user(self, job_id: str, *, user_id: Optional[str] = None, proxy_user: str) -> None:
        # TODO #275 this "proxy_user" is a pretty implementation (YARN/VITO) specific field. Generalize this in some way?
        self._update(job_id=job_id, data={"proxy_user": proxy_user})

    def set_application_id(self, job_id: str, *, user_id: Optional[str] = None, application_id: str) -> None:
        self._update(job_id=job_id, data={"application_id": application_id})

    def _search(self, query: dict, fields: Optional[List[str]] = None) -> List[JobDict]:
        # TODO: sorting, pagination?
        fields = set(fields or [])
        # Make sure to include some basic fields by default
        fields.update(["job_id", "user_id", "created", "status", "updated"])
        body = {
            "query": query,
            "_source": list(fields),
        }
        self._log.debug(f"Doing search with query {json.dumps(body)}")
        return self._do_request("POST", "/jobs/search", json=body, retry=True)

    def _search_paginated(
        self,
        query: dict,
        *,
        fields: Optional[List[str]] = None,
        page_size: Optional[int] = None,
        page_number: Optional[int] = None,
    ) -> _PaginatedSearchResult:
        fields = set(fields or [])
        # Make sure to include some basic fields by default
        # TODO #332 avoid duplication of this default field set
        fields.update(["job_id", "user_id", "created", "status", "updated"])
        params = {}
        if page_size:
            params["size"] = page_size
        if page_number is not None and page_number >= 0:
            params["page"] = page_number
        body = {
            "query": query,
            "_source": list(fields),
        }
        self._log.debug(f"Doing search with query {json.dumps(body)} and {params=}")
        response = self._do_request("POST", "/jobs/search/paginated", params=params, json=body, retry=True)
        # Response structure:
        #   {
        #       "jobs": [list of job docs],
        #       "pagination": {"previous": {"size": 5, "page": 6}, "next": {"size": 5, "page": 8}}}
        #    }
        next_page_number = self._parse_response_pagination(
            pagination=response.get("pagination"),
            expected_size=page_size,
        )
        return _PaginatedSearchResult(
            jobs=response.get("jobs", []),
            next_page=next_page_number,
        )

    def _parse_response_pagination(self, pagination: Union[dict, None], expected_size: int) -> Union[int, None]:
        """Extract page number from pagination construct"""
        next_page_params = (pagination or {}).get("next")
        if (
            isinstance(next_page_params, dict)
            and "size" in next_page_params
            and next_page_params["size"] == expected_size
            and "page" in next_page_params
        ):
            next_page_number = next_page_params["page"]
        else:
            next_page_number = None
        self._log.debug(f"_search_paginated: parsed {next_page_number=} from {pagination=}")
        return next_page_number

    def list_user_jobs(
        self,
        user_id: str,
        *,
        fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        request_parameters: Optional[dict] = None,
        # TODO #332 settle on returning just `JobListing` and eliminate other options/code paths.
    ) -> Union[JobListing, List[JobDict]]:
        query = {
            "bool": {
                "filter": [
                    {"term": {"backend_id": self.backend_id}},
                    {"term": {"user_id": user_id}},
                ]
            }
        }

        if limit:
            # Do paginated search
            # TODO #332 make this the one and only code path
            page_number = (request_parameters or {}).get(self.PAGINATION_URL_PARAM)
            if page_number is not None:
                page_number = int(page_number)

            data = self._search_paginated(query=query, fields=fields, page_size=limit, page_number=page_number)

            jobs = [ejr_job_info_to_metadata(j, full=False) for j in data.jobs]
            if data.next_page:
                next_parameters = {"limit": limit, self.PAGINATION_URL_PARAM: data.next_page}
            else:
                next_parameters = None
            return JobListing(jobs=jobs, next_parameters=next_parameters)
        else:
            # Deprecated non-paginated search
            # TODO #332 eliminate this code path
            return self._search(query=query, fields=fields)

    def list_active_jobs(
        self,
        *,
        fields: Optional[List[str]] = None,
        max_age: Optional[int] = None,
        max_updated_ago: Optional[int] = None,
        require_application_id: bool = False,
    ) -> List[JobDict]:
        active = [JOB_STATUS.CREATED, JOB_STATUS.QUEUED, JOB_STATUS.RUNNING]
        query = {
            "bool": {
                "filter": [
                    {"term": {"backend_id": self.backend_id}},
                    {"terms": {"status": active}},
                ]
            },
        }
        if max_age:
            query["bool"]["filter"].append({"range": {"created": {"gte": f"now-{max_age}d"}}})
        if max_updated_ago:
            query["bool"]["filter"].append({"range": {"updated": {"gte": f"now-{max_updated_ago}d"}}})
        if require_application_id:
            query["bool"]["must"] = {
                # excludes null values as well as property missing altogether
                "exists": {"field": "application_id"}
            }
        return self._search(query=query, fields=fields)

    def set_results_metadata(
        self,
        job_id: str,
        *,
        user_id: Optional[str] = None,
        costs: Optional[float],
        usage: dict,
        results_metadata: Dict[str, Any],
    ) -> None:
        self._update(
            job_id=job_id,
            data={
                "costs": costs,
                "usage": usage,
                "results_metadata": results_metadata,
            },
        )


class CliApp:
    """
    Simple CLI to Elastic Job Registry API for testing, development and debugging

    Example usage:

        # First: log in to vault to get a vault token (for obtaining EJR credentials)
        vault login -method=ldap username=john

        # Dummy usage:
        # Create a dummy job for user
        python openeo_driver/jobregistry.py create john
        # List jobs from user
        python openeo_driver/jobregistry.py list-user john
        # Get job metadata
        python openeo_driver/jobregistry.py get j-231208662fa3450da54e1987394f7ed0

        # Real usage, e.g. with backend id 'mep-dev', specified through `--backend-id` option
        # List jobs from a user
        python openeo_driver/jobregistry.py list-user --backend-id mep-dev abc123@egi.eu
        # Get all metadata from a job
        python openeo_driver/jobregistry.py get --backend-id mep-dev j-231206cbc43a4ae6a3fe58173c0f45f6

        # Debug tools:
        # deeper log level, e.g. DEBUG level:
        python openeo_driver/jobregistry.py -vv ...
        # Show equivalent curl commands to query EJR API:
        python openeo_driver/jobregistry.py --show-curl ...
    """

    _DEFAULT_BACKEND_ID = "test_cli"

    def __init__(self, environ: Optional[dict] = None):
        self.environ = environ or os.environ

    def main(self):
        cli_args = self._parse_cli()
        log_level = {0: logging.WARNING, 1: logging.INFO}.get(cli_args.verbose, logging.DEBUG)
        if cli_args.show_curl:
            log_level = logging.DEBUG
        logging.basicConfig(level=log_level)
        cli_args.func(cli_args)

    def _parse_cli(self) -> argparse.Namespace:
        cli = argparse.ArgumentParser(
            description=textwrap.dedent(self.__doc__), formatter_class=argparse.RawTextHelpFormatter
        )

        cli.add_argument("-v", "--verbose", action="count", default=0)
        cli.add_argument("--show-curl", action="store_true")

        # Sub-commands
        subparsers = cli.add_subparsers(required=True)

        health_check = subparsers.add_parser("health", help="Do health check request.")
        health_check.add_argument(
            "--no-auth", dest="do_auth", action="store_false", default=True
        )
        health_check.set_defaults(func=self.health_check)

        cli_list_user = subparsers.add_parser("list-user", help="List jobs for given user.")
        cli_list_user.add_argument("--backend-id", help="Backend id to filter on.")
        cli_list_user.add_argument("user_id", help="User id to filter on.")
        cli_list_user.add_argument("--limit", type=int, default=None, help="Page size of job listing")
        cli_list_user.add_argument("--page", default=0)
        cli_list_user.set_defaults(func=self.list_user_jobs)

        cli_create = subparsers.add_parser("create", help="Create a new job.")
        cli_create.add_argument("--backend-id", help="Backend-id to create the job for.")
        cli_create.add_argument("user_id", help="User id to create the job for.")
        cli_create.add_argument("--process-graph", help="JSON representation of process graph")
        cli_create.set_defaults(func=self.create_dummy_job)

        cli_set_status = subparsers.add_parser("set-status", help="Set status of a job")
        cli_set_status.add_argument("job_id", help="Job id")
        cli_set_status.add_argument("status", help="New status (queued, running, ...)")
        cli_set_status.set_defaults(func=self.set_status)

        cli_list_active = subparsers.add_parser("list-active", help="List active jobs.")
        cli_list_active.add_argument("--backend-id", help="Backend id to filter on.")
        cli_list_active.add_argument(
            "--max-age", default=30, help="Maximum age (in days) of job. (default: %(default)s)"
        )
        cli_list_active.add_argument(
            "--max-updated-ago",
            default=7,
            help="Only return jobs 'updated' at most this number of days ago. (default: %(default)s)",
        )
        cli_list_active.add_argument(
            "--require-application-id", action="store_true", help="Toggle to only list jobs with an application_id"
        )
        cli_list_active.set_defaults(func=self.list_active_jobs)

        cli_get_job = subparsers.add_parser("get", help="Get job metadata of a single job")
        cli_get_job.add_argument("--backend-id", help="Backend id to filter on.")
        cli_get_job.add_argument("job_id")
        cli_get_job.set_defaults(func=self.get_job)

        cli_delete = subparsers.add_parser("delete", help="Mark job as deleted")
        cli_delete.add_argument("--backend-id", help="Backend id to filter on.")
        cli_delete.add_argument("job_id", help="Job id")
        cli_delete.set_defaults(func=self.delete_job)

        return cli.parse_args()

    def _get_job_registry(self, setup_auth=True, cli_args: Optional[argparse.Namespace] = None) -> ElasticJobRegistry:
        # TODO #275 avoid hardcoded VITO reference, instead require env var or cli option?
        api_url = self.environ.get("OPENEO_EJR_API", "https://jobregistry.vgt.vito.be")
        if "backend_id" in cli_args and cli_args.backend_id:
            backend_id = cli_args.backend_id
        else:
            _log.warning("No backend id specified, using default %r", self._DEFAULT_BACKEND_ID)
            backend_id = self._DEFAULT_BACKEND_ID
        ejr = ElasticJobRegistry(
            api_url=api_url,
            backend_id=backend_id,
            _debug_show_curl=cli_args.show_curl if cli_args else False,
        )

        if setup_auth:
            _log.info("Trying to get EJR credentials from Vault")
            try:
                import hvac, hvac.exceptions
            except ImportError as e:
                raise RuntimeError(
                    "Package `hvac` (HashiCorp Vault client) is required for this functionality"
                ) from e

            try:
                # Note: this flow assumes the default token resolution of vault client,
                # (using `VAULT_TOKEN` env variable or local `~/.vault-token` file)
                vault_client = hvac.Client(url=self.environ.get("VAULT_ADDR"))
                ejr_vault_path = self.environ.get(
                    "OPENEO_EJR_CREDENTIALS_VAULT_PATH",
                    # TODO #275 eliminate this hardcoded VITO specific default value
                    "TAP/big_data_services/openeo/openeo-job-registry-elastic-api",
                )
                secret = vault_client.secrets.kv.v2.read_secret_version(
                    ejr_vault_path,
                    mount_point="kv",
                )
            except hvac.exceptions.Forbidden as e:
                raise RuntimeError(
                    f"No permissions to read EJR credentials from vault: {e}."
                    " Make sure `hvac.Client` can find a valid vault token"
                    " through environment variable `VAULT_TOKEN`"
                    " or local file `~/.vault-token` (e.g. created with `vault login -method=ldap username=john`)."
                )
            credentials = ClientCredentials.from_mapping(secret["data"]["data"])
            ejr.setup_auth_oidc_client_credentials(credentials=credentials)
        return ejr

    def health_check(self, args: argparse.Namespace):
        ejr = self._get_job_registry(setup_auth=args.do_auth, cli_args=args)
        print(ejr.health_check(use_auth=args.do_auth))

    def list_user_jobs(self, args: argparse.Namespace):
        user_id = args.user_id
        ejr = self._get_job_registry(cli_args=args)
        request_parameters = {}
        if args.page:
            request_parameters[ElasticJobRegistry.PAGINATION_URL_PARAM] = str(args.page)
        jobs = ejr.list_user_jobs(
            user_id=user_id,
            # TODO: option to return more fields?
            fields=["started", "finished", "title"],
            limit=args.limit,
            request_parameters=request_parameters,
        )
        print(f"Found {len(jobs)} jobs for user {user_id!r}:")
        if isinstance(jobs, JobListing):
            pprint.pp(jobs.to_response_dict(build_url=lambda d: "/jobs?" + urllib.parse.urlencode(d)))
        elif isinstance(jobs, list):
            pprint.pp(jobs)
        else:
            raise ValueError(jobs)

    def list_active_jobs(self, args: argparse.Namespace):
        ejr = self._get_job_registry(cli_args=args)
        # TODO: option to return more fields?
        jobs = ejr.list_active_jobs(
            max_age=args.max_age,
            max_updated_ago=args.max_updated_ago,
            require_application_id=args.require_application_id,
        )
        print(f"Found {len(jobs)} active jobs (backend {ejr.backend_id!r}):")
        pprint.pp(jobs)

    def create_dummy_job(self, args: argparse.Namespace):
        user_id = args.user_id
        ejr = self._get_job_registry(cli_args=args)
        if args.process_graph:
            process = json.loads(args.process_graph)
            if "process_graph" not in process:
                process = {"process_graph": process}
        else:
            # Default process graph
            process = {
                "summary": "calculate 3+5, please",
                "process_graph": {
                    "add": {
                        "process_id": "add",
                        "arguments": {"x": 3, "y": 5},
                        "result": True,
                    },
                },
            }
        result = ejr.create_job(process=process, user_id=user_id)
        print("Created job:")
        pprint.pprint(result)

    def get_job(self, args: argparse.Namespace):
        job_id = args.job_id
        ejr = self._get_job_registry(cli_args=args)
        job = ejr.get_job(job_id=job_id)
        pprint.pprint(job)

    def set_status(self, args: argparse.Namespace):
        ejr = self._get_job_registry(cli_args=args)
        result = ejr.set_status(job_id=args.job_id, status=args.status)
        pprint.pprint(result)

    def delete_job(self, args: argparse.Namespace):
        job_id = args.job_id
        ejr = self._get_job_registry(cli_args=args)
        ejr.delete_job(job_id=job_id)
        print(f"Deleted {job_id}")


if __name__ == "__main__":
    CliApp().main()
