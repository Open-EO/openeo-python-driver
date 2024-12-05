from __future__ import annotations

import argparse
import json
import logging
import os
import pprint
import shlex
import textwrap
import time
import typing
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Union

import requests
import reretry
from deprecated.classic import deprecated
from openeo.rest.connection import url_join
from openeo.util import TimingLogger, repr_truncate, rfc3339

import openeo_driver._version
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

    def get_job(self, job_id: str, user_id: Optional[str] = None) -> JobDict:
        raise NotImplementedError

    def delete_job(self, job_id: str, user_id: Optional[str] = None) -> None:
        raise NotImplementedError

    def set_status(
        self,
        job_id: str,
        status: str,
        *,
        updated: Optional[str] = None,
        started: Optional[str] = None,
        finished: Optional[str] = None,
    ) -> JobDict:
        raise NotImplementedError

    def set_dependencies(
        self, job_id: str, dependencies: List[Dict[str, str]]
    ) -> JobDict:
        raise NotImplementedError

    def remove_dependencies(self, job_id: str) -> JobDict:
        raise NotImplementedError

    def set_dependency_status(self, job_id: str, dependency_status: str) -> JobDict:
        raise NotImplementedError

    def set_dependency_usage(self, job_id: str, dependency_usage: Decimal) -> JobDict:
        raise NotImplementedError

    def set_proxy_user(self, job_id: str, proxy_user: str) -> JobDict:
        # TODO #275 this "proxy_user" is a pretty implementation (YARN/VITO) specific field. Generalize this in some way?
        raise NotImplementedError

    def set_application_id(self, job_id: str, application_id: str) -> JobDict:
        raise NotImplementedError

    @deprecated("call set_results_metadata instead", version="0.82.0")
    def set_usage(self, job_id: str, costs: float, usage: dict) -> JobDict:
        raise NotImplementedError

    # TODO: improve name?
    def set_results_metadata(self, job_id: str, costs: Optional[float], usage: dict,
                             results_metadata: Dict[str, Any]) -> JobDict:
        raise NotImplementedError

    def list_user_jobs(
        self, user_id: str, fields: Optional[List[str]] = None
    ) -> List[JobDict]:
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
            msg=f"EJR API error: {response.status_code} {response.reason!r} on `{request.method} {request.url!r}`: {response.text}",
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


class ElasticJobRegistry(JobRegistryInterface):
    """
    (Base)class to manage storage of batch job metadata
    using the Job Registry Elastic API (openeo-job-tracker-elastic-api)
    """

    _REQUEST_TIMEOUT = 20

    logger = logging.getLogger(f"{__name__}.elastic")

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

        self.logger.debug(f"Creating ElasticJobRegistry with {backend_id=} and {api_url=}")
        self._backend_id: Optional[str] = backend_id
        self._api_url = api_url
        self._access_token_helper = ClientCredentialsAccessTokenHelper(session=session)

        if session:
            self._session = session
        else:
            self._session = requests.Session()
            self.set_user_agent()

        self._debug_show_curl = _debug_show_curl

    def set_user_agent(self):
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
        self._access_token_helper.setup_credentials(credentials)

    def _do_request(
        self,
        method: str,
        path: str,
        json: Union[dict, list, None] = None,
        use_auth: bool = True,
        expected_status: int = 200,
        log_response_errors: bool = True,
        retry: bool = False,
    ) -> Union[dict, list, None]:
        """Do an HTTP request to Elastic Job Tracker service."""
        with TimingLogger(logger=self.logger.debug, title=f"EJR Request `{method} {path}`"):
            headers = {}
            if use_auth:
                access_token = self._access_token_helper.get_access_token()
                headers["Authorization"] = f"Bearer {access_token}"

            url = url_join(self._api_url, path)
            self.logger.debug(f"Doing EJR request `{method} {url}` {headers.keys()=}")
            if self._debug_show_curl:
                curl_command = self._as_curl(method=method, url=url, data=json, headers=headers)
                self.logger.debug(f"Equivalent curl command: {curl_command}")
            try:
                do_request = lambda: self._session.request(
                    method=method,
                    url=url,
                    json=json,
                    headers=headers,
                    timeout=self._REQUEST_TIMEOUT,
                )
                if retry:
                    response = reretry.retry_call(
                        do_request,
                        exceptions=requests.exceptions.RequestException,
                        logger=self.logger,
                        **get_backend_config().ejr_retry_settings,
                    )
                else:
                    response = do_request()
            except Exception as e:
                self.logger.exception(f"Failed to do EJR API request `{method} {path}`: {e!r}")
                raise EjrApiError(f"Failed to do EJR API request `{method} {path}`") from e
            self.logger.debug(f"EJR response on `{method} {path}`: {response.status_code!r}")
            if expected_status and response.status_code != expected_status:
                exc = EjrApiResponseError.from_response(response=response)
                if log_response_errors:
                    self.logger.error(str(exc))
                raise exc
            else:
                response.raise_for_status()

            if response.content:
                return response.json()

    def _as_curl(self, method: str, url: str, data: dict, headers: dict):
        cmd = ["curl", "-i", "-X", method.upper()]
        cmd += ["-H", "Content-Type: application/json"]
        for k, v in headers.items():
            cmd += ["-H", f"{k}: {v}"]
        cmd += ["--data", json.dumps(data, separators=(",", ":"))]
        cmd += [url]
        return " ".join(shlex.quote(c) for c in cmd)

    def health_check(self, use_auth: bool = True, log: bool = True) -> dict:
        response = self._do_request("GET", "/health", use_auth=use_auth)
        if log:
            self.logger.info(f"EJR health check {response}")
        return response

    def create_job(
        self,
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
        created = rfc3339.utcnow()
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
            self.logger.info(f"EJR creating {job_id=} {created=}")
            result = self._do_request("POST", "/jobs", json=job_data, expected_status=201)
            return result

    def get_job(self, job_id: str, user_id: Optional[str] = None, fields: Optional[List[str]] = None) -> JobDict:
        with ExtraLoggingFilter.with_extra_logging(job_id=job_id, user_id=user_id):
            self.logger.debug(f"EJR get job data {job_id=} {user_id=}")

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
                self.logger.warning(f"Found no jobs for {job_id=} {user_id=}")
                raise JobNotFoundException(job_id=job_id)
            else:
                summary = [{k: j.get(k) for k in ["user_id", "created"]} for j in jobs]
                self.logger.error(
                    f"Found multiple ({len(jobs)}) jobs for {job_id=} {user_id=}: {repr_truncate(summary, width=200)}"
                )
                raise InternalException(message=f"Found {len(jobs)} jobs for {job_id=} {user_id=}")

    def delete_job(self, job_id: str, user_id: Optional[str] = None) -> None:
        with ExtraLoggingFilter.with_extra_logging(job_id=job_id, user_id=user_id):
            try:
                self.get_job(job_id=job_id, user_id=user_id, fields=["job_id"])  # assert own job
                self._do_request(method="DELETE", path=f"/jobs/{job_id}", log_response_errors=False)
                self.logger.info(f"EJR deleted {job_id=}")
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
            self.logger.debug(f"_verify_job_existence {job_id=} {user_id=} {exists=} {backoff=}")
            time.sleep(backoff)
            try:
                self.get_job(job_id=job_id, user_id=user_id, fields=["job_id"])
                if exists:
                    return
            except JobNotFoundException:
                if not exists:
                    return
            except Exception as e:
                # TODO: fail hard instead of just logging?
                self.logger.exception(f"Unexpected error while verifying {job_id=} {user_id=} {exists=}: {e=}")
                return
        # TODO: fail hard instead of just logging?
        self.logger.error(f"Verification of {job_id=} {user_id=} {exists=} unsure after {len(backoffs)} attempts")

    def set_status(
        self,
        job_id: str,
        status: str,
        *,
        updated: Optional[str] = None,
        started: Optional[str] = None,
        finished: Optional[str] = None,
    ) -> JobDict:
        data = {
            "status": status,
            "updated": rfc3339.datetime(updated) if updated else rfc3339.utcnow(),
        }
        if started:
            data["started"] = rfc3339.datetime(started)
        if finished:
            data["finished"] = rfc3339.datetime(finished)
        return self._update(job_id=job_id, data=data)

    def _update(self, job_id: str, data: dict) -> JobDict:
        """Generic update method"""
        with ExtraLoggingFilter.with_extra_logging(job_id=job_id):
            self.logger.info(f"EJR update {job_id=} {data=}")
            return self._do_request("PATCH", f"/jobs/{job_id}", json=data)

    def set_dependencies(
        self, job_id: str, dependencies: List[Dict[str, str]]
    ) -> JobDict:
        return self._update(job_id=job_id, data={"dependencies": dependencies})

    def remove_dependencies(self, job_id: str) -> JobDict:
        return self._update(
            job_id=job_id, data={"dependencies": None, "dependency_status": None}
        )

    def set_dependency_status(self, job_id: str, dependency_status: str) -> JobDict:
        return self._update(
            job_id=job_id, data={"dependency_status": dependency_status}
        )

    def set_dependency_usage(self, job_id: str, dependency_usage: Decimal) -> JobDict:
        return self._update(
            job_id=job_id, data={"dependency_usage": str(dependency_usage)}
        )

    def set_usage(self, job_id: str, costs: float, usage: dict) -> JobDict:
        return self._update(
            job_id=job_id, data={"costs": costs, "usage": usage}
        )

    def set_proxy_user(self, job_id: str, proxy_user: str) -> JobDict:
        # TODO #275 this "proxy_user" is a pretty implementation (YARN/VITO) specific field. Generalize this in some way?
        return self._update(job_id=job_id, data={"proxy_user": proxy_user})

    def set_application_id(self, job_id: str, application_id: str) -> JobDict:
        return self._update(job_id=job_id, data={"application_id": application_id})

    def _search(self, query: dict, fields: Optional[List[str]] = None) -> List[JobDict]:
        # TODO: sorting, pagination?
        fields = set(fields or [])
        # Make sure to include some basic fields by default
        fields.update(["job_id", "user_id", "created", "status", "updated"])
        query = {
            "query": query,
            "_source": list(fields),
        }
        self.logger.debug(f"Doing search with query {json.dumps(query)}")
        return self._do_request("POST", "/jobs/search", query, retry=True)

    def list_user_jobs(
        self, user_id: Optional[str], fields: Optional[List[str]] = None
    ) -> List[JobDict]:
        query = {
            "bool": {
                "filter": [
                    {"term": {"backend_id": self.backend_id}},
                    {"term": {"user_id": user_id}},
                ]
            }
        }
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
        self, job_id: str, costs: Optional[float], usage: dict, results_metadata: Dict[str, Any]
    ) -> JobDict:
        return self._update(job_id=job_id, data={
            "costs": costs,
            "usage": usage,
            "results_metadata": results_metadata,
        })


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

        # Real usage, e.g. with backend id 'mep-dev', specidied through `--backend-id` option
        # List jobs from a user
        python openeo_driver/jobregistry.py list-user --backend-id mep-dev abc123@egi.eu
        # Get all metadata from a job
        python openeo_driver/jobregistry.py get --backend-id mep-dev j-231206cbc43a4ae6a3fe58173c0f45f6
    """

    _DEFAULT_BACKEND_ID = "test_cli"

    def __init__(self, environ: Optional[dict] = None):
        self.environ = environ or os.environ

    def main(self):
        cli_args = self._parse_cli()
        log_level = {0: logging.WARNING, 1: logging.INFO}.get(
            cli_args.verbose, logging.DEBUG
        )
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
        # TODO: option to return more fields?
        jobs = ejr.list_user_jobs(user_id=user_id, fields=["started", "finished", "title"])
        print(f"Found {len(jobs)} jobs for user {user_id!r}:")
        pprint.pp(jobs)

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
