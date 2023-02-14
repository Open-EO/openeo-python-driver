import argparse
import datetime as dt
import json
import logging
import os
import pprint
import time
from decimal import Decimal
import typing
from typing import Dict, List, NamedTuple, Optional, Union, Any

import requests
from openeo.rest.auth.oidc import (
    OidcClientCredentialsAuthenticator,
    OidcClientInfo,
    OidcProviderInfo,
)
from openeo.rest.connection import url_join
from openeo.util import TimingLogger, rfc3339

import openeo_driver._version
from openeo_driver.datastructs import secretive_repr
from openeo_driver.util.caching import TtlCache
from openeo_driver.util.logging import just_log_exceptions
from openeo_driver.utils import generate_unique_id

_log = logging.getLogger(__name__)


class JOB_STATUS:
    """
    Container of batch job status constants.

    Allows to easily find places where batch job status is checked/set/updated.
    """

    # TODO: move this to a module for API-related constants?

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    CANCELED = "canceled"
    FINISHED = "finished"
    ERROR = "error"


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
        # TODO: this is a pretty implementation specific field. Generalize this in some way?
        raise NotImplementedError

    def set_application_id(self, job_id: str, application_id: str) -> JobDict:
        raise NotImplementedError

    # TODO: methods to list jobs (filtering on timeframe, userid, ...)?

    def list_active_jobs(self, max_age: Optional[int] = None) -> List[JobDict]:
        """
        List active jobs (created, queued, running)

        :param max_age: optional filter to only return recently created jobs:
            maximum number of days creation date.
        """
        raise NotImplementedError


class EjrError(Exception):
    """Elastic Job Registry error (base class)."""

    pass


class EjrHttpError(EjrError):
    def __init__(self, msg: str, status_code: Optional[int]):
        super().__init__(msg)
        self.status_code = status_code

    @classmethod
    def from_response(cls, response: requests.Response) -> "EjrHttpError":
        request = response.request
        return cls(
            msg=f"EJR API error: {response.status_code} {response.reason!r} on `{request.method} {request.url!r}`: {response.text}",
            status_code=response.status_code,
        )


class ElasticJobRegistryCredentials(NamedTuple):
    """Container of Elastic Job Registry related credentials."""

    oidc_issuer: str
    client_id: str
    client_secret: str
    __repr__ = __str__ = secretive_repr()

    @staticmethod
    def get(
        *,
        oidc_issuer: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        config: Optional[typing.Mapping] = None,
        env: Optional[typing.Mapping] = None,
    ) -> "ElasticJobRegistryCredentials":
        """Best effort factory to build ElasticJobRegistryCredentials from given args, config or env"""
        # Start from args
        kwargs = {
            "oidc_issuer": oidc_issuer,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        # fallback on config (if any) and env
        env_var_mapping = {
            "oidc_issuer": "OPENEO_EJR_OIDC_ISSUER",
            "client_id": "OPENEO_EJR_OIDC_CLIENT_ID",
            "client_secret": "OPENEO_EJR_OIDC_CLIENT_SECRET",
        }
        if env is None:
            env = os.environ
        for key in [k for (k, v) in kwargs.items() if not v]:
            if config and key in config:
                kwargs[key] = config[key]
            elif env_var_mapping[key] in env:
                kwargs[key] = env[env_var_mapping[key]]
            elif key == "oidc_issuer":
                # TODO #153 eliminate hardcoded oidc_issuer default
                _log.warning(
                    "Deprecated ElasticJobRegistryCredentials.oidc_issuer fallback"
                )
                kwargs[key] = "https://sso.terrascope.be/auth/realms/terrascope"
            else:
                raise EjrError(
                    f"Failed to obtain {key} field for building {ElasticJobRegistryCredentials.__name__}"
                )
        return ElasticJobRegistryCredentials(**kwargs)


class ElasticJobRegistry(JobRegistryInterface):
    """
    (Base)class to manage storage of batch job metadata
    using the Job Registry Elastic API (openeo-job-tracker-elastic-api)
    """

    _REQUEST_TIMEOUT = 20

    logger = logging.getLogger(f"{__name__}.elastic")

    def __init__(self, api_url: str, backend_id: Optional[str] = None):
        self.logger.info(
            f"Creating ElasticJobRegistry with {backend_id=} and {api_url=}"
        )
        self._backend_id: Optional[str] = backend_id
        self._api_url = api_url
        self._authenticator: Optional[OidcClientCredentialsAuthenticator] = None
        self._cache = TtlCache(default_ttl=60 * 60)
        self._session = requests.Session()

        user_agent = f"openeo_driver-{openeo_driver._version.__version__}/{self.__class__.__name__}"
        if self._backend_id:
            user_agent += f"/{self._backend_id}"
        self._session.headers["User-Agent"] = user_agent

    @property
    def backend_id(self) -> str:
        assert self._backend_id
        return self._backend_id

    def setup_auth_oidc_client_credentials(
        self, credentials: ElasticJobRegistryCredentials
    ) -> None:
        """Set up OIDC client credentials authentication."""
        self.logger.info(
            f"Setting up EJR OIDC Client Credentials Authentication with {credentials.client_id=}, {credentials.oidc_issuer=}, {len(credentials.client_secret)=}"
        )
        oidc_provider = OidcProviderInfo(issuer=credentials.oidc_issuer)
        client_info = OidcClientInfo(
            client_id=credentials.client_id,
            provider=oidc_provider,
            client_secret=credentials.client_secret,
        )
        self._authenticator = OidcClientCredentialsAuthenticator(
            client_info=client_info
        )

    def _get_access_token(self) -> str:
        if not self._authenticator:
            raise EjrError("No authentication set up")
        with TimingLogger(
            title=f"Requesting EJR OIDC access_token ({self._authenticator.__class__.__name__})",
            logger=self.logger.info,
        ):
            tokens = self._authenticator.get_tokens()
        return tokens.access_token

    def _do_request(
        self,
        method: str,
        path: str,
        json: Union[dict, list, None] = None,
        use_auth: bool = True,
        expected_status: int = 200,
        logging_extra: Optional[dict] = None,
    ) -> Union[dict, list]:
        """Do an HTTP request to Elastic Job Tracker service."""
        with TimingLogger(
            logger=(lambda m: self.logger.debug(m, extra=logging_extra)),
            title=f"EJR Request `{method} {path}`",
        ):
            headers = {}
            if use_auth:
                access_token = self._cache.get_or_call(
                    key="api_access_token",
                    callback=self._get_access_token,
                    # TODO: finetune/optimize caching TTL? Detect TTl/expiry from JWT access token itself?
                    ttl=30 * 60,
                )
                headers["Authorization"] = f"Bearer {access_token}"

            url = url_join(self._api_url, path)
            self.logger.debug(
                f"Doing EJR request `{method} {url}` {headers.keys()=}",
                extra=logging_extra,
            )
            response = self._session.request(
                method=method,
                url=url,
                json=json,
                headers=headers,
                timeout=self._REQUEST_TIMEOUT,
            )
            self.logger.info(
                f"EJR response on `{method} {path}`: {response.status_code!r}",
                extra=logging_extra,
            )
            if expected_status and response.status_code != expected_status:
                raise EjrHttpError.from_response(response=response)
            else:
                response.raise_for_status()
            return response.json()

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
        logging_extra = {"job_id": job_id}
        self.logger.info(f"EJR creating {job_id=} {created=}", extra=logging_extra)
        return self._do_request(
            "POST",
            "/jobs",
            json=job_data,
            expected_status=201,
            logging_extra=logging_extra,
        )

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
        logging_extra = {"job_id": job_id}
        self.logger.info(f"EJR update {job_id=} {data=}", extra=logging_extra)
        return self._do_request(
            "PATCH", f"/jobs/{job_id}", json=data, logging_extra=logging_extra
        )

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

    def set_proxy_user(self, job_id: str, proxy_user: str) -> JobDict:
        return self._update(job_id=job_id, data={"proxy_user": proxy_user})

    def set_application_id(self, job_id: str, application_id: str) -> JobDict:
        return self._update(job_id=job_id, data={"application_id": application_id})

    def _search(self, query: dict) -> List[JobDict]:
        # TODO: sorting, pagination?
        # TODO: How to return field subset (e.g. not process graph) to reduce payload size
        #   https://github.com/Open-EO/openeo-job-registry-elastic-api/issues/25
        self.logger.info(f"Doing search with query {json.dumps(query)}")
        return self._do_request("POST", "/jobs/search", json=query)

    def list_user_jobs(self, user_id: Optional[str]) -> List[JobDict]:
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"backend_id": self.backend_id}},
                        {"term": {"user_id": user_id}},
                    ]
                }
            }
        }
        return self._search(query=query)

    def list_active_jobs(self, max_age: Optional[int] = None) -> List[JobDict]:
        active = [JOB_STATUS.CREATED, JOB_STATUS.QUEUED, JOB_STATUS.RUNNING]
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"backend_id": self.backend_id}},
                        {"terms": {"status": active}},
                        {"range": {"created": {"gte": f"now-{max_age or 7}d"}}},
                    ]
                }
            }
        }
        return self._search(query=query)

    @staticmethod
    def just_log_errors(name: str = "EJR"):
        """
        Shortcut to easily and compactly guard all experimental, new ElasticJobRegistry logic
        with a "just_log_errors" context.
        """
        # TODO #153: remove all usage when ElasticJobRegistry is ready for production
        return just_log_exceptions(log=ElasticJobRegistry.logger.warning, name=name)


class CliApp:
    """
    Simple toy CLI to Elastic Job Registry API for development and testing

    Example usage:

        # First: log in to vault to get a vault token (for obtaining EJR credentials)
        vault login -method=ldap username=john

        # Create a dummy job for user
        python openeo_driver/jobregistry.py create john
        # List jobs from user
        python openeo_driver/jobregistry.py list john

    """

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
        cli = argparse.ArgumentParser()

        cli.add_argument("-v", "--verbose", action="count", default=0)

        # Sub-commands
        subparsers = cli.add_subparsers(required=True)

        health_check = subparsers.add_parser("health", help="Do health check request.")
        health_check.add_argument(
            "--no-auth", dest="do_auth", action="store_false", default=True
        )
        health_check.set_defaults(func=self.health_check)

        cli_list_user = subparsers.add_parser(
            "list-user", help="List jobs for given user."
        )
        cli_list_user.add_argument("user_id", help="User id to filter on.")
        cli_list_user.set_defaults(func=self.list_user_jobs)

        cli_create = subparsers.add_parser("create", help="Create a new job.")
        cli_create.add_argument("user_id", help="User id to filter on.")
        cli_create.set_defaults(func=self.create_dummy_job)

        cli_set_status = subparsers.add_parser("set_status", help="Set status of a job")
        cli_set_status.add_argument("job_id", help="Job id")
        cli_set_status.add_argument("status", help="New status (queued, running, ...)")
        cli_set_status.set_defaults(func=self.set_status)

        cli_list_active = subparsers.add_parser("list-active", help="List active jobs.")
        cli_list_active.add_argument("--backend-id", help="Override back-end ID")
        cli_list_active.set_defaults(func=self.list_active_jobs)

        return cli.parse_args()

    def _get_job_registry(
        self, backend_id: Optional[str] = None, setup_auth=True
    ) -> ElasticJobRegistry:
        api_url = self.environ.get(
            "OPENEO_EJR_API", "https://jobregistry.openeo.vito.be"
        )
        ejr = ElasticJobRegistry(api_url=api_url, backend_id=backend_id or "test_cli")

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
                secret = vault_client.secrets.kv.v2.read_secret_version(
                    # TODO: avoid this hardcoded path?
                    f"TAP/big_data_services/openeo/openeo-job-registry-elastic-api",
                    mount_point="kv",
                )
            except hvac.exceptions.Forbidden as e:
                raise RuntimeError(
                    f"No permissions to read EJR credentials from vault: {e}."
                    " Make sure `hvac.Client` can find a valid vault token"
                    " through environment variable `VAULT_TOKEN`"
                    " or local file `~/.vault-token` (e.g. created with `vault login -method=ldap username=john`)."
                )
            credentials = ElasticJobRegistryCredentials.get(
                config=secret["data"]["data"]
            )
            ejr.setup_auth_oidc_client_credentials(credentials=credentials)
        return ejr

    def health_check(self, args: argparse.Namespace):
        ejr = self._get_job_registry(setup_auth=args.do_auth)
        print(ejr.health_check(use_auth=args.do_auth))

    def list_user_jobs(self, args: argparse.Namespace):
        user_id = args.user_id
        ejr = self._get_job_registry()
        jobs = ejr.list_user_jobs(user_id=user_id)
        print(f"Found {len(jobs)} jobs for user {user_id!r}:")
        pprint.pp(jobs)

    def list_active_jobs(self, args: argparse.Namespace):
        ejr = self._get_job_registry(backend_id=args.backend_id)
        jobs = ejr.list_active_jobs()
        print(f"Found {len(jobs)} active jobs (backend {ejr.backend_id!r}):")
        pprint.pp(jobs)

    def create_dummy_job(self, args: argparse.Namespace):
        user_id = args.user_id
        ejr = self._get_job_registry()
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

    def set_status(self, args: argparse.Namespace):
        ejr = self._get_job_registry()
        result = ejr.set_status(job_id=args.job_id, status=args.status)
        pprint.pprint(result)


if __name__ == "__main__":
    CliApp().main()
