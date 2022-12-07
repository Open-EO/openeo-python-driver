import argparse
import datetime as dt
import logging
import os
import pprint
import requests
from typing import Optional, Union

import openeo_driver._version
from openeo.rest.auth.oidc import (
    OidcClientCredentialsAuthenticator,
    OidcClientInfo,
    OidcProviderInfo,
)
from openeo.rest.connection import url_join
from openeo.util import TimingLogger, rfc3339
from openeo_driver.util.caching import TtlCache
from openeo_driver.utils import generate_unique_id

_log = logging.getLogger(__name__)


class ElasticJobRegistry:
    """
    (Base)class to manage storage of batch job metadata
    using the Job Registry Elastic API (openeo-job-tracker-elastic-api)
    """

    logger = logging.getLogger(f"{__name__}.elastic")

    def __init__(self, backend_id: str, api_url: str):
        self.logger.info(f"Creating ElasticJobRegistry with {backend_id=} and {api_url=}")
        self._backend_id = backend_id
        self._api_url = api_url
        self._authenticator = None
        self._cache = TtlCache(default_ttl=60 * 60)

    def authenticate_oidc_client_credentials(
        self, oidc_issuer: str, client_id: str, client_secret: str
    ):
        """Set up OIDC client credentials authentication."""
        self.logger.info(
            f"Setting up OIDC Client Credentials Authentication with {client_id=}, {oidc_issuer=}, {len(client_secret)=}"
        )
        oidc_provider = OidcProviderInfo(issuer=oidc_issuer)
        client_info = OidcClientInfo(
            client_id=client_id,
            provider=oidc_provider,
            client_secret=client_secret,
        )
        self._authenticator = OidcClientCredentialsAuthenticator(
            client_info=client_info
        )

    @staticmethod
    def from_environ(
        environ: Optional[dict] = None, backend_id: Optional[str] = None
    ) -> "ElasticJobRegistry":
        """
        Factory to build an ElasticJobRegistry from an environment dict (or equivalent config).
        """
        # TODO: get most settings from a config file and secrets from env vars or the vault?
        environ = environ or os.environ

        backend_id = backend_id or environ.get("OPENEO_EJR_BACKEND_ID", "undefined")
        # TODO: HTTPS url for api? https://github.com/Open-EO/openeo-job-tracker-elastic-api/issues/7
        api_url = environ.get("OPENEO_EJR_API", "http://jobtracker.openeo.vgt.vito.be/")
        # TODO: get authentication settings and secrets from Vault?
        oidc_issuer = environ.get(
            "OPENEO_EJR_OIDC_ISSUER", "https://sso.terrascope.be/auth/realms/terrascope"
        )
        client_id = environ.get("OPENEO_EJR_OIDC_CLIENT_ID", "openeo-job-tracker")
        client_secret = environ.get("OPENEO_EJR_OIDC_CLIENT_SECRET")
        if not client_secret:
            raise RuntimeError("Env var 'OPENEO_EJR_OIDC_CLIENT_SECRET' must be set")

        ejr = ElasticJobRegistry(backend_id=backend_id, api_url=api_url)
        ejr.authenticate_oidc_client_credentials(
            oidc_issuer=oidc_issuer, client_id=client_id, client_secret=client_secret
        )
        return ejr

    def _get_access_token(self) -> str:
        if not self._authenticator:
            raise RuntimeError("No authentication set up")
        with TimingLogger(
            title=f"Requesting OIDC access_token ({self._authenticator.__class__.__name__})",
            logger=self.logger.info,
        ):
            tokens = self._authenticator.get_tokens()
        return tokens.access_token

    def _do_request(self, method: str, path: str, json: Union[dict, list]):
        """Do an HTTP request to Elastic Job Tracker service."""
        with TimingLogger(logger=self.logger.info, title=f"Request `{method} {path}`"):
            access_token = self._cache.get_or_call(
                key="api_access_token",
                callback=self._get_access_token,
                # TODO: finetune/optimize caching TTL? Detect TTl/expiry from JWT access token itself?
                ttl=30 * 60,
            )
            headers = {
                "User-Agent": f"openeo_driver/{self.__class__.__name__}/{openeo_driver._version.__version__}/{self._backend_id}",
                "Authorization": f"Bearer {access_token}",
            }
            url = url_join(self._api_url, path)
            self.logger.debug(f"Doing request to {url=} {headers.keys()=}")
            response = requests.request(
                method=method, url=url, json=json, headers=headers
            )
            self.logger.debug(f"Response on `{method} {path}`: {response!r}")
            response.raise_for_status()
            return response.json()

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
    ):
        """
        Store/Initialize a new job
        """
        if not job_id:
            job_id = generate_unique_id(prefix="j")
        # TODO: dates: as UTC ISO format or unix timestamp?
        created = rfc3339.datetime(dt.datetime.utcnow())
        job_data = {
            # Essential identifiers
            "backend_id": self._backend_id,
            "user_id": user_id,
            "job_id": job_id,
            # Essential job creation fields (as defined by openEO API)
            "process": process,
            "title": title,
            "description": description,
            # TODO: fields plan, budget?
            # Initialize essential status fields (as defined by openEO API)
            "status": "created",
            "created": created,
            "updated": created,
            # TODO: costs and usage?
            # job options: important, but not standardized (yet?)
            "job_options": job_options,
            # various internal/optional housekeeping fields
            "parent_id": parent_id,
            "api_version": api_version,
            # TODO: additional technical metadata, see https://github.com/Open-EO/openeo-api/issues/472
        }
        # TODO: keep multi-job support? https://github.com/Open-EO/openeo-job-tracker-elastic-api/issues/3
        # TODO: what to return? What does API return?  https://github.com/Open-EO/openeo-job-tracker-elastic-api/issues/3
        return self._do_request("POST", "/jobs", json=[job_data])

    def list_user_jobs(self, user_id: Optional[str]):
        # TODO: sorting, pagination?
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"backend_id": self._backend_id}},
                        {"term": {"user_id": user_id}},
                    ]
                }
            }
        }
        # TODO: what to return? What does API return?
        return self._do_request("POST", "/jobs/search", json=query)


class Main:
    """
    Simple toy CLI for development and testing

    Example usage:

        export OPENEO_EJR_OIDC_CLIENT_SECRET=foobar

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

        cli_list = subparsers.add_parser("list", help="List jobs for given user.")
        cli_list.add_argument("user_id", help="User id to filter on.")
        cli_list.set_defaults(func=self.list_user_jobs)

        cli_create = subparsers.add_parser("create", help="Create a new job.")
        cli_create.add_argument("user_id", help="User id to filter on.")
        cli_create.set_defaults(func=self.create_dummy_job)

        return cli.parse_args()

    def _get_job_registry(self) -> ElasticJobRegistry:
        return ElasticJobRegistry.from_environ(self.environ, backend_id="test")

    def list_user_jobs(self, args: argparse.Namespace):
        user_id = args.user_id
        ejr = self._get_job_registry()
        jobs = ejr.list_user_jobs(user_id=user_id)
        print(f"Found {len(jobs)} jobs for user {user_id!r}:")
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


if __name__ == "__main__":
    Main().main()
