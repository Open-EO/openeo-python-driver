"""Job registry client for the EJR jobs-v2 domain API."""

import datetime
import json
import urllib.parse
from typing import Dict, List, Optional, Tuple, Union

import requests
from openeo.util import TimingLogger, rfc3339, url_join

from openeo_driver.backend import JobListing
from openeo_driver.constants import JOB_STATUS
from openeo_driver.errors import JobNotFoundException
from openeo_driver.jobregistry import (
    ElasticJobRegistry,
    EjrApiError,
    EjrApiResponseError,
    JobDict,
    ejr_job_info_to_metadata,
)
from openeo_driver.util.logging import ExtraLoggingFilter


class ElasticJobRegistryV2(ElasticJobRegistry):
    """Use jobs-v2 without changing the legacy ElasticJobRegistry client.

    Paginated listings encode the API's structured ``next_page`` continuation
    in the ``cursor`` URL parameter. Non-paginated listings consume all pages.
    Updates and deletions read the current job ETag and use If-Match. Concurrent
    changes are reported as EjrApiResponseError (HTTP 412), not overwritten.
    """

    PAGINATION_URL_PARAM = "cursor"

    def _jobs_path(self, job_id: Optional[str] = None) -> str:
        path = "/v2/jobs"
        if job_id is not None:
            path += f"/{urllib.parse.quote(job_id, safe='')}"
        return path

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
        job_id = job_id or self.generate_job_id()
        created = rfc3339.now_utc()
        data = {
            "job_id": job_id,
            "backend_id": self.backend_id,
            "user_id": user_id,
            "process": self._preserialize(process) if self._preserialize_process else process,
            "title": title,
            "description": description,
            "status": JOB_STATUS.CREATED,
            "created": created,
            "updated": created,
            "job_options": job_options,
            "parent_id": parent_id,
            "application_id": None,
            "api_version": api_version,
        }
        with ExtraLoggingFilter.with_extra_logging(job_id=job_id):
            self._log.info(f"EJR creating {job_id=} {created=}")
            return self._do_request("POST", self._jobs_path(), json=data, expected_status=201)

    def _get_job(self, job_id: str, *, user_id: Optional[str] = None, fields: Optional[List[str]] = None) -> JobDict:
        job, _ = self._read_job(job_id, user_id=user_id)
        return job

    def _versioned_request(
        self,
        method: str,
        job_id: str,
        *,
        data: Optional[dict] = None,
        etag: Optional[str] = None,
        expected_status: int = 200,
    ) -> requests.Response:
        path = self._jobs_path(job_id)
        url = url_join(self._api_url, path)
        headers = {"Authorization": f"Bearer {self._access_token_helper.get_access_token()}"}
        if etag is not None:
            headers["If-Match"] = etag
        with TimingLogger(logger=self._log.debug, title=f"EJR Request `{method} {path}`"):

            def do_request():
                return self._session.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers,
                    timeout=self._REQUEST_TIMEOUT,
                )

            if self._debug_show_curl:
                self._log.debug(
                    "Equivalent curl command: %s",
                    self._as_curl(method=method, url=url, data=data, headers=headers),
                )
            try:
                response = do_request()
                if response.status_code == 401:
                    self._log.warning("EJR request got 401; invalidating token cache and retrying with fresh token")
                    self._access_token_helper.invalidate_cache()
                    headers["Authorization"] = f"Bearer {self._access_token_helper.get_access_token()}"
                    response = do_request()
            except Exception as exc:
                self._log.exception(f"Failed to do EJR API request `{method} {url}`: {exc!r}")
                raise EjrApiError(f"Failed to do EJR API request `{method} {url}`") from exc
            self._log.debug(f"EJR response on `{method} {path}`: {response.status_code!r}")
            if response.status_code != expected_status:
                raise EjrApiResponseError.from_response(response)
            return response

    def _read_job(
        self,
        job_id: str,
        user_id: Optional[str] = None,
        *,
        require_etag: bool = False,
    ) -> Tuple[JobDict, Optional[str]]:
        with ExtraLoggingFilter.with_extra_logging(job_id=job_id, user_id=user_id):
            try:
                response = self._versioned_request("GET", job_id)
            except EjrApiResponseError as exc:
                if exc.status_code == 404:
                    raise JobNotFoundException(job_id=job_id) from exc
                raise
            job = response.json()
            assert job["job_id"] == job_id
            if job.get("backend_id") != self.backend_id:
                raise JobNotFoundException(job_id=job_id)
            assert user_id is None or job["user_id"] == user_id
            etag = response.headers.get("ETag")
            if require_etag and not etag:
                raise EjrApiError("EJR jobs-v2 response is missing an ETag; refusing an unconditional write")
            if "process" in job:
                job["process"] = self._postdeserialize(job["process"])
            return job, etag

    def _update(self, job_id: str, data: dict) -> JobDict:
        with ExtraLoggingFilter.with_extra_logging(job_id=job_id):
            _, etag = self._read_job(job_id, require_etag=True)
            self._log.info(f"EJR update {job_id=} {data=}")
            return self._versioned_request("PATCH", job_id, data=data, etag=etag).json()

    def delete_job(self, job_id: str, *, user_id: Optional[str] = None, verify_deletion: bool = True) -> None:
        with ExtraLoggingFilter.with_extra_logging(job_id=job_id):
            _, etag = self._read_job(job_id, user_id=user_id, require_etag=True)
            try:
                self._versioned_request("DELETE", job_id, etag=etag, expected_status=204)
            except EjrApiResponseError as exc:
                if exc.status_code == 404:
                    raise JobNotFoundException(job_id=job_id) from exc
                raise
            self._log.info(f"EJR deleted {job_id=}")
            if verify_deletion:
                self._verify_job_existence(job_id=job_id, user_id=user_id, exists=False)

    def _list_jobs(
        self,
        endpoint: str,
        parameters: dict,
        *,
        fields: Optional[List[str]] = None,
        limit: int = 1000,
        next_page: Optional[dict] = None,
    ) -> dict:
        if not 1 <= limit <= 1000:
            raise ValueError("jobs-v2 listing limit must be between 1 and 1000")
        body = {
            **parameters,
            "backend_id": self.backend_id,
            "fields": sorted(set(fields or []) | {"job_id", "user_id", "created", "status", "updated"}),
            "limit": limit,
        }
        if fields and "*" in fields:
            del body["fields"]
        if next_page is not None:
            body["next_page"] = next_page
        return self._do_request("POST", self._jobs_path() + "/" + endpoint, json=body, retry=True)

    def _collect_jobs(
        self,
        endpoint: str,
        parameters: dict,
        *,
        fields: Optional[List[str]] = None,
        max_count: Optional[int] = None,
    ) -> List[JobDict]:
        jobs = []
        next_page = None
        limit = 1000 if max_count is None else min(1000, max_count)
        while max_count is None or len(jobs) < max_count:
            page = self._list_jobs(endpoint, parameters, fields=fields, limit=limit, next_page=next_page)
            jobs.extend(page["jobs"])
            next_page = page.get("next_page")
            if not next_page:
                break
        return jobs if max_count is None else jobs[:max_count]

    def list_user_jobs(
        self,
        user_id: str,
        *,
        fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        request_parameters: Optional[dict] = None,
    ) -> Union[JobListing, List[JobDict]]:
        parameters = {"user_id": user_id}
        if limit is None:
            return self._collect_jobs("user", parameters, fields=fields)
        cursor = (request_parameters or {}).get(self.PAGINATION_URL_PARAM)
        page = self._list_jobs(
            "user",
            parameters,
            fields=fields,
            limit=limit,
            next_page=json.loads(cursor) if cursor is not None else None,
        )
        next_page = page.get("next_page")
        return JobListing(
            jobs=[ejr_job_info_to_metadata(job, full=False) for job in page["jobs"]],
            next_parameters=(
                {"limit": limit, self.PAGINATION_URL_PARAM: json.dumps(next_page, separators=(",", ":"))}
                if next_page
                else None
            ),
        )

    def list_active_jobs(
        self,
        *,
        fields: Optional[List[str]] = None,
        max_age: Optional[int] = None,
        max_updated_ago: Optional[int] = None,
        require_application_id: bool = False,
        infra_id: Optional[str] = None,
    ) -> List[JobDict]:
        parameters = {"require_application_id": require_application_id}
        if max_age is not None:
            parameters["max_age"] = max_age
        if max_updated_ago is not None:
            parameters["max_updated_ago"] = max_updated_ago
        if infra_id is not None:
            parameters["infra_id"] = infra_id
        return self._collect_jobs("active", parameters, fields=fields)

    def get_all_started_jobs_before(
        self,
        upper: datetime.datetime,
        user_ids: Optional[List[str]] = None,
        field_whitelist: Optional[List[str]] = None,
        max_count: int = 1000,
    ) -> List[Dict]:
        parameters = {"updated_until": rfc3339.datetime(upper)}
        if user_ids:
            parameters["user_ids"] = user_ids
        return self._collect_jobs("cleanup-candidates", parameters, fields=field_whitelist, max_count=max_count)
