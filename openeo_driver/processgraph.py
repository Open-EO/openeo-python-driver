import logging
import requests
from typing import NamedTuple, List, Optional
from openeo_driver.errors import OpenEOApiException

_log = logging.getLogger(__name__)


class ProcessDefinition(NamedTuple):
    """
    Like `UserDefinedProcessMetadata`, but with different defaults
    (e.g. process graph and parameters are required).
    """

    # Process id
    id: str
    # Flat-graph representation of the process
    process_graph: dict
    # List of parameters expected by the process
    parameters: List[dict]
    # Definition what the process returns
    returns: Optional[dict] = None


def get_process_definition_from_url(process_id: str, url: str) -> ProcessDefinition:
    """
    Get process definition (process graph, parameters, title, ...) from URL,
    which should provide:
    -   a JSON document with the process definition, compatible with
        the `GET /process_graphs/{process_graph_id}` openEO API endpoint.
    -   a JSON doc with process listing, compatible with
        the `GET /process_graphs` openEO API endpoint.
    """
    _log.debug(f"Trying to load process definition for {process_id=} from {url=}")
    # TODO: send headers, e.g. with custom user agent?
    # TODO: add/support caching. Add retrying too?
    res = requests.get(url=url)
    res.raise_for_status()
    doc = res.json()
    if not isinstance(doc, dict):
        raise ValueError(f"Process definition should be a JSON object, but got {type(doc)}.")

    # TODO: deeper validation (e.g. JSON Schema based)?
    if "id" in doc and "process_graph" in doc:
        _log.debug(f"Detected single process definition for {process_id=} at {url=}")
        spec = doc
        if spec["id"] != process_id:
            raise OpenEOApiException(
                status_code=400,
                code="ProcessIdMismatch",
                message=f"Mismatch between expected process {process_id!r} and process {spec['id']!r} defined at {url!r}.",
            )
    elif "processes" in doc and "links" in doc:
        _log.debug(f"Searching for {process_id=} at process listing {url=}")
        found = [
            p for p in doc["processes"] if isinstance(p, dict) and p.get("id") == process_id and p.get("process_graph")
        ]
        if len(found) != 1:
            raise OpenEOApiException(
                status_code=400,
                code="ProcessNotFound",
                message=f"Process {process_id!r} not found in process listing at {url!r}.",
            )
        spec = found[0]
    else:
        raise OpenEOApiException(
            status_code=400,
            code="ProcessNotFound",
            message=f"No valid process definition for {process_id!r} found at {url!r}.",
        )

    return ProcessDefinition(
        id=process_id,
        process_graph=spec["process_graph"],
        parameters=spec.get("parameters", []),
        returns=spec.get("returns"),
    )
