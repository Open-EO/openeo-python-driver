import dataclasses
import logging
from typing import List, NamedTuple, Optional, Union

import requests

from openeo_driver.errors import OpenEOApiException
from openeo_driver.util.http import is_http_url

_log = logging.getLogger(__name__)


class ProcessGraphFlatDict(dict):
    """
    Wrapper for the classic "flat dictionary" representation
    of an openEO process graph, e.g.

        {
            "lc1": {"process_id": "load_collection", "arguments": {...
            "rd1": {"process_id": "reduce_dimension", "arguments": {...
            ...
        }

    - To be used as type annotation where one wants to clarify
      what exact kind of dictionary-based process graph representation is expected.
    - Implemented as a subclass of `dict` to be directly compatible
      with existing, legacy code that expects a simple dictionary.
    """

    # TODO: move this to openeo python client library?
    pass


@dataclasses.dataclass(frozen=True)
class ProcessDefinition:
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

    # Default processing options as defined by the Processing Parameters Extension
    # (conformance class https://api.openeo.org/extensions/processing-parameters/0.1.0)
    default_job_options: Optional[dict] = None
    default_synchronous_options: Optional[dict] = None


def get_process_definition_from_url(process_id: str, url: str) -> ProcessDefinition:
    """
    Get a remote process definition (process graph, parameters, title, ...) from URL,
    which should provide:
    -   a JSON document with the process definition, compatible with
        the `GET /process_graphs/{process_graph_id}` openEO API endpoint.
    -   a JSON doc with process listing, compatible with
        the `GET /process_graphs` openEO API endpoint.
    """
    try:
        process_definition: ProcessDefinition = _get_process_definition_from_url(process_id=process_id, url=url)
    except OpenEOApiException:
        raise
    except Exception as e:
        raise OpenEOApiException(
            status_code=400,
            code="ProcessNamespaceInvalid",
            message=f"Process '{process_id}' specified with invalid namespace '{url}': {e!r}",
        ) from e

    return process_definition


def _get_process_definition_from_url(process_id: str, url: str) -> ProcessDefinition:
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

    # Support for fields from Processing Parameters Extension
    default_job_options = spec.get("default_job_options", None)
    default_synchronous_options = spec.get("default_synchronous_options", None)

    return ProcessDefinition(
        id=process_id,
        process_graph=spec["process_graph"],
        parameters=spec.get("parameters", []),
        returns=spec.get("returns"),
        default_job_options=default_job_options,
        default_synchronous_options=default_synchronous_options,
    )


def extract_default_job_options_from_process_graph(
    process_graph: ProcessGraphFlatDict, processing_mode: str = "batch_job"
) -> Union[dict, None]:
    """
    Extract default job options from a process definitions in process graph.
    based on "Processing Parameters" extension.

    :param process_graph: process graph in flat graph format
    :param processing_mode: "batch_job" or "synchronous"
    """

    job_options = []
    for node in process_graph.values():
        namespace = node.get("namespace")
        process_id = node["process_id"]
        if is_http_url(namespace):
            process_definition = get_process_definition_from_url(process_id=process_id, url=namespace)
            if processing_mode == "batch_job" and process_definition.default_job_options:
                job_options.append(process_definition.default_job_options)
            elif processing_mode == "synchronous" and process_definition.default_synchronous_options:
                job_options.append(process_definition.default_synchronous_options)

    if len(job_options) == 0:
        return None
    elif len(job_options) == 1:
        return job_options[0]
    else:
        # TODO: how to combine multiple default for same parameters?
        raise NotImplementedError(
            "Merging multiple default job options from different process definitions is not yet implemented."
        )
