"""
Process graph evaluation engine.

Extracted from openeo_driver/ProcessGraphDeserializer.py.
"""
import copy
import dataclasses
import functools
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from openeo.internal.process_graph_visitor import ProcessGraphVisitor, ProcessGraphVisitException
try:
    from openeo.internal.process_graph_visitor import ORIG_NODE_ID_KEY
except ImportError:
    # Older openeo versions use a different constant name or don't have it at all
    try:
        from openeo.internal.process_graph_visitor import DEREFERENCED_NODE_KEY as ORIG_NODE_ID_KEY
    except ImportError:
        ORIG_NODE_ID_KEY = "_node_id"
from openeo.util import TimingLogger

from openeo_driver.backend import ErrorSummary, UserDefinedProcessMetadata
from openeo_driver.dry_run import (
    DryRunDataCube,
    DryRunDataTracer,
    SourceConstraint,
    deduplicate_source_constraints,
)
from openeo_driver.errors import (
    OpenEOApiException,
    ProcessParameterRequiredException,
    ProcessUnsupportedException,
)
from openeo_driver.processes import ProcessArgs
from openeo_driver.processgraph.definitions import ProcessDefinition, get_process_definition_from_url
from openeo_driver.processgraph.registry import (
    ENV_DRY_RUN_TRACER,
    ENV_FINAL_RESULT,
    ENV_MAX_BUFFER,
    ENV_SAVE_RESULT,
    ENV_SOURCE_CONSTRAINTS,
)
from openeo_driver.util import UNSET
from openeo_driver.util.http import is_http_url
from openeo_driver.utils import EvalEnv, smart_bool
from openeo_driver.views import OPENEO_API_VERSION_DEFAULT

_log = logging.getLogger(__name__)

# TODO: Eliminate usage and definition of DEFAULT_TEMPORAL_EXTENT
#       (openeo-python-driver should stick to `None` to signal "unspecified", not prescribe some arbitrary value).
DEFAULT_TEMPORAL_EXTENT = ["1970-01-01", "2070-01-01"]


def _collect_end_nodes(process_graph: dict) -> (dict, str):
    end_node_ids = _end_node_ids(process_graph)
    top_level_node_id = "collect1"

    collected_process_graph = dict(
        process_graph,
        **{
            top_level_node_id: {
                "process_id": "collect",
                "arguments": {"end_nodes": [{"from_node": end_node_id} for end_node_id in sorted(end_node_ids)]},
            }
        },
    )

    ProcessGraphVisitor.dereference_from_node_arguments(collected_process_graph)
    return collected_process_graph, top_level_node_id


def _end_node_ids(process_graph: dict) -> set:
    all_node_ids = set(process_graph.keys())

    def get_from_node_ids(value) -> set:
        if isinstance(value, dict):
            if "from_node" in value:
                return {value["from_node"]}
            else:
                return {node_id for v in value.values() for node_id in get_from_node_ids(v)}
        if isinstance(value, list):
            return {node_id for v in value for node_id in get_from_node_ids(v)}
        return set()

    from_node_ids = {node_id
                     for node in process_graph.values()
                     for argument_value in node["arguments"].values()
                     for node_id in get_from_node_ids(argument_value)}

    return all_node_ids - from_node_ids


def evaluate(
        process_graph: dict,
        env: EvalEnv,
        do_dry_run: Union[bool, DryRunDataTracer] = True
):
    """
    Converts the json representation of a (part of a) process graph into the corresponding Python data cube.

    Warning: this function could manipulate the given process graph dict in-place (see `convert_node`).
    """
    if "version" not in env:
        env = env.push({"version": OPENEO_API_VERSION_DEFAULT})
    if "openeo_api_version" not in env:
        _log.warning(f"No 'openeo_api_version' in `evaluate()` env. Blindly assuming {OPENEO_API_VERSION_DEFAULT}.")
        env = env.push({"openeo_api_version": OPENEO_API_VERSION_DEFAULT})

    collected_process_graph, top_level_node_id = _collect_end_nodes(process_graph)
    top_level_node = collected_process_graph[top_level_node_id]
    if ENV_SAVE_RESULT not in env:
        env = env.push({ENV_SAVE_RESULT: []})

    env = env.push({ENV_FINAL_RESULT: [None], ENV_MAX_BUFFER: {}})

    if do_dry_run:
        dry_run_tracer = do_dry_run if isinstance(do_dry_run, DryRunDataTracer) else DryRunDataTracer()
        _log.info("Doing dry run")
        dry_run_env = env.push(
            {
                ENV_DRY_RUN_TRACER: dry_run_tracer,
                ENV_SAVE_RESULT: [],
                "node_caching": False
            }
        )
        dry_run_result = convert_node(top_level_node, env=dry_run_env)
        source_constraints = dry_run_tracer.get_source_constraints()
        _log.info(f"Dry run extracted {len(source_constraints)} source constraints: {source_constraints}")

        env = env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})

        with TimingLogger("evaluate:post_dry_run", logger=_log):
            post_dry_run_data = env.backend_implementation.post_dry_run(
                dry_run_result=dry_run_result,
                dry_run_tracer=dry_run_tracer,
                source_constraints=deduplicate_source_constraints(source_constraints),
            )
            if post_dry_run_data:
                env = env.push(post_dry_run_data)

    result = convert_node(top_level_node, env=env)
    if len(env[ENV_SAVE_RESULT]) > 0:
        if len(env[ENV_SAVE_RESULT]) == 1:
            return env[ENV_SAVE_RESULT][0]
        else:
            return env[ENV_SAVE_RESULT]
    else:
        return result


@dataclasses.dataclass(frozen=True)
class _ResultCachingMode:
    """
    How the reuse/caching of the processing result of a process graph node should be handled.
    """

    enabled: bool = True
    lazy_and_compare: bool = False

    @classmethod
    def from_env(cls, env: EvalEnv, process_id: str) -> "_ResultCachingMode":
        node_caching = env.get("node_caching", None)

        if node_caching is None:
            node_caching = "geopyspark-conservative"
            cls._warn(f"No 'node_caching' specified in env. Using {node_caching!r}.")
        elif node_caching is True:
            node_caching = "geopyspark-conservative"
            cls._warn(f"Legacy 'node_caching' value True. Using {node_caching!r}.")

        if node_caching == "geopyspark-conservative":
            return cls(enabled=process_id != "load_collection", lazy_and_compare=True)
        elif node_caching == "geopyspark-progressive":
            return cls(enabled=process_id != "load_collection", lazy_and_compare=False)
        elif node_caching == "full":
            return cls(enabled=True, lazy_and_compare=False)
        elif node_caching in (False, "off"):
            return cls(enabled=False, lazy_and_compare=False)
        else:
            raise ValueError(f"Unrecognized {node_caching=}")

    @staticmethod
    @functools.lru_cache
    def _warn(message: str):
        _log.warning(message)


def convert_node(processGraph: Union[dict, list], *, env: EvalEnv):
    """
    :param processGraph: process graph in nested representation
    Warning: this function could manipulate the given process graph dict in-place.
    """
    if isinstance(processGraph, dict):
        if 'process_id' in processGraph:
            process_id = processGraph['process_id']
            caching_mode = _ResultCachingMode.from_env(env=env, process_id=process_id)

            cached = processGraph.get("result_cache", UNSET) if caching_mode.enabled else UNSET

            if cached is UNSET or caching_mode.lazy_and_compare:
                process_result = apply_process(
                    process_id=process_id,
                    args=processGraph.get("arguments", {}),
                    namespace=processGraph.get("namespace", None),
                    env=env,
                    pg_node_id=processGraph.get(ORIG_NODE_ID_KEY),
                )
            else:
                process_result = cached

            if caching_mode.lazy_and_compare:
                if cached is not UNSET:
                    comparison = cached == process_result
                    if isinstance(comparison, bool) and comparison:
                        _log.info(f"Reusing an already evaluated subgraph for process {process_id}")
                        return cached

            if caching_mode.enabled:
                processGraph["result_cache"] = process_result

            if processGraph.get("result", False) and ENV_FINAL_RESULT in env:
                env[ENV_FINAL_RESULT][0] = process_result

            return process_result
        elif 'node' in processGraph:
            return convert_node(processGraph['node'], env=env)
        elif 'callback' in processGraph or 'process_graph' in processGraph:
            return processGraph
        elif 'from_parameter' in processGraph:
            try:
                parameters = env.collect_parameters()
                return parameters[processGraph['from_parameter']]
            except KeyError:
                raise ProcessParameterRequiredException(process="n/a", parameter=processGraph['from_parameter'])
        else:
            return {k: convert_node(v, env=env) for k, v in processGraph.items()}
    elif isinstance(processGraph, list):
        return [convert_node(x, env=env) for x in processGraph]
    return processGraph


def flatten_children_node_types(process_graph: Union[dict, list]):
    children_node_types = set()

    def recurse(graph):
        process_id = graph["node"]["process_id"]
        children_node_types.add(process_id)

        arguments = graph["node"]["arguments"]
        for arg_name in arguments:
            arg_value = arguments[arg_name]
            if isinstance(arg_value, dict) and "node" in arg_value:
                recurse(arg_value)

    recurse(process_graph)
    return children_node_types


def flatten_children_node_names(process_graph: Union[dict, list]):
    children_node_names = set()

    def recurse(graph):
        process_id = graph["from_node"]
        children_node_names.add(process_id)

        arguments = graph["node"]["arguments"]
        for arg_name in arguments:
            arg_value = arguments[arg_name]
            if isinstance(arg_value, dict) and "node" in arg_value:
                recurse(arg_value)

    recurse(process_graph)
    return children_node_names


def check_subgraph_for_data_mask_optimization(args: dict) -> bool:
    """
    Check if it is safe to early apply a mask on load_collection.
    """
    whitelist = {
        "drop_dimension",
        "filter_bands",
        "filter_bbox",
        "filter_spatial",
        "filter_temporal",
        "load_collection",
        "resample_spatial"
    }

    children_node_types = flatten_children_node_types(args["data"])
    if len(children_node_types.intersection(whitelist)) != len(children_node_types):
        return False

    data_children_node_names = flatten_children_node_names(args["data"])
    mask_children_node_names = flatten_children_node_names(args["mask"])
    if not data_children_node_names.isdisjoint(mask_children_node_names):
        _log.info("Overlap between data and mask node. Will not pre-apply mask on load_collections.")
        return False

    return True


def apply_process(
    process_id: str,
    args: dict,
    *,
    namespace: Union[str, None],
    env: EvalEnv,
    pg_node_id: Optional[str] = None,
):
    _log.debug(f"apply_process {process_id=}")
    parameters = env.collect_parameters()

    if process_id == "mask" and args.get("replacement", None) is None \
            and smart_bool(env.get("data_mask_optimization", True)):
        mask_node = args.get("mask", None)
        the_mask = convert_node(mask_node, env=env)
        dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
        if not dry_run_tracer and check_subgraph_for_data_mask_optimization(args):
            if not env.get("data_mask"):
                _log.debug(f"data_mask: env.push: {the_mask}")
                env = env.push(data_mask=the_mask)
                the_data = convert_node(args["data"], env=env)
                return the_data
        the_data = convert_node(args["data"], env=env)
        args = {"data": the_data, "mask": the_mask}
    elif process_id == "if":
        value = args.get("value")
        if convert_node(value, env=env):
            return convert_node(args.get("accept"), env=env)
        else:
            return convert_node(args.get("reject"), env=env)
    else:
        args = {name: convert_node(expr, env=env) for (name, expr) in sorted(args.items())}

    if is_http_url(namespace):
        if namespace.startswith("http://"):
            _log.warning(f"HTTP protocol for namespace based remote process definitions is discouraged: {namespace!r}")
        return evaluate_process_from_url(
            process_id=process_id, namespace=namespace, args=args, env=env
        )

    process_registry = env.backend_implementation.processing.get_process_registry(api_version=env.openeo_api_version())

    try:
        process_function = process_registry.get_function(process_id, namespace=(namespace or "backend"))
        return process_function(args=ProcessArgs(args, process_id=process_id, pg_node_id=pg_node_id), env=env)
    except ProcessUnsupportedException as e:
        pass
    except OpenEOApiException:
        raise
    except Exception as e:
        errorsummary = env.backend_implementation.summarize_exception(e)
        detail = f"{e!r}"
        if isinstance(errorsummary, ErrorSummary):
            detail = errorsummary.summary
        raise OpenEOApiException(f"Unexpected error during {process_id!r}: {detail}. The process had these arguments: {args!r} ") from e

    if namespace in ["user", None]:
        user = env.get("user")
        if user:
            udp = env.backend_implementation.user_defined_processes.get(user_id=user.user_id, process_id=process_id)
            if udp:
                if namespace is None:
                    _log.debug("Using process {p!r} from namespace 'user'.".format(p=process_id))
                return evaluate_udp(process_id=process_id, udp=udp, args=args, env=env)

    raise ProcessUnsupportedException(process=process_id, namespace=namespace)


def _evaluate_process_graph_process(
        process_id: str, process_graph: dict, parameters: List[dict], args: dict, env: EvalEnv
):
    """Evaluate a process specified as a process graph (e.g. user-defined process)"""
    args = args.copy()
    for param in parameters or []:
        name = param["name"]
        if name not in args:
            if "default" in param:
                args[name] = param["default"]
            else:
                raise ProcessParameterRequiredException(process=process_id, parameter=name)
    env = env.push_parameters(args)

    process_graph = copy.deepcopy(process_graph)
    return evaluate(process_graph, env=env, do_dry_run=False)


def evaluate_udp(process_id: str, udp: UserDefinedProcessMetadata, args: dict, env: EvalEnv):
    return _evaluate_process_graph_process(
        process_id=process_id, process_graph=udp.process_graph, parameters=udp.parameters,
        args=args, env=env
    )


def evaluate_process_from_url(process_id: str, namespace: str, args: dict, env: EvalEnv):
    """
    Load remote process definition from URL (provided through `namespace` property)
    """
    process_definition: ProcessDefinition = get_process_definition_from_url(process_id=process_id, url=namespace)

    return _evaluate_process_graph_process(
        process_id=process_id,
        process_graph=process_definition.process_graph,
        parameters=process_definition.parameters,
        args=args,
        env=env,
    )


# The `collect` process is registered as a hidden process in SimpleProcessing.
# It must be defined here (in evaluator) since it references ENV_FINAL_RESULT.
from openeo_driver.processes import ProcessArgs as _ProcessArgs
from openeo_driver.processgraph.registry import custom_process as _custom_process


@_custom_process
def collect(args: _ProcessArgs, env: EvalEnv):
    return env[ENV_FINAL_RESULT][0]
