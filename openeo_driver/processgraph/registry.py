"""
Process registry setup, decorators, and standard process wiring.

Extracted from openeo_driver/ProcessGraphDeserializer.py.
"""
import logging
import math
from pathlib import Path
from typing import Any, Callable, List, Sequence, Union

import openeo_processes
from openeo.util import load_json
from openeo.utils.version import ComparableVersion

from openeo_driver.backend import OpenEoBackendImplementation, Processing
from openeo_driver.errors import OpenEOApiException
from openeo_driver.processes import DEFAULT_NAMESPACE, ProcessArgs, ProcessRegistry, ProcessSpec
from openeo_driver.specs import SPECS_ROOT
from openeo_driver.utils import EvalEnv
from openeo_driver.views import OPENEO_API_VERSION_DEFAULT

_log = logging.getLogger(__name__)


class NoPythonImplementationError(NotImplementedError):
    """
    Exception to use with the `ProcessRegistry.add_function` registration pattern
    for openEO processes that don't have an actual Python implementation.
    Typically used for callback processes that have an actual implementation elsewhere.
    (e.g in openeo-geotrellis-extensions).
    """
    pass


# Set up process registries (version dependent)

# Process registry based on 1.x version of openeo-processes, to be used with api_version 1.0 an 1.1
process_registry_100 = ProcessRegistry(
    spec_root=SPECS_ROOT / "openeo-processes/1.x", argument_names=["args", "env"], target_version="1.2.0"
)

# Process registry based on 2.x version of openeo-processes, to be used starting with api_version 1.2
process_registry_2xx = ProcessRegistry(
    spec_root=SPECS_ROOT / "openeo-processes/2.x", argument_names=["args", "env"], target_version="2.0.0-rc.1"
)


def _add_standard_processes(process_registry: ProcessRegistry, process_ids: List[str]):
    """
    Add standard processes as implemented by the openeo-processes-python project.
    """
    def wrap(process: Callable):
        """Adapter to connect the kwargs style of openeo-processes-python with ProcessArgs/EvalEnv"""

        def wrapped(args: ProcessArgs, env: EvalEnv):
            return process(**args.as_dict())

        return wrapped

    for pid in set(process_ids):
        if openeo_processes.has_process(pid):
            proc = openeo_processes.get_process(pid)
            wrapped = wrap(proc)
            spec = process_registry.load_predefined_spec(pid)
            process_registry.add_process(name=pid, function=wrapped, spec=spec)
        elif pid in _openeo_processes_extra:
            proc = _openeo_processes_extra[pid]
            wrapped = wrap(proc)
            spec = process_registry.load_predefined_spec(pid)
            process_registry.add_process(name=pid, function=wrapped, spec=spec)
        else:
            _log.warning("Adding process {p!r} without implementation".format(p=pid))
            process_registry.add_spec_by_name(pid)


_OPENEO_PROCESSES_PYTHON_WHITELIST = [
    'array_contains', 'array_element', 'array_filter', 'array_find', 'array_labels',
    'count', 'first', 'last', 'order', 'rearrange', 'sort',
    'between', 'eq', 'gt', 'gte', 'is_nan', 'is_nodata', 'is_valid', 'lt', 'lte', 'neq',
    'all', 'and', 'any', 'not', 'or', 'xor',
    'absolute', 'add', 'clip', 'divide', 'extrema', 'int', 'max', 'mean',
    'median', 'min', 'mod', 'multiply', 'power', 'product', 'quantiles', 'sd', 'sgn', 'sqrt',
    'subtract', 'sum', 'variance', 'e', 'pi', 'exp', 'ln', 'log',
    'ceil', 'floor', 'int', 'round',
    'arccos', 'arcosh', 'arcsin', 'arctan', 'arctan2', 'arsinh', 'artanh', 'cos', 'cosh', 'sin', 'sinh', 'tan', 'tanh',
    'all', 'any', 'count', 'first', 'last', 'max', 'mean', 'median', 'min', 'product', 'sd', 'sum', 'variance'
]

_openeo_processes_extra = {
    "pi": lambda: math.pi,
    "e": lambda: math.e,
}

_add_standard_processes(process_registry_100, _OPENEO_PROCESSES_PYTHON_WHITELIST)
_add_standard_processes(process_registry_2xx, _OPENEO_PROCESSES_PYTHON_WHITELIST)


# Type hint alias for a "process function":
# a Python function that implements some openEO process (as used in `apply_process`)
ProcessFunction = Callable[[Union[dict, ProcessArgs], EvalEnv], Any]


def process(f: ProcessFunction) -> ProcessFunction:
    """
    Decorator for registering a process function in the process registries.
    """
    process_registry_100.add_function(f)
    process_registry_2xx.add_function(f)
    return f


def simple_function(f: Callable) -> Callable:
    """
    Decorator for registering a simple process function in the process registries.
    """
    process_registry_100.add_simple_function(f)
    process_registry_2xx.add_simple_function(f)
    return f


def non_standard_process(
    spec: ProcessSpec, namespace: str = DEFAULT_NAMESPACE
) -> Callable[[ProcessFunction], ProcessFunction]:
    """Decorator for registering non-standard process functions"""

    def decorator(f: ProcessFunction) -> ProcessFunction:
        process_registry_100.add_function(f=f, spec=spec.to_dict_100(), namespace=namespace)
        process_registry_2xx.add_function(f=f, spec=spec.to_dict_100(), namespace=namespace)
        return f

    return decorator


def custom_process(f: ProcessFunction):
    """Decorator for custom processes (e.g. in custom_processes.py)."""
    process_registry_100.add_hidden(f)
    process_registry_2xx.add_hidden(f)
    return f


def custom_process_from_process_graph(
    process_spec: Union[dict, Path],
    *,
    process_registries: Sequence[ProcessRegistry] = (process_registry_100, process_registry_2xx),
    namespace: str = DEFAULT_NAMESPACE,
    hidden: bool = False,
):
    """
    Register a custom process from a process spec containing a "process_graph" definition
    """
    if isinstance(process_spec, Path):
        process_spec = load_json(process_spec)
    process_id = process_spec["id"]
    process_function = _process_function_from_process_graph(process_spec)
    for reg in process_registries:
        if hidden:
            reg.add_hidden(process_function, name=process_id, namespace=namespace)
        else:
            reg.add_function(process_function, name=process_id, spec=process_spec, namespace=namespace)


def _process_function_from_process_graph(process_spec: dict) -> ProcessFunction:
    """
    Build a process function (to be used in `apply_process`) from a given process spec with process graph
    """
    process_id = process_spec["id"]
    process_graph = process_spec["process_graph"]
    parameters = process_spec.get("parameters")

    def process_function(args: ProcessArgs, env: EvalEnv):
        # Import here to avoid circular import: evaluator imports registry, process_function uses evaluator
        from openeo_driver.processgraph.evaluator import _evaluate_process_graph_process
        return _evaluate_process_graph_process(
            process_id=process_id,
            process_graph=process_graph,
            parameters=parameters,
            args=args.as_dict(),
            env=env,
        )

    return process_function


def _register_fallback_implementations_by_process_graph(process_registry: ProcessRegistry):
    """
    Register process functions for (yet undefined) processes that have
    a process graph based fallback implementation in their spec
    """
    for name in process_registry.list_predefined_specs():
        spec = process_registry.load_predefined_spec(name)
        if "process_graph" in spec and not process_registry.contains(name):
            _log.debug(f"Registering fallback implementation of {name!r} by process graph ({process_registry})")
            custom_process_from_process_graph(process_spec=spec, process_registries=[process_registry])


# Some (env) string constants to simplify code navigation
ENV_SOURCE_CONSTRAINTS = "source_constraints"
ENV_DRY_RUN_TRACER = "dry_run_tracer"
ENV_FINAL_RESULT = "final_result"
ENV_SAVE_RESULT = "save_result"
ENV_MAX_BUFFER = "max_buffer"


class SimpleProcessing(Processing):
    """
    Simple graph processing: just implement basic math/logic operators
    (based on openeo-processes-python implementation)
    """

    # For lazy loading of (global) process registry
    _registry_cache = {}

    def get_process_registry(self, api_version: Union[str, ComparableVersion]) -> ProcessRegistry:
        api_version = ComparableVersion(api_version)
        if api_version.at_least("1.2.0"):
            spec = "openeo-processes/2.x"
        elif api_version.at_least("1.0.0"):
            spec = "openeo-processes/1.x"
        else:
            raise OpenEOApiException(message=f"No process support for openEO version {api_version}")
        if spec not in self._registry_cache:
            registry = ProcessRegistry(spec_root=SPECS_ROOT / spec, argument_names=["args", "env"])
            _add_standard_processes(registry, _OPENEO_PROCESSES_PYTHON_WHITELIST)
            # Import collect lazily to avoid circular import
            from openeo_driver.processgraph.evaluator import collect
            registry.add_hidden(collect)
            self._registry_cache[spec] = registry
        return self._registry_cache[spec]

    def get_basic_env(self, api_version: str = OPENEO_API_VERSION_DEFAULT) -> EvalEnv:
        return EvalEnv(
            {
                "backend_implementation": OpenEoBackendImplementation(processing=self),
                # TODO #382 Deprecated field "version", use "openeo_api_version" instead
                "version": api_version,
                "openeo_api_version": api_version,
                "node_caching": False,
            }
        )

    def evaluate(self, process_graph: dict, env: EvalEnv = None):
        from openeo_driver.processgraph.evaluator import evaluate
        return evaluate(process_graph=process_graph, env=env or self.get_basic_env(), do_dry_run=False)


class ConcreteProcessing(Processing):
    """
    Concrete process graph processing: (most) processes have concrete Python implementation
    (manipulating `DriverDataCube` instances)
    """

    def get_process_registry(self, api_version: Union[str, ComparableVersion]) -> ProcessRegistry:
        if ComparableVersion(api_version).at_least("1.2.0"):
            return process_registry_2xx
        elif ComparableVersion(api_version).at_least("1.0.0"):
            return process_registry_100
        else:
            raise OpenEOApiException(message=f"No process support for openEO version {api_version}")

    def evaluate(self, process_graph: dict, env: EvalEnv = None):
        from openeo_driver.processgraph.evaluator import evaluate
        return evaluate(process_graph=process_graph, env=env)

    def validate(self, process_graph: dict, env: EvalEnv = None):
        from openeo_driver.processgraph.evaluator import evaluate, _collect_end_nodes, convert_node
        from openeo_driver.dry_run import DryRunDataTracer
        from openeo.internal.process_graph_visitor import ProcessGraphVisitException, ProcessGraphVisitor

        dry_run_tracer = DryRunDataTracer()
        env = env.push({ENV_DRY_RUN_TRACER: dry_run_tracer, ENV_FINAL_RESULT: [None]})

        try:
            ProcessGraphVisitor.dereference_from_node_arguments(process_graph)
        except ProcessGraphVisitException as e:
            return [{"code": "ProcessGraphInvalid", "message": str(e)}]

        try:
            _log.info("Doing dry run")
            collected_process_graph, top_level_node_id = _collect_end_nodes(process_graph)
            top_level_node = collected_process_graph[top_level_node_id]
            result = convert_node(top_level_node, env=env.push({
                ENV_SAVE_RESULT: [],
                "node_caching": False
            }))
            source_constraints = dry_run_tracer.get_source_constraints()
            _log.info(f"Dry run extracted {len(source_constraints)} source constraints: {source_constraints}")
            env = env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
        except OpenEOApiException as e:
            _log.error(f"dry run phase of validation failed: {e!r}", exc_info=True)
            return [{"code": e.code, "message": str(e)}]
        except Exception as e:
            _log.error(f"dry run phase of validation failed: {e!r}", exc_info=True)
            return [{"code": "Internal", "message": str(e)}]

        errors = []
        source_constraints = dry_run_tracer.get_source_constraints()
        errors.extend(self.extra_validation(
            process_graph=process_graph,
            env=env,
            result=result,
            source_constraints=source_constraints
        ))
        return errors

    def extra_validation(self, process_graph, env, result, source_constraints):
        return []


# Import process implementations to trigger @process / @process_registry decorator registrations.
# This must happen AFTER the registry objects are defined above.
from openeo_driver.processgraph import process_implementations  # noqa: E402, F401

# Register fallback implementations from process specs that have a process_graph
_register_fallback_implementations_by_process_graph(process_registry_100)
_register_fallback_implementations_by_process_graph(process_registry_2xx)
