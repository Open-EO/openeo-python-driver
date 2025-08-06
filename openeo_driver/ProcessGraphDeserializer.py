# TODO: rename this module to something in snake case? It doesn't even implement a ProcessGraphDeserializer class.
# TODO: and related: separate generic process graph handling from more concrete openEO process implementations

# pylint: disable=unused-argument

import calendar
import copy
import datetime
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np

import openeo.udf
import openeo_processes
import pandas as pd
import pyproj
import shapely.geometry
import shapely.ops
from dateutil.relativedelta import relativedelta
from openeo.internal.process_graph_visitor import ProcessGraphVisitException, ProcessGraphVisitor
from openeo.metadata import CollectionMetadata
from openeo.util import load_json, rfc3339, str_truncate
from openeo.utils.version import ComparableVersion
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import GeometryCollection, MultiPolygon, mapping, shape

from openeo_driver import dry_run
from openeo_driver.backend import (
    LoadParameters,
    OpenEoBackendImplementation,
    Processing,
    UserDefinedProcessMetadata, ErrorSummary,
)
from openeo_driver.constants import RESAMPLE_SPATIAL_ALIGNS, RESAMPLE_SPATIAL_METHODS
from openeo_driver.datacube import (
    DriverDataCube,
    DriverMlModel,
    DriverVectorCube,
    SupportsRunUdf,
)
from openeo_driver.datastructs import ResolutionMergeArgs, SarBackscatterArgs
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.dry_run import DryRunDataCube, DryRunDataTracer, SourceConstraint
from openeo_driver.errors import (
    CollectionNotFoundException,
    FeatureUnsupportedException,
    OpenEOApiException,
    ProcessParameterInvalidException,
    ProcessParameterRequiredException,
    ProcessUnsupportedException,
)
from openeo_driver.processes import DEFAULT_NAMESPACE, ProcessArgs, ProcessRegistry, ProcessSpec
from openeo_driver.processgraph import ProcessDefinition, get_process_definition_from_url
from openeo_driver.save_result import (
    AggregatePolygonResult,
    AggregatePolygonSpatialResult,
    JSONResult,
    MlModelResult,
    NullResult,
    SaveResult,
    to_save_result,
)
from openeo_driver.specs import SPECS_ROOT, read_spec
from openeo_driver.util.date_math import month_shift
from openeo_driver.util.geometry import BoundingBox, geojson_to_geometry, geojson_to_multipolygon, spatial_extent_union
from openeo_driver.util.http import is_http_url
from openeo_driver.util.utm import auto_utm_epsg_for_geometry
from openeo_driver.utils import EvalEnv, smart_bool
from openeo_driver.views import OPENEO_API_VERSION_DEFAULT

DEFAULT_TEMPORAL_EXTENT = ["1970-01-01", "2070-01-01"]  # also a sentinel value for load_stac

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
        """Adapter to connect the kwargs style of openeo-processes-python with args/EvalEnv"""

        def wrapped(args: dict, env: EvalEnv):
            return process(**args)

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
            # TODO: this warning is triggered before logging is set up usually
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
    To be used as shortcut for all simple cases of

        @process_registry_100.add_function
        @process_registry_2xx.add_function
        def foo(args, env):
            ...
    """
    process_registry_100.add_function(f)
    process_registry_2xx.add_function(f)
    return f


def simple_function(f: Callable) -> Callable:
    """
    Decorator for registering a process function in the process registries.
    To be used as shortcut for all simple cases of

        @process_registry_100.add_simple_function
        @process_registry_2xx.add_simple_function
        def foo(args, env):
            ...
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

    :param process_spec: process spec dict or path to a JSON file,
        containing keys like "id", "process_graph", "parameter"
    :param process_registries: process registries to register to
    :param namespace: process namespace
    :param hidden: whether to register as hidden process
    """
    if isinstance(process_spec, Path):
        process_spec = load_json(process_spec)
    process_id = process_spec["id"]
    process_function = _process_function_from_process_graph(process_spec)
    for process_registry in process_registries:
        if hidden:
            process_registry.add_hidden(process_function, name=process_id, namespace=namespace)
        else:
            process_registry.add_function(process_function, name=process_id, spec=process_spec, namespace=namespace)


def _process_function_from_process_graph(process_spec: dict) -> ProcessFunction:
    """
    Build a process function (to be used in `apply_process`) from a given process spec with process graph

    :param process_spec: process spec dict, containing keys like "id", "process_graph", "parameter"
    :return: process function
    """
    process_id = process_spec["id"]
    process_graph = process_spec["process_graph"]
    parameters = process_spec.get("parameters")

    def process_function(args: dict, env: EvalEnv):
        return _evaluate_process_graph_process(
            process_id=process_id, process_graph=process_graph, parameters=parameters,
            args=args, env=env
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
        # Lazy load registry.
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
        return evaluate(process_graph=process_graph, env=env)

    def validate(self, process_graph: dict, env: EvalEnv = None) -> List[dict]:
        dry_run_tracer = DryRunDataTracer()
        env = env.push({ENV_DRY_RUN_TRACER: dry_run_tracer, ENV_FINAL_RESULT: [None]})

        try:
            ProcessGraphVisitor.dereference_from_node_arguments(process_graph)
        except ProcessGraphVisitException as e:
            return [{"code": "ProcessGraphInvalid", "message": str(e)}]

        try:
            # Same dry run logic as in evaluate()
            _log.info("Doing dry run")
            collected_process_graph, top_level_node_id = _collect_end_nodes(process_graph)
            top_level_node = collected_process_graph[top_level_node_id]
            result = convert_node(top_level_node, env=env.push({
                ENV_SAVE_RESULT: [],  # otherwise dry run and real run append to the same mutable result list
                "node_caching": False
            }))
            # TODO: work with a dedicated DryRunEvalEnv?
            source_constraints = dry_run_tracer.get_source_constraints()
            _log.info("Dry run extracted these source constraints: {s}".format(s=source_constraints))
            env = env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})
        except OpenEOApiException as e:
            _log.error(f"dry run phase of validation failed: {e!r}", exc_info=True)
            return [{"code": e.code, "message": str(e)}]
        except Exception as e:
            _log.error(f"dry run phase of validation failed: {e!r}", exc_info=True)
            return [{"code": "Internal", "message": str(e)}]

        errors = []
        # TODO: check other resources for errors, warnings?

        source_constraints = dry_run_tracer.get_source_constraints()
        errors.extend(self.extra_validation(
            process_graph=process_graph,
            env=env,
            result=result,
            source_constraints=source_constraints
        ))

        return errors

    def extra_validation(
            self, process_graph: dict, env: EvalEnv, result, source_constraints: List[SourceConstraint]
    ) -> Iterable[dict]:
        """
        Extra process graph validation

        :return: List (or generator) of validation error dicts (having at least a "code" and "message" field)
        """
        return []


def _collect_end_nodes(process_graph: dict) -> (dict, str):
    end_node_ids = _end_node_ids(process_graph)
    top_level_node_id = "collect1"  # the node where evaluation starts (not necessarily the result node)

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
) -> Union[DriverDataCube, Any]:
    """
    Converts the json representation of a (part of a) process graph into the corresponding Python data cube.

    Warning: this function could manipulate the given process graph dict in-place (see `convert_node`).
    """

    if "version" not in env:
        # TODO #382 Deprecated field "version", use "openeo_api_version" instead
        env = env.push({"version": OPENEO_API_VERSION_DEFAULT})
    if "openeo_api_version" not in env:
        _log.warning(f"No 'openeo_api_version' in `evaluate()` env. Blindly assuming {OPENEO_API_VERSION_DEFAULT}.")
        env = env.push({"openeo_api_version": OPENEO_API_VERSION_DEFAULT})

    collected_process_graph, top_level_node_id = _collect_end_nodes(process_graph)
    top_level_node = collected_process_graph[top_level_node_id]
    if ENV_SAVE_RESULT not in env:
        env = env.push({ENV_SAVE_RESULT: []})

    env = env.push({ENV_FINAL_RESULT: [None], ENV_MAX_BUFFER:{}})  # mutable, holds final result of process graph

    if do_dry_run:
        dry_run_tracer = do_dry_run if isinstance(do_dry_run, DryRunDataTracer) else DryRunDataTracer()
        _log.info("Doing dry run")
        convert_node(top_level_node, env=env.push({
            ENV_DRY_RUN_TRACER: dry_run_tracer,
            ENV_SAVE_RESULT: [],  # otherwise dry run and real run append to the same mutable result list
            "node_caching": False
        }))
        # TODO: work with a dedicated DryRunEvalEnv?
        source_constraints = dry_run_tracer.get_source_constraints()
        _log.info("Dry run extracted these source constraints: {s}".format(s=source_constraints))
        env = env.push({ENV_SOURCE_CONSTRAINTS: source_constraints})

    result = convert_node(top_level_node, env=env)
    if len(env[ENV_SAVE_RESULT]) > 0:
        if len(env[ENV_SAVE_RESULT]) == 1:
            return env[ENV_SAVE_RESULT][0]
        else:
            # unpack to remain consistent with previous behaviour of returning results
            return env[ENV_SAVE_RESULT]
    else:
        return result


def convert_node(processGraph: Union[dict, list], env: EvalEnv = None):
    """

    Warning: this function could manipulate the given process graph dict in-place,
    e.g. by adding a "result_cache" key (see lower).
    """
    if isinstance(processGraph, dict):
        if 'process_id' in processGraph:
            process_id = processGraph['process_id']
            caching_flag = smart_bool(env.get("node_caching", True)) and process_id != "load_collection"
            cached = None
            if caching_flag and "result_cache" in processGraph:
                cached =  processGraph["result_cache"]

            process_result = apply_process(process_id=process_id, args=processGraph.get('arguments', {}),
                                           namespace=processGraph.get("namespace", None), env=env)
            if caching_flag:
                if cached is not None:
                    comparison = cached == process_result
                    #numpy arrays have a custom eq that requires this weird check
                    if isinstance(comparison,bool) and comparison:
                        _log.info(f"Reusing an already evaluated subgraph for process {process_id}")
                        return cached
                # TODO: this manipulates the process graph, while we often assume it's immutable.
                #       Adding complex data structures could also interfere with attempts to (re)encode the process graph as JSON again.
                processGraph["result_cache"] = process_result

            if processGraph.get("result", False) and ENV_FINAL_RESULT in env:
                env[ENV_FINAL_RESULT][0] = process_result

            return process_result
        elif 'node' in processGraph:
            return convert_node(processGraph['node'], env=env)
        elif 'callback' in processGraph or 'process_graph' in processGraph:
            # a "process_graph" object is a new process graph, don't evaluate it in the parent graph
            return processGraph
        elif 'from_parameter' in processGraph:
            try:
                parameters = env.collect_parameters()
                return parameters[processGraph['from_parameter']]
            except KeyError:
                raise ProcessParameterRequiredException(process="n/a", parameter=processGraph['from_parameter'])
        else:
            # TODO: Don't apply `convert_node` for some special cases (e.g. geojson objects)?
            return {k: convert_node(v, env=env) for k, v in processGraph.items()}
    elif isinstance(processGraph, list):
        return [convert_node(x, env=env) for x in processGraph]
    return processGraph


def _as_process_args(args: Union[dict, ProcessArgs], process_id: str = "n/a") -> ProcessArgs:
    """Adapter for legacy style args"""
    if not isinstance(args, ProcessArgs):
        args = ProcessArgs(args, process_id=process_id)
    elif process_id not in {args.process_id, "n/a"}:
        _log.warning(f"Inconsistent {process_id=} in extract_arg(): expected {args.process_id=}")
    return args


def extract_arg(args: ProcessArgs, name: str, process_id="n/a"):
    # TODO: eliminate this function, use `ProcessArgs.get_required()` directly
    return _as_process_args(args, process_id=process_id).get_required(name=name)

def _collection_crs(collection_id, env) -> str:
    metadata = None
    try:
        metadata = env.backend_implementation.catalog.get_collection_metadata(collection_id)
    except CollectionNotFoundException:
        return None
    crs = metadata.get('cube:dimensions', {}).get('x', {}).get('reference_system', None)
    return crs

def _collection_resolution(collection_id, env) -> str:
    try:
        metadata = env.backend_implementation.catalog.get_collection_metadata(collection_id)
    except CollectionNotFoundException:
        return None
    x = metadata.get('cube:dimensions', {}).get('x', {})
    y = metadata.get('cube:dimensions', {}).get('y', {})
    if ( "step" in x and "step" in y):
        return [x['step'], y['step']]
    else:
        return None


def _align_extent(extent,collection_id,env,target_resolution=None):
    metadata = None
    try:
        metadata = env.backend_implementation.catalog.get_collection_metadata(collection_id)
    except CollectionNotFoundException:
        pass

    # TODO #275 eliminate this VITO specific handling?
    if metadata is None or not metadata.get("_vito",{}).get("data_source", {}).get("realign", True):
        return extent

    crs = _collection_crs(collection_id, env)
    collection_resolution = _collection_resolution(collection_id, env)
    isUTM = crs == "AUTO:42001" or "Auto42001" in str(crs)

    x = metadata.get('cube:dimensions', {}).get('x', {})
    y = metadata.get('cube:dimensions', {}).get('y', {})

    if (target_resolution == None and collection_resolution == None):
        return extent


    if (    crs == 4326
            and extent.get('crs','') == "EPSG:4326"
            and "extent" in x and "extent" in y
            and (target_resolution == None or  target_resolution == collection_resolution)
    ):
        #only align to collection resolution
        target_resolution = collection_resolution
        def align(v, dimension, rounding, resolution):
            range = dimension.get('extent', [])
            if v < range[0]:
                v = range[0]
            elif v > range[1]:
                v = range[1]
            else:
                index = rounding((v - range[0]) / resolution)
                v = range[0] + index * resolution
            return v

        new_extent = {
            'west': align(extent['west'], x, math.floor,target_resolution[0]),
            'east': align(extent['east'], x, math.ceil,target_resolution[0]),
            'south': align(extent['south'], y, math.floor,target_resolution[1]),
            'north': align(extent['north'], y, math.ceil,target_resolution[1]),
            'crs': extent['crs']
        }
        _log.info(f"Realigned input extent {extent} into {new_extent}")

        return new_extent
    elif(isUTM):
        bbox = BoundingBox.from_dict(extent,default_crs=4326)
        bbox_utm = bbox.reproject_to_best_utm()

        new_extent = bbox_utm.round_to_resolution(target_resolution[0],target_resolution[1])

        _log.info(f"Realigned input extent {extent} into {new_extent}")
        return new_extent.as_dict()
    else:
        return extent

def _extract_load_parameters(env: EvalEnv, source_id: tuple) -> LoadParameters:
    """
    This is a side effect method that also removes source constraints from the list, which needs to happen in the right order!!
    Args:
        env:
        source_id:

    Returns:

    """
    source_constraints: List[SourceConstraint] = env[ENV_SOURCE_CONSTRAINTS]
    global_extent = None
    process_types = set()

    filtered_constraints = [c for c in source_constraints if c[0] == source_id]

    if len(filtered_constraints) == 0:
        raise Exception(f"Could not find source constraints for source {source_id}, available constraints are: {set([id for id,_ in source_constraints])}")

    if "global_extent" not in source_constraints[0][1]:

        for collection_id, constraint in source_constraints:
            extent = None
            if "spatial_extent" in constraint:
                extent = constraint["spatial_extent"]
            elif "weak_spatial_extent" in constraint:
                extent = constraint["weak_spatial_extent"]
            if extent is not None:
                collection_crs = _collection_crs(collection_id[1][0], env)
                crs = constraint.get("resample", {}).get("target_crs", collection_crs) or collection_crs
                target_resolution = constraint.get("resample", {}).get("resolution", None) or _collection_resolution(collection_id[1][0], env)

                if "pixel_buffer" in constraint:

                    buffer = constraint["pixel_buffer"]["buffer_size"]

                    if (crs is not None) and target_resolution:
                        bbox = BoundingBox.from_dict(extent, default_crs=4326)
                        extent = bbox.reproject(crs).as_dict()

                        extent = {
                            "west": extent["west"] - target_resolution[0] * math.ceil(buffer[0]),
                            "east": extent["east"] + target_resolution[0] * math.ceil(buffer[0]),
                            "south": extent["south"] - target_resolution[1] * math.ceil(buffer[1]),
                            "north": extent["north"] + target_resolution[1] * math.ceil(buffer[1]),
                            "crs": extent["crs"]
                        }
                    else:
                        _log.warning("Not applying buffer to extent because the target CRS is not known.")

                load_collection_in_native_grid = "resample" not in constraint or crs == collection_crs
                if (not load_collection_in_native_grid) and collection_crs is not None and ("42001" in str(collection_crs)):
                    #resampling auto utm to utm means we are loading in native grid
                    try:
                        load_collection_in_native_grid = "UTM zone" in CRS.from_user_input(crs).to_wkt()
                    except CRSError as e:
                        pass


                if  load_collection_in_native_grid:
                    # Ensure that the extent that the user provided is aligned with the collection's native grid.

                    extent = _align_extent(extent, collection_id[1][0], env,target_resolution)

                global_extent = spatial_extent_union(global_extent, extent) if global_extent else extent

        #store global extent in all constraints, too ensure the same extent everywhere
        for collection_id, constraint in source_constraints:
            constraint["global_extent"] = global_extent

    process_types = filtered_constraints[0][1].get("process_types", None)
    if not process_types:
        process_types = set()
        for _, constraint in filtered_constraints:
            if "process_type" in constraint:
                process_types |= set(constraint["process_type"])
        for _, constraint in filtered_constraints:
            constraint["process_types"] = process_types


    max_buffer_cache = env[ENV_MAX_BUFFER]

    max_buffer = None
    if source_id not in max_buffer_cache:
        #cache is important for correctness, because we need to compute over all source constraints

        for _, constraint in filtered_constraints:

            buffer = constraint.get("pixel_buffer", {}).get("buffer_size", None)
            if buffer:
                if max_buffer is None:
                    max_buffer = buffer
                else:
                    max_buffer = [max(max_buffer[0], buffer[0]), max(max_buffer[1], buffer[1])]
        max_buffer_cache[source_id] = max_buffer
    else:
        max_buffer = max_buffer_cache[source_id]

    _, constraints = filtered_constraints.pop(0)
    source_constraints.remove((source_id,constraints))  # Side effect!

    params = LoadParameters()
    params.temporal_extent = constraints.get("temporal_extent", DEFAULT_TEMPORAL_EXTENT)
    labels_args = constraints.get("filter_labels", {})
    if("dimension" in labels_args and labels_args["dimension"] == "t"):
        params.filter_temporal_labels = labels_args.get("condition")
    params.spatial_extent = constraints.get("spatial_extent", {})
    params.global_extent = constraints.get("global_extent", {})
    params.bands = constraints.get("bands", None)
    params.properties = constraints.get("properties", {})
    params.aggregate_spatial_geometries = constraints.get("aggregate_spatial", {}).get("geometries")
    if params.aggregate_spatial_geometries is None:
        params.aggregate_spatial_geometries = constraints.get("filter_spatial", {}).get("geometries")
    params.sar_backscatter = constraints.get("sar_backscatter", None)
    params.process_types = process_types
    params.custom_mask = constraints.get("custom_cloud_mask", {})
    params.data_mask = env.get("data_mask", None)
    if params.data_mask:
        _log.debug(f"extracted data_mask {params.data_mask}")
    params.target_crs = constraints.get("resample", {}).get("target_crs",None)
    params.target_resolution = constraints.get("resample", {}).get("resolution", None)
    params.resample_method = constraints.get("resample", {}).get("method", "near")
    params.pixel_buffer = max_buffer
    return params


@process
def load_collection(args: dict, env: EvalEnv) -> DriverDataCube:
    collection_id = extract_arg(args, 'id')

    # Sanitized arguments
    arguments = {}
    if args.get("temporal_extent"):
        arguments["temporal_extent"] = _extract_temporal_extent(
            args, field="temporal_extent", process_id="load_collection"
        )
    if args.get("spatial_extent"):
        arguments["spatial_extent"] = _extract_bbox_extent(
            args, field="spatial_extent", process_id="load_collection", handle_geojson=True
        )
        # TODO when spatial_extent is geojson: additional mask_polygon operation? https://github.com/Open-EO/openeo-python-driver/issues/49
    if args.get("bands"):
        arguments["bands"] = extract_arg(args, "bands", process_id="load_collection")
    if args.get("properties"):
        arguments["properties"] = extract_arg(args, 'properties', process_id="load_collection")

    if args.get("featureflags"):
        arguments["featureflags"] = extract_arg(args, 'featureflags', process_id="load_collection")

    metadata = env.backend_implementation.catalog.get_collection_metadata(collection_id)

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        return dry_run_tracer.load_collection(
            collection_id=collection_id, arguments=arguments, metadata=metadata, env=env
        )
    else:
        # Extract basic source constraints.
        # TODO #275: eliminate this VITO specific handling?
        properties = {**CollectionMetadata(metadata).get("_vito", "properties", default={}),
                      **arguments.get("properties", {})}

        source_id = dry_run.DataSource.load_collection(
            collection_id=collection_id, properties=properties, bands=arguments.get("bands", []), env=env
        ).get_source_id()
        load_params = _extract_load_parameters(env, source_id=source_id)
        # Override with explicit arguments
        load_params.update(arguments)
        return env.backend_implementation.catalog.load_collection(collection_id, load_params=load_params, env=env)

@non_standard_process(
    ProcessSpec(id='load_disk_data', description="Loads arbitrary from disk. This process is deprecated, considering using load_uploaded_files or load_stac.")
        .param(name='format', description="the file format, e.g. 'GTiff'", schema={"type": "string"}, required=True)
        .param(name='glob_pattern', description="a glob pattern that matches the files to load from disk",
               schema={"type": "string"}, required=True)
        .param(name='options', description="options specific to the file format", schema={"type": "object"})
        .returns(description="the data as a data cube", schema={})
)
def load_disk_data(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    """
    Deprecated, use load_uploaded_files or load_stac
    """
    _log.warning("DEPRECATED: load_disk_data usage")
    kwargs = dict(
        glob_pattern=args.get_required("glob_pattern", expected_type=str),
        format=args.get_required("format", expected_type=str),
        options=args.get_optional("options", default={}, expected_type=dict),
    )
    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        return dry_run_tracer.load_disk_data(**kwargs)
    else:
        source_id = dry_run.DataSource.load_disk_data(**kwargs).get_source_id()
        load_params = _extract_load_parameters(env, source_id=source_id)
        return env.backend_implementation.load_disk_data(**kwargs, load_params=load_params, env=env)


def _check_geometry_path_assumption(path: str, process: str, parameter: str):
    if isinstance(path, str) and path.lstrip().startswith("{"):
        raise ProcessParameterInvalidException(
            parameter=parameter,
            process=process,
            reason=f"provided a string (to be handled as path/URL), but it looks like (Geo)JSON encoded data: {str_truncate(path, width=32)!r}.",
        )


@non_standard_process(
    ProcessSpec(id='vector_buffer', description="Add a buffer around a geometry.")
        .param(name='geometries', description="Input geometry to add buffer to, vector-cube or GeoJSON(deprecated).",
               schema=[{
                "type": "object",
                "subtype": "datacube",
                "dimensions": [
                    {
                        "type": "geometry"
                    }
                ]
            },{"type": "object", "subtype": "geojson"}]
        )
        .param(name='distance', description="The distance of the buffer in meters. A positive distance expands the geometries, resulting in outward buffering (dilation), while a negative distance shrinks the geometries, resulting in inward buffering (erosion).\n\nIf the unit of the spatial reference system is not meters, a `UnitMismatch` error is thrown. Use ``vector_reproject()`` to convert the geometries to a suitable spatial reference system.",
               schema={"type": "number"}, required=True)
        .returns(description="Output geometry (GeoJSON object) with the added or subtracted buffer",
                 schema={"type": "object", "subtype": "geojson"})
)
def vector_buffer(args: ProcessArgs, env: EvalEnv) -> dict:
    if "geometry" in args and "geometries" not in args:
        # TODO drop legacy support for non-standard arg
        _log.warning("DEPRECATED: vector_buffer expects `geometries` argument, not `geometry`")
        geometry = args.get_required("geometry")
    else:
        geometry = args.get_required("geometries")
    distance = args.get_required("distance", expected_type=(int, float))
    if "unit" in args:
        # TODO resolve/eliminate non-official unit argument
        _log.warning("vector_buffer: usage of non-standard 'unit' parameter")
    unit = args.get_optional("unit", default="meter")
    input_crs = output_crs = 'epsg:4326'
    buffer_resolution = 3

    # TODO #114 EP-3981 convert `geometry` to vector cube and move buffer logic to there
    if isinstance(geometry,DriverVectorCube):
        geoms = geometry.get_geometries()
        input_crs = geometry.get_crs()
    elif isinstance(geometry, str):
        _check_geometry_path_assumption(
            path=geometry, process="vector_buffer", parameter="geometry"
        )
        # TODO: assumption here that `geometry` is a path/url
        geoms = list(DelayedVector(geometry).geometries)
    elif isinstance(geometry, dict) and "type" in geometry:
        geometry_type = geometry["type"]
        if geometry_type == "FeatureCollection":
            geoms = [shape(feat["geometry"]) for feat in geometry["features"]]
        elif geometry_type == "GeometryCollection":
            geoms = [shape(geom) for geom in geometry["geometries"]]
        elif geometry_type in {"Polygon", "MultiPolygon", "Point", "MultiPoint", "LineString"}:
            geoms = [shape(geometry)]
        elif geometry_type == "Feature":
            geoms = [shape(geometry["geometry"])]
        else:
            raise ProcessParameterInvalidException(
                parameter="geometry", process="vector_buffer", reason=f"Invalid geometry type {geometry_type}."
            )

        if "crs" in geometry:
            _log.warning("Handling GeoJSON dict with (non-standard) crs field")
            try:
                crs_name = geometry["crs"]["properties"]["name"]
                input_crs = pyproj.crs.CRS.from_string(crs_name)
            except Exception:
                _log.error(f"Failed to parse input geometry CRS {crs_name!r}", exc_info=True)
                raise ProcessParameterInvalidException(
                    parameter="geometry", process="vector_buffer", reason=f"Failed to parse input geometry CRS."
                )
    else:
        raise ProcessParameterInvalidException(
            parameter="geometry", process="vector_buffer", reason="The input geometry cannot be parsed"
        )
    geoms = gpd.GeoSeries(geoms, crs=input_crs)

    unit_scaling = {"meter": 1, "kilometer": 1000}
    if unit not in unit_scaling:
        raise ProcessParameterInvalidException(
            parameter="unit", process="vector_buffer",
            reason=f"Invalid unit {unit!r}. Should be one of {list(unit_scaling.keys())}."
        )
    distance = distance * unit_scaling[unit]

    epsg_utmzone = auto_utm_epsg_for_geometry(geoms.geometry[0])

    #TODO in the official spec, we have to throw an exception rather than reproject implicitly
    poly_buff_latlon = geoms.to_crs(epsg_utmzone).buffer(distance, resolution=buffer_resolution).to_crs(output_crs)

    empty_result_indices = np.where(poly_buff_latlon.is_empty)[0]
    if empty_result_indices.size > 0:
        raise ProcessParameterInvalidException(
            parameter="geometry", process="vector_buffer",
            reason=f"Buffering with distance {distance} {unit} resulted in empty geometries "
                   f"at position(s) {empty_result_indices}"
        )

    return mapping(poly_buff_latlon[0]) if len(poly_buff_latlon) == 1 else mapping(poly_buff_latlon)


@process
def apply_neighborhood(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    process = args.get_deep("process", "process_graph", expected_type=dict)
    size = args.get_required("size")
    overlap = args.get_optional("overlap")
    context = args.get_optional("context", default=None)
    return data_cube.apply_neighborhood(process=process, size=size, overlap=overlap, env=env, context=context)


@process
def apply_dimension(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=(DriverDataCube, DriverVectorCube))
    process = args.get_deep("process", "process_graph", expected_type=dict)
    dimension = args.get_required(
        "dimension", expected_type=str, validator=ProcessArgs.validator_one_of(data_cube.get_dimension_names())
    )
    target_dimension = args.get_optional("target_dimension", default=None, expected_type=str)
    context = args.get_optional("context", default=None)

    cube = data_cube.apply_dimension(
        process=process, dimension=dimension, target_dimension=target_dimension, context=context, env=env
    )
    if target_dimension is not None and target_dimension not in cube.metadata.dimension_names():
        cube = cube.rename_dimension(dimension, target_dimension)
    return cube


@process
def save_result(args: ProcessArgs, env: EvalEnv) -> SaveResult:  # TODO: return type no longer holds
    data = args.get_required("data")
    format = args.get_required("format", expected_type=str)
    options = args.get_optional("options", expected_type=dict, default={})

    if isinstance(data, SaveResult):
        # TODO: Is this an expected code path? `save_result` should be terminal node in a graph
        #       so chaining `save_result` calls should not be valid
        # https://github.com/Open-EO/openeo-geopyspark-driver/issues/295
        data = data.with_format(format, options)
        if ENV_SAVE_RESULT in env:
            env[ENV_SAVE_RESULT].append(data)
        return data
    else:
        result = to_save_result(data, format=format, options=options)
        if ENV_SAVE_RESULT in env:
            env[ENV_SAVE_RESULT].append(result)
            return data
        else:
            return result


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/save_ml_model.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/save_ml_model.json"))
def save_ml_model(args: ProcessArgs, env: EvalEnv) -> MlModelResult:
    data = args.get_required("data", expected_type=DriverMlModel)
    options = args.get_optional("options", default={}, expected_type=dict)
    return MlModelResult(ml_model=data, options=options)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/load_ml_model.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/load_ml_model.json"))
def load_ml_model(args: ProcessArgs, env: EvalEnv) -> DriverMlModel:
    if env.get(ENV_DRY_RUN_TRACER):
        return DriverMlModel()
    job_id = args.get_required("id", expected_type=str)
    return env.backend_implementation.load_ml_model(job_id)


@process
def apply(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    """
    Applies a unary process (a local operation) to each value of the specified or all dimensions in the data cube.
    """
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    apply_pg = args.get_deep("process", "process_graph", expected_type=dict)
    context = args.get_optional("context", default=None)
    return data_cube.apply(process=apply_pg, context=context, env=env)


@process
def reduce_dimension(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    reduce_pg = args.get_deep("reducer", "process_graph", expected_type=dict)
    dimension = args.get_required(
        "dimension", expected_type=str, validator=ProcessArgs.validator_one_of(data_cube.get_dimension_names())
    )
    context = args.get_optional("context", default=None)
    return data_cube.reduce_dimension(reducer=reduce_pg, dimension=dimension, context=context, env=env)


@process_registry_100.add_function(
    spec=read_spec("openeo-processes/experimental/chunk_polygon.json"), name="chunk_polygon"
)
def chunk_polygon(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    # TODO #229 deprecate this process and promote the "apply_polygon" name.
    #       See https://github.com/Open-EO/openeo-processes/issues/287, https://github.com/Open-EO/openeo-processes/pull/298
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    reduce_pg = args.get_deep("process", "process_graph", expected_type=dict)
    chunks = args.get_required("chunks")
    mask_value = args.get_optional("mask_value", expected_type=(int, float), default=None)
    context = args.get_optional("context", default=None)

    # Chunks parameter check.
    # TODO #114 EP-3981 normalize first to vector cube and simplify logic
    if isinstance(chunks, DelayedVector):
        polygons = list(chunks.geometries)
        for p in polygons:
            if not isinstance(p, shapely.geometry.Polygon):
                reason = "{m!s} is not a polygon.".format(m=p)
                raise ProcessParameterInvalidException(parameter='chunks', process='chunk_polygon', reason=reason)
        polygon = MultiPolygon(polygons)
    elif isinstance(chunks, shapely.geometry.base.BaseGeometry):
        polygon = MultiPolygon(chunks)
    elif isinstance(chunks, dict):
        polygon = geojson_to_multipolygon(chunks)
        if isinstance(polygon, shapely.geometry.Polygon):
            polygon = MultiPolygon([polygon])
    elif isinstance(chunks, str):
        # Delayed vector is not supported yet.
        reason = "Polygon of type string is not yet supported."
        raise ProcessParameterInvalidException(parameter='chunks', process='chunk_polygon', reason=reason)
    else:
        reason = "Polygon type is not supported."
        raise ProcessParameterInvalidException(parameter='chunks', process='chunk_polygon', reason=reason)
    if polygon.area == 0:
        reason = "Polygon {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
        raise ProcessParameterInvalidException(parameter='chunks', process='chunk_polygon', reason=reason)

    return data_cube.chunk_polygon(reducer=reduce_pg, chunks=polygon, mask_value=mask_value, context=context, env=env)


@process_registry_100.add_function(spec=read_spec("openeo-processes/2.x/proposals/apply_polygon.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/apply_polygon.json"))
def apply_polygon(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    process = args.get_deep("process", "process_graph", expected_type=dict)
    if "polygons" in args and "geometries" not in args:
        # TODO remove this deprecated  "polygons" parameter handling when not used anymore
        _log.warning(
            "DEPRECATED: In process 'apply_polygon': parameter 'polygons' is deprecated, use 'geometries' instead."
        )
        geometries = args.get_required("polygons")
    else:
        geometries = args.get_required("geometries")
    mask_value = args.get_optional("mask_value", expected_type=(int, float), default=None)
    context = args.get_optional("context", default=None)

    # TODO #114 EP-3981 normalize first to vector cube and simplify logic
    # TODO #288: this logic (copied from original chunk_polygon implementation) coerces the input polygons
    #       to a single MultiPolygon of pure (non-multi) polygons, which is conceptually wrong.
    #       Instead it should normalize to a feature collection or vector cube.
    if isinstance(geometries, DelayedVector):
        geometries = list(geometries.geometries)
        for p in geometries:
            if not isinstance(p, shapely.geometry.Polygon):
                reason = "{m!s} is not a polygon.".format(m=p)
                raise ProcessParameterInvalidException(parameter="polygons", process="apply_polygon", reason=reason)
        polygon = MultiPolygon(geometries)
    elif isinstance(geometries, DriverVectorCube):
        # TODO #288: I know it's wrong to coerce to MultiPolygon here, but we stick to this ill-defined API for now.
        polygon = geometries.to_multipolygon()
    elif isinstance(geometries, shapely.geometry.base.BaseGeometry):
        polygon = MultiPolygon(geometries)
    elif isinstance(geometries, dict):
        polygon = geojson_to_multipolygon(geometries)
        if isinstance(polygon, shapely.geometry.Polygon):
            polygon = MultiPolygon([polygon])
    else:
        reason = f"unsupported type: {type(geometries).__name__}"
        raise ProcessParameterInvalidException(parameter="polygons", process="apply_polygon", reason=reason)

    if polygon.area == 0:
        reason = "Polygon {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
        raise ProcessParameterInvalidException(parameter="polygons", process="apply_polygon", reason=reason)

    return data_cube.apply_polygon(polygons=polygon, process=process, mask_value=mask_value, context=context, env=env)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/fit_class_random_forest.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/fit_class_random_forest.json"))
def fit_class_random_forest(args: ProcessArgs, env: EvalEnv) -> DriverMlModel:
    # Keep it simple for dry run
    if env.get(ENV_DRY_RUN_TRACER):
        return DriverMlModel()

    predictors = extract_arg(args, 'predictors')
    if not isinstance(predictors, (AggregatePolygonSpatialResult, DriverVectorCube)):
        # TODO #114 EP-3981 drop AggregatePolygonSpatialResult support.
        raise ProcessParameterInvalidException(
            parameter="predictors",
            process="fit_class_random_forest",
            reason=f"should be non-temporal vector-cube, but got {type(predictors)}.",
        )

    target: Union[dict, DriverVectorCube] = extract_arg(args, "target")
    if isinstance(target, DriverVectorCube):
        # Convert target to geojson feature collection.
        target: dict = shapely.geometry.mapping(target.get_geometries())
    if not (isinstance(target, dict) and target.get("type") == "FeatureCollection"):
        raise ProcessParameterInvalidException(
            parameter="target",
            process="fit_class_random_forest",
            reason=f"expected feature collection or vector-cube value, but got {type(target)}.",
        )

    # TODO: get defaults from process spec?
    # TODO: do parameter checks automatically based on process spec?
    num_trees = args.get("num_trees", 100)
    if not isinstance(num_trees, int) or num_trees < 0:
        raise ProcessParameterInvalidException(
            parameter="num_trees", process="fit_class_random_forest",
            reason="should be an integer larger than 0."
        )
    max_variables = args.get("max_variables") or args.get('mtry')
    seed = args.get("seed")
    if not (seed is None or isinstance(seed, int)):
        raise ProcessParameterInvalidException(
            parameter="seed", process="fit_class_random_forest", reason="should be an integer"
        )

    return predictors.fit_class_random_forest(
        target=target, num_trees=num_trees, max_variables=max_variables, seed=seed,
    )


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/fit_class_catboost.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/fit_class_catboost.json"))
def fit_class_catboost(args: ProcessArgs, env: EvalEnv) -> DriverMlModel:
    process = "fit_class_catboost"
    if env.get(ENV_DRY_RUN_TRACER):
        return DriverMlModel()

    predictors = extract_arg(args, "predictors")
    if not isinstance(predictors, (AggregatePolygonSpatialResult, DriverVectorCube)):
        raise ProcessParameterInvalidException(
            parameter="predictors",
            process=process,
            reason=f"should be non-temporal vector-cube, but got {type(predictors)}.",
        )

    target: Union[dict, DriverVectorCube] = extract_arg(args, "target")
    if isinstance(target, DriverVectorCube):
        # Convert target to geojson feature collection.
        target: dict = shapely.geometry.mapping(target.get_geometries())
    if not (isinstance(target, dict) and target.get("type") == "FeatureCollection"):
        raise ProcessParameterInvalidException(
            parameter="target",
            process=process,
            reason=f"expected feature collection or vector-cube value, but got {type(target)}.",
        )

    # TODO: get defaults from process spec?
    # TODO: do parameter checks automatically based on process spec?
    def get_validated_parameter(args, param_name, default_value, expected_type, min_value=1, max_value=1000):
        return args.get_optional(
            param_name,
            default=default_value,
            expected_type=expected_type,
            validator=ProcessArgs.validator_generic(
                lambda v: v >= min_value and v <= max_value,
                error_message=f"The `{param_name}` parameter should be an integer between {min_value} and {max_value}.",
            ),
        )

    iterations = get_validated_parameter(args, "iterations", 5, int, 1, 500)
    depth = get_validated_parameter(args, "depth", 5, int, 1, 16)
    seed = get_validated_parameter(args, "seed", 0, int, 0, 2**31 - 1)

    return predictors.fit_class_catboost(target=target, iterations=iterations, depth=depth, seed=seed)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/predict_random_forest.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/predict_random_forest.json"))
def predict_random_forest(args: ProcessArgs, env: EvalEnv):
    raise NoPythonImplementationError


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/predict_catboost.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/predict_catboost.json"))
def predict_catboost(args: ProcessArgs, env: EvalEnv):
    raise NoPythonImplementationError


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/predict_probabilities.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/predict_probabilities.json"))
def predict_probabilities(args: ProcessArgs, env: EvalEnv):
    raise NoPythonImplementationError


@process
def add_dimension(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    return data_cube.add_dimension(
        name=args.get_required("name", expected_type=str),
        label=args.get_required("label", expected_type=str),
        type=args.get_optional("type", default="other", expected_type=str),
    )


@process
def drop_dimension(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    name: str = args.get_required("name", expected_type=str)
    return cube.drop_dimension(name=name)


@process
def dimension_labels(args: ProcessArgs, env: EvalEnv) -> List[str]:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    dimension: str = args.get_required("dimension", expected_type=str)
    return cube.dimension_labels(dimension=dimension)


@process
def rename_dimension(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    source: str = args.get_required("source", expected_type=str)
    target: str = args.get_required("target", expected_type=str)
    return cube.rename_dimension(source=source, target=target)


@process
def rename_labels(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    dimension: str = args.get_required("dimension", expected_type=str)
    target: List = args.get_required("target", expected_type=list)
    source: Optional[list] = args.get_optional("source", default=None, expected_type=list)
    return cube.rename_labels(dimension=dimension, target=target, source=source)


@process
def aggregate_temporal(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    intervals = args.get_required("intervals")
    reduce_pg = args.get_deep("reducer", "process_graph", expected_type=dict)
    labels = args.get_optional("labels", default=None)
    dimension = args.get_optional(
        "dimension",
        default=lambda: data_cube.metadata.temporal_dimension.name,
        validator=ProcessArgs.validator_one_of(data_cube.get_dimension_names()),
    )
    context = args.get_optional("context", default=None)

    return data_cube.aggregate_temporal(
        intervals=intervals, labels=labels, reducer=reduce_pg, dimension=dimension, context=context
    )


@process
def aggregate_temporal_period(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    period = args.get_required("period")
    reduce_pg = args.get_deep("reducer", "process_graph", expected_type=dict)
    dimension = args.get_optional(
        "dimension",
        default=lambda: data_cube.metadata.temporal_dimension.name,
        validator=ProcessArgs.validator_one_of(data_cube.get_dimension_names()),
    )
    context = args.get_optional("context", default=None)

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        intervals = []
    else:
        temporal_extent = data_cube.metadata.temporal_dimension.extent
        start = temporal_extent[0]
        end = temporal_extent[1]
        intervals = _period_to_intervals(start, end, period)

    return data_cube.aggregate_temporal(intervals=intervals, labels=None, reducer=reduce_pg, dimension=dimension,
                                        context=context)


def _period_to_intervals(start, end, period) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    from datetime import timedelta
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if start.tzinfo:
        start = start.tz_convert(None)
    if end.tzinfo:
        end = end.tz_convert(None)

    # TODO: "hour" support?
    if "day" == period:
        offset = timedelta(days=1)
        start_dates = pd.date_range(start - offset, end, freq="D", inclusive="left")
        end_dates = pd.date_range(start, end + offset, freq="D", inclusive="left")
        intervals = zip(start_dates, end_dates)
    elif "week" == period:
        offset = timedelta(weeks=1)
        start_dates = pd.date_range(start - offset, end, freq="W", inclusive="left")
        end_dates = pd.date_range(start, end + offset, freq="W", inclusive="left")
        intervals = zip(start_dates, end_dates)
    elif "dekad" == period:
        offset = timedelta(days=10)
        start_dates = pd.date_range(start - offset, end, freq="MS", inclusive="left")
        ten_days = pd.Timedelta(days=10)
        first_dekad_month = [(date, date + ten_days) for date in start_dates]
        second_dekad_month = [(date + ten_days, date + ten_days + ten_days) for date in start_dates]
        end_month = [date + pd.Timedelta(days=calendar.monthrange(date.year, date.month)[1]) for date in start_dates]
        third_dekad_month = list(zip([date + ten_days + ten_days for date in start_dates], end_month))
        intervals = (first_dekad_month + second_dekad_month + third_dekad_month)
        intervals.sort(key=lambda t: t[0])
    elif "month" == period:
        periods = pd.period_range(start, end, freq="M")
        intervals = [(p.to_timestamp(), month_shift(p.to_timestamp(), months=1)) for p in periods]
    elif "season" == period:
        # Shift start to season start months (Mar=3, Jun=6, Sep=9, Dec=12)
        season_start = month_shift(start, months=-(start.month % 3))
        periods = pd.period_range(season_start, end, freq="3M")
        intervals = [(p.to_timestamp(), month_shift(p.to_timestamp(), months=3)) for p in periods]
    elif "tropical-season" == period:
        # Shift start to season start months (May=5, Nov=11)
        season_start = month_shift(start, months=-((start.month - 5) % 6))
        periods = pd.period_range(season_start, end, freq="6M")
        intervals = [(p.to_timestamp(), month_shift(p.to_timestamp(), months=6)) for p in periods]
    elif "year" == period:
        offset = timedelta(weeks=52)
        start_dates = pd.date_range(
            start - offset, end, freq="A-DEC", inclusive="left"
        ) + timedelta(days=1)
        end_dates = pd.date_range(
            start, end + offset, freq="A-DEC", inclusive="left"
        ) + timedelta(days=1)
        intervals = zip(start_dates, end_dates)
    # TODO: "decade" support?
    # TODO: "decade-ad" support?
    else:
        raise ProcessParameterInvalidException('period', 'aggregate_temporal_period',
                                               'No support for a period of type: ' + str(period))
    intervals = [i for i in intervals if i[0] < end]
    _log.info(f"aggregate_temporal_period input: [{start},{end}] - {period} intervals: {intervals}")
    return intervals


@process
def aggregate_spatial(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube = args.get_required("data", expected_type=DriverDataCube)
    reduce_pg = args.get_deep("reducer", "process_graph", expected_type=dict)
    target_dimension = args.get_optional("target_dimension", default=None)

    geoms = args.get_required("geometries")

    # TODO #114: convert all cases to DriverVectorCube first and just work with that
    if isinstance(geoms, DriverVectorCube):
        pass
    elif isinstance(geoms, DryRunDataCube):
        # TODO: properly support DriverVectorCube in dry run
        geoms = DriverVectorCube(geometries=gpd.GeoDataFrame(geometry=[]), cube=None)
    elif isinstance(geoms, dict):
        try:
            # Automatically convert inline GeoJSON to a vector cube #114/#141
            geoms = env.backend_implementation.vector_cube_cls.from_geojson(geoms)
        except Exception as e:
            _log.error(
                f"Failed to parse inline GeoJSON geometries in aggregate_spatial: {e!r}",
                exc_info=True,
            )
            raise ProcessParameterInvalidException(
                parameter="geometries",
                process="aggregate_spatial",
                reason="Failed to parse inline GeoJSON",
            )
    elif isinstance(geoms, (AggregatePolygonResult, DelayedVector)):
        geoms = geoms.to_driver_vector_cube()
    else:
        raise ProcessParameterInvalidException(
            parameter="geometries", process="aggregate_spatial", reason=f"Invalid type: {type(geoms)} ({geoms!r})"
        )
    return cube.aggregate_spatial(geometries=geoms, reducer=reduce_pg, target_dimension=target_dimension)


@process
def mask(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    mask: DriverDataCube = args.get_required("mask", expected_type=DriverDataCube)
    replacement = args.get_optional("replacement", default=None)
    return cube.mask(mask=mask, replacement=replacement)


@process
def mask_polygon(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube = args.get_required("data", expected_type=DriverDataCube)
    mask = args.get_required("mask")
    replacement = args.get_optional("replacement", default=None)
    inside = args.get_optional("inside", default=False)

    # TODO #114: instead of if-elif-else chain: generically "cast" to VectorCube first (e.g. for wide input
    #       support: GeoJSON, WKT, ...) and then convert to MultiPolygon?
    if isinstance(mask, DelayedVector):
        # TODO: avoid reading DelayedVector twice due to dry-run?
        # TODO #114 EP-3981 embed DelayedVector in VectorCube implementation
        polygon = shapely.ops.unary_union(list(mask.geometries))
    elif isinstance(mask, DriverVectorCube):
        polygon = mask.to_multipolygon()
    elif isinstance(mask, dict) and "type" in mask:
        # Assume GeoJSON
        polygon = geojson_to_multipolygon(mask)
    else:
        reason = f"Unsupported mask type {type(mask)}"
        raise ProcessParameterInvalidException(parameter="mask", process="mask_polygon", reason=reason)

    if polygon.area == 0:
        reason = "mask {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
        raise ProcessParameterInvalidException(parameter='mask', process='mask_polygon', reason=reason)

    cube = cube.mask_polygon(mask=polygon, replacement=replacement, inside=inside)
    return cube


def _extract_temporal_extent(args: dict, field="extent", process_id="filter_temporal") -> Tuple[str, str]:
    extent = extract_arg(args, name=field, process_id=process_id)
    if len(extent) != 2:
        raise ProcessParameterInvalidException(
            process=process_id, parameter=field, reason="should have length 2, but got {e!r}".format(e=extent)
        )

    start, end = extent[0], extent[1]

    if start is None and end is None:
        raise ProcessParameterInvalidException(
            process=process_id, parameter=field, reason="both start and end are null"
        )

    if (start is not None and end is not None
            and rfc3339.parse_date_or_datetime(end) < rfc3339.parse_date_or_datetime(start)):
        raise ProcessParameterInvalidException(
            process=process_id, parameter=field, reason="end '{e}' is before start '{s}'".format(e=end, s=start)
        )

    # TODO: convert to datetime? or at least normalize?
    return tuple(extent)


@process
def filter_temporal(args: dict, env: EvalEnv) -> DriverDataCube:
    cube = extract_arg(args, 'data')
    if not isinstance(cube, DriverDataCube):
        raise ProcessParameterInvalidException(
            parameter="data", process="filter_temporal",
            reason=f"Invalid data type {type(cube)!r} expected raster-cube."
        )
    extent = _extract_temporal_extent(args, field="extent", process_id="filter_temporal")
    return cube.filter_temporal(start=extent[0], end=extent[1])


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/filter_labels.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/filter_labels.json"))
def filter_labels(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    # TODO: validation that condition is a process graph construct
    condition = args.get_required("condition", expected_type=dict)
    dimension = args.get_required("dimension", expected_type=str)
    context = args.get_optional("context", default=None)
    return cube.filter_labels(condition=condition, dimension=dimension, context=context, env=env)


def _extract_bbox_extent(args: dict, field="extent", process_id="filter_bbox", handle_geojson=False) -> dict:
    extent = extract_arg(args, name=field, process_id=process_id)
    # TODO #114: support vector cube
    if handle_geojson and extent.get("type") in [
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",  # TODO #71 #114: deprecate GeometryCollection
        "Feature",
        "FeatureCollection",
    ]:
        if not _contains_only_polygons(extent):
            raise ProcessParameterInvalidException(
                parameter=field,
                process=process_id,
                reason="unsupported GeoJSON; requires at least one Polygon or MultiPolygon",
            )
        try:
            w, s, e, n = DriverVectorCube.from_geojson(extent).get_bounding_box()
        except Exception as e:
            raise ProcessParameterInvalidException(
                parameter=field,
                process=process_id,
                reason="GeoJSON is not valid: {e!r}".format(e=e),
            )
        # TODO: support (non-standard) CRS field in GeoJSON?
        d = {"west": w, "south": s, "east": e, "north": n, "crs": "EPSG:4326"}
    else:
        d = {
            k: extract_arg(extent, name=k, process_id=process_id)
            for k in ["west", "south", "east", "north"]
        }
        crs = extent.get("crs") or "EPSG:4326"
        if isinstance(crs, int):
            crs = "EPSG:{crs}".format(crs=crs)
        d["crs"] = crs
    return d


def _contains_only_polygons(geojson: dict) -> bool:
    if geojson["type"] in ["Polygon", "MultiPolygon"]:
        return True

    if geojson["type"] == "Feature":
        return _contains_only_polygons(geojson["geometry"])

    if geojson["type"] == "FeatureCollection":
        return all(_contains_only_polygons(feature) for feature in geojson["features"])

    return False


@process
def filter_bbox(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    spatial_extent = _extract_bbox_extent(args, "extent", process_id="filter_bbox")
    return cube.filter_bbox(**spatial_extent)


@process
def filter_spatial(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    geometries = args.get_required("geometries")

    if isinstance(geometries, dict):
        if "type" in geometries and geometries["type"] != "GeometryCollection":
            geometries = env.backend_implementation.vector_cube_cls.from_geojson(geometries)
        else:
            # TODO #71 #114 #268 EP-3981 phase out special handling of GeometryCollection
            geometries = geojson_to_geometry(geometries)
            if isinstance(geometries, GeometryCollection):
                polygons = [
                    geom.geoms[0] if isinstance(geom, MultiPolygon) else geom
                    for geom in geometries.geoms
                ]
                geometries = MultiPolygon(polygons)


    elif isinstance(geometries, DelayedVector):
        geometries = DriverVectorCube.from_fiona([geometries.path]).to_multipolygon()
    elif isinstance(geometries, DriverVectorCube):
        pass
    else:
        # TODO #114: support DriverVectorCube
        raise NotImplementedError(
            "filter_spatial does not support {g!r}".format(g=geometries)
        )
    return cube.filter_spatial(geometries)


@process
def filter_bands(args: ProcessArgs, env: EvalEnv) -> Union[DriverDataCube, DriverVectorCube]:
    cube: Union[DriverDataCube, DriverVectorCube] = args.get_required(
        "data", expected_type=(DriverDataCube, DriverVectorCube)
    )
    bands = args.get_required("bands", expected_type=list)
    return cube.filter_bands(bands=bands)


@process
def apply_kernel(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    kernel = np.asarray(args.get_required("kernel", expected_type=list))
    factor = args.get_optional("factor", default=1.0, expected_type=(int, float))
    border = args.get_optional("border", default=0, expected_type=int)
    replace_invalid = args.get_optional("replace_invalid", default=0, expected_type=(int, float))
    return cube.apply_kernel(kernel=kernel, factor=factor, border=border, replace_invalid=replace_invalid)


@process
def ndvi(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube = args.get_required("data", expected_type=DriverDataCube)
    nir = args.get_optional("nir", default="nir")
    red = args.get_optional("red", default="red")
    target_band = args.get_optional("target_band", default=None)
    return cube.ndvi(nir=nir, red=red, target_band=target_band)


@process
def resample_spatial(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    resolution = args.get_optional(
        "resolution",
        default=0,
        validator=lambda v: isinstance(v, (int, float)) or (isinstance(v, (tuple, list)) and len(v) == 2),
    )
    projection = args.get_optional("projection", default=None)
    method = args.get_enum("method", options=RESAMPLE_SPATIAL_METHODS, default="near")
    align = args.get_enum("align", options=RESAMPLE_SPATIAL_ALIGNS, default="upper-left")
    return cube.resample_spatial(resolution=resolution, projection=projection, method=method, align=align)


@process
def resample_cube_spatial(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    target: DriverDataCube = args.get_required("target", expected_type=DriverDataCube)
    method = args.get_enum("method", options=RESAMPLE_SPATIAL_METHODS, default="near")
    return cube.resample_cube_spatial(target=target, method=method)


@process
def merge_cubes(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube1 = args.get_required("cube1", expected_type=DriverDataCube)
    cube2 = args.get_required("cube2", expected_type=DriverDataCube)
    # TODO raise check if cubes overlap and raise exception if resolver is missing
    resolver_process = None
    if "overlap_resolver" in args:
        pg = args.get_deep("overlap_resolver", "process_graph")
        if len(pg) != 1:
            raise ProcessParameterInvalidException(
                parameter='overlap_resolver', process='merge_cubes',
                reason='This backend only supports overlap resolvers with exactly one process for now.')
        resolver_process = next(iter(pg.values()))["process_id"]
    return cube1.merge_cubes(cube2, resolver_process)


@process
def run_udf(args: dict, env: EvalEnv):
    # TODO: note: this implements a non-standard usage of `run_udf`: processing "vector" cube (direct JSON or from aggregate_spatial, ...)
    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    data = extract_arg(args, 'data')
    udf, runtime = _get_udf(args, env=env)
    context = args.get('context',{})

    # TODO: this is simple heuristic about skipping `run_udf` in dry-run mode. Does this have to be more advanced?
    # TODO: would it be useful to let user hook into dry-run phase of run_udf (e.g. hint about result type/structure)?
    if dry_run_tracer and isinstance(data, DryRunDataCube):
        # Note: Other data types do execute the UDF during the dry-run.
        # E.g. A DelayedVector (when the user directly provides geometries as input).
        # This way a weak_spatial_extent can be calculated from the UDF's output.
        return data.run_udf()

    if env.get("validation", False):
        raise FeatureUnsupportedException("run_udf is not supported in validation mode.")

    if isinstance(data, SupportsRunUdf) and data.supports_udf(udf=udf, runtime=runtime):
        _log.info(f"run_udf: data of type {type(data)} has direct run_udf support")
        return data.run_udf(udf=udf, runtime=runtime, context=context, env=env)

    # TODO #114 add support for DriverVectorCube
    if isinstance(data, AggregatePolygonResult):
        pass
    if isinstance(data, DriverVectorCube):
        # TODO: this is temporary stopgap measure, converting to old-style save results to stay backward compatible.
        #       Better have proper DriverVectorCube support in run_udf?
        #       How does that fit in UdfData and UDF function signatures?
        data = data.to_legacy_save_result()

    if isinstance(data, (DelayedVector, dict)):
        if isinstance(data, dict):
            data = DelayedVector.from_json_dict(data)
        collection = openeo.udf.FeatureCollection(id='VectorCollection', data=data.as_geodataframe())
        data = openeo.udf.UdfData(
            proj={"EPSG": data.crs.to_epsg()}, feature_collection_list=[collection], user_context=context
        )
    elif isinstance(data, JSONResult):
        st = openeo.udf.StructuredData(description="Dictionary data", data=data.get_data(), type="dict")
        data = openeo.udf.UdfData(structured_data_list=[st], user_context=context)
    elif isinstance(data, list):
        data = openeo.udf.UdfData(
            structured_data_list=[openeo.udf.StructuredData(description="Data list", data=data, type="list")],
            user_context=context
        )
    else:
        raise ProcessParameterInvalidException(
            parameter="data",
            process="run_udf",
            reason=f"Unsupported data type {type(data)}.",
        )

    _log.info(f"[run_udf] Running UDF {str_truncate(udf, width=256)!r} on {data!r}")
    result_data = env.backend_implementation.processing.run_udf(udf, data)
    _log.info(f"[run_udf] UDF resulted in {result_data!r}")

    result_collections = result_data.get_feature_collection_list()
    if result_collections != None and len(result_collections) > 0:
        geo_data = result_collections[0].data
        dataframe = geo_data
        if isinstance(geo_data, gpd.GeoSeries):
            dataframe = gpd.GeoDataFrame(geometry=geo_data)
        invalid_indexes = dataframe.index[~dataframe.is_valid].tolist()
        if len(invalid_indexes) > 0:
            raise OpenEOApiException(
                status_code=400,
                code="InvalidGeometry",
                message="UDF returned invalid polygons. This could "
                        + f"be due to the input or the code. Invalid index(es): {invalid_indexes}"
            )
        return DriverVectorCube.from_geodataframe(data=dataframe)
    structured_result = result_data.get_structured_data_list()
    if structured_result != None and len(structured_result)>0:
        if(len(structured_result)==1):
            return structured_result[0].data
        else:
            return [s.data for s in structured_result]

    raise ProcessParameterInvalidException(
            parameter='udf', process='run_udf',
            reason='The provided UDF should return exactly either a feature collection or a structured result but got: %s .'%str(result_data) )


@process
def linear_scale_range(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    # TODO: eliminate this top-level linear_scale_range process implementation (should be used as `apply` callback)
    _log.warning("DEPRECATED: linear_scale_range usage directly on cube is deprecated/non-standard.")
    cube: DriverDataCube = args.get_required("x", expected_type=DriverDataCube)
    # Note: non-standard camelCase parameter names (https://github.com/Open-EO/openeo-processes/issues/302)
    input_min = args.get_required("inputMin")
    input_max = args.get_required("inputMax")
    output_min = args.get_optional("outputMin", default=0.0)
    output_max = args.get_optional("outputMax", default=1.0)
    # TODO linear_scale_range is defined on GeopysparkDataCube, but not on DriverDataCube
    return cube.linear_scale_range(input_min, input_max, output_min, output_max)


@process
def constant(args: ProcessArgs, env: EvalEnv):
    return args.get_required("x")

@process
def array_apply(args: ProcessArgs, env: EvalEnv):
    data = args.get_required("data")
    p = args.get_required("process")
    c = args.get_optional("context",None)
    if not isinstance(p, dict) and not "process_graph" in p:
        raise ProcessParameterInvalidException(parameter="process", process="array_apply", reason=f"Parameter should be a process graph, but got {p}")
    if not isinstance(data,list):
        raise ProcessParameterInvalidException(parameter="data", process="array_apply", reason=f"Parameter should be a list, but got {data}")
    result = [ evaluate(p.get("process_graph"),env.push_parameters(dict(context=c,x=d,index=index))) for index,d in enumerate(data)]

    return result


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
    Check if it is safe to early apply a mask on load_collection. When the mask is applied early,
    some data tiles may be discarded before being loaded. But there may be no special filters between
    the load_collection and the mask node.
    """
    whitelist = {
        "drop_dimension",
        "filter_bands",
        "filter_bbox",
        "filter_spatial",
        "filter_temporal",
        "load_collection",
        "resample_spatial"#resampling will also be preapplied at load time, so can go together with masking
    }

    children_node_types = flatten_children_node_types(args["data"])
    # If children_node_types exists only out of whitelisted nodes, an intersection should have no effect.
    if len(children_node_types.intersection(whitelist)) != len(children_node_types):
        return False

    data_children_node_names = flatten_children_node_names(args["data"])
    mask_children_node_names = flatten_children_node_names(args["mask"])
    if not data_children_node_names.isdisjoint(mask_children_node_names):
        # To avoid an issue in integration tests:
        _log.info("Overlap between data and mask node. Will not pre-apply mask on load_collections.")
        return False

    return True


def apply_process(process_id: str, args: dict, namespace: Union[str, None], env: EvalEnv) -> DriverDataCube:
    _log.debug(f"apply_process {process_id} with {args}")
    parameters = env.collect_parameters()

    if process_id == "mask" and args.get("replacement", None) is None \
            and smart_bool(env.get("data_mask_optimization", True)):
        mask_node = args.get("mask", None)
        # evaluate the mask
        _log.debug(f"data_mask: convert_node(mask_node): {mask_node}")
        the_mask = convert_node(mask_node, env=env)
        dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
        if not dry_run_tracer and check_subgraph_for_data_mask_optimization(args):
            if not env.get("data_mask"):
                _log.debug(f"data_mask: env.push: {the_mask}")
                env = env.push(data_mask=the_mask)
                the_data = convert_node(args["data"], env=env)
                return the_data  # masking happens in scala when loading data
        the_data = convert_node(args["data"], env=env)
        args = {"data": the_data, "mask": the_mask}
    elif process_id == "if":
        #special handling: we only want to evaluate the branch that gets accepted
        value = args.get("value")
        if convert_node(value, env=env):
            return convert_node(args.get("accept"), env=env)
        else:
            return convert_node(args.get("reject"),env=env)
    else:
        # first we resolve child nodes and arguments in an arbitrary but deterministic order
        args = {name: convert_node(expr, env=env) for (name, expr) in sorted(args.items())}


    # when all arguments and dependencies are resolved, we can run the process
    if is_http_url(namespace):
        if namespace.startswith("http://"):
            _log.warning(f"HTTP protocol for namespace based remote process definitions is discouraged: {namespace!r}")
        # TODO: security aspects: only allow for certain users, only allow whitelisted domains, support content hash verification ...?

        return evaluate_process_from_url(
            process_id=process_id, namespace=namespace, args=args, env=env
        )

    process_registry = env.backend_implementation.processing.get_process_registry(api_version=env.openeo_api_version())

    try:
        process_function = process_registry.get_function(process_id, namespace=(namespace or "backend"))
        _log.debug(f"Applying process {process_id} to arguments {args}")
        #TODO: for API compliance, we would actually first need to check if a UDP with same name exists.
        # we would however prefer to avoid overriding predefined functions with UDP's.
        # if we want to do this, we require caching in UDP registry to avoid expensive UDP lookups. We only need to cache the list of UDP names for a given user.
        return process_function(args=ProcessArgs(args, process_id=process_id), env=env)
    except ProcessUnsupportedException as e:
        pass
    except OpenEOApiException:
        raise
    except Exception as e:
        errorsummary = env.backend_implementation.summarize_exception(e)
        detail = f"{e!r}"
        if isinstance(errorsummary,ErrorSummary):
            detail = errorsummary.summary
        raise OpenEOApiException(f"Unexpected error during {process_id!r} with {args!r}: {detail}") from e

    if namespace in ["user", None]:
        user = env.get("user")
        if user:
            # the DB-call can be cached if necessary, but how will a user be able to use a new pre-defined process of the same
            # name without renaming his UDP?
            udp = env.backend_implementation.user_defined_processes.get(user_id=user.user_id, process_id=process_id)
            if udp:
                if namespace is None:
                    _log.debug("Using process {p!r} from namespace 'user'.".format(p=process_id))
                return evaluate_udp(process_id=process_id, udp=udp, args=args, env=env)

    raise ProcessUnsupportedException(process=process_id, namespace=namespace)



@non_standard_process(
    ProcessSpec("read_vector", description="Reads vector data from a file or a URL.")
        .param(
            "filename",
            description="Vector file reference: a HTTP(S) URL or a file path.",
            schema=[
                {
                    "description": "Public URL to a vector file.",
                    "type": "string", "subtype": "uri",
                    "pattern": "^https?://",
                },
                {
                    "description": "File path (resolvable back-end-side) to a vector file.",
                    "type": "string", "subtype": "file-path",
                    "pattern": "^[^\r\n\\:'\"]+$",
                },
            ])
        .returns("GeoJSON-style feature collection", schema={"type": "object", "subtype": "geojson"})
)
def read_vector(args: ProcessArgs, env: EvalEnv) -> DelayedVector:
    # TODO #114 EP-3981: deprecated in favor of load_uploaded_files/load_external? https://github.com/Open-EO/openeo-processes/issues/322
    # TODO: better argument name than `filename`?
    _log.warning("DEPRECATED: read_vector usage")
    path = args.get_required("filename")
    _check_geometry_path_assumption(
        path=path, process="read_vector", parameter="filename"
    )
    return DelayedVector(path)


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/load_uploaded_files.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_uploaded_files.json"))
def load_uploaded_files(args: ProcessArgs, env: EvalEnv) -> Union[DriverVectorCube, DriverDataCube]:
    # TODO #114 EP-3981 process name is still under discussion https://github.com/Open-EO/openeo-processes/issues/322
    paths = args.get_required("paths", expected_type=list)
    format = args.get_required(
        "format",
        expected_type=str,
        validator=ProcessArgs.validator_file_format(formats=env.backend_implementation.file_formats()["input"]),
    )
    options = args.get_optional("options", default={})

    if DriverVectorCube.from_fiona_supports(format):
        return DriverVectorCube.from_fiona(paths, driver=format, options=options)
    elif format.lower() in {"GTiff"}:
        if len(paths) != 1:
            raise FeatureUnsupportedException(
                f"load_uploaded_files only supports a single raster of format {format!r}, you provided {paths}"
            )
        kwargs = dict(glob_pattern=paths[0], format=format, options=options)
        dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
        if dry_run_tracer:
            return dry_run_tracer.load_disk_data(**kwargs)
        else:
            source_id = dry_run.DataSource.load_disk_data(**kwargs).get_source_id()
            load_params = _extract_load_parameters(env, source_id=source_id)
            return env.backend_implementation.load_disk_data(**kwargs, load_params=load_params, env=env)

    else:
        raise FeatureUnsupportedException(f"Loading format {format!r} is not supported")


@non_standard_process(
    ProcessSpec(
        id="to_vector_cube",
        description="[EXPERIMENTAL:] Converts given data (e.g. GeoJson object) to a vector cube."
    )
    .param('data', description="GeoJson object.", schema={"type": "object", "subtype": "geojson"})
    .returns("vector-cube", schema={"type": "object", "subtype": "vector-cube"})
)
def to_vector_cube(args: ProcessArgs, env: EvalEnv):
    _log.warning("DEPRECATED: process to_vector_cube is deprecated, use load_geojson instead")
    # TODO: remove this experimental/deprecated process
    data = args.get_required("data")
    if isinstance(data, dict) and data.get("type") in {"Polygon", "MultiPolygon", "Feature", "FeatureCollection"}:
        return env.backend_implementation.vector_cube_cls.from_geojson(data)
    raise FeatureUnsupportedException(f"Converting {type(data)} to vector cube is not supported")


@process_registry_100.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_geojson.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_geojson.json"))
def load_geojson(args: ProcessArgs, env: EvalEnv) -> DriverVectorCube:
    data = args.get_required(
        "data",
        validator=ProcessArgs.validator_geojson_dict(
            # TODO: also allow LineString and MultiLineString?
            allowed_types=["Point", "MultiPoint", "Polygon", "MultiPolygon", "Feature", "FeatureCollection"]
        ),
    )
    # TODO: better default value for `properties`? https://github.com/Open-EO/openeo-processes/issues/448
    properties = args.get_optional("properties", default=[], expected_type=(list, tuple))
    vector_cube = env.backend_implementation.vector_cube_cls.from_geojson(data, columns_for_cube=properties)
    return vector_cube


@process_registry_100.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_url.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_url.json"))
def load_url(args: ProcessArgs, env: EvalEnv) -> DriverVectorCube:
    # TODO: Follow up possible `load_url` changes https://github.com/Open-EO/openeo-processes/issues/450 ?
    url = args.get_required("url", expected_type=str, validator=re.compile("^https?://").match)
    format = args.get_required(
        "format",
        expected_type=str,
        validator=ProcessArgs.validator_file_format(formats=env.backend_implementation.file_formats()["input"]),
    )
    options = args.get_optional("options", default={})

    if DriverVectorCube.from_fiona_supports(format):
        # TODO: for GeoJSON (and related) support `properties` option like load_geojson? https://github.com/Open-EO/openeo-processes/issues/450
        return DriverVectorCube.from_fiona(paths=[url], driver=format, options=options)
    else:
        raise FeatureUnsupportedException(f"Loading format {format!r} is not supported")


@non_standard_process(
    ProcessSpec("get_geometries", description="Reads vector data from a file or a URL or get geometries from a FeatureCollection")
        .param('filename', description="filename or http url of a vector file", schema={"type": "string"}, required=False)
        .param('feature_collection', description="feature collection", schema={"type": "object"}, required=False)
        .returns("TODO", schema={"type": "object", "subtype": "vector-cube"})
)
def get_geometries(args: Dict, env: EvalEnv) -> Union[DelayedVector, dict]:
    # TODO: standardize or deprecate this? #114 EP-3981 https://github.com/Open-EO/openeo-processes/issues/322
    feature_collection = args.get('feature_collection', None)
    path = args.get('filename', None)
    if path is not None:
        _check_geometry_path_assumption(
            path=path, process="get_geometries", parameter="filename"
        )
        return DelayedVector(path)
    else:
        return feature_collection


@non_standard_process(
    ProcessSpec("raster_to_vector", description="Converts this raster data cube into a vector data cube. The bounding polygon of homogenous areas of pixels is constructed.\n"
                                                "Only the first band is considered the others are ignored.", extra={"experimental": True})
        .param('data', description="A raster data cube.", schema={"type": "object", "subtype": "raster-cube"})
        .returns("vector-cube", schema={"type": "object", "subtype": "vector-cube"})
)
def raster_to_vector(args: ProcessArgs, env: EvalEnv):
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    # TODO: raster_to_vector is only defined on GeopysparkDataCube, not DriverDataCube
    return cube.raster_to_vector()


@non_standard_process(
    ProcessSpec("vector_to_raster", description="Creates a raster cube as output based on a vector cube. The values in the output raster cube are based on the numeric properties in the input vector cube.", extra={"experimental": True})
        .param('data', description="A vector data cube.", schema={"type": "object", "subtype": "vector-cube"})
        .param('target', description = "A raster data cube used as reference.", schema = {"type": "object", "subtype": "raster-cube"})
        .returns("raster-cube", schema={"type": "object", "subtype": "raster-cube"})
)
def vector_to_raster(args: dict, env: EvalEnv) -> DriverDataCube:
    input_vector_cube = extract_arg(args, "data")
    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        if not isinstance(input_vector_cube, DryRunDataCube):
            raise ProcessParameterInvalidException(
                parameter="data",
                process="vector_to_raster",
                reason=f"Invalid data type {type(input_vector_cube)!r} expected vector-cube.",
            )
        return input_vector_cube

    if "target_data_cube" in args:
        target = extract_arg(args, "target_data_cube")  # TODO: remove after full migration to use of 'target'
    else:
        target = extract_arg(args, "target")
    # TODO: to_driver_vector_cube is temporary. Remove it when vector cube is fully supported.
    if not isinstance(input_vector_cube, DriverVectorCube) and not hasattr(input_vector_cube, "to_driver_vector_cube"):
        raise ProcessParameterInvalidException(
            parameter="data",
            process="vector_to_raster",
            reason=f"Invalid data type {type(input_vector_cube)!r} expected vector-cube.",
        )
    if not isinstance(target, DriverDataCube):
        raise ProcessParameterInvalidException(
            parameter="target",
            process="vector_to_raster",
            reason=f"Invalid data type {type(target)!r} expected raster-cube.",
        )
    return env.backend_implementation.vector_to_raster(input_vector_cube, target)


def _get_udf(args, env: EvalEnv) -> Tuple[str, str]:
    udf = extract_arg(args, "udf")
    runtime = extract_arg(args, "runtime")
    version = args.get("version", None)

    available_runtimes = env.backend_implementation.udf_runtimes.get_udf_runtimes()
    available_runtime_names = list(available_runtimes.keys())
    # Make lookup case insensitive
    available_runtimes.update({k.lower(): v for k, v in available_runtimes.items()})

    if not runtime or runtime.lower() not in available_runtimes:
        raise OpenEOApiException(
            status_code=400, code="InvalidRuntime",
            message=f"Unsupported UDF runtime {runtime!r}. Should be one of {available_runtime_names}"
        )
    available_versions = list(available_runtimes[runtime.lower()]["versions"].keys())
    if version and version not in available_versions:
        raise OpenEOApiException(
            status_code=400, code="InvalidVersion",
            message=f"Unsupported UDF runtime version {runtime} {version!r}. Should be one of {available_versions} or null"
        )

    return udf, runtime


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

    # Make a deep copy of the process graph (as the `evaluate` pipeline might modify it)
    process_graph = copy.deepcopy(process_graph)
    return evaluate(process_graph, env=env, do_dry_run=False)


def evaluate_udp(process_id: str, udp: UserDefinedProcessMetadata, args: dict, env: EvalEnv):
    return _evaluate_process_graph_process(
        process_id=process_id, process_graph=udp.process_graph, parameters=udp.parameters,
        args=args, env=env
    )


def evaluate_process_from_url(process_id: str, namespace: str, args: dict, env: EvalEnv):
    """
    Load remote process definition from URL (provided through `namespace` property
    :param process_id: process id of process that should be available at given URL (namespace)
    :param namespace: URL of process definition
    """
    process_definition: ProcessDefinition = get_process_definition_from_url(process_id=process_id, url=namespace)

    return _evaluate_process_graph_process(
        process_id=process_id,
        process_graph=process_definition.process_graph,
        parameters=process_definition.parameters,
        args=args,
        env=env,
    )


@non_standard_process(
    ProcessSpec("sleep", description="Sleep for given amount of seconds (and just pass-through given data).")
        .param('data', description="Data to pass through.", schema={}, required=False)
        .param('seconds', description="Number of seconds to sleep.", schema={"type": "number"}, required=True)
        .returns("Original data", schema={})
)
def sleep(args: ProcessArgs, env: EvalEnv):
    data = args.get_required("data")
    seconds = args.get_required("seconds", expected_type=(int, float))
    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if not dry_run_tracer:
        _log.info("Sleeping {s} seconds".format(s=seconds))
        time.sleep(seconds)
    return data


@non_standard_process(
    # TODO: get spec directly from @process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/atmospheric_correction.json"))
    ProcessSpec(
        id='atmospheric_correction',
        description="Applies an atmospheric correction that converts top of atmosphere reflectance values into bottom of atmosphere/top of canopy reflectance values.",
        extra={
            "summary": "Apply atmospheric correction",
            "categories": ["cubes", "optical"],
            "experimental": True,
            "links": [
                {
                    "rel": "about",
                    "href": "https://bok.eo4geo.eu/IP1-7-1",
                    "title": "Atmospheric correction explained by EO4GEO body of knowledge."
                }
            ],
            "exceptions": {
                "DigitalElevationModelInvalid": {
                    "message": "The digital elevation model specified is either not a DEM or can't be used with the data cube given."
                }
            },
        }
    )
        .param('data', description="Data cube containing multi-spectral optical top of atmosphere reflectances to be corrected.", schema={"type": "object", "subtype": "raster-cube"})
        .param(name='method', description="The atmospheric correction method to use. To get reproducible results, you have to set a specific method.\n\nSet to `null` to allow the back-end to choose, which will improve portability, but reduce reproducibility as you *may* get different results if you run the processes multiple times.",                      schema={"type": "string"}, required=False)
        .param(name='elevation_model', description="The digital elevation model to use, leave empty to allow the back-end to make a suitable choice.", schema={"type": "string"}, required=False)
         # TODO #91 the following parameters deviate from the official atmospheric_correction spec
         # TODO: process parameters should be snake_case, not camelCase
        .param(name='missionId', description="non-standard mission Id, currently defaults to sentinel2",                      schema={"type": "string"}, required=False)
        .param(name='sza',       description="non-standard if set, overrides sun zenith angle values [deg]",                  schema={"type": "number"}, required=False)
        .param(name='vza',       description="non-standard if set, overrides sensor zenith angle values [deg]",               schema={"type": "number"}, required=False)
        .param(name='raa',       description="non-standard if set, overrides rel. azimuth angle values [deg]",                schema={"type": "number"}, required=False)
        .param(name='gnd',       description="non-standard if set, overrides ground elevation [km]",                          schema={"type": "number"}, required=False)
        .param(name='aot',       description="non-standard if set, overrides aerosol optical thickness [], usually 0.1..0.2", schema={"type": "number"}, required=False)
        .param(name='cwv',       description="non-standard if set, overrides water vapor [], usually 0..7",                   schema={"type": "number"}, required=False)
        # TODO: process parameters should be snake_case, not camelCase
        .param(name='appendDebugBands', description="non-standard if set to 1, saves debug bands",                            schema={"type": "number"}, required=False)
        .returns(description="the corrected data as a data cube", schema={"type": "object", "subtype": "raster-cube"})
)
def atmospheric_correction(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    method = args.get_optional("method", expected_type=str)
    elevation_model = args.get_optional("elevation_model", expected_type=str)
    mission_id = args.get_optional("missionId", expected_type=str)
    sza = args.get_optional("sza", expected_type=float)
    vza = args.get_optional("vza", expected_type=float)
    raa = args.get_optional("raa", expected_type=float)
    gnd = args.get_optional("gnd", expected_type=float)
    aot = args.get_optional("aot", expected_type=float)
    cwv = args.get_optional("cwv", expected_type=float)
    append_debug_bands = args.get_optional("appendDebugBands", expected_type=int)
    return cube.atmospheric_correction(
        method=method,
        elevation_model=elevation_model,
        options={
            "mission_id": mission_id,
            "sza": sza,
            "vza": vza,
            "raa": raa,
            "gnd": gnd,
            "aot": aot,
            "cwv": cwv,
            "append_debug_bands": append_debug_bands,
        },
    )


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/sar_backscatter.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/sar_backscatter.json"))
def sar_backscatter(args: ProcessArgs, env: EvalEnv):
    # Note: this default `sar_backscatter` implementation can be subject
    #       to deployment-specific overrides (e.g. through "custom_processes" functionality).
    #       For example to change possible coefficient values, defaults, etc.
    #       Also see https://github.com/Open-EO/openeo-python-driver/issues/376
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    kwargs = args.get_subset(
        names=[
            "coefficient",
            "elevation_model",
            "mask",
            "contributing_area",
            "local_incidence_angle",
            "ellipsoid_incidence_angle",
            "noise_removal",
            "options",
        ]
    )
    return cube.sar_backscatter(SarBackscatterArgs(**kwargs))


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/resolution_merge.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/resolution_merge.json"))
def resolution_merge(args: ProcessArgs, env: EvalEnv):
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    kwargs = args.get_subset(names=["method", "high_resolution_bands", "low_resolution_bands", "options"])
    return cube.resolution_merge(ResolutionMergeArgs(**kwargs))


@non_standard_process(
    ProcessSpec("discard_result", description="Discards given data. Used for side-effecting purposes.")
        .param('data', description="Data to discard.", schema={}, required=False)
        .returns("Nothing", schema={})
)
def discard_result(args: ProcessArgs, env: EvalEnv):
    # TODO: keep a reference to the discarded result?
    return NullResult()


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/mask_scl_dilation.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/mask_scl_dilation.json"))
def mask_scl_dilation(args: ProcessArgs, env: EvalEnv):
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    if hasattr(cube, "mask_scl_dilation"):
        the_args = args.copy()
        del the_args["data"]
        return cube.mask_scl_dilation(**the_args)
    else:
        return cube


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/to_scl_dilation_mask.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/to_scl_dilation_mask.json"))
def to_scl_dilation_mask(args: ProcessArgs, env: EvalEnv):
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    # Get default values for other args from spec
    spec = read_spec("openeo-processes/experimental/to_scl_dilation_mask.json")
    defaults = {param["name"]: param["default"] for param in spec["parameters"] if "default" in param}
    optionals = {
        arg: args.get_optional(arg, default=defaults[arg])
        for arg in [
            "erosion_kernel_size",
            "mask1_values",
            "mask2_values",
            "kernel1_size",
            "kernel2_size",
        ]
    }
    return cube.to_scl_dilation_mask(**optionals)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/mask_l1c.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/mask_l1c.json"))
def mask_l1c(args: ProcessArgs, env: EvalEnv):
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    if hasattr(cube, "mask_l1c"):
        return cube.mask_l1c()
    else:
        return cube


custom_process_from_process_graph(read_spec("openeo-processes/1.x/proposals/ard_normalized_radar_backscatter.json"))


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/array_append.json"))
@process_registry_2xx.add_function
def array_append(args: ProcessArgs, env: EvalEnv) -> list:
    raise NoPythonImplementationError


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/array_interpolate_linear.json"))
@process_registry_2xx.add_function
def array_interpolate_linear(args: ProcessArgs, env: EvalEnv) -> list:
    raise NoPythonImplementationError


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/date_shift.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/date_shift.json"))
def date_shift(args: ProcessArgs, env: EvalEnv) -> str:
    date = rfc3339.parse_date_or_datetime(args.get_required("date", expected_type=str))
    value = int(args.get_required("value", expected_type=int))
    unit = args.get_enum("unit", options={"year", "month", "week", "day", "hour", "minute", "second", "millisecond"})
    if unit == "millisecond":
        raise FeatureUnsupportedException(message="Millisecond unit is not supported in date_shift")
    shifted = date + relativedelta(**{unit + "s": value})
    if type(date) is datetime.date and type(shifted) is datetime.datetime:
        shifted = shifted.date()
    return rfc3339.normalize(shifted)


@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/date_between.json"))
def date_between(args: ProcessArgs, env: EvalEnv) -> bool:
    raise NoPythonImplementationError


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/array_concat.json"))
@process_registry_2xx.add_function
def array_concat(args: ProcessArgs, env: EvalEnv) -> list:
    array1 = args.get_required(name="array1", expected_type=list)
    array2 = args.get_required(name="array2", expected_type=list)
    return list(array1) + list(array2)


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/array_create.json"))
@process_registry_2xx.add_function
def array_create(args: ProcessArgs, env: EvalEnv) -> list:
    data = args.get_required("data", expected_type=list)
    repeat = args.get_optional(
        name="repeat",
        default=1,
        expected_type=int,
        validator=ProcessArgs.validator_generic(
            lambda v: v >= 1, error_message="The `repeat` parameter should be an integer of at least value 1."
        ),
    )
    return list(data) * repeat


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/load_result.json"))
def load_result(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    _log.warning("DEPRECATED: load_result usage")
    job_id = args.get_required("id", expected_type=str)
    user = env.get("user")

    arguments = {}
    if args.get("temporal_extent"):
        arguments["temporal_extent"] = _extract_temporal_extent(
            args, field="temporal_extent", process_id="load_result"
        )
    if args.get("spatial_extent"):
        arguments["spatial_extent"] = _extract_bbox_extent(
            args, field="spatial_extent", process_id="load_result", handle_geojson=True
        )
    if args.get("bands"):
        arguments["bands"] = extract_arg(args, "bands", process_id="load_result")

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        return dry_run_tracer.load_result(job_id, arguments)
    else:
        source_id = dry_run.DataSource.load_result(job_id).get_source_id()
        load_params = _extract_load_parameters(env, source_id=source_id)

        return env.backend_implementation.load_result(job_id=job_id, user_id=user.user_id if user is not None else None,
                                                      load_params=load_params, env=env)


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/inspect.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/inspect.json"))
def inspect(args: ProcessArgs, env: EvalEnv):
    data = args.get_required("data")
    message = args.get_optional("message", default="")
    code = args.get_optional("code", default="User")
    level = args.get_optional("level", default="info")
    if message:
        _log.log(level=logging.getLevelName(level.upper()), msg=message)
    data_message = str(data)
    if isinstance(data, DriverDataCube):
        data_message = str(data.metadata)
    _log.log(level=logging.getLevelName(level.upper()), msg=data_message)
    return data


@simple_function
def text_begins(data: str, pattern: str, case_sensitive: bool = True) -> Union[bool, None]:
    if data is None:
        return None
    if not case_sensitive:
        data = data.lower()
        pattern = pattern.lower()
    return data.startswith(pattern)


@simple_function
def text_contains(data: str, pattern: str, case_sensitive: bool = True) -> Union[bool, None]:
    if data is None:
        return None
    if not case_sensitive:
        data = data.lower()
        pattern = pattern.lower()
    return pattern in data


@simple_function
def text_ends(data: str, pattern: str, case_sensitive: bool = True) -> Union[bool, None]:
    if data is None:
        return None
    if not case_sensitive:
        data = data.lower()
        pattern = pattern.lower()
    return data.endswith(pattern)


@process_registry_100.add_simple_function
def text_merge(
        data: List[Union[str, int, float, bool, None]],
        separator: Union[str, int, float, bool, None] = ""
) -> str:
    # TODO #196 text_merge is deprecated if favor of test_concat
    return str(separator).join(str(d) for d in data)


@process_registry_100.add_simple_function(spec=read_spec("openeo-processes/2.x/text_concat.json"))
@process_registry_2xx.add_simple_function
def text_concat(
    data: List[Union[str, int, float, bool, None]],
    separator: str = "",
) -> str:
    return str(separator).join(str(d) for d in data)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/load_stac.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_stac.json"))
def load_stac(args: Dict, env: EvalEnv) -> DriverDataCube:
    url = extract_arg(args, "url", process_id="load_stac")

    arguments = {}
    if args.get("temporal_extent"):
        arguments["temporal_extent"] = _extract_temporal_extent(
            args, field="temporal_extent", process_id="load_stac"
        )
    if args.get("spatial_extent"):
        arguments["spatial_extent"] = _extract_bbox_extent(
            args, field="spatial_extent", process_id="load_stac", handle_geojson=True
        )
    if args.get("bands"):
        arguments["bands"] = extract_arg(args, "bands", process_id="load_stac")
    if args.get("properties"):
        arguments["properties"] = extract_arg(args, "properties", process_id="load_stac")
    if args.get("featureflags"):
        arguments["featureflags"] = extract_arg(args, "featureflags", process_id="load_stac")

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        return dry_run_tracer.load_stac(url, arguments, env)
    else:
        source_id = dry_run.DataSource.load_stac(
            url, properties=arguments.get("properties", {}), bands=arguments.get("bands", []), env=env
        ).get_source_id()
        load_params = _extract_load_parameters(env, source_id=source_id)
        load_params.update(arguments)

        return env.backend_implementation.load_stac(url=url, load_params=load_params, env=env)


@process_registry_100.add_simple_function(name="if")
@process_registry_2xx.add_simple_function(name="if")
def if_(value: Union[bool, None], accept, reject=None):
    return accept if value else reject


@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/export_workspace.json"))
def export_workspace(args: ProcessArgs, env: EvalEnv) -> SaveResult:
    data = args.get_required("data")
    workspace_id = args.get_required("workspace", expected_type=str)
    merge = args.get_optional("merge", expected_type=str)

    if isinstance(data, SaveResult):
        result = data
    else:
        # TODO: work around save_result returning a data cube instead of a SaveResult (#295)
        results = env[ENV_SAVE_RESULT]
        result = results[-1]

    result.add_workspace_export(workspace_id, merge=merge)
    return result


@custom_process
def collect(args: ProcessArgs, env: EvalEnv):
    return env[ENV_FINAL_RESULT][0]


# Finally: register some fallback implementation if possible
_register_fallback_implementations_by_process_graph(process_registry_100)
_register_fallback_implementations_by_process_graph(process_registry_2xx)
