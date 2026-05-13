"""User-defined process (UDP) evaluation."""
from openeo_driver.processes import ProcessArgs
from openeo_driver.processgraph.evaluator import evaluate_udp, evaluate_process_from_url, _evaluate_process_graph_process
from openeo_driver.processgraph.registry import custom_process, custom_process_from_process_graph, ENV_FINAL_RESULT
from openeo_driver.utils import EvalEnv

# Re-export key functions so they can be imported from this module for symmetry
__all__ = [
    "evaluate_udp",
    "evaluate_process_from_url",
    "_evaluate_process_graph_process",
    "custom_process_from_process_graph",
]


def _get_udf(args: ProcessArgs, env: EvalEnv):
    from openeo_driver.errors import OpenEOApiException
    udf_code = args.get_required(name="udf", expected_type=str)
    runtime = args.get_required(name="runtime", expected_type=str)
    version = args.get_optional(name="version", expected_type=str)

    available_runtimes = env.backend_implementation.udf_runtimes.get_udf_runtimes()
    available_runtime_names = list(available_runtimes.keys())
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

    return udf_code, runtime
