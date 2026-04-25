"""Array operation process implementations."""
from openeo_driver.errors import ProcessParameterInvalidException
from openeo_driver.processgraph.registry import (
    NoPythonImplementationError,
    process,
    process_registry_100,
    process_registry_2xx,
)
from openeo_driver.processes import ProcessArgs
from openeo_driver.specs import read_spec
from openeo_driver.utils import EvalEnv


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/array_append.json"))
@process_registry_2xx.add_function
def array_append(args: ProcessArgs, env: EvalEnv) -> list:
    raise NoPythonImplementationError


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/array_interpolate_linear.json"))
@process_registry_2xx.add_function
def array_interpolate_linear(args: ProcessArgs, env: EvalEnv) -> list:
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


@process
def array_apply(args: ProcessArgs, env: EvalEnv):
    data = args.get_required("data")
    p = args.get_required("process")
    c = args.get_optional("context", None)
    if not isinstance(p, dict) and not "process_graph" in p:
        raise ProcessParameterInvalidException(
            parameter="process", process="array_apply",
            reason=f"Parameter should be a process graph, but got {p}"
        )
    if not isinstance(data, list):
        raise ProcessParameterInvalidException(
            parameter="data", process="array_apply",
            reason=f"Parameter should be a list, but got {data}"
        )
    from openeo_driver.processgraph.evaluator import evaluate
    result = [
        evaluate(p.get("process_graph"), env.push_parameters(dict(context=c, x=d, index=index)))
        for index, d in enumerate(data)
    ]
    return result
