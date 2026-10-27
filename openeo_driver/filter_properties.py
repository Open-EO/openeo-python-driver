from typing import Any, Dict, Union

from openeo.internal.process_graph_visitor import ProcessGraphVisitor

from openeo_driver.errors import OpenEOApiException
from openeo_driver.utils import EvalEnv


class PropertyConditionException(OpenEOApiException):
    status_code = 400
    code = "PropertyConditionInvalid"
    message = "Property condition is invalid."


def resolve_condition_parameters(properties: Dict[str, dict], env: EvalEnv) -> Dict[str, dict]:
    """
    Replace parameter references (other than the callback parameter "value") in property conditions
    with their actual values from the evaluation env, so the conditions can be evaluated without that env later on.
    """
    callback_parameter_name = "value"
    env_parameters = env.collect_parameters()

    def resolve(x):
        if isinstance(x, dict):
            if set(x.keys()) == {"from_parameter"}:
                name = x["from_parameter"]
                if name != callback_parameter_name and name in env_parameters:
                    return env_parameters[name]
                return x
            resolved = {k: resolve(v) for k, v in x.items()}
            return x if all(resolved[k] is x[k] for k in x) else resolved
        elif isinstance(x, list):
            resolved = [resolve(v) for v in x]
            return x if all(r is v for r, v in zip(resolved, x)) else resolved
        return x

    resolved = {name: resolve(condition) for name, condition in properties.items()}
    return properties if all(resolved[k] is properties[k] for k in properties) else resolved


def extract_literal_match(condition: dict, env=EvalEnv()) -> Dict[str, Any]:
    """
    Turns a condition as defined by the load_collection process into a set of criteria ((operator, value) pairs).
    Conditions are currently limited to processes "eq", "lte" and "gte" so they will be turned into a single criterion.
    """
    callback_parameter_name = "value"  # as in: {"from_parameter": "value"}

    class LiteralMatchExtractingGraphVisitor(ProcessGraphVisitor):
        SUPPORTED_PROCESSES = ["eq", "lte", "gte", "array_contains"]

        def __init__(self):
            super().__init__()
            self.result = {}

        def enterProcess(self, process_id: str, arguments: dict, namespace: Union[str, None]):
            if process_id not in self.SUPPORTED_PROCESSES:
                raise PropertyConditionException(
                    f"Property filtering only supports {self.SUPPORTED_PROCESSES}, not {process_id!r}."
                )

            self.result["operator"] = "in" if process_id == "array_contains" else process_id

        def enterArgument(self, argument_id: str, value: dict):
            parameter_name = value.get("from_parameter")

            if parameter_name == callback_parameter_name:
                self.result["parameter"] = parameter_name
            else:
                env_parameters = env.collect_parameters()
                if parameter_name not in env_parameters:
                    raise PropertyConditionException(f"Unknown parameter {parameter_name}")

                self.result["value"] = env_parameters[parameter_name]

        def constantArgument(self, argument_id: str, value):
            self.result["value"] = value

            if argument_id == "x":
                if self.result["operator"] == "lte":
                    self.result["operator"] = "gte"
                elif self.result["operator"] == "gte":
                    self.result["operator"] = "lte"

        def constantArrayElement(self, value):
            if "value" not in self.result:
                self.result["value"] = []

            self.result["value"].append(value)

    visitor = LiteralMatchExtractingGraphVisitor()
    visitor.accept_process_graph(condition["process_graph"])

    if "parameter" not in visitor.result:
        raise PropertyConditionException(f"No parameter {callback_parameter_name!r} found")
    if visitor.result["parameter"] != callback_parameter_name:
        raise PropertyConditionException(
            f"Expected parameter {callback_parameter_name!r} but got {visitor.result['parameter']!r}"
        )
    if "value" not in visitor.result:
        raise PropertyConditionException(f"No comparison with constant/value from parameter")

    return {visitor.result["operator"]: visitor.result["value"]}
