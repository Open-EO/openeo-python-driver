from typing import Any, Dict, Union

from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo_driver.errors import OpenEOApiException
from openeo_driver.utils import EvalEnv


class PropertyConditionException(OpenEOApiException):
    status_code = 400
    code = "PropertyConditionInvalid"
    message = "Property condition is invalid."


def extract_literal_match(
    condition: dict, env: EvalEnv = EvalEnv(), parameter_name="value"
) -> Dict[str, Any]:  # TODO: drop parameter_name?
    """
    Turns a condition as defined by the load_collection process into a set of criteria ((operator, value) pairs).
    Conditions are currently limited to processes "eq", "lte" and "gte" so they will be turned into a single criterion.
    """

    class LiteralMatchExtractingGraphVisitor(ProcessGraphVisitor):

        SUPPORTED_PROCESSES = ['eq', 'lte', 'gte', 'array_contains']

        def __init__(self):
            super().__init__()
            self.result = {}
            self._parameters = env.collect_parameters()

        def enterProcess(self, process_id: str, arguments: dict, namespace: Union[str, None]):
            if process_id not in self.SUPPORTED_PROCESSES:
                raise PropertyConditionException(
                    f"Property filtering only supports {self.SUPPORTED_PROCESSES}, not {process_id!r}."
                )

            self.result["operator"] = "in" if process_id == "array_contains" else process_id

        def enterArgument(self, argument_id: str, value):
            from_parameter = value.get("from_parameter")

            if from_parameter == parameter_name:
                self.result["parameter"] = from_parameter
            else:
                self.result["value"] = self._parameters[from_parameter]

        def constantArgument(self, argument_id: str, value):
            self.result["value"] = value

            if argument_id == 'x':
                if self.result["operator"] == "lte":
                    self.result["operator"] = "gte"
                elif self.result["operator"] == "gte":
                    self.result["operator"] = "lte"

        def constantArrayElement(self, value):
            if "value" not in self.result:
                self.result["value"] = []

            self.result["value"].append(value)

    visitor = LiteralMatchExtractingGraphVisitor()
    visitor.accept_process_graph(condition['process_graph'])

    if "parameter" not in visitor.result:
        raise PropertyConditionException(f"No parameter {parameter_name!r} found")
    if visitor.result["parameter"] != parameter_name:
        raise PropertyConditionException(
            f"Expected parameter {parameter_name!r} but got {visitor.result['parameter']!r}"
        )
    if "value" not in visitor.result:
        raise PropertyConditionException(f"No comparison with constant")  # TODO: extend error message?

    return {visitor.result["operator"]: visitor.result["value"]}
