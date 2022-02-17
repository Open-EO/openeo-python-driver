from typing import Any, Dict, Union

from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo_driver.errors import OpenEOApiException


class PropertyConditionException(OpenEOApiException):
    status_code = 400
    code = "PropertyConditionInvalid"
    message = "Property condition is invalid."


def extract_literal_match(condition: dict, parameter_name="value") -> Dict[str, Any]:
    """
    Turns a condition as defined by the load_collection process into a set of criteria ((operator, value) pairs).
    Conditions are currently limited to processes "eq", "lte" and "gte" so they will be turned into a single criterion.
    """

    class LiteralMatchExtractingGraphVisitor(ProcessGraphVisitor):

        SUPPORTED_PROCESSES = ['eq', 'lte', 'gte']

        def __init__(self):
            super().__init__()
            self.result = {}

        def enterProcess(self, process_id: str, arguments: dict, namespace: Union[str, None]):
            if process_id not in self.SUPPORTED_PROCESSES:
                raise PropertyConditionException(
                    f"Property filtering only supports {self.SUPPORTED_PROCESSES}, not {process_id!r}."
                )

            self.result["operator"] = process_id

        def enterArgument(self, argument_id: str, value):
            self.result["parameter"] = value.get("from_parameter")

        def constantArgument(self, argument_id: str, value):
            self.result["constant"] = value

            if argument_id == 'x':
                if self.result["operator"] == "lte":
                    self.result["operator"] = "gte"
                elif self.result["operator"] == "gte":
                    self.result["operator"] = "lte"

    visitor = LiteralMatchExtractingGraphVisitor()
    visitor.accept_process_graph(condition['process_graph'])

    if "parameter" not in visitor.result:
        raise PropertyConditionException(f"No parameter {parameter_name!r} found")
    if visitor.result["parameter"] != parameter_name:
        raise PropertyConditionException(
            f"Expected parameter {parameter_name!r} but got {visitor.result['parameter']!r}"
        )
    if "constant" not in visitor.result:
        raise PropertyConditionException(f"No comparison with constant")

    return {visitor.result["operator"]: visitor.result["constant"]}
