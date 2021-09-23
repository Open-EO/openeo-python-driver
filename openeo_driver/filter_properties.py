from typing import Union

from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo_driver.errors import OpenEOApiException


class PropertyConditionException(OpenEOApiException):
    status_code = 400
    code = "PropertyConditionInvalid"
    message = "Property condition is invalid."


def extract_literal_match(condition: dict, parameter_name="value") -> str:
    """
    Turns a condition as defined by the load_collection process into a (key, value) pair; therefore, conditions are
    currently limited to exact matches ("eq").
    """

    class LiteralMatchExtractingGraphVisitor(ProcessGraphVisitor):
        def __init__(self):
            super().__init__()
            self.result = {}

        def enterProcess(self, process_id: str, arguments: dict, namespace: Union[str, None]):
            if process_id != 'eq':
                raise NotImplementedError("process %s is not supported" % process_id)

        def enterArgument(self, argument_id: str, value):
            self.result["parameter"] = value.get("from_parameter")

        def constantArgument(self, argument_id: str, value):
            if argument_id in ['x', 'y']:
                self.result["constant"] = value

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

    return visitor.result["constant"]
