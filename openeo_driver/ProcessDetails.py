from typing import List
from typing import Dict

class ProcessDetails:
    """
    API reference:
    https://open-eo.github.io/openeo-api/apireference/#tag/Process-Discovery/paths/~1processes/get
    """
    class Arg:
        def __init__(self, name: str, description: str, required: bool = True,schema:Dict={}):
            self.name = name
            self.description = description
            self.required = required
            self.schema = schema

    def __init__(self, process_id: str, description: str, args: List[Arg] = [], returns:Dict = {
                "description" : "Raster Data Cube",
                "schema": {
                    "type": "object",
                    "format": "raster-cube"
                }
            } ):
        self.returns = returns
        self.process_id = process_id
        self.description = description
        self.args = args

    def serialize(self) -> dict:
        """Convert to JSON-able dictionary."""
        # TODO give this function a better name?
        serialized = {
            "name": self.process_id,  # Pre 0.4.0 style id field  # TODO DEPRECATED
            "id": self.process_id,  # id field since 0.4.0
            "description": self.description,
            "returns": self.returns,
            "parameters": {
                arg.name: {"description": arg.description, "required": arg.required, "schema": arg.schema}
                for arg in self.args
            },
        }
        if len(self.args) >= 2:
            serialized["parameter_order"] = [a.name for a in self.args]

        return serialized
