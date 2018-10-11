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

    def __init__(self, process_id: str, description: str, args: List[Arg] = []):
        self.process_id = process_id
        self.description = description
        self.args = args

    def serialize(self):
        serialized = {
            "process_id": self.process_id,
            "description": self.description
        }

        if self.args:
            serialized["parameters"] = {arg.name: {"description": arg.description,"required": arg.required,"schema":arg.schema} for arg in self.args}

        return serialized
