from typing import List


class ProcessDetails:
    class Arg:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description

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
            serialized["args"] = {arg.name: {"description": arg.description} for arg in self.args}

        return serialized
