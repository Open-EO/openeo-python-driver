import dataclasses
from typing import Optional


class NotASingleRunUDFProcessGraph(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class SingleRunUDFProcessGraph:
    """
    Container (and parser) for a callback process graph containing only a single `run_udf` node.
    """

    data: dict
    udf: str
    runtime: str
    version: Optional[str] = None
    context: Optional[dict] = None

    @classmethod
    def parse(cls, process_graph: dict) -> "SingleRunUDFProcessGraph":
        try:
            (node,) = process_graph.values()
            assert node["process_id"] == "run_udf"
            assert node["result"] is True
            arguments = node["arguments"]
            assert {"data", "udf", "runtime"}.issubset(arguments.keys())

            return cls(
                data=arguments["data"],
                udf=arguments["udf"],
                runtime=arguments["runtime"],
                version=arguments.get("version"),
                context=arguments.get("context") or {},
            )
        except Exception as e:
            raise NotASingleRunUDFProcessGraph(str(e)) from e

    @classmethod
    def parse_or_none(cls, process_graph: dict) -> Optional["SingleNodeRunUDFProcessGraph"]:
        try:
            return cls.parse(process_graph=process_graph)
        except NotASingleRunUDFProcessGraph:
            return None
