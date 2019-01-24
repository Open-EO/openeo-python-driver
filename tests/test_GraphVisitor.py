from unittest import TestCase
from unittest.mock import MagicMock

from openeo_driver.ProcessGraphVisitor import ProcessGraphVisitor

class GraphVisitorTest(TestCase):

    def test_visit_nodes(self):
        graph = {
                    "abs":{
                        "arguments":{
                            "data": {
                                "from_argument": "dimension_data"
                            }
                        },
                        "process_id":"abs"
                    },
                    "cos": {
                        "arguments":{
                            "data": {
                                "from_node": "abs"
                            }
                        },
                        "process_id": "cos",
                        "result": True
                    }
                }
        original = ProcessGraphVisitor()

        leaveProcess = MagicMock(original.leaveProcess)
        original.leaveProcess = leaveProcess
        original.accept_process_graph(graph)


        print(leaveProcess)