from unittest import TestCase
from unittest.mock import MagicMock,call,ANY

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

        enterArgument = MagicMock(original.enterArgument)
        original.enterArgument = enterArgument

        original.accept_process_graph(graph)
        self.assertEquals(2, leaveProcess.call_count)
        leaveProcess.assert_has_calls([
            call('abs',ANY),
            call('cos', ANY)
        ])

        self.assertEquals(2, enterArgument.call_count)
        enterArgument.assert_has_calls([
            call('data', ANY),
            call('data', ANY)
        ])

        print(leaveProcess)