from unittest import TestCase
from openeo_driver.ProcessGraphDeserializer import list_to_graph

import os
os.environ["DRIVER_IMPLEMENTATION_PACKAGE"] = "dummy_impl"

class Test(TestCase):

    def test_simple_graph_to_list(self):
        graph = {
            "node1": {

        },
            "node2": {
                "arguments": {
                    "data1": {
                        "from_node": "node1"
                    },
                    "data2": {
                        "from_node": "node3"
                    }
                },
                "result": True
            },
            "node3": {
                "arguments": {
                    "data": {
                        "from_node": "node4"
                    }
                }
            },
            "node4": {

            }
        }
        result = list_to_graph(graph)

        self.assertEqual(result,"node2")
        self.assertEqual(graph["node1"],graph[result]["arguments"]["data1"]["node"])
        self.assertEqual(graph["node3"], graph[result]["arguments"]["data2"]["node"])
        self.assertEqual(graph["node4"], graph["node3"]["arguments"]["data"]["node"])


    def test_simple_graph_to_list_no_result(self):
        with self.assertRaises(ValueError):
            result = list_to_graph({
                "node1":{

                },
                "node2":{
                    "result":False
                }
            })

    def test_simple_graph_invalid_node(self):
        graph = {
            "node1": {

        },
            "node2": {
                "arguments": {
                    "data": {
                        "from_node": "node3"
                    }
                },
                "result": True
            }
        }
        with self.assertRaises(ValueError):

            result = list_to_graph(graph)

