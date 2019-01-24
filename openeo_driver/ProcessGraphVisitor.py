from abc import  ABC
from typing import Dict

class ProcessGraphVisitor(ABC):
    """
    Hierarchical Visitor for process graphs, to allow different tools to traverse the graph.
    """


    def __init__(self):
        self.process_stack = []


    def accept_process_graph(self,graph:Dict):
        """
        Traverse a process graph, provided as a flat Dict of nodes that are not referencing each other.
        :param graph:
        :return:
        """
        from .ProcessGraphDeserializer import list_to_graph
        top_level_node = list_to_graph(graph)
        self.accept(graph[top_level_node])



    def accept(self, node:Dict):
        if 'process_id' in node:
            pid = node['process_id']
            arguments = node.get('arguments',{})
            self.process_stack.append(pid)
            self.enterProcess(pid, arguments)
            for arg in arguments:
                value = arguments[arg]
                self.enterArgument(arg,value)
                if 'node' in value and 'from_node' in value:
                    self.accept(value['node'])
                else:
                    self.accept(value)
                self.leaveArgument(arg,value)

            self.leaveProcess(pid, arguments)
            self.process_stack.pop()


    def enterProcess(self,process_id, arguments:Dict):
        pass

    def leaveProcess(self, process_id, arguments: Dict):
        pass

    def enterArgument(self,process_id,node:Dict):
        pass

    def leaveArgument(self, process_id, node: Dict):
        pass