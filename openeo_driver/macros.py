def expand_macros(process_graph: dict) -> dict:
    """
    Expands macro nodes in a process graph by replacing them with other nodes. The implementation is aimed towards
    supporting processes that can be written in terms of other processes and therefore it currently only considers dicts
    in the tree.

    Make sure that newly introduced node identifiers don't clash with existing ones (make_unique).

    :param process_graph:
    :return: a copy of the input process graph with the macros expanded
    """
    # TODO: remove this deprecated approach: through `custom_process_from_process_graph` and related
    #       we can now inject process graph based implementations directly instead of manually writing macros.

    def expand_macros_recursively(tree: dict) -> dict:
        def make_unique(node_identifier: str) -> str:
            return node_identifier if node_identifier not in tree else make_unique(node_identifier + '_')

        result = {}

        for key, value in tree.items():
            if isinstance(value, dict):
                if 'process_id' in value:
                    original_node = value
                    original_arguments = original_node['arguments']

                    result[key] = expand_macros_recursively(value)
                else:
                    result[key] = expand_macros_recursively(value)
            else:
                result[key] = value

        return result

    return expand_macros_recursively(process_graph)
