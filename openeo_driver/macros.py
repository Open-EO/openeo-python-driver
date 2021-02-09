def expand_macros(process_graph: dict) -> dict:
    """
    Expands macro nodes in a process graph by replacing them with other nodes, making sure their node identifiers don't
    clash with existing ones.

    :param process_graph:
    :return: a copy of the input process graph with the macros expanded
    """
    # TODO: can this system be combined with user defined processes (both kind of replace a single "virtual" node with a replacement process graph)

    def expand_macros_recursively(tree: dict) -> dict:
        def make_unique(node_identifier: str) -> str:
            return node_identifier if node_identifier not in tree else make_unique(node_identifier + '_')

        result = {}

        for key, value in tree.items():
            if isinstance(value, dict):
                if 'process_id' in value:
                    original_node = value
                    original_arguments = original_node['arguments']

                    if value['process_id'] == 'normalized_difference':
                        subtract_key = make_unique(key + "_subtract")
                        add_key = make_unique(key + "_add")

                        # add "subtract" and "add"/"sum" processes
                        result[subtract_key] = {'process_id': 'subtract',
                                                'arguments': original_arguments}
                        result[add_key] = {'process_id': 'sum' if 'data' in original_arguments else 'add',
                                           'arguments': original_arguments}

                        # replace "normalized_difference" with "divide" under the original key (it's being referenced)
                        result[key] = {
                            'process_id': 'divide',
                            'arguments': {
                                'x': {'from_node': subtract_key},
                                'y': {'from_node': add_key}
                            },
                            "result": original_node.get('result', False)
                        }
                    elif value['process_id'] == 'ard_normalized_radar_backscatter':
                        result[key] = {
                            'process_id': 'sar_backscatter',
                            'arguments': {
                                'data': original_arguments['data'],
                                'orthorectify': True,
                                'rtc': True,
                                'elevation_model': original_arguments['elevation_model'],
                                'mask': True,
                                'contributing_area': True,
                                'local_incidence_angle': True,
                                'ellipsoid_incidence_angle': original_arguments['ellipsoid_incidence_angle'],
                                'noise_removal': original_arguments['noise_removal']
                            },
                            "result": original_node.get('result', False)
                        }
                    else:
                        result[key] = expand_macros_recursively(value)
                else:
                    result[key] = expand_macros_recursively(value)
            else:
                result[key] = value

        return result

    return expand_macros_recursively(process_graph)
