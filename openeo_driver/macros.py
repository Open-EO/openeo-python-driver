def expand_macros(process_graph: dict) -> dict:
    """
    Expands macro nodes in a process graph by replacing them with other nodes. The implementation is aimed towards
    supporting processes that can be written in terms of other processes and therefore it currently only considers dicts
    in the tree.

    Make sure that newly introduced node identifiers don't clash with existing ones (make_unique).

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

                    if value['process_id'] == 'ard_normalized_radar_backscatter':
                        result[key] = {
                            'process_id': 'sar_backscatter',
                            'arguments': {
                                'data': original_arguments['data'],
                                'orthorectify': True,
                                'rtc': True,
                                'elevation_model': original_arguments.get('elevation_model'),
                                'mask': True,
                                'contributing_area': True,
                                'local_incidence_angle': True,
                                'ellipsoid_incidence_angle': original_arguments.get('ellipsoid_incidence_angle', False),
                                'noise_removal': original_arguments.get('noise_removal', True)
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
