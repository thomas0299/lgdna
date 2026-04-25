import logging
from operator import itemgetter
from typing import Callable

import numpy as np
from graph_tool import Graph


def static_generator(
    network: Graph,
    sorting_function: Callable,
    logger: logging.Logger = logging.getLogger("dummy"),
):
    values = None
    try:
        sorting_function_name = sorting_function.__name__.replace("get_", "")
        if sorting_function_name in network.vertex_properties.keys():
            logger.info("{} values already computed!".format(sorting_function_name))

            values = network.vertex_properties[sorting_function_name].get_array()
    except Exception:
        pass

    if values is None:
        values = sorting_function(
            network,
            logger=logger,
        )

    pairs = dict(zip([network.vertex_index[v] for v in network.vertices()], values))
    # Get the highest predicted value
    sorted_predictions = sorted(pairs.items(), key=itemgetter(1), reverse=True)

    for node, value in sorted_predictions:
        print(node, value)
        yield node, value


def dynamic_generator(
    network: Graph,
    sorting_function: Callable,
    logger: logging.Logger = logging.getLogger("dummy"),
):
    for _ in range(network.num_vertices()):
        values = sorting_function(network, logger=logger)
        index = np.argmax(values)
        print(network.vertex_index[index], values[index])
        yield network.vertex_index[index], values[index]
