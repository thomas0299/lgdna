import logging
from typing import Callable

import cupy as cp
from cugraph import Graph


def static_generator(
    network: Graph,
    sorting_function: Callable,
    logger: logging.Logger = logging.getLogger("dummy"),
):
    values = sorting_function(network, logger=logger)

    vertex_indices = cp.sort(network.nodes().to_cupy())
    values = cp.asarray(values)
    values = cp.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    pairs = cp.stack((vertex_indices, values), axis=-1)

    sorted_pairs = pairs[cp.argsort(pairs[:, 1], axis=0)[::-1]]

    for node, value in sorted_pairs:
        print(int(node), float(value))
        yield int(node), float(value)


def dynamic_generator(
    network: Graph,
    sorting_function: Callable,
    logger: logging.Logger = logging.getLogger("dummy"),
):
    vertex_ids = cp.sort(network.nodes().to_cupy())
    values = sorting_function(network, logger=logger)
    index = cp.argmax(values).item()
    print(vertex_ids[index], values[index].item())
    yield vertex_ids[index], values[index].item()
