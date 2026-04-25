from itertools import product
from typing import Optional

import numpy as np
from graph_tool import EdgePropertyMap, Graph
from graph_tool.centrality import betweenness, eigenvector, pagerank
from gwi.geometric_weights_inference import (
    VALID_WEIGHTINGS,
    geometric_weights_inference,
)
from scipy.sparse import csr_array

from ra_dismantling.domirank import domirank_fast
from ra_dismantling.helpers.graph_tool import to_adjacency
from ra_dismantling.helpers.sparse import _sparse_sum


def get_domirank(G: Graph, **kwargs):
    sparse_array = csr_array(to_adjacency(G))
    _, domiranks, sigma = domirank_fast(sparse_array, sigma_numerator=0.5)
    print(f"{sigma=}")
    return domiranks


def get_fitness_centrality(
    network: Graph, max_iter=100, δ=1.0, ϵ=1e-6, **kwargs
):  # https://iopscience.iop.org/article/10.1088/2632-072X/ada845
    """
    Calculate fitness centrality in the undirected case using iterative updates.

    Parameters:
    - A: np.ndarray, adjacency matrix of the graph.
    - δ: float, constant added to the fitness centrality values in each iteration.
    - ϵ: float, convergence tolerance for the stopping criterion.

    Returns:
    - F1: np.ndarray, the final fitness centrality values.
    """
    A = to_adjacency(network)

    c, p = A.shape
    err = 1000.0
    nr_of_iterations = 0

    F0 = np.ones(c)  # Initial fitness centrality values
    F1 = np.ones(c)  # Copy for new fitness centrality values

    while err > ϵ and nr_of_iterations <= max_iter:
        nr_of_iterations += 1
        F1 = δ + np.dot(
            A, 1 / F0
        )  # Element-wise inverse of F0 and matrix multiplication

        err = np.max(np.abs((F1 / F0) - 1))  # Relative error between F1 and F0

        F0 = F1.copy()
    return F1


def get_degree(network: Graph, **kwargs):
    degree = network.get_out_degrees(network.get_vertices())
    return degree


def get_resilience_centrality(
    network: Graph, **kwargs
):  # https://journals.aps.org/pre/abstract/10.1103/PhysRevE.101.022304
    A = to_adjacency(network)

    # Number of nodes
    N = A.shape[0]

    # Compute degrees of each node (d_i)
    d = np.sum(A, axis=1)

    # Compute mean degree (c)
    c = np.mean(d)

    # Compute degree variance (σ²)
    sigma_sq = np.mean(d**2) - c**2

    # Compute effective β (β_eff)
    beta_eff = (c**2 + sigma_sq) / c

    # Compute weighted nearest-neighbor degree (D_i)
    D = A @ d

    # Compute R_i for each node
    R = (1 / (N * (c**2 + sigma_sq))) * (2 * D + d * (d - 2 * beta_eff))

    return R


def get_betweenness_centrality(
    network: Graph, weight: Optional[EdgePropertyMap] = None, **kwargs
):
    betweenness_out, _ = betweenness(network, weight=weight)
    return betweenness_out.get_array()


def get_eigenvector_centrality(network: Graph, **kwargs):
    _, eigenvectors = eigenvector(network, max_iter=100)

    return eigenvectors.get_array()


def get_pagerank(network: Graph, **kwargs):
    return pagerank(network).get_array()


def _create_and_export_functions():
    for c in product(VALID_WEIGHTINGS):
        func_name_1 = "get_%s_sum" % (c[0])

        def dynamic_function_1(
            network: Graph,
            weighting: str = c[0],
            **kwargs,
        ):
            res = geometric_weights_inference(
                graph=network,
                weightings=[weighting],
            )
            return _sparse_sum(res.weighting_results[weighting], axis=1)

        globals()[func_name_1] = dynamic_function_1
        globals()[func_name_1].__name__ = func_name_1


_create_and_export_functions()

all_funcs = [
    (name.replace("get_", ""), thing)
    for (name, thing) in globals().items()
    if callable(thing) and thing.__name__[0] != "_"  # and thing.__module__ == __name__
]

__all_dict__ = dict(all_funcs)
