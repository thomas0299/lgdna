import numpy as np
from graph_tool import Graph
from graph_tool.spectral import adjacency
from scipy.sparse import csr_matrix



def check_and_get_adj(network: Graph | csr_matrix) -> csr_matrix:
    adjacency_matrix: csr_matrix

    if isinstance(network, Graph):
        if network.is_directed():
            raise ValueError("Network must be undirected")

        adjacency_matrix = adjacency(network, csr=True).astype(
            copy=False, dtype=np.float64
        )

    elif isinstance(network, csr_matrix):
        adjacency_matrix = network.tocsr(copy=False)

        if (adjacency_matrix.toarray() != adjacency_matrix.T.toarray()).any():
            raise ValueError("Network must be symmetric to be undirected")

    if adjacency_matrix.diagonal().sum() != 0:
        raise ValueError("No self-loops are allowed")

    if adjacency_matrix.min() < 0 or adjacency_matrix.max() > 1:
        raise ValueError("Network must be unweighted")

    return adjacency_matrix
