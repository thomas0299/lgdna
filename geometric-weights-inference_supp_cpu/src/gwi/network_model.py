from typing import Optional

import numpy as np
from graph_tool import Graph
from scipy.sparse import csr_matrix

from .graph_utils import check_and_get_adj


class Network:
    def __init__(self, graph: Graph | csr_matrix):
        self.network: csr_matrix = check_and_get_adj(graph)
        self.network_dense: np.matrix = self.network.todense()
        self.n: int = self.network.shape[0]
        self.edges: tuple[np.ndarray, np.ndarray] = self.network.nonzero()
        self.CN_L2_matrix: Optional[csr_matrix] = None
        self.d_matrix: Optional[csr_matrix] = None
        self.e_L2_matrix: Optional[csr_matrix] = None
        self.eRA_L2_matrix: Optional[csr_matrix] = None
        self.weighting_results: dict[str, csr_matrix] = {}

    def get_CN_L2_matrix(self) -> csr_matrix:
        if self.CN_L2_matrix is None:
            self.CN_L2_matrix = csr_matrix(
                np.multiply(self.network_dense, self.network_dense**2)
            )
        return self.CN_L2_matrix

    def get_d_matrix(self) -> csr_matrix:
        if self.d_matrix is None:
            self.d_matrix = csr_matrix(np.tile(np.sum(self.network_dense, 1), self.n))
        return self.d_matrix

    def get_e_L2_matrix(self) -> csr_matrix:
        if self.e_L2_matrix is None:
            self.e_L2_matrix = csr_matrix(
                np.multiply(
                    self.network_dense,
                    (
                        self.get_d_matrix()
                        - self.get_CN_L2_matrix()
                        - self.network_dense
                    ),
                )
            )
        return self.e_L2_matrix

    def get_eRA_L2_matrix(self) -> csr_matrix:
        if self.eRA_L2_matrix is None:
            self.eRA_L2_matrix = csr_matrix(
                np.add(self.get_e_L2_matrix(), self.get_e_L2_matrix().T)
            )
        return self.eRA_L2_matrix
