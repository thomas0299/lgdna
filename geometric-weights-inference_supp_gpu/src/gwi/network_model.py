from typing import Optional

import cugraph
import cupy as cp
from cupyx.scipy.sparse import csr_matrix

from .graph_utils import check_and_get_adj


class Network:
    def __init__(self, graph: cugraph.Graph | csr_matrix):
        self.network: csr_matrix = check_and_get_adj(graph)
        self.n: int = self.network.shape[0]
        self.network_dense: Optional[cp.ndarray] = None
        self.edges: tuple[cp.ndarray, cp.ndarray] = self.get_edges()
        self.CN_L2_matrix: Optional[cp.ndarray] = None
        self.d_matrix: Optional[cp.ndarray] = None
        self.e_L2_matrix: Optional[cp.ndarray] = None
        self.eRA_L2_matrix: Optional[cp.ndarray] = None
        self.weighting_results: dict = {}

    def get_network_dense(self):
        if self.network_dense is None:
            self.network_dense = cp.asarray(
                self.network.todense() + self.network.todense().T, dtype=cp.float16
            )
        return self.network_dense

    def get_edges(self):
        src, dst = self.network.get().nonzero()
        return (cp.asarray(src), cp.asarray(dst))

    def get_CN_L2_matrix(self) -> cp.ndarray:
        if self.CN_L2_matrix is None:
            elementwise_multiply = cp.ElementwiseKernel(
                "float16 x, float16 y", "float16 z", "z = x * y", "elementwise_multiply"
            )
            network_dense_squared = cp.matmul(
                self.get_network_dense(), self.get_network_dense()
            )
            self.CN_L2_matrix = elementwise_multiply(
                self.get_network_dense(), network_dense_squared
            )

        return self.CN_L2_matrix

    def get_d_matrix(self) -> cp.ndarray:
        if self.d_matrix is None:
            self.d_matrix = cp.tile(
                cp.sum(
                    self.get_network_dense(), axis=1, keepdims=True, dtype=cp.float16
                ),
                (1, self.get_network_dense().shape[1]),
            )
        return self.d_matrix

    def get_e_L2_matrix(self) -> cp.ndarray:
        if self.e_L2_matrix is None:
            network_dense = self.get_network_dense()
            d_matrix = self.get_d_matrix()
            CN_L2_matrix = self.get_CN_L2_matrix()

            self.e_L2_matrix = cp.multiply(
                network_dense, (d_matrix - CN_L2_matrix - network_dense)
            )

        return self.e_L2_matrix

    def get_eRA_L2_matrix(self) -> cp.ndarray:
        if self.eRA_L2_matrix is None:
            e_L2_matrix = self.get_e_L2_matrix()
            self.eRA_L2_matrix = (
                e_L2_matrix + e_L2_matrix.T + e_L2_matrix * e_L2_matrix.T
            )
        return self.eRA_L2_matrix
