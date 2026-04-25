import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .network_model import Network


def get_RA2(network: Network) -> csr_matrix:
    CN_L2: np.matrix = network.get_CN_L2_matrix().todense()
    e_L2: np.matrix = network.get_e_L2_matrix().todense()
    eRA_L2: np.matrix = network.get_eRA_L2_matrix().todense()

    RA2: np.matrix | np.ndarray = np.nan_to_num(
        np.divide(
            np.multiply(
                network.network_dense, (1 + eRA_L2 + (np.multiply(e_L2, e_L2.T)))
            ),
            (1 + CN_L2),
        )
    )
    return csr_matrix(RA2)


def get_RA2num(network: Network) -> csr_matrix:
    e_L2: np.matrix = network.get_e_L2_matrix().todense()
    eRA_L2: np.matrix = network.get_eRA_L2_matrix().todense()

    RA2num: np.matrix | np.ndarray = np.nan_to_num(
        np.multiply(network.network_dense, (1 + eRA_L2 + (np.multiply(e_L2, e_L2.T)))),
    )
    return csr_matrix(RA2num)


def get_CND(network: Network) -> csr_matrix:
    CND: lil_matrix = lil_matrix(network.network.shape, dtype=np.float64)
    CN_L2: np.matrix = network.get_CN_L2_matrix()

    CND[network.edges] = np.divide((1), (1 + CN_L2[network.edges]))
    return CND.tocsr()


def compute_weighting(
    network: Network,
    weight: str,
) -> None:
    if network.weighting_results.get(weight) is None:
        if weight == "RA2":
            result = get_RA2(network)
        elif weight == "RA2num":
            result = get_RA2num(network)
        elif weight == "CND":
            result = get_CND(network)
        else:
            raise ValueError(f"{weight} is not a valid weighting.")

        network.weighting_results[weight] = result
