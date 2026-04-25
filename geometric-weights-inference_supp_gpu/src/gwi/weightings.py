import cupy as cp

from .network_model import Network


def get_RA2(network: Network) -> cp.ndarray:
    CN_L2 = network.get_CN_L2_matrix()
    eRA_L2 = network.get_eRA_L2_matrix()
    edges = cp.column_stack(network.edges)
    row_indices, col_indices = edges[:, 0], edges[:, 1]
    CN_L2_values = CN_L2[row_indices, col_indices]
    eRA_L2 = eRA_L2[row_indices, col_indices]
    values = cp.divide(1 + eRA_L2, 1 + CN_L2_values)
    RA2 = cp.zeros_like(network.network_dense, dtype=cp.float16)
    RA2[row_indices, col_indices] = values
    RA2 = RA2 + RA2.T

    return RA2


def get_RA2num(network: Network) -> cp.ndarray:
    eRA_L2 = network.get_eRA_L2_matrix()
    edges = cp.column_stack(network.edges)
    row_indices, col_indices = edges[:, 0], edges[:, 1]
    eRA_L2 = eRA_L2[row_indices, col_indices]
    values = cp.add(1, eRA_L2)
    RA2num = cp.zeros_like(network.network_dense, dtype=cp.float16)
    RA2num[row_indices, col_indices] = values
    RA2num = RA2num + RA2num.T

    return RA2num


def get_CND(network: Network) -> cp.ndarray:
    CN_L2 = network.get_CN_L2_matrix()
    edges = cp.column_stack(network.edges)
    row_indices, col_indices = edges[:, 0], edges[:, 1]
    CN_L2_values = CN_L2[row_indices, col_indices]
    values = cp.divide(1, (1 + CN_L2_values))
    CND = cp.zeros_like(network.network_dense, dtype=cp.float16)
    CND[row_indices, col_indices] = values
    CND = CND + CND.T

    return CND


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
