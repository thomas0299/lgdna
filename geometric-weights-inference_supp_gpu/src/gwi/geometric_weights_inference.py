import logging

import cugraph
import cupy as cp
from cupyx.scipy.sparse import csr_matrix

from .network_model import Network
from .utils import block_timing
from .weightings import compute_weighting

log = logging.getLogger(__name__)

VALID_WEIGHTINGS = [
    "RA2",
    "RA2num",
    "CND",
]


def _geometric_weights_inference(
    network: Network,
    weightings: list[str],
) -> None:
    for current_weighting in weightings:

        with block_timing(
            log.debug,
            f"Computing weighting for {current_weighting=}",
        ):
            compute_weighting(
                network=network,
                weight=current_weighting,
            )


def geometric_weights_inference(
    graph: cugraph.Graph | csr_matrix,
    weightings: list[str],
) -> Network:

    cp.matmul(cp.zeros((10, 10)), cp.zeros((10, 10)))

    current_network: Network = Network(graph=graph)

    if weightings is None or len(weightings) == 0:
        raise ValueError("Must specify a method.")

    if weightings is not None and not set(weightings).issubset(VALID_WEIGHTINGS):
        raise ValueError(f"All elements of {weightings=} must be in {VALID_WEIGHTINGS}")

    _geometric_weights_inference(current_network, weightings)

    return current_network
