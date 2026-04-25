from itertools import product

import cugraph
from gwi.geometric_weights_inference import (
    VALID_WEIGHTINGS,
    geometric_weights_inference,
)

from ra_dismantling.helpers.sparse import _sparse_sum


def get_betweenness_centrality(network: cugraph.Graph, **kwargs):
    bc_df = cugraph.betweenness_centrality(network, normalized=True)
    sorted_df = bc_df.sort_values("vertex", ascending=True)
    return sorted_df["betweenness_centrality"].to_cupy()


def _create_and_export_functions():
    for c in product(VALID_WEIGHTINGS):
        func_name_1 = f"get_{c[0]}_sum"

        def dynamic_function_1(
            network: cugraph.Graph,
            weighting: str = c[0],
            **kwargs,
        ):
            method = f"{weighting}"
            res = geometric_weights_inference(
                graph=network,
                weightings=[method],
            )
            return _sparse_sum(res.weighting_results[method], axis=1)

        globals()[func_name_1] = dynamic_function_1
        globals()[func_name_1].__name__ = func_name_1


_create_and_export_functions()

all_funcs = [
    (name.replace("get_", ""), thing)
    for (name, thing) in globals().items()
    if callable(thing) and thing.__name__[0] != "_" and thing.__module__ == __name__
]

__all_dict__ = dict(all_funcs)
