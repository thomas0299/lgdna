import logging
from typing import Any, Callable

import cugraph
import cupy as cp
from scipy.integrate import simpson


def filter_edges(graph, condition, edge_list):
    filtered_edges = edge_list[condition]
    if filtered_edges.empty:
        print("no edges to filter")
        new_graph = cugraph.Graph(directed=graph.is_directed())
    try:
        new_graph = cugraph.DiGraph() if graph.is_directed() else cugraph.Graph()
        filtered_edges["src"] = filtered_edges["src"].astype("int32")
        filtered_edges["dst"] = filtered_edges["dst"].astype("int32")

        new_graph.from_cudf_edgelist(
            filtered_edges, source="src", destination="dst", edge_attr="weight"
        )
    except Exception as e:
        print("Error during graph construction:")
        print(e)
        new_graph = cugraph.Graph(directed=graph.is_directed())

    return new_graph


def check_stopping_conditions(
    local_network_lcc_size,
    stop_condition,
    removals,
    i,
    early_stopping_removals,
    early_stopping_auc,
    logger,
):
    if local_network_lcc_size <= stop_condition:
        return True

    current_auc = simpson(list(r[2] for r in removals), dx=1)

    if (i > early_stopping_removals) and (current_auc > early_stopping_auc):
        removals.append((-1, -1, -1, -1, -1))
        logger.debug("EARLY STOPPING")
        return True

    return False


def get_lcc_slcc(network):
    # Networks are undirected, and this is checked after load phase
    # Forcing directed = False triggers a GraphView call which is expensive

    components_df = cugraph.connected_components(network)

    component_labels = cp.asarray(components_df["labels"])
    _, counts = cp.unique(component_labels, return_counts=True)

    counts = counts.astype(cp.int32, copy=False)
    if len(counts) < 2:
        return component_labels, int(counts[0]) if counts.size > 0 else 0, 0, 0

    lcc_index, slcc_index = cp.argpartition(-counts, 1)[:2]

    return (
        component_labels,
        int(counts[lcc_index]),
        int(counts[slcc_index]),
        int(lcc_index),
    )


# def recover_original_indices(removed_nodes: list, total_vertices: int, logger):
#     current_indices = list(range(total_vertices))
#     removed_original_indices = []
#     for r in removed_nodes:
#         if r >= len(current_indices):
#             raise IndexError(
#                 f"Index {r} out of range. Current indices length: {len(current_indices)}"
#             )
#         original_index = current_indices[r]
#         removed_original_indices.append(original_index)
#         del current_indices[r]
#     return removed_original_indices


def threshold_dismantler(
    network: cugraph.Graph,
    node_generator: Callable,
    generator_args: dict,
    stop_condition: int,
    early_stopping_auc=cp.inf,
    early_stopping_removals=cp.inf,
    logger=logging.getLogger("dummy"),
):
    removals: list[Any] = []

    network_size = network.number_of_nodes()

    generator_args.setdefault("logger", logger)

    generator = node_generator(network=network, **generator_args)
    i = 0
    while network.number_of_nodes() > 0:
        generator = node_generator(network=network, **generator_args)
        v_i_static, p = next(generator)
        v_gt = v_i_static.get() if isinstance(v_i_static, cp.ndarray) else v_i_static
        edge_list = network.view_edge_list()

        condition = ~((edge_list["src"] == v_gt) | (edge_list["dst"] == v_gt))
        network = filter_edges(network, condition, edge_list)
        i += 1

        _, local_network_lcc_size, local_network_slcc_size, _ = get_lcc_slcc(network)

        removals.append(
            (
                i,
                v_gt,
                float(p),
                local_network_lcc_size / network_size,
                local_network_slcc_size / network_size,
            )
        )

        if check_stopping_conditions(
            local_network_lcc_size,
            stop_condition,
            removals,
            i,
            early_stopping_removals,
            early_stopping_auc,
            logger,
        ):
            break
        reinserted_nodes = None

    return removals, None, None, reinserted_nodes
