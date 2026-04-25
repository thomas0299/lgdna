import copy
import logging
from typing import Any, Callable

import numpy as np
from graph_tool import Graph
from graph_tool.topology import label_components
from scipy.integrate import simpson


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

    current_auc = simpson(list(r[3] for r in removals), dx=1)
    if (i > early_stopping_removals) and (current_auc > early_stopping_auc):
        removals.append((-1, -1, -1, -1, -1))
        logger.debug("EARLY STOPPING")
        return True

    return False


def get_lcc_slcc(network):
    # Networks are undirected, and this is checked after load phase
    # Forcing directed = False triggers a GraphView call which is expensive
    belongings, counts = label_components(network)  # , directed=False)
    counts = counts.astype(int, copy=False)
    if len(counts) < 2:
        return belongings, counts[0] if counts else 0, 0, 0
    lcc_index, slcc_index = np.argpartition(np.negative(counts), 1)[:2]
    return belongings, counts[lcc_index], counts[slcc_index], lcc_index


def recover_original_indices(removed_nodes: list, total_vertices: int, logger):
    current_indices = list(range(total_vertices))
    removed_original_indices = []
    for r in removed_nodes:
        if r >= len(current_indices):
            raise IndexError(
                f"Index {r} out of range. Current indices length: {len(current_indices)}"
            )
        original_index = current_indices[r]
        removed_original_indices.append(original_index)
        del current_indices[r]
    return removed_original_indices


def process_removals(
    network: Graph,
    removed_nodes: list,
    stop_condition: int,
):
    network_size = network.num_vertices()
    removals: list[Any] = []
    i = 1

    for i, v_i_static in enumerate(removed_nodes, start=1):

        v_gt = network.vertex(v_i_static, use_index=True, add_missing=False)
        network.clear_vertex(v_gt)

        _, local_network_lcc_size, local_network_slcc_size, _ = get_lcc_slcc(network)

        removals.append(
            (
                i,
                v_i_static,
                None,
                local_network_lcc_size / network_size,
                local_network_slcc_size / network_size,
            )
        )

        if local_network_lcc_size < stop_condition:
            print("Premature break of removals process")
            break

    _, local_network_lcc_size, local_network_slcc_size, _ = get_lcc_slcc(network)
    if local_network_lcc_size > stop_condition:
        raise RuntimeError(
            f"Reinsertion phase increased LCC {local_network_lcc_size=} above threshold {stop_condition=}"
        )

    return network, removals


def reinsert_nodes_r1(
    network: Graph,
    removals: list[Any],
    stop_condition: int,
    logger: logging.Logger,
):
    original_network = copy.deepcopy(network)

    all_neighbors = {
        i: set(original_network.get_all_neighbors(i))
        for i in original_network.iter_vertices()
    }
    reinserted_nodes = []

    try:
        original_indices = recover_original_indices(
            removals, original_network.num_vertices(), logger
        )
        dismantled_network, _ = process_removals(
            copy.deepcopy(original_network),
            original_indices,
            stop_condition,
        )
    except Exception:
        original_indices = (
            [x[1] for x in removals] if isinstance(removals[0], tuple) else removals
        )

        dismantled_network, _ = process_removals(
            copy.deepcopy(original_network),
            original_indices,
            stop_condition,
        )

    _, local_network_lcc_size, _, _ = get_lcc_slcc(dismantled_network)

    while original_indices:
        reinsertion_candidates = []

        for removed_node in original_indices:
            node_to_add = removed_node
            temp_network = copy.deepcopy(dismantled_network)
            edges_to_add = [
                x for x in all_neighbors[node_to_add] if x not in original_indices
            ]
            temp_network.add_vertex(node_to_add)
            edge_list = [(node_to_add, x) for x in edges_to_add]
            temp_network.add_edge_list(edge_list)
            _, temp_local_network_lcc_size, _, _ = get_lcc_slcc(temp_network)

            components = label_components(temp_network)[0]
            component_label = components[node_to_add]
            component_size = np.sum(components.a == component_label)

            reinsertion_candidates.append(
                (
                    node_to_add,
                    original_indices.index(node_to_add),
                    edge_list,
                    component_size,
                    temp_local_network_lcc_size,
                )
            )

        if len(reinsertion_candidates) == 0:
            break

        reinsertion_candidates.sort(key=lambda x: (x[3], -x[1]))
        found = False

        for candidate in reinsertion_candidates:
            if candidate[4] <= stop_condition:
                dismantled_network.add_vertex(candidate[0])
                dismantled_network.add_edge_list(candidate[2])
                _, local_network_lcc_size, _, _ = get_lcc_slcc(dismantled_network)
                original_indices.remove(candidate[0])
                reinserted_nodes.append(candidate[0])
                found = True
                break

        if not found:
            logger.info("No reinsertion candidate satisfies the stop_condition.")
            break

    logger.info(
        f"Stopping reinsertion at {local_network_lcc_size=} vs {stop_condition=}"
    )

    _, removals_reinsertion = process_removals(
        original_network,
        original_indices,
        stop_condition,
    )

    return removals_reinsertion, reinserted_nodes


def reinsert_nodes_r2(
    network: Graph,
    removals: list[Any],
    stop_condition: int,
    logger: logging.Logger,
):
    original_network = copy.deepcopy(network)

    all_neighbors = {
        i: set(original_network.get_all_neighbors(i))
        for i in original_network.iter_vertices()
    }
    reinserted_nodes = []

    try:
        original_indices = recover_original_indices(
            removals, original_network.num_vertices(), logger
        )
        dismantled_network, _ = process_removals(
            copy.deepcopy(original_network),
            original_indices,
            stop_condition,
        )
    except Exception:
        original_indices = (
            [x[1] for x in removals] if isinstance(removals[0], tuple) else removals
        )

        dismantled_network, _ = process_removals(
            copy.deepcopy(original_network),
            original_indices,
            stop_condition,
        )

    _, local_network_lcc_size, _, _ = get_lcc_slcc(dismantled_network)

    while original_indices:
        reinsertion_candidates = []

        for removed_node in original_indices:
            node_to_add = removed_node
            temp_network = copy.deepcopy(dismantled_network)
            edges_to_add = [
                x for x in all_neighbors[node_to_add] if x not in original_indices
            ]
            temp_network.add_vertex(node_to_add)
            edge_list = [(node_to_add, x) for x in edges_to_add]
            temp_network.add_edge_list(edge_list)
            _, temp_local_network_lcc_size, _, _ = get_lcc_slcc(temp_network)

            components = label_components(temp_network)[0]
            neighbor_components = {components[x] for x in edges_to_add}
            num_clusters_joined = len(neighbor_components)

            reinsertion_candidates.append(
                (
                    node_to_add,
                    original_indices.index(node_to_add),
                    edge_list,
                    num_clusters_joined,
                    temp_local_network_lcc_size,
                )
            )

        if len(reinsertion_candidates) == 0:
            break

        reinsertion_candidates.sort(key=lambda x: (x[3], -x[1]))
        found = False

        for candidate in reinsertion_candidates:
            if candidate[4] <= stop_condition:
                dismantled_network.add_vertex(candidate[0])
                dismantled_network.add_edge_list(candidate[2])
                _, local_network_lcc_size, _, _ = get_lcc_slcc(dismantled_network)
                original_indices.remove(candidate[0])
                reinserted_nodes.append(candidate[0])
                found = True
                break

        if not found:
            logger.info("No reinsertion candidate satisfies the stop_condition.")
            break

    logger.info(
        f"Stopping reinsertion at {local_network_lcc_size=} vs {stop_condition=}"
    )

    _, removals_reinsertion = process_removals(
        original_network,
        original_indices,
        stop_condition,
    )

    return removals_reinsertion, reinserted_nodes


def reinsert_nodes_r3(
    network: Graph,
    removals: list[Any],
    stop_condition: int,
    logger: logging.Logger,
):
    original_network = copy.deepcopy(network)

    all_neighbors = {
        i: set(original_network.get_all_neighbors(i))
        for i in original_network.iter_vertices()
    }
    reinserted_nodes = []

    try:
        original_indices = recover_original_indices(
            removals, original_network.num_vertices(), logger
        )
        dismantled_network, _ = process_removals(
            copy.deepcopy(original_network),
            original_indices,
            stop_condition,
        )
    except Exception:
        original_indices = (
            [x[1] for x in removals] if isinstance(removals[0], tuple) else removals
        )

        dismantled_network, _ = process_removals(
            copy.deepcopy(original_network),
            original_indices,
            stop_condition,
        )

    _, local_network_lcc_size, _, _ = get_lcc_slcc(dismantled_network)

    while original_indices:
        reinsertion_candidates = []

        for removed_node in original_indices:
            node_to_add = removed_node
            temp_network = copy.deepcopy(dismantled_network)
            edges_to_add = [
                x for x in all_neighbors[node_to_add] if x not in original_indices
            ]
            temp_network.add_vertex(node_to_add)
            edge_list = [(node_to_add, x) for x in edges_to_add]
            temp_network.add_edge_list(edge_list)
            _, temp_local_network_lcc_size, _, _ = get_lcc_slcc(temp_network)

            reinsertion_candidates.append(
                (
                    node_to_add,
                    original_indices.index(node_to_add),
                    edge_list,
                    temp_local_network_lcc_size,
                )
            )

        if len(reinsertion_candidates) == 0:
            break

        reinsertion_candidates.sort(key=lambda x: (x[3], -x[1]))
        found = False

        for candidate in reinsertion_candidates:
            if candidate[3] <= stop_condition:
                dismantled_network.add_vertex(candidate[0])
                dismantled_network.add_edge_list(candidate[2])
                _, local_network_lcc_size, _, _ = get_lcc_slcc(dismantled_network)
                original_indices.remove(candidate[0])
                reinserted_nodes.append(candidate[0])
                found = True
                break

        if not found:
            logger.info("No reinsertion candidate satisfies the stop_condition.")
            break

    logger.info(
        f"Stopping reinsertion at {local_network_lcc_size=} vs {stop_condition=}"
    )

    _, removals_reinsertion = process_removals(
        original_network,
        original_indices,
        stop_condition,
    )

    return removals_reinsertion, reinserted_nodes


def threshold_dismantler(
    network: Graph,
    node_generator: Callable,
    generator_args: dict,
    stop_condition: int,
    reinsertion: bool,
    reinsertion_type: str,
    removals_for_reinsertion: list[Any] | None,
    early_stopping_auc=np.inf,
    early_stopping_removals=np.inf,
    logger=logging.getLogger("dummy"),
):
    removals: list[Any] = []

    network.set_fast_edge_removal(fast=True)

    network_size = network.num_vertices()

    generator_args.setdefault("logger", logger)

    # there is a big difference between removing (index might reset) and clearing (index doesn't reset because vertex still exists, it just doesn't have more edges
    vertex_remover: Callable = (
        network.clear_vertex
        if "static" in node_generator.__name__
        else network.remove_vertex
    )

    if not reinsertion:
        for i, (v_i_static, p) in enumerate(
            node_generator(network, **generator_args), start=1
        ):
            # Find the vertex in graph-tool and remove it
            v_gt = network.vertex(v_i_static, use_index=True, add_missing=False)

            if v_gt.in_degree() == 0 and v_gt.out_degree() == 0:
                print(f"Vertex {v_i_static} is isolated.")
            vertex_remover(v_gt)

            _, local_network_lcc_size, local_network_slcc_size, _ = get_lcc_slcc(
                network
            )
            removals.append(
                (
                    i,
                    v_i_static,
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
    else:
        if removals_for_reinsertion:
            if reinsertion_type == "R1":
                reinsertion_function = reinsert_nodes_r1
            elif reinsertion_type == "R2":
                reinsertion_function = reinsert_nodes_r2
            elif reinsertion_type == "R3":
                reinsertion_function = reinsert_nodes_r3
            else:
                raise ValueError(f"Reinsertion type {reinsertion_type} not valid.")

            removals, reinserted_nodes = reinsertion_function(
                network,
                removals_for_reinsertion,
                stop_condition,
                logger,
            )

    return removals, None, None, reinserted_nodes
