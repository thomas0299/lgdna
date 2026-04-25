import cugraph
import cupy as cp
from cupyx.scipy.sparse import csr_matrix


def check_and_get_adj(network: cugraph.Graph | csr_matrix) -> csr_matrix:
    if isinstance(network, cugraph.Graph):
        if network.is_directed():
            raise ValueError("Network must be undirected")

        edge_df = network.to_pandas_edgelist()
        src = cp.asarray(edge_df["src"])
        dst = cp.asarray(edge_df["dst"])
        weight = (
            cp.asarray(edge_df["weights"])
            if "weights" in edge_df.columns
            else cp.ones(len(src))
        )
        all_unique_nodes = cp.unique(cp.concatenate([src, dst]))

        node_id_mapping = {node: i for i, node in enumerate(all_unique_nodes.get())}

        src_remapped = cp.asarray([node_id_mapping[node] for node in src.get()])
        dst_remapped = cp.asarray([node_id_mapping[node] for node in dst.get()])

        new_shape = (len(all_unique_nodes), len(all_unique_nodes))

        adjacency_matrix = csr_matrix(
            (weight, (src_remapped, dst_remapped)),
            shape=new_shape,
        )

    elif isinstance(network, csr_matrix):
        adjacency_matrix = network.tocsr(copy=False)

        if (adjacency_matrix != adjacency_matrix.T).nnz != 0:
            raise ValueError("Network must be symmetric to be undirected")

    if cp.sum(adjacency_matrix.diagonal()) != 0:
        raise ValueError("No self-loops are allowed")

    if cp.min(adjacency_matrix.data) < 0 or cp.max(adjacency_matrix.data) > 1:
        raise ValueError("Network must be unweighted")

    return adjacency_matrix
