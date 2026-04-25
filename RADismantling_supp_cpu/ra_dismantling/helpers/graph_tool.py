from pathlib import Path

import numpy as np
from graph_tool import Graph, load_graph_from_csv


def to_adjacency(g: Graph) -> np.ndarray:
    num_vertices = g.num_vertices()
    adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

    for e in g.edges():
        u, v = e.source(), e.target()
        u_index, v_index = int(u), int(v)
        adj_matrix[u_index, v_index] = 1
        adj_matrix[v_index, u_index] = 1
    return adj_matrix


def load_graph(
    file: Path | str,
    fmt="auto",
    ignore_vp=None,
    ignore_ep=None,
    ignore_gp=None,
    directed=True,
    **kwargs
):
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        from graph_tool import load_graph

    if (
        fmt == "auto"
        and isinstance(file, str)
        and Path(file).suffix[1:] in ["csv", "edgelist", "edge", "edges", "el", "txt"]
    ):

        delimiter = kwargs.get("delimiter", None)
        if delimiter is None:
            delimiter = "," if Path(file).suffix == ".csv" else " "

        g = load_graph_from_csv(
            file,
            directed=directed,
            eprop_types=kwargs.get("eprop_types", None),
            eprop_names=kwargs.get("eprop_names", None),
            hashed=kwargs.get("hashed", False),
            hash_type=kwargs.get("hash_type", "string"),
            skip_first=kwargs.get("skip_first", False),
            ecols=kwargs.get("ecols", (0, 1)),
            csv_options=kwargs.get(
                "csv_options", {"delimiter": delimiter, "quotechar": '"'}
            ),
        )
    else:
        g = load_graph(
            file,
            fmt=fmt,
            ignore_vp=ignore_vp,
            ignore_ep=ignore_ep,
            ignore_gp=ignore_gp,
        )

    return g


def to_networkx(g):
    from io import BytesIO

    from networkx import read_graphml

    with BytesIO() as io_buffer:
        g.save(io_buffer, fmt="graphml")

        io_buffer.seek(0)

        try:
            gn = read_graphml(io_buffer)
        except Exception as e:
            raise e
    return gn
