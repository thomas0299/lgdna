from pathlib import Path

import cudf
import cugraph
import cupy as cp
from graph_tool import load_graph_from_csv
from graph_tool.spectral import adjacency


def load_graph(
    file: Path | str,
    fmt="auto",
    ignore_vp=None,
    ignore_ep=None,
    ignore_gp=None,
    directed=False,
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

    adj = adjacency(g, csr=True).astype(cp.float64)
    sources, destinations = adj.nonzero()
    weights = adj.data

    edge_df = cudf.DataFrame(
        {
            "src": cp.asarray(sources),
            "dst": cp.asarray(destinations),
            "weight": cp.asarray(weights),
        }
    )

    G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(edge_df, source="src", destination="dst", edge_attr="weight")

    return G
