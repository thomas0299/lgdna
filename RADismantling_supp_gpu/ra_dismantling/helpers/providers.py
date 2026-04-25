from glob import glob
from pathlib import Path

import cupy as cp

from ra_dismantling.helpers.graph_tool import load_graph


def storage_provider(
    location,
    max_num_vertices=None,
    max_num_edges=None,  # Additional parameter to filter by edges
    filter="*",
    extensions: list | tuple | str = ("graphml", "gt"),
    callback=None,
):

    if not location.is_absolute():
        location = location.resolve()

    if not location.exists():
        raise FileNotFoundError(f"Location {location} does not exist.")
    elif not location.is_dir():
        raise FileNotFoundError(f"Location {location} is not a directory.")

    if not isinstance(filter, list):
        filter = [filter]

    if not isinstance(extensions, (list, tuple)):
        extensions = [extensions]

    files = []
    for extension in extensions:
        for f in filter:
            loc = location / f"{f}.{extension}"
            files += glob(str(loc))
    files = sorted(files)

    if len(files) == 0:
        raise FileNotFoundError("No matching graph files found.")

    networks = []
    for i, file in enumerate(files):
        filename = Path(file).stem

        try:
            network = load_graph(file)

            num_vertices = network.number_of_nodes()
            num_edges = network.number_of_edges()
            if (max_num_vertices is not None and num_vertices > max_num_vertices) or (
                max_num_edges is not None and num_edges > max_num_edges
            ):
                print(f"Skipping graph {filename}: {num_vertices=} {num_edges=}")
                continue

            assert not network.is_directed()

            cp._default_memory_pool.free_all_blocks()

            if callback:
                callback(filename, network)

            networks.append((filename, network))

        except RuntimeError as e:
            print(f"RuntimeError for graph {filename}: {e}. Skipping this graph.")
        except Exception as e:
            print(
                f"Exception while processing graph {filename}: {e}. Skipping this graph {i=}."
            )

    return networks


def init_network_provider(location, filter="*"):
    networks = storage_provider(location, filter=filter)
    return networks
