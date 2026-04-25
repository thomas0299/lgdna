from glob import glob
from pathlib import Path

from ra_dismantling.helpers.graph_tool import load_graph


def storage_provider(
    location,
    max_num_vertices=None,
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
        raise FileNotFoundError

    networks = list()
    for file in files:
        filename = Path(file).stem

        network = load_graph(file)

        if (max_num_vertices is not None) and (
            network.num_vertices() > max_num_vertices
        ):
            continue

        assert not network.is_directed()

        if "static_id" not in network.vertex_properties:
            network.vertex_properties["static_id"] = network.new_vertex_property(
                "int",
                vals=network.vertex_index,
            )

        network.graph_properties["filename"] = network.new_graph_property(
            "string", filename
        )

        if callback:
            callback(filename, network)

        networks.append((filename, network))

    return networks


def init_network_provider(location, filter="*"):
    networks = storage_provider(location, filter=filter)
    return networks
