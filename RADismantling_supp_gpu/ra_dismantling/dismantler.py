import argparse
import logging
import os
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from functools import partial
from itertools import product
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

import cupy as cp
import pandas as pd
from cugraph import Graph
from scipy.integrate import simpson
from tqdm.auto import tqdm

from ra_dismantling import sorters as sorters
from ra_dismantling.helpers.common import product_dict
from ra_dismantling.helpers.df_helpers import df_reader
from ra_dismantling.helpers.multiprocessing import TqdmLoggingHandler
from ra_dismantling.helpers.providers import init_network_provider
from ra_dismantling.review_dismantlers import threshold_dismantler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# available_heuristics = list(sorters.__all_dict__.keys())
available_heuristics = [
    "RA2_sum",
    "RA2num_sum",
    "CND_sum",
    "betweenness_centrality",
]
available_heuristics = sorted(available_heuristics)


def setup_threads_and_queues(args, logger):
    df_queue = []
    log_queue = []

    return df_queue, log_queue


def get_networks_provider(args):
    networks = init_network_provider(args.location, filter=args.test_filter)
    return sorted(
        [
            (name, network)
            for name, network in networks
            if (
                (
                    args.max_num_vertices
                    >= network.number_of_nodes()
                    > args.min_num_vertices
                )
                and (
                    args.max_num_edges >= network.number_of_edges() > args.min_num_edges
                )
            )
        ],
        key=lambda x: x[1].number_of_nodes(),
        reverse=True,
    )


def load_or_create_dataframe(args):
    if args.output_file.exists():
        df = df_reader(
            args.output_file,
            include_removals=True,
            expected_columns=None,
        )
        for col in args.output_df_columns:
            if col not in df.columns:
                df[col] = False
    else:
        df = pd.DataFrame(columns=args.output_df_columns)
    return df.fillna("NaN")


def process_heuristic(
    args,
    logger,
    df,
    networks_provider,
):
    for heuristic in tqdm(args.heuristics, position=0):
        heuristic_function = sorters.__all_dict__[heuristic]
        parameter_names = [
            param[1]
            for param in string.Formatter().parse(heuristic)
            if param[1] is not None
        ]
        parameters_to_try = (
            {param: getattr(args, param) for param in parameter_names}
            if parameter_names
            else [{}]
        )

        if isinstance(parameters_to_try, dict):
            parameters_to_try = list(product_dict(**parameters_to_try))

        for heuristic_kwargs in tqdm(parameters_to_try, position=1):
            local_heuristic_name = (
                heuristic.format(**heuristic_kwargs) if heuristic_kwargs else heuristic
            )
            local_heuristic_function = (
                partial(heuristic_function, logger=logger, **heuristic_kwargs)
                if heuristic_kwargs
                else heuristic_function
            )
            local_heuristic_function.__name__ = local_heuristic_name

            process_networks(
                args,
                logger,
                df,
                networks_provider,
                local_heuristic_function,
                local_heuristic_name,
                heuristic_kwargs,
            )


def process_network_wrapper(
    args,
    heuristic_function,
    heuristic_kwargs,
    mode,
    name,
    network,
    stop_condition,
    logger,
    gpu_id,
):
    device = cp.cuda.Device(gpu_id)

    props = device.attributes

    print(f"Streaming Multiprocessors: {props['MultiProcessorCount']}")
    print(f"Warp Size: {props['WarpSize']}")
    print(f"Max Threads Per Block: {props['MaxThreadsPerBlock']}")
    print(f"Max Threads Per Multiprocessor: {props['MaxThreadsPerMultiProcessor']}")

    print("Device:", device)

    with cp.cuda.Device(gpu_id):
        logging.info(f"Starting task for network '{name}' on GPU {gpu_id}")
        stop_condition = cp.ceil(network.number_of_nodes() * args.threshold)

        result = process_network(
            args=args,
            heuristic_function=heuristic_function,
            heuristic_kwargs=heuristic_kwargs,
            mode=mode,
            name=name,
            network=network,
            stop_condition=stop_condition,
            logger=logger,
            log_level=args.verbose,
        )
        logging.info(f"Finished task for network '{name}' on GPU {gpu_id}")
        return result


def parallel_process_networks(
    args,
    logger,
    networks_provider,
    heuristic_function,
    mode,
    name,
    network,
    stop_condition,
    heuristic_kwargs,
    num_gpus,
):
    results = []
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures_to_task = {}
        gpu_cycle = iter(range(num_gpus))

        for i, network_data in enumerate(networks_provider):
            gpu_id = next(gpu_cycle)
            future = executor.submit(
                process_network_wrapper,
                args=args,
                heuristic_function=heuristic_function,
                heuristic_kwargs=heuristic_kwargs,
                mode=mode,
                name=name,
                network=network,
                stop_condition=stop_condition,
                logger=logger,
                gpu_id=gpu_id,
            )
            futures_to_task[future] = (gpu_id, network_data)

            if len(futures_to_task) >= num_gpus:
                break

        for future in as_completed(futures_to_task):
            gpu_id, completed_task = futures_to_task[future]
            result = future.result()
            results.append(result)

            logger.info(f"Task completed for {completed_task[0]} on GPU {gpu_id}")
            next_task_index = len(futures_to_task) + len(results)

            if next_task_index < len(networks_provider):
                network_data = networks_provider[next_task_index]
                new_future = executor.submit(
                    process_network_wrapper,
                    args=args,
                    heuristic_function=heuristic_function,
                    heuristic_kwargs=heuristic_kwargs,
                    mode=mode,
                    name=name,
                    network=network,
                    stop_condition=stop_condition,
                    logger=logger,
                    gpu_id=gpu_id,
                )
                futures_to_task[new_future] = (gpu_id, network_data)

    return results


def process_networks(
    args,
    logger,
    df,
    networks_provider,
    heuristic_function,
    heuristic_name,
    heuristic_kwargs,
):
    num_gpus = cp.cuda.runtime.getDeviceCount()
    logger.info(f"Detected {num_gpus} GPU(s).")

    modes = []
    if args.static_dismantling:
        modes.append(True)
    if args.dynamic_dismantling:
        modes.append(False)

    for name, network in tqdm(networks_provider, position=2):
        logger.info(
            f"Loaded network: {name}, Nodes: {network.number_of_nodes()}, Edges: {network.number_of_edges()}"
        )
        stop_condition = cp.ceil(network.number_of_nodes() * args.threshold)

        for mode in modes:
            if should_skip(df, mode, name, heuristic_name, args):
                logger.info(
                    f"Skipping already processed {name}, static={mode}, {heuristic_name}, "
                    f"reinsertion={args.reinsertion}, reinsertion_type={args.reinsertion_type}"
                )
                continue
            results = parallel_process_networks(
                args=args,
                heuristic_function=heuristic_function,
                heuristic_kwargs=heuristic_kwargs,
                mode=mode,
                networks_provider=networks_provider,
                name=name,
                network=network,
                stop_condition=stop_condition,
                logger=logger,
                num_gpus=1,
            )

            for result in results:
                if result is not None:
                    result.to_csv(
                        args.output_file,
                        mode="a",
                        header=not args.output_file.exists(),
                        index=False,
                    )


def should_skip(df, mode, name, heuristic_name, args):
    filtered_df = df.loc[
        (df["static"] == mode)
        & (df["network"] == name)
        & (df["heuristic"] == heuristic_name)
    ]
    return len(filtered_df.loc[(filtered_df["reinsertion"] == args.reinsertion)]) != 0


def main(args: argparse.Namespace):

    def get_memory_usage():
        free, total = cp.cuda.runtime.memGetInfo()
        used = total - free
        return used, free, total

    used, free, total = get_memory_usage()
    print("Memory usage before computation:")
    print(f"Used: {used / 1024**2:.2f} MB")
    print(f"Free: {free / 1024**2:.2f} MB")
    print(f"Total: {total / 1024**2:.2f} MB")

    if not (args.static_dismantling or args.dynamic_dismantling):
        exit("No generators chosen!")

    logger.info("Initializing resources...")
    networks_provider = get_networks_provider(args)
    df = load_or_create_dataframe(args)

    logger.info("Starting processing...")
    process_heuristic(
        args,
        logger,
        df,
        networks_provider,
    )

    used, free, total = get_memory_usage()
    print("Memory usage after computation:")
    print(f"Used: {used / 1024**2:.2f} MB")
    print(f"Free: {free / 1024**2:.2f} MB")
    print(f"Total: {total / 1024**2:.2f} MB")


def process_network(
    args,
    heuristic_function: Callable | list[Callable],
    heuristic_kwargs: dict,
    mode,
    name: str,
    network: Graph,
    stop_condition: int,
    logger=logging.getLogger("dummy"),
    log_level=logging.INFO,
):
    from operator import itemgetter

    import pandas as pd

    from ra_dismantling.helpers.generators import dynamic_generator, static_generator

    if isinstance(log_level, str):
        log_level = logging.getLevelName(log_level)
    logger.setLevel(log_level)

    heuristic_name = heuristic_function.__name__  # type: ignore
    generator_args = dict(
        **heuristic_kwargs,
        **{
            "sorting_function": heuristic_function,
        },
    )

    generator = static_generator if mode else dynamic_generator

    logger.debug(f"Generator {generator} for {heuristic_name}")
    if generator is None:
        raise NotImplementedError(f"Generator not implemented for {heuristic_name}")

    display_name = " ".join(heuristic_name.split("_"))

    logger.info(
        f"Dismantling {name} according to "
        f"{('static' if mode is True else 'dynamic').upper()} {display_name.upper()} "
        f"{('reinsertion' if args.reinsertion else 'no-reinsertion').upper()} "
        f"Aiming to LCC size {stop_condition} "
        f"({stop_condition / network.number_of_nodes()})"
    )

    n_v = network.number_of_nodes()

    d0 = perf_counter_ns()
    removals, _, _, reinserted_nodes = threshold_dismantler(
        network=network,
        node_generator=generator,
        generator_args=generator_args,
        stop_condition=stop_condition,
        logger=logger,
    )
    delta = perf_counter_ns() - d0

    peak_slcc = max(removals, key=itemgetter(4))
    auc = simpson(list(r[3] for r in removals), dx=1)

    logger.info(
        f"Dismantling {name} according to "
        f"{('static' if mode is True else 'dynamic').upper()} {display_name.upper()} "
        f"{('reinsertion' if args.reinsertion else 'no-reinsertion').upper()} "
        f"took {delta / 1e9:.2f} s with {len(removals)} number of removals and r_auc_n of {auc / n_v} ."
    )

    run = {
        "network": name,
        "heuristic": heuristic_name,
        "removals": removals,
        "slcc_peak_at": peak_slcc[0],
        "lcc_size_at_peak": peak_slcc[3],
        "slcc_size_at_peak": peak_slcc[4],
        "static": mode,
        "r_auc": auc,
        "rem_num": len(removals),
        "r_auc_n": auc / n_v,
        "time": delta / 1e9,
        "date": dt.now().strftime("%d/%m/%Y, %H:%M:%S"),
        "threshold": args.threshold,
        "reinsertion": args.reinsertion,
        "reinsertion_type": args.reinsertion_type,
        "reinserted_nodes": reinserted_nodes,
    }

    run_dataframe = pd.DataFrame(
        data=[run],
        columns=args.output_df_columns,
    )

    return run_dataframe


def get_df_columns():
    return [
        "network",
        "heuristic",
        "slcc_peak_at",
        "lcc_size_at_peak",
        "slcc_size_at_peak",
        "removals",
        "static",
        "r_auc",
        "rem_num",
        "r_auc_n",
        "time",
        "date",
        "threshold",
        "reinsertion",
        "reinsertion_type",
        "reinserted_nodes",
    ]


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)-8s | %(message)s",
        level=logging.INFO,
        handlers=[TqdmLoggingHandler()],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./out/df/heuristics.csv"),
        required=False,
        help="Output DataFrame file location to store the results. Default: ./out/df/heuristics.csv",
    )
    parser.add_argument(
        "-l",
        "--location",
        type=Path,
        required=True,
        default=None,
        help="Path to the dataset (directory)",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
        help="Dismantling target threshold. Fraction of the network size. Default: 0.1",
    )

    parser.add_argument(
        "-H",
        "--heuristics",
        type=str,
        choices=available_heuristics + ["all", "all_new"],
        default="",
        metavar="",
        nargs="*",
        help=f"Space separated list of heuristics to test. "
        f"Allowed values are {', '.join(available_heuristics)}",
    )

    parser.add_argument(
        "-HF",
        "--heuristics_file",
        type=str,
        default="",
        help="If specified, use heuristics names stored within this file. Each heuristics"
        " can be separated by blankspace or newline.",
    )

    parser.add_argument(
        "-SD",
        "--static_dismantling",
        default=False,
        action="store_true",
        help="Static computation of the heuristics (no recomputation)",
    )

    parser.add_argument(
        "-DD",
        "--dynamic_dismantling",
        default=False,
        action="store_true",
        help="Enables recomputation of heuristics after each removal",
    )

    parser.add_argument(
        "-Ft",
        "--test_filter",
        type=str,
        default="*",
        required=False,
        help="Test folder filter (default: *). Example: 'LFR*'",
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel",
    )

    parser.add_argument(
        "-mnv",
        "--max_num_vertices",
        type=int,
        default=float("inf"),
        help="Filter the networks given the maximum number of vertices",
    )

    parser.add_argument(
        "-miv",
        "--min_num_vertices",
        type=int,
        default=0,
        help="Filter the networks given the minimum number of vertices",
    )

    parser.add_argument(
        "-mne",
        "--max_num_edges",
        type=int,
        default=float("inf"),
        help="Filter the networks given the maximum number of edges",
    )

    parser.add_argument(
        "-mie",
        "--min_num_edges",
        type=int,
        default=0,
        help="Filter the networks given the minimum number of edges",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=str.upper,
        choices=["INFO", "DEBUG", "WARNING", "ERROR"],
        default="info",
        help="Verbosity level (case insensitive)",
    )

    args, cmdline_args = parser.parse_known_args()

    args.reinsertion = False
    args.reinsertion_type = None

    logger.setLevel(logging.getLevelName(args.verbose))

    if not args.location.is_absolute():
        args.location = args.location.resolve()

    if not args.output.absolute():
        output_path = Path("./out/").resolve()
        base_dataframes_path = output_path / "df"

        args.output_file = base_dataframes_path / args.output

    else:
        args.output_file = args.output

    if not args.output_file.is_absolute():
        args.output_file = args.output_file.resolve()

    if not args.output_file.parent.exists():
        args.output_file.parent.mkdir(parents=True)

    args.output_df_columns = get_df_columns()

    if args.heuristics and args.heuristics == ["all"]:
        args.heuristics = available_heuristics
    elif args.heuristics and args.heuristics == ["all_new"]:
        comb = [sorters.VALID_WEIGHTINGS]
        all_weighting_methods = ["_".join(c) for c in comb if c[1] != "none"]
        prefix = [""]
        suffix = ["sum"]
        args.heuristics = [
            "_".join(filter(None, c))
            for c in product(prefix, all_weighting_methods, suffix)
        ]

    if args.heuristics_file and len(args.heuristics_file) > 0:
        logger.info("Read from heuristics file %s" % args.heuristics_file)
        try:
            with open(args.heuristics_file, "r") as f:
                res = f.read()
            all_heuristics = res.split()
            for heuristics in all_heuristics:
                if heuristics not in available_heuristics:
                    logger.error(
                        "Heuristics %s not available, please check your heuristics input file!"
                        % heuristics
                    )
                    raise RuntimeError(
                        "Some heuristics specified in file are not allowed."
                    )
            if args.heuristics and len(args.heuristics) > 0:
                logger.warning(
                    "Using heuristics specified in the file to overwrite heuristics specified in command"
                    " line arguments."
                )
            args.heuristics = all_heuristics
        except FileNotFoundError:
            logger.error("Specified heuristics file not found! exiting...")
            raise

    if not args.heuristics or len(args.heuristics) == 0:
        logger.error("No heuristics specified in arguments! Exiting...")
        raise RuntimeError("Heuristics not specified.")
    else:
        logger.info("All heuristics methods are %s" % ", ".join(args.heuristics))

    logger.info("Output file {}".format(args.output_file))

    main(args)
