import argparse
import ast
import logging
import multiprocessing
import string
import threading
from datetime import datetime as dt
from functools import partial
from itertools import product
from logging.handlers import QueueHandler
from pathlib import Path
from queue import Queue
from time import perf_counter_ns
from typing import Callable

import numpy as np
import pandas as pd
from graph_tool import Graph
from scipy.integrate import simpson
from tqdm.auto import tqdm

from ra_dismantling import sorters as sorters
from ra_dismantling.helpers.common import product_dict
from ra_dismantling.helpers.df_helpers import df_reader
from ra_dismantling.helpers.multiprocessing import (
    TqdmLoggingHandler,
    bounded_apply_async,
    dataset_writer,
    logger_thread,
)
from ra_dismantling.helpers.providers import init_network_provider
from ra_dismantling.review_dismantlers import threshold_dismantler
from ra_dismantling.wrappers.logger import logged

# available_heuristics = list(sorters.__all_dict__.keys())
available_heuristics = [
    "RA2_sum",
    "RA2num_sum",
    "CND_sum",
    "betweenness_centrality",
    "domirank",
    "pagerank",
    "degree",
    "fitness_centrality",
    "resilience_centrality",
    "eigenvector_centrality",
]
available_heuristics = sorted(available_heuristics)


def pool_initializer(
    log_queue,
    log_level=logging.INFO,
    lock=None,
):
    import logging

    global logger

    if isinstance(log_level, str):
        log_level = logging.getLevelName(log_level)

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=[QueueHandler(log_queue)],
    )

    logger = logging.getLogger(__name__)

    if lock is not None:
        tqdm.set_lock(lock)


def setup_threads_and_queues(args, logger):
    mp_manager = multiprocessing.Manager()
    df_queue, log_queue = mp_manager.Queue(), mp_manager.Queue()
    rand_removal_data = mp_manager.dict()

    for target, args in [
        (logger_thread, (logger, log_queue)),
        (dataset_writer, (df_queue, args.output_file)),
    ]:
        threading.Thread(target=target, args=args, daemon=True).start()

    return mp_manager, df_queue, log_queue, rand_removal_data


def get_networks_provider(args):
    networks = init_network_provider(args.location, filter=args.test_filter)
    return sorted(
        [
            (name, network)
            for name, network in networks
            if (args.max_num_vertices >= network.num_vertices() > args.min_num_vertices)
            & (args.max_num_edges >= network.num_edges() > args.min_num_edges)
        ],
        key=lambda x: x[1].num_vertices(),
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
    pool,
    queue_semaphore,
    df_queue,
    logger,
    df,
    networks_provider,
    rand_removal_data,
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
                pool,
                queue_semaphore,
                df_queue,
                logger,
                df,
                networks_provider,
                rand_removal_data,
                local_heuristic_function,
                local_heuristic_name,
                heuristic_kwargs,
            )


def process_networks(
    args,
    pool,
    queue_semaphore,
    df_queue,
    logger,
    df,
    networks_provider,
    rand_removal_data,
    heuristic_function,
    heuristic_name,
    heuristic_kwargs,
):

    modes = []
    if args.static_dismantling:
        modes.append(True)
    if args.dynamic_dismantling:
        modes.append(False)

    for name, network in tqdm(networks_provider, position=2):
        stop_condition = np.ceil(network.num_vertices() * args.threshold)

        for mode in modes:
            if should_skip(df, mode, name, heuristic_name, args):
                logger.info(
                    f"Skipping already processed {name}, static={mode}, {heuristic_name}, "
                    f"reinsertion={args.reinsertion}, reinsertion_type={args.reinsertion_type}"
                )
                continue
            try:
                removals_for_reinsertion = get_removals_for_reinsertion(
                    df, mode, name, heuristic_name, args
                )

            except RuntimeError as e:
                logger.info(e)
                break

            bounded_apply_async(
                semaphore=queue_semaphore,
                pool=pool,
                func=process_network,
                kwargs={
                    "args": args,
                    "heuristic_function": heuristic_function,
                    "heuristic_kwargs": heuristic_kwargs,
                    "mode": mode,
                    "reinsert": args.reinsertion,
                    "removals_for_reinsertion": removals_for_reinsertion,
                    "reinsertion_type": args.reinsertion_type,
                    "name": name,
                    "network": network,
                    "stop_condition": stop_condition,
                    "df_queue": df_queue,
                    "logger": logger,
                    "log_level": args.verbose,
                },
                error_callback=partial(logger.exception, exc_info=True),
            )


def should_skip(df, mode, name, heuristic_name, args):
    filtered_df = df.loc[
        (df["static"] == mode)
        & (df["network"] == name)
        & (df["heuristic"] == heuristic_name)
    ]

    return (
        len(
            filtered_df.loc[
                (filtered_df["reinsertion"] == args.reinsertion)
                & (filtered_df["reinsertion_type"] == args.reinsertion_type)
            ]
        )
        != 0
    )


def get_removals_for_reinsertion(df, mode, name, heuristic_name, args):
    if not args.reinsertion:
        return None
    reinsert_filtered_df = df.loc[
        (df["static"] == mode)
        & (df["network"] == name)
        & (df["heuristic"] == heuristic_name)
        & (df["reinsertion"].isin([False, "NaN"]))
    ]

    if len(reinsert_filtered_df) == 0:
        raise RuntimeError(
            "Method has not yet been computed without reinsertion, skipping, not getting removals for reinsertion."
        )
    return reinsert_filtered_df["removals"].values[0]


def main(args: argparse.Namespace):
    mp_manager, df_queue, log_queue, rand_removal_data = setup_threads_and_queues(
        args, logger
    )
    queue_semaphore = mp_manager.BoundedSemaphore(value=args.jobs)

    if not (args.static_dismantling or args.dynamic_dismantling):
        exit("No generators chosen!")

    networks_provider = get_networks_provider(args)
    df = load_or_create_dataframe(args)

    with multiprocessing.Pool(
        processes=args.jobs,
        initializer=pool_initializer,
        initargs=(log_queue, args.verbose, multiprocessing.Lock()),
    ) as pool:
        process_heuristic(
            args,
            pool,
            queue_semaphore,
            df_queue,
            logger,
            df,
            networks_provider,
            rand_removal_data,
        )

        pool.close()
        pool.join()

    df_queue.put(None)
    log_queue.put(None)


@logged
def process_network(
    args,
    heuristic_function: Callable | list[Callable],
    heuristic_kwargs: dict,
    mode,
    reinsert,
    reinsertion_type: str,
    name: str,
    network: Graph,
    stop_condition: int,
    df_queue: Queue,
    removals_for_reinsertion=None,
    logger=logging.getLogger("dummy"),
    log_level=logging.INFO,
):
    import re
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
        raise NotImplementedError(
            "Generator not implemented for {}".format(heuristic_name)
        )

    display_name = " ".join(heuristic_name.split("_"))

    logger.info(
        "Dismantling {} according to {} {} {} {}. Aiming to LCC size {} ({})".format(
            name,
            ("static" if mode is True else "dynamic").upper(),
            display_name.upper(),
            ("reinsertion" if reinsert is True else "no-reinsertion").upper(),
            reinsertion_type,
            stop_condition,
            stop_condition / network.num_vertices(),
        )
    )

    n_v = network.num_vertices()

    if removals_for_reinsertion is not None:
        removals_for_reinsertion = removals_for_reinsertion.replace("nan", "0")
        removals_for_reinsertion = removals_for_reinsertion.replace("inf", "0")

    if reinsert:
        try:
            removals_for_reinsertion = ast.literal_eval(removals_for_reinsertion)
            removals_for_reinsertion = [
                int(match[1]) for match in removals_for_reinsertion
            ]
        except Exception:
            matches = re.findall(
                r"\((\d+), array\((\d+), dtype=[^)]*\), (.*?), (.*?), (.*?)\)",
                removals_for_reinsertion,
            )

            removals_for_reinsertion = [int(match[1]) for match in matches]
    else:
        removals_for_reinsertion = None

    d0 = perf_counter_ns()
    removals, _, _, reinserted_nodes = threshold_dismantler(
        network=network,
        node_generator=generator,
        generator_args=generator_args,
        stop_condition=stop_condition,
        logger=logger,
        reinsertion=reinsert,
        removals_for_reinsertion=removals_for_reinsertion,
        reinsertion_type=reinsertion_type,
    )
    delta = perf_counter_ns() - d0

    auc = simpson(list(r[3] for r in removals), dx=1)

    logger.info(
        f"Dismantling {name} according to "
        f"{('static' if mode is True else 'dynamic').upper()} {display_name.upper()} "
        f"{('reinsertion' if reinsert else 'no-reinsertion').upper()} "
        f"{reinsertion_type} took {delta / 1e9:.2f} s with {len(removals)} number of removals and r_auc_n of {auc / n_v} ."
    )

    peak_slcc = max(removals, key=itemgetter(4))

    heuristic_name = heuristic_name.replace("get_", "", 1)

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
        "time": delta / 1e9,
        "date": dt.now().strftime("%d/%m/%Y, %H:%M:%S"),
        "threshold": args.threshold,
        "reinsertion": reinsert,
        "reinserted_nodes": reinserted_nodes,
        "reinsertion_type": reinsertion_type,
    }

    run_dataframe = pd.DataFrame(
        data=[run],
        columns=args.output_df_columns,
    )
    df_queue.put(run_dataframe)


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
        help="Filter the networks given the maximum number of vertedgesices",
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

    parser.add_argument(
        "-R",
        "--reinsertion",
        default=False,
        action="store_true",
        help="Performs reinsertion at end of the attack.",
    )

    parser.add_argument(
        "-RT",
        "--reinsertion_type",
        choices=["R1", "R2", "R3"],
        default=False,
        help="Reinsertion type to use at the end of the attack. Options: R1, R2, R3.",
    )

    args, cmdline_args = parser.parse_known_args()

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
