from contextlib import contextmanager
from time import perf_counter
from typing import Callable


@contextmanager
def block_timing(
    output: Callable[[str], None] | None = None,
    msg: str = "Timer",
    suffix: str = " took {:.2f} s",
):
    if output is None:
        output = print
    d0 = perf_counter()
    yield
    delta = perf_counter() - d0
    print(msg + suffix.format(delta))
