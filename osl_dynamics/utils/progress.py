"""Progress bars.

All progress bars printed by osl-dynamics are made with :func:`progress`
or :func:`parallel_map`, so they look the same and can be turned off
together. Progress bars are shown when the :code:`osl-dynamics` logger is
enabled for :code:`INFO` messages.
"""

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Optional

from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

_logger = logging.getLogger("osl-dynamics")

# Minimum number of seconds between updates of a progress bar that is
# written to a file (e.g. the log of a job on a cluster)
_NON_INTERACTIVE_MININTERVAL = 30


def _interactive() -> bool:
    """Is the output going to a terminal or a notebook?"""
    try:
        if sys.stderr.isatty():
            return True
    except (AttributeError, ValueError):
        pass
    if "IPython" in sys.modules:
        try:
            from IPython import get_ipython

            return "IPKernelApp" in get_ipython().config
        except (AttributeError, ImportError):
            pass
    return False


def progress(
    iterable: Optional[Iterable] = None,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    **kwargs,
) -> tqdm:
    """Progress bar.

    Parameters
    ----------
    iterable : iterable, optional
        Iterable to wrap with a progress bar.
    desc : str, optional
        Description shown before the progress bar.
    total : int, optional
        Number of iterations. Defaults to :code:`len(iterable)`.
    kwargs : keyword arguments, optional
        Keyword arguments for :code:`tqdm`.

    Returns
    -------
    bar : tqdm.tqdm
        Progress bar.
    """
    if not _logger.isEnabledFor(logging.INFO):
        kwargs["disable"] = True

    # Update a progress bar written to a file less often, unless the user
    # has chosen how often with tqdm's environment variables. maxinterval
    # is also set because tqdm refreshes a progress bar after maxinterval
    # seconds without an update
    if not _interactive():
        for name in ["mininterval", "maxinterval"]:
            if f"TQDM_{name.upper()}" not in os.environ:
                kwargs.setdefault(name, _NON_INTERACTIVE_MININTERVAL)

    return tqdm(iterable, desc=desc, total=total, **kwargs)


def _split_arguments(item: Any, argument_type: str) -> tuple:
    """Arguments and keyword arguments to call a function with an item."""
    if argument_type == "direct":
        return (item,), {}
    if argument_type == "args":
        return tuple(item), {}
    if argument_type == "kwargs":
        return (), item
    raise ValueError("argument_type must be 'direct', 'args' or 'kwargs'.")


def parallel_map(
    function: Callable,
    items: Iterable,
    desc: Optional[str] = None,
    n_jobs: int = 1,
    argument_type: str = "direct",
    backend: str = "threads",
    label: str = "item",
) -> list:
    """Apply a function to each item of an iterable, with a progress bar.

    Parameters
    ----------
    function : callable
        Function to apply.
    items : iterable
        Items to apply the function to.
    desc : str, optional
        Description shown before the progress bar.
    n_jobs : int, optional
        Number of items to process in parallel.
    argument_type : str, optional
        How each item is passed to the function: :code:`'direct'`
        (:code:`function(item)`), :code:`'args'` (:code:`function(*item)`)
        or :code:`'kwargs'` (:code:`function(**item)`).
    backend : str, optional
        Run in parallel with :code:`'threads'` or :code:`'processes'`.
        When using threads, numpy/scipy are limited to one thread each.
    label : str, optional
        What an item is, used in the error messages
        (e.g. :code:`'session'`).

    Returns
    -------
    results : list
        Result for each item, in the same order as :code:`items`.
    """
    items = list(items)
    calls = [_split_arguments(item, argument_type) for item in items]

    if n_jobs == 1 or len(items) <= 1:
        return [
            function(*args, **kwargs) for args, kwargs in progress(calls, desc=desc)
        ]

    if backend == "threads":
        executor = ThreadPoolExecutor
        limits = 1
    elif backend == "processes":
        executor = ProcessPoolExecutor
        limits = None
    else:
        raise ValueError("backend must be 'threads' or 'processes'.")

    with threadpool_limits(limits=limits), executor(max_workers=n_jobs) as pool:
        futures = [pool.submit(function, *args, **kwargs) for args, kwargs in calls]
        for _ in progress(as_completed(futures), desc=desc, total=len(futures)):
            pass

    # Report every item that failed and raise the first error
    errors = [(i, future.exception()) for i, future in enumerate(futures)]
    errors = [(i, e) for i, e in errors if e is not None]
    for i, e in errors:
        name = label[0].upper() + label[1:]
        _logger.error(f"{name} {i}: {type(e).__name__}: {e}")
    if errors:
        raise errors[0][1]

    return [future.result() for future in futures]
