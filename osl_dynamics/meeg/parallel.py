"""Run a processing function over multiple items in parallel."""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import time
import traceback
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import report
from osl_dynamics.utils.logger import MEEGSessionLogger

_THREAD_LIMIT_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_MAX_THREADS",
]


def _limit_onnx_threads() -> None:
    """Make ONNX Runtime use 1 thread, like the BLAS/LAPACK libraries."""
    try:
        import onnxruntime as ort
    except ImportError:
        return

    class _SingleThreadSession(ort.InferenceSession):
        def __init__(self, *args, **kwargs):
            if len(args) < 2 and kwargs.get("sess_options") is None:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                kwargs["sess_options"] = opts
            super().__init__(*args, **kwargs)

    ort.InferenceSession = _SingleThreadSession


def _get_id(item: object, index: int) -> str:
    """Get the ID for an item."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "id" in item:
            return item["id"]
        return f"session_{index}"
    return str(index)


def _items_to_sessions_dict(items: list) -> dict:
    """Convert a list of items to a sessions dict for the report module."""
    sessions = {}
    for i, item in enumerate(items):
        id = _get_id(item, i)
        if isinstance(item, dict):
            sessions[id] = item
        else:
            sessions[id] = None
    return sessions


# Logger of the item being processed, set in each worker
_logger = None


def log(msg: str) -> None:
    """Print a message to the screen and to the log file of the item.

    For use in the function passed to :func:`run`. Anything it prints with
    :code:`print` only goes to the log file.

    Parameters
    ----------
    msg : str
        Message to print.
    """
    if _logger is None:
        print(msg)
    else:
        _logger.log(msg)


def warning(msg: str) -> None:
    """Print a warning to the screen and to the log file of the item.

    For use in the function passed to :func:`run`, when an item finishes
    without everything it was meant to produce. The item is reported as
    partial.

    Parameters
    ----------
    msg : str
        Warning to print.
    """
    if _logger is None:
        print(f"WARNING: {msg}")
    else:
        _logger.warning(msg)


def _init_worker(counter) -> None:
    """Give a worker the count of finished items, shared by all workers."""
    global _n_finished
    _n_finished = counter
    _limit_onnx_threads()


def _format_elapsed(seconds: float) -> str:
    """Format a duration, e.g. 42s, 13m25s or 2h07m."""
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _worker(
    args: tuple[Callable, str, object, Path, int],
) -> tuple[str, str, float, str]:
    """Wrapper that handles logging and error catching for a single item.

    Returns the ID, the status ("ok", "partial" or "failed"), the time taken
    in seconds and the warnings or the error.
    """
    global _logger
    func, id, item, log_dir, n_items = args
    log_file = log_dir / f"{id}.log"
    err_file = log_dir / f"{id}.log.err"

    # The log of an earlier run that failed
    err_file.unlink(missing_ok=True)

    status, message = "ok", ""
    start = time.time()
    with MEEGSessionLogger(id, log_dir) as logger:
        _logger = logger
        try:
            func(item)
            if logger.warnings:
                status, message = "partial", "; ".join(
                    w.split("\n")[0] for w in logger.warnings
                )
        except Exception as e:
            status, message = "failed", " ".join(f"{type(e).__name__}: {e}".split())
            logger.error(str(e))
            traceback.print_exc()
        elapsed = time.time() - start
        with _n_finished.get_lock():
            _n_finished.value += 1
            progress = f"{_n_finished.value}/{n_items}"
        outcome = {"ok": "Done", "partial": "Done with warnings", "failed": "Failed"}
        logger.log(f"{outcome[status]} in {_format_elapsed(elapsed)} ({progress}).")
    _logger = None
    if status == "failed":
        log_file.rename(err_file)
    return id, status, elapsed, message


def _summarise(results: list[tuple[str, str, float, str]], log_dir: Path) -> None:
    """Print a summary of a run and save it to ``log_dir/summary.csv``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_file = log_dir / "summary.csv"
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "status", "seconds", "message"])
        for id, status, elapsed, message in results:
            writer.writerow([id, status, round(elapsed, 1), message])

    counts = Counter(status for _, status, _, _ in results)
    print(
        f"\nFinished {len(results)} items: {counts['ok']} ok, "
        f"{counts['partial']} partial, {counts['failed']} failed."
    )

    # Items that stopped for the same reason are counted together
    max_messages = 10
    for status in ["partial", "failed"]:
        messages = Counter(
            message.replace(id, "<id>")
            for id, item_status, _, message in results
            if item_status == status
        )
        if messages:
            print(f"\n{status.capitalize()}:")
            for message, count in messages.most_common(max_messages):
                print(f"  {count} x {message}")
            if len(messages) > max_messages:
                print(f"  ... and {len(messages) - max_messages} other messages")
    print(f"\nSummary saved: {summary_file}")


def run(
    func: Callable,
    items: list,
    n_workers: int,
    log_dir: str | Path,
    output_dir: str | Path | None = None,
    qc_dir: str | Path | None = None,
    maxtasksperchild: int | None = 50,
) -> None:
    """Run a function over items in parallel.

    Parameters
    ----------
    func : callable
        Function to call for each item. Signature: ``func(item)``. Everything
        it prints goes to the log file of the item. Use :func:`log` to print a
        message to the screen as well. An item whose function raises an
        exception is reported as failed. One whose function calls
        :func:`warning`, for an item that finishes without everything it was
        meant to produce, is reported as partial.
    items : list
        Items to process. Each item is passed as the first argument to
        :code:`func`. Items can be strings or dicts. If a dict contains
        an :code:`"id"` key, it is used as the session ID for logging;
        otherwise, a numeric index is used.
    n_workers : int
        Number of parallel workers.
    log_dir : str or Path
        Directory for per-item log files (:code:`<id>.log`, or
        :code:`<id>.log.err` for an item that failed) and for
        :code:`summary.csv`, which holds the status, time taken and warnings
        or error of each item.
    output_dir : str or Path, optional
        Derivatives directory. Passed to report generation for
        reading the surface extraction and coregistration QC and copying
        their plots.
    qc_dir : str or Path, optional
        If provided, generate a QC report after processing.
    maxtasksperchild : int, optional
        Number of items a worker processes before it is replaced by a fresh
        one. Some steps (e.g. rhino coregistration and forward modelling) leak
        on the order of tens of MB per item, which over a large dataset is
        enough to exhaust memory in a long-lived worker. Recycling caps that at
        :code:`maxtasksperchild` times the per-item leak. Pass :code:`None` to
        let workers live for the whole run.
    """
    log_dir = Path(log_dir)
    worker_args = [
        (func, _get_id(item, i), item, log_dir, len(items))
        for i, item in enumerate(items)
    ]

    # Limit BLAS/LAPACK threads to 1 per worker to avoid over-subscription.
    # We set these before spawning workers so each child process picks them
    # up before NumPy/SciPy initialise their threading backends.
    old_env = {var: os.environ.get(var) for var in _THREAD_LIMIT_VARS}
    for var in _THREAD_LIMIT_VARS:
        os.environ[var] = "1"

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        maxtasksperchild=maxtasksperchild,
        initializer=_init_worker,
        initargs=(ctx.Value("i", 0),),
    ) as pool:
        results = pool.map(_worker, worker_args, chunksize=1)

    # Restore original environment
    for var, val in old_env.items():
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val

    _summarise(results, log_dir)

    if qc_dir is not None:
        report.generate_report(
            qc_dir,
            _items_to_sessions_dict(items),
            output_dir=output_dir,
        )
