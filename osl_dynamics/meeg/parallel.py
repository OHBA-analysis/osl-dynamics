"""Run a processing function over multiple items in parallel."""

import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from . import report
from osl_dynamics.utils.logger import MEEGSessionLogger

_THREAD_LIMIT_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_MAX_THREADS",
]


def _limit_onnx_threads():
    """Patch ONNX Runtime to use 1 thread per session."""
    try:
        import onnxruntime as ort

        _OriginalSession = ort.InferenceSession

        class _SingleThreadSession(_OriginalSession):
            def __init__(self, *args, **kwargs):
                if "sess_options" not in kwargs or kwargs["sess_options"] is None:
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    kwargs["sess_options"] = opts
                super().__init__(*args, **kwargs)

        ort.InferenceSession = _SingleThreadSession
    except ImportError:
        pass


def _get_id(item: Any, index: int) -> str:
    """Get the ID for an item."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "id" in item:
            return item["id"]
        return f"session_{index}"
    return str(index)


def _items_to_sessions_dict(items: list) -> Dict:
    """Convert a list of items to a sessions dict for the report module."""
    sessions = {}
    for i, item in enumerate(items):
        id = _get_id(item, i)
        if isinstance(item, dict):
            sessions[id] = item
        else:
            sessions[id] = None
    return sessions


def _worker(
    args: Tuple[Callable, str, Any, Path],
) -> Tuple[str, bool]:
    """Wrapper that handles logging and error catching for a single item."""
    _limit_onnx_threads()
    func, id, item, log_dir = args
    with MEEGSessionLogger(id, log_dir) as logger:
        try:
            func(item, logger)
            return id, True
        except Exception as e:
            logger.error(str(e))
            traceback.print_exc()
            return id, False


def run(
    func: Callable,
    items: List[Any],
    n_workers: int,
    log_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    plots_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Run a function over items in parallel.

    Parameters
    ----------
    func : callable
        Function to call for each item. Signature:
        ``func(item, logger)``.
    items : list
        Items to process. Each item is passed as the first argument to
        :code:`func`. Items can be strings or dicts. If a dict contains
        an :code:`"id"` key, it is used as the session ID for logging;
        otherwise, a numeric index is used.
    n_workers : int
        Number of parallel workers.
    log_dir : str or Path
        Directory for per-item log files.
    output_dir : str or Path, optional
        Derivatives directory. Passed to report generation for
        copying surface extraction plots.
    plots_dir : str or Path, optional
        If provided, generate a QC report after processing.
    """
    log_dir = Path(log_dir)
    worker_args = [
        (func, _get_id(item, i), item, log_dir) for i, item in enumerate(items)
    ]

    # Limit BLAS/LAPACK threads to 1 per worker to avoid over-subscription.
    # We set these before spawning workers so each child process picks them
    # up before NumPy/SciPy initialise their threading backends.
    old_env = {var: os.environ.get(var) for var in _THREAD_LIMIT_VARS}
    for var in _THREAD_LIMIT_VARS:
        os.environ[var] = "1"

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_worker, worker_args)

    # Restore original environment
    for var, val in old_env.items():
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val

    failed = [id for id, ok in results if not ok]
    if failed:
        print(f"\nFinished with errors in: {', '.join(failed)}")
    else:
        print(f"\nComplete.")

    if plots_dir is not None:
        report.generate_report(
            plots_dir,
            _items_to_sessions_dict(items),
            output_dir=output_dir,
        )
