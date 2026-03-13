"""Run a processing function over multiple items in parallel."""

import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from multiprocessing import Pool

from . import report
from osl_dynamics.utils.logger import MEEGSessionLogger


def _worker(
    args: Tuple[Callable, str, Any, Path, dict],
) -> Tuple[str, bool]:
    """Wrapper that handles logging and error catching for a single item."""
    func, id, item, log_dir, kwargs = args
    with MEEGSessionLogger(id, log_dir) as logger:
        try:
            func(id, item, logger, **kwargs)
            return id, True
        except Exception as e:
            logger.error(str(e))
            traceback.print_exc()
            return id, False


def run(
    func: Callable,
    items: Union[Dict[str, Any], List[str]],
    n_workers: int,
    log_dir: Union[str, Path],
    plots_dir: Optional[Union[str, Path]] = None,
    sessions: Optional[Dict] = None,
    output_dir: Optional[Union[str, Path]] = None,
    step_name: str = "Step",
    **kwargs,
) -> None:
    """Run a function over items in parallel.

    Parameters
    ----------
    func : callable
        Function to call for each item. Signature:
        ``func(id, item, logger, **kwargs)``.
    items : dict or list
        Items to process. If a dict, each key is an ID and each value
        is passed as the second argument to func. If a list, each
        element is used as the ID with None as the value.
    n_workers : int
        Number of parallel workers.
    log_dir : str or Path
        Directory for per-item log files.
    plots_dir : str or Path, optional
        If provided, generate a QC report after processing.
    sessions : dict, optional
        Sessions dictionary for the QC report. If None, uses items.
    output_dir : str or Path, optional
        Derivatives directory. Passed to report generation for
        copying surface extraction plots.
    step_name : str, optional
        Name for the step (used in summary message).
    **kwargs
        Extra keyword arguments passed to func.
    """
    if isinstance(items, list):
        items = {item: None for item in items}

    log_dir = Path(log_dir)
    worker_args = [(func, id, item, log_dir, kwargs) for id, item in items.items()]

    with Pool(processes=n_workers) as pool:
        results = pool.map(_worker, worker_args)

    failed = [id for id, ok in results if not ok]
    if failed:
        print(f"\n{step_name} finished with errors in: {', '.join(failed)}")
    else:
        print(f"\n{step_name} complete.")

    if plots_dir is not None:
        report.generate_report(
            plots_dir, sessions or items, output_dir=output_dir,
        )
