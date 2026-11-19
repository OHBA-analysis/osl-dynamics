"""Functions for running full pipelines via the config API.

See the documentation for :mod:`osl_dynamics.config_api` for example usage.
"""

import argparse
import logging
import pprint
from pathlib import Path
from typing import Callable, List, Optional, Union

import yaml

from osl_dynamics.config_api import wrappers
from osl_dynamics.utils.misc import override_dict_defaults

_logger = logging.getLogger("osl-dynamics")


def load_config(config: Union[str, dict]) -> dict:
    """Load config.

    Parameters
    ----------
    config : str or dict
        Path to yaml file, :code:`str` to convert to :code:`dict`,
        or :code:`dict` containing the config.

    Returns
    -------
    config : dict
        Config for a full pipeline.
    """
    if type(config) not in [str, dict]:
        raise TypeError("config must be a str or dict, got {}.".format(type(config)))

    if isinstance(config, str):
        try:
            # See if we have a filepath
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            # We have a string
            config = yaml.load(config, Loader=yaml.FullLoader)

    return config


def find_function(
    name: str, extra_funcs: Optional[List[Callable]] = None
) -> Optional[Callable]:
    """Find a function to execute via the config API.

    Parameters
    ----------
    name : str
        Function name.
    extra_funcs : list of functions, optional
        Custom functions passed by the user.

    Returns
    -------
    func : function
        Function to execute. :code:`None` if the function is not found.
    """
    # Custom functions take precedence over osl_dynamics.config_api.wrappers
    for func in reversed(extra_funcs or []):
        if func.__name__ == name:
            return func

    if name.startswith("_"):
        return None
    func = getattr(wrappers, name, None)
    return func if callable(func) else None


def run_pipeline(
    config: Union[str, dict],
    output_dir: str,
    restrict: Optional[Union[int, str]] = None,
    data=None,
    extra_funcs: Optional[List[Callable]] = None,
) -> None:
    """Run a full pipeline.

    Parameters
    ----------
    config : str or dict
        Path to yaml file, :code:`str` to convert to :code:`dict`,
        or :code:`dict` containing the config.
    output_dir : str
        Path to output directory.
    restrict : int or str, optional
        GPU to use. If a str is passed it will be cast to an int.
    data : osl_dynamics.data.Data, optional
        Data object.
    extra_funcs : list of functions, optional
        User-defined functions referenced in the config.

    The pipeline stops at the first function that raises an error.
    """
    if restrict is not None:
        from osl_dynamics.inference import tf_ops

        tf_ops.select_gpu(int(restrict))
        tf_ops.gpu_growth()

    # Load config
    config = load_config(config)
    config_id = str(id(config))[3:7]
    _logger.info(
        "Using config:\n {}".format(
            pprint.pformat(config, sort_dicts=False, compact=True)
        )
    )

    # Check every function exists before running any
    load_data_kwargs = config.pop("load_data", None)
    funcs = {name: find_function(name, extra_funcs) for name in config}
    missing = [name for name, func in funcs.items() if func is None]
    if missing:
        raise ValueError(f"Functions not found: {', '.join(missing)}.")

    # Load data via the config
    if load_data_kwargs is not None:
        # Make sure the Data class uses a unique temporary directory
        kwargs = load_data_kwargs.pop("kwargs", {})
        default_kwargs = {"store_dir": f"tmp_{config_id}"}
        kwargs = override_dict_defaults(default_kwargs, kwargs)
        load_data_kwargs["kwargs"] = kwargs

        # Load data
        _logger.info(f"load_data: {load_data_kwargs}")
        data = wrappers.load_data(**load_data_kwargs)

    # Run each function in the config
    try:
        for name, kwargs in config.items():
            _logger.info(f"{name}: {kwargs}")
            funcs[name](data=data, output_dir=output_dir, **kwargs)
    finally:
        # Delete the temporary directory created by the Data class
        if data is not None:
            data.delete_dir()


def run_pipeline_from_file(
    config_file: str, output_directory: str, restrict: Optional[Union[int, str]] = None
) -> None:
    """Run a pipeline from a config file.

    Parameters
    ----------
    config_file : str
        Path to the config file.
    output_directory : str
        Path to the output directory.
    restrict : int or str, optional
        GPU to use. If a str is passed it will be cast to an int.
    """
    if restrict is not None:
        from osl_dynamics.inference import tf_ops

        tf_ops.select_gpu(int(restrict))
        tf_ops.gpu_growth()

    config_path = Path(config_file)
    config = config_path.read_text()

    run_pipeline(config, output_directory)


def osl_dynamics_cli() -> None:
    """Command line interface function for running a pipeline from a config file."""

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the config file.",
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--restrict",
        type=str,
        help="GPU to use. Optional.",
    )
    args = parser.parse_args()

    # Run pipeline
    run_pipeline_from_file(
        args.config_file,
        args.output_directory,
        restrict=args.restrict,
    )
