"""Functions for running full pipelines via the config API.

See the `toolbox examples
<https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/toolbox_paper>`_
for scripts that use the config API.
"""

import logging
import os
import pickle
import pprint
from pathlib import Path

import numpy as np
import yaml

from osl_dynamics.config_api import wrappers
from osl_dynamics.utils.misc import override_dict_defaults

_logger = logging.getLogger("osl-dynamics")


def load_config(config):
    """Load config.

    Parameters
    ----------
    config : str or dict
        Path to yaml file, string to convert to dict, or dict
        containing the config.

    Returns
    -------
    config : dict
        Config for a full pipeline.
    """
    if type(config) not in [str, dict]:
        raise ValueError("config must be a str or dict, got {}.".format(type(config)))

    if isinstance(config, str):
        try:
            # See if we have a filepath
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            # We have a string
            config = yaml.load(config, Loader=yaml.FullLoader)

    return config


def find_function(name, extra_funcs=None):
    """Find a function to execute via the config API.

    Parameters
    ----------
    name : str
        Function name.
    extra_funcs : list of function
        Custom functions passed by the user.

    Returns
    -------
    func : function
        Function to execute.
    """
    func = None

    # Check if the requested function is one of the custom functions
    if extra_funcs is not None:
        func_ind = [
            idx if (f.__name__ == name) else -1 for idx, f in enumerate(extra_funcs)
        ]
        if np.max(func_ind) > -1:
            func = extra_funcs[np.argmax(func_ind)]

    # Check osl_dynamics.config_api.wrappers
    if func is None and hasattr(wrappers, name):
        func = getattr(wrappers, name)

    if func is None:
        _logger.warn(f"{name} not found.")

    return func


def run_pipeline(config, output_dir, data=None, extra_funcs=None):
    """Run a full pipeline.

    Parameters
    ----------
    config : str or dict
        Path to yaml file, string to convert to dict, or dict containing the config.
    output_dir : str
        Path to output directory.
    data : osl_dynamics.data.Data
        Data object. Optional.
    extra_funcs : list of functions
        User-defined functions referenced in the config.
    """

    # Load config
    config = load_config(config)
    config_id = str(id(config))[3:7]
    _logger.info(
        "Using config:\n {}".format(
            pprint.pformat(config, sort_dicts=False, compact=True)
        )
    )

    # Load data via the config
    load_data_kwargs = config.pop("load_data", None)
    if load_data_kwargs is not None:
        # Make sure the Data class uses a unique temporary directory
        data_kwargs = load_data_kwargs.pop("data_kwargs", {})
        default_data_kwargs = {"store_dir": f"tmp_{config_id}"}
        data_kwargs = override_dict_defaults(default_data_kwargs, data_kwargs)
        load_data_kwargs["data_kwargs"] = data_kwargs

        # Load data
        _logger.info(f"load_data: {load_data_kwargs}")
        data = wrappers.load_data(**load_data_kwargs)

    # Loop through each item in the config
    for name, kwargs in config.items():
        func = find_function(name, extra_funcs)
        if func is not None:
            try:
                _logger.info(f"{name}: {kwargs}")
                func(data=data, output_dir=output_dir, **kwargs)
            except Exception as e:
                _logger.exception(e)

    # Delete the temporary directory created by the Data class
    if data is not None:
        data.delete_dir()


def run_pipeline_from_file(
    config_file,
    output_directory,
    restrict: str,
):
    """Run a pipeline from a config file.

    Parameters
    ----------
    config_file : str
        Path to the config file.
    output_directory : str
        Path to the output directory.
    restrict : str
        GPU to use. Optional.
    """
    if restrict is not None:
        from osl_dynamics.inference.tf_ops import gpu_growth, select_gpu

        select_gpu(int(restrict))
        gpu_growth()
    config_path = Path(config_file)
    config = config_path.read_text()

    run_pipeline(config, output_directory)
