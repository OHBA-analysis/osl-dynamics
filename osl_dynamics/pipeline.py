"""Wrapper functions for full pipelines.

"""

import os
import logging
import yaml
import pprint
import pickle
import numpy as np

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


def run_pipeline(config, inputs, savedir="./"):
    """Run a full pipeline.

    Parameters
    ----------
    config : str or dict
        Path to yaml file, string to convert to dict, or dict
        containing the config.
    inputs : str or list
        Inputs to pass to osl_dynamics.data.Data.
    savedir : str
        Output directory to save to.
    """

    # Load config
    config = load_config(config)
    _logger.info("Using config:\n {}".format(pprint.pformat(config)))

    # Validation
    available_models = ["hmm", "dynemo"]
    model_count = 0
    for section in config:
        if section in available_models:
            model_count += 1
            model_name = section
    if model_count == 0:
        e = ValueError(
            "Please pass a model section in the config. "
            + f"Available models are: {available_models}."
        )
        _logger.error(e)
    if model_count > 1:
        e = ValueError("Multiple models are specified in the config. Please pass one.")
        _logger.error(e)

    if "multitaper_spectra" in config and "regression_spectra" in config:
        e = ValueError("Please only specify one spectra section.")
        _logger.error(e)
    elif "multitaper_spectra" in config:
        spectra_name = "multitaper_spectra"
    elif "regression_spectra" in config:
        spectra_name = "regression_spectra"
    else:
        spectra_name = None

    # Load data
    _logger.info("Loading data")
    from osl_dynamics.data import Data  # moved inside the function for fast imports

    training_data = Data(inputs)

    # Prepare data
    if "data_prep" in config:
        training_data.prepare(**config["data_prep"])

    # Create the model
    _logger.info("Building model")

    from osl_dynamics import models  # moved inside the function for fast imports

    module_files = {
        "hmm": models.hmm,
        "dynemo": models.dynemo,
    }

    model_config = module_files[model_name].Config
    model_class = module_files[model_name].Model
    model = model_class(model_config(**config[model_name]))
    model.summary()

    # Train the model
    _logger.info("Training model")
    history = model.fit(training_data)

    # Save trained model
    trained_model_dir = savedir + "/trained_model"
    _logger.info(f"Saving model to: {trained_model_dir}")
    model.save(trained_model_dir)
    pickle.dump(history, open(trained_model_dir + "/history.pkl", "wb"))

    # Get inferred parameters
    _logger.info("Getting inferred parameters")
    alpha = model.get_alpha(training_data)
    means, covs = model.get_means_covariances()

    # Save inferred parameters
    inf_params_dir = savedir + "/inf_params"
    _logger.info(f"Saving inferred parameters to: {inf_params_dir}")
    os.makedirs(inf_params_dir, exist_ok=True)
    pickle.dump(alpha, open(inf_params_dir + "/alp.pkl", "wb"))
    np.save(inf_params_dir + "/means.npy", means)
    np.save(inf_params_dir + "/covs.npy", covs)

    # Post-hoc spectra
    if spectra_name is not None:
        from osl_dynamics.analysis import (
            spectral,
        )  # moved inside the function for fast imports

        # Use unprepared data to calculate the spectra
        data = model.get_training_time_series(training_data, prepared=True)

        spectra_functions = {
            "multitaper_spectra": spectral.multitaper_spectra,
            "regression_spectra": spectral.regression_spectra,
        }

        # Calculate spectra
        _logger.info(f"Calculating {spectra_name.replace('_', ' ')}")
        f, psd, coh, w = spectra_functions[spectra_name](
            data, alpha, return_weights=True, **config[spectra_name]
        )

        # Save
        spectra_dir = savedir + "/spectra"
        os.makedirs(spectra_dir, exist_ok=True)
        _logger.info(f"Saving spectra to: {spectra_dir}")
        np.save(spectra_dir + "/f.npy", f)
        np.save(spectra_dir + "/psd.npy", psd)
        np.save(spectra_dir + "/coh.npy", coh)
        np.save(spectra_dir + "/w.npy", w)

    # Delete temporary directory
    training_data.delete_dir()
