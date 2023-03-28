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

    See the `toolbox examples
    <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/toolbox_paper>`_
    for scripts that use the config API this.

    Parameters
    ----------
    config : str or dict
        Path to yaml file, string to convert to dict, or dict
        containing the config. Recommended defaults are given in the `toolbox examples
        <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/toolbox_paper>`_.
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

    # See if we pass parameters related to initialisation (n_init, n_init_epochs)
    n_init = config[model_name].pop("n_init", None)
    n_init_epochs = config[model_name].pop("n_init_epochs", None)
    init_take = config[model_name].pop("init_take", None)
    if n_init is not None and n_init_epochs is None:
        n_init_epochs = 1
    if n_init_epochs is not None and n_init is None:
        n_init = 1
    if init_take is None:
        init_take = 1

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

    # See if we're loading the data in parallel
    if "data_prep" in config:
        data_n_jobs = config["data_prep"].pop("n_jobs", 1)

    # Create the Data object
    training_data = Data(inputs, n_jobs=data_n_jobs)

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
    config[model_name]["n_channels"] = training_data.n_channels
    model = model_class(model_config(**config[model_name]))
    model.summary()

    # Initialisation
    if n_init is not None and model_name == "hmm":
        model.random_state_time_course_initialization(
            training_data, n_init_epochs, n_init, take=init_take
        )
    if n_init is not None and model_name == "dynemo":
        model.random_subset_initialization(
            training_data, n_init_epochs, n_init, take=init_take
        )

    # Train the model
    _logger.info("Training model")
    history = model.fit(training_data)

    # Save trained model
    trained_model_dir = savedir + "/trained_model"
    _logger.info(f"Saving model to: {trained_model_dir}")
    model.save(trained_model_dir)

    # Save the free energy
    #
    # Note, for DyNeMo the 'loss' in the history object is the
    # free energy, whereas for the HMM the 'loss' in the history
    # object is the log-likelihood
    if model_name == "hmm":
        free_energy = model.free_energy(training_data)
        history["free_energy"] = free_energy

    # Save the free energy and training history
    pickle.dump(history, open(trained_model_dir + "/history.pkl", "wb"))

    # Get inferred parameters
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
        data = model.get_training_time_series(training_data, prepared=False)

        spectra_functions = {
            "multitaper_spectra": spectral.multitaper_spectra,
            "regression_spectra": spectral.regression_spectra,
        }

        # Calculate spectra
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
