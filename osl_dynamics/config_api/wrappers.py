"""Wrapper functions for use in the config API.

See the `toolbox examples
<https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/toolbox_paper>`_
for scripts that use the config API.

All of the functions in this module can be listed in the config passed to
:code:`osl_dynamics.run_pipeline`.

All wrapper functions have the structure::

    func(data, output_dir, **kwargs)

where:

- :code:`data` is an :code:`osl_dynamics.data.Data` object
- :code:`output_dir` is the path to save output to.
- :code:`kwargs` are keyword arguments for function specific options.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from osl_dynamics import array_ops
from osl_dynamics.utils.misc import load, override_dict_defaults, save

_logger = logging.getLogger("osl-dynamics")


def load_data(data_dir, data_kwargs={}, prepare_kwargs={}):
    """Load and prepare data.

    Parameters
    ----------
    data_dir : str
        Path to directory containing npy files.
    data_kwargs: dict
        Keyword arguments to pass to the Data class.
        Useful keyword arguments to pass are the :code:`sampling_frequency`,
        :code:`mask_file` and :code:`parcellation_file`.
    prepare_kwargs : dict
        Keyword arguments to pass to the prepare method.

    Returns
    -------
    data : osl_dynamics.data.Data
        Data object.
    """
    from osl_dynamics.data import Data

    data = Data(data_dir, **data_kwargs)
    data.prepare(**prepare_kwargs)
    return data


def train_hmm(
    data,
    output_dir,
    config_kwargs,
    init_kwargs={},
    fit_kwargs={},
    save_inf_params=True,
):
    """Train a Hidden Markov Model.

    This function will:

    1. Build an :code:`hmm.Model` object.
    2. Initialize the parameters of the model using
       :code:`Model.random_state_time_course_initialization`.
    3. Perform full training.
    4. Save the inferred parameters (state probabilities, means and covariances)
       if :code:`save_inf_params=True`.

    This function will create two directories:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object for training the model.
    output_dir : str
        Path to output directory.
    config_kwargs : dict
        Keyword arguments to pass to hmm.Config. Defaults to::

            {'sequence_length': 2000,
             'batch_size': 32,
             'learning_rate': 0.01,
             'n_epochs': 20}.
    init_kwargs : dict
        Keyword arguments to pass to :code:`Model.random_state_time_course_initialization`.
        Optional, defaults to::

            {'n_init': 3, 'n_epochs': 1}.
    fit_kwargs : dict
        Keyword arguments to pass to the :code:`Model.fit`. Optional, no defaults.
    save_inf_params : bool
        Should we save the inferred parameters? Optional, defaults to :code:`True`.
    """
    if data is None:
        raise ValueError("data must be passed.")

    from osl_dynamics.models import hmm

    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"
    os.makedirs(inf_params_dir, exist_ok=True)

    # Create the model object
    _logger.info("Building model")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "sequence_length": 2000,
        "batch_size": 32,
        "learning_rate": 0.01,
        "n_epochs": 20,
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
    _logger.info(f"Using config_kwargs: {config_kwargs}")
    config = hmm.Config(**config_kwargs)
    model = hmm.Model(config)
    model.summary()

    # Initialisation
    default_init_kwargs = {"n_init": 3, "n_epochs": 1}
    init_kwargs = override_dict_defaults(default_init_kwargs, init_kwargs)
    _logger.info(f"Using init_kwargs: {init_kwargs}")
    init_history = model.random_state_time_course_initialization(data, **init_kwargs)

    # Training
    history = model.fit(data, **fit_kwargs)

    # Get the variational free energy
    history["free_energy"] = model.free_energy(data)

    # Save trained model
    _logger.info(f"Saving model to: {model_dir}")
    model.save(model_dir)
    save(f"{model_dir}/init_history.pkl", init_history)
    save(f"{model_dir}/history.pkl", history)

    if save_inf_params:
        # Get the inferred parameters
        alpha = model.get_alpha(data)
        means, covs = model.get_means_covariances()

        # Save inferred parameters
        save(f"{inf_params_dir}/alp.pkl", alpha)
        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/covs.npy", covs)


def train_dynemo(
    data,
    output_dir,
    config_kwargs,
    init_kwargs={},
    fit_kwargs={},
    save_inf_params=True,
):
    """Train DyNeMo.

    This function will:

    1. Build a :code:`dynemo.Model` object.
    2. Initialize the parameters of the model using
       :code:`Model.random_subset_initialization`.
    3. Perform full training.
    4. Save the inferred parameters (mode mixing coefficients, means and covariances)
       if :code:`save_inf_params=True`.

    This function will create two directories:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object for training the model.
    output_dir : str
        Path to output directory.
    config_kwargs : dict
        Keyword arguments to pass to dynemo.Config.
        Defaults to::

            {'n_channels': data.n_channels.
             'sequence_length': 200,
             'inference_n_units': 64,
             'inference_normalization': 'layer',
             'model_n_units': 64,
             'model_normalization': 'layer',
             'learn_alpha_temperature': True,
             'initial_alpha_temperature': 1.0,
             'do_kl_annealing': True,
             'kl_annealing_curve': 'tanh',
             'kl_annealing_sharpness': 10,
             'n_kl_annealing_epochs': 20,
             'batch_size': 128,
             'learning_rate': 0.01,
             'n_epochs': 40}
    init_kwargs : dict
        Keyword arguments to pass to :code:`Model.random_subset_initialization`.
        Optional, defaults to::

            {'n_init': 5, 'n_epochs': 1, 'take': 0.25}.
    fit_kwargs : dict
        Keyword arguments to pass to the :code:`Model.fit`. Optional, no defaults.
    save_inf_params : bool
        Should we save the inferred parameters? Optional, defaults to :code:`True`.
    """
    if data is None:
        raise ValueError("data must be passed.")

    from osl_dynamics.models import dynemo

    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"

    # Create the model object
    _logger.info("Building model")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "sequence_length": 200,
        "inference_n_units": 64,
        "inference_normalization": "layer",
        "model_n_units": 64,
        "model_normalization": "layer",
        "learn_alpha_temperature": True,
        "initial_alpha_temperature": 1.0,
        "do_kl_annealing": True,
        "kl_annealing_curve": "tanh",
        "kl_annealing_sharpness": 10,
        "n_kl_annealing_epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.01,
        "n_epochs": 40,
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
    _logger.info(f"Using config_kwargs: {config_kwargs}")
    config = dynemo.Config(**config_kwargs)
    model = dynemo.Model(config)
    model.summary()

    # Initialisation
    default_init_kwargs = {"n_init": 5, "n_epochs": 1, "take": 0.25}
    init_kwargs = override_dict_defaults(default_init_kwargs, init_kwargs)
    _logger.info(f"Using init_kwargs: {init_kwargs}")
    init_history = model.random_subset_initialization(data, **init_kwargs)

    # Training
    history = model.fit(data, **fit_kwargs)

    # Add free energy to the history object
    history["free_energy"] = history["loss"][-1]

    # Save trained model
    _logger.info(f"Saving model to: {model_dir}")
    model.save(model_dir)
    save(f"{model_dir}/init_history.pkl", init_history)
    save(f"{model_dir}/history.pkl", history)

    if save_inf_params:
        os.makedirs(inf_params_dir, exist_ok=True)

        # Get the inferred parameters
        alpha = model.get_alpha(data)
        means, covs = model.get_means_covariances()

        # Save inferred parameters
        save(f"{inf_params_dir}/alp.pkl", alpha)
        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/covs.npy", covs)


def calc_subject_ae_hmm_networks(data, output_dir):
    """Calculate subject-specific AE-HMM networks.

    This function expects a model has already been trained and the following
    directory to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create the following directory:

    - :code:`<output_dir>/networks`, which contains the subject-specific networks.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    """
    if data is None:
        raise ValueError("data must be passed.")

    # Directories
    inf_params_dir = output_dir + "/inf_params"
    networks_dir = output_dir + "/networks"
    os.makedirs(networks_dir, exist_ok=True)

    # Load the inferred state probabilities
    alpha = load(f"{inf_params_dir}/alp.pkl")

    # Get the prepared data
    # This should be the data after calculating the amplitude envelope
    data = data.time_series()

    # Calculate subject-specific means and AECs
    from osl_dynamics.analysis import modes

    means, aecs = modes.ae_hmm_networks(data, alpha)

    # Save
    save(f"{networks_dir}/subj_means.npy", means)
    save(f"{networks_dir}/subj_aecs.npy", aecs)


def multitaper_spectra(data, output_dir, kwargs, nnmf_components=None):
    """Calculate multitaper spectra.

    This function expects a model has already been trained and the following
    directories exist:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create the following directory:

    - :code:`<output_dir>/spectra`, which contains the post-hoc spectra.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    kwargs : dict
        Keyword arguments to pass to osl_dynamics.analysis.spectral.multitaper_spectra.
        Defaults to::

            {'sampling_frequency': data.sampling_frequency,
             'time_half_bandwidth': 4,
             'n_tapers': 7,
             'keepdims': True}
    nnmf_components : int
        Number of non-negative matrix factorization (NNMF) components to fit to
        the stacked subject-specific coherence spectra.
    """
    if data is None:
        raise ValueError("data must be passed.")

    default_kwargs = {
        "sampling_frequency": data.sampling_frequency,
        "time_half_bandwidth": 4,
        "n_tapers": 7,
        "keepdims": True,
    }
    kwargs = override_dict_defaults(default_kwargs, kwargs)
    _logger.info(f"Using kwargs: {kwargs}")

    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"
    spectra_dir = output_dir + "/spectra"
    os.makedirs(spectra_dir, exist_ok=True)

    # Load the inferred state probabilities
    alpha = load(f"{inf_params_dir}/alp.pkl")

    # Get the config used to create the model
    from osl_dynamics.models.mod_base import ModelBase

    model_config, _ = ModelBase.load_config(model_dir)

    # Get unprepared data (i.e. the data before calling Data.prepare)
    # We also trim the data to account for the data points lost to
    # time embedding or applying a sliding window
    data = data.trim_time_series(
        sequence_length=model_config["sequence_length"], prepared=False
    )

    # Calculate multitaper
    from osl_dynamics.analysis import spectral

    spectra = spectral.multitaper_spectra(data, alpha, **kwargs)

    # Unpack spectra and save
    return_weights = kwargs.pop("return_weights", False)
    if return_weights:
        f, psd, coh, w = spectra
        save(f"{spectra_dir}/f.npy", f)
        save(f"{spectra_dir}/psd.npy", psd)
        save(f"{spectra_dir}/coh.npy", coh)
        save(f"{spectra_dir}/w.npy")
    else:
        f, psd, coh = spectra
        save(f"{spectra_dir}/f.npy", f)
        save(f"{spectra_dir}/psd.npy", psd)
        save(f"{spectra_dir}/coh.npy", coh)

    if nnmf_components is not None:
        # Calculate NNMF and save
        nnmf = spectral.decompose_spectra(coh, n_components=nnmf_components)
        save(f"{spectra_dir}/nnmf_{nnmf_components}.npy", nnmf)


def nnmf(data, output_dir, n_components):
    """Calculate non-negative matrix factorization (NNMF).

    This function expects spectra have already been calculated and are in:

    - :code:`<output_dir>/spectra`, which contains multitaper spectra.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    n_components : int
        Number of components to fit.
    """
    coh = load(f"{spectra_dir}/coh.npy")
    nnmf = spectral.decompose_spectra(coh, n_components=n_components)
    save(f"{spectra_dir}/nnmf_{n_components}.npy", nnmf)


def regression_spectra(data, output_dir, kwargs):
    """Calculate regression spectra.

    This function expects a model has already been trained and the following
    directories exist:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create the following directory:

    - :code:`<output_dir>/spectra`, which contains the post-hoc spectra.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    kwargs : dict
        Keyword arguments to pass to osl_dynamics.analysis.spectral.multitaper_spectra.
        Defaults to::

            {'sampling_frequency': data.sampling_frequency,
             'window_length': 4 * sampling_frequency,
             'step_size': 20,
             'n_sub_windows': 8,
             'return_coef_int': True,
             'keepdims': True}
    """
    if data is None:
        raise ValueError("data must be passed.")

    sampling_frequency = (
        kwargs.pop("sampling_frequency", None) or data.sampling_frequency
    )
    default_kwargs = {
        "sampling_frequency": sampling_frequency,
        "window_length": int(4 * sampling_frequency),
        "step_size": 20,
        "n_sub_windows": 8,
        "return_coef_int": True,
        "keepdims": True,
    }
    kwargs = override_dict_defaults(default_kwargs, kwargs)
    _logger.info(f"Using kwargs: {kwargs}")

    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"
    spectra_dir = output_dir + "/spectra"
    os.makedirs(spectra_dir, exist_ok=True)

    # Load the inferred mixing coefficients
    alpha = load(f"{inf_params_dir}/alp.pkl")

    # Get the config used to create the model
    from osl_dynamics.models.mod_base import ModelBase

    model_config, _ = ModelBase.load_config(model_dir)

    # Get unprepared data (i.e. the data before calling Data.prepare)
    # We also trim the data to account for the data points lost to
    # time embedding or applying a sliding window
    data = data.trim_time_series(
        sequence_length=model_config["sequence_length"], prepared=False
    )

    # Calculate regression spectra
    from osl_dynamics.analysis import spectral

    spectra = spectral.regression_spectra(data, alpha, **kwargs)

    # Unpack spectra and save
    return_weights = kwargs.pop("return_weights", False)
    if return_weights:
        f, psd, coh, w = spectra
        save(f"{spectra_dir}/f.npy", f)
        save(f"{spectra_dir}/psd.npy", psd)
        save(f"{spectra_dir}/coh.npy", coh)
        save(f"{spectra_dir}/w.npy")
    else:
        f, psd, coh = spectra
        save(f"{spectra_dir}/f.npy", f)
        save(f"{spectra_dir}/psd.npy", psd)
        save(f"{spectra_dir}/coh.npy", coh)


def plot_group_ae_networks(
    data,
    output_dir,
    mask_file=None,
    parcellation_file=None,
    aec_abs=True,
    power_save_kwargs={},
    conn_save_kwargs={},
):
    """Plot group-level amplitude envelope networks.

    This function expects a model has been trained and the following directory to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/networks`, which contains plots of the networks.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    mask_file : str
        Mask file used to preprocess the training data. Optional. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str
        Parcellation file used to parcellate the training data. Optional. If
        :code:`None`, we use :code:`data.parcellation_file`.
    aec_abs : bool
        Should we take the absolute value of the amplitude envelope correlations?
        Optional, defaults to :code:`True`.
    power_save_kwargs : dict
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'filename': '<output_dir>/networks/mean_.png',
             'mask_file': data.mask_file,
             'parcellation_file': data.parcellation_file}
    conn_save_kwargs : dict
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_.
        Defaults to::

            {'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/aec_.png',
             'threshold': 0.97}
    """
    # Validation
    if mask_file is None:
        if data is None or data.mask_file is None:
            raise ValueError(
                "mask_file must be passed or specified in the Data object."
            )
        else:
            mask_file = data.mask_file

    if parcellation_file is None:
        if data is None or data.parcellation_file is None:
            raise ValueError(
                "parcellation_file must be passed or specified in the Data object."
            )
        else:
            parcellation_file = data.parcellation_file

    # Directories
    inf_params_dir = output_dir + "/inf_params"
    networks_dir = output_dir + "/networks"
    os.makedirs(networks_dir, exist_ok=True)

    # Load inferred means and covariances
    means = load(f"{inf_params_dir}/means.npy")
    covs = load(f"{inf_params_dir}/covs.npy")
    aecs = array_ops.cov2corr(covs)
    if aec_abs:
        aecs = abs(aecs)

    # Save mean activity maps
    from osl_dynamics.analysis import power

    default_power_save_kwargs = {
        "filename": f"{networks_dir}/mean_.png",
        "mask_file": mask_file,
        "parcellation_file": parcellation_file,
    }
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, power_save_kwargs
    )
    _logger.info(f"Using power_save_kwargs: {power_save_kwargs}")
    power.save(means, **power_save_kwargs)

    # Save AEC networks
    from osl_dynamics.analysis import connectivity

    default_conn_save_kwargs = {
        "parcellation_file": parcellation_file,
        "filename": f"{networks_dir}/aec_.png",
        "threshold": 0.97,
    }
    conn_save_kwargs = override_dict_defaults(
        default_conn_save_kwargs, conn_save_kwargs
    )
    _logger.info(f"Using conn_save_kwargs: {conn_save_kwargs}")
    connectivity.save(aecs, **conn_save_kwargs)


def plot_group_tde_hmm_networks(
    data,
    output_dir,
    mask_file=None,
    parcellation_file=None,
    frequency_range=None,
    percentile=97,
    power_save_kwargs={},
    conn_save_kwargs={},
):
    """Plot group-level TDE-HMM networks for a specified frequency band.

    This function will:

    1. Plot state PSDs.
    2. Plot the power maps.
    3. Plot coherence networks.

    This function expects spectra have already been calculated and are in:

    - :code:`<output_dir>/spectra`, which contains multitaper spectra.

    This function will create:

    - :code:`<output_dir>/networks`, which contains plots of the networks.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    mask_file : str
        Mask file used to preprocess the training data. Optional. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str
        Parcellation file used to parcellate the training data. Optional. If
        :code:`None`, we use :code:`data.parcellation_file`.
    frequency_range : list
        List of length 2 containing the minimum and maximum frequency to integrate
        spectra over. Optional, defaults to the full frequency range.
    percentile : float
        Percentile for thresholding the coherence networks. Default is 97, which
        corresponds to the top 3% of edges (relative to the mean across states).
    plot_save_kwargs : dict
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'mask_file': mask_file,
             'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/pow_.png',
             'subtract_mean': True}
    conn_save_kwargs : dict
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_.
        Defaults to::

            {'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/coh_.png',
             'plot_kwargs': {'edge_cmap': 'Reds'}}
    """
    # Validation
    if mask_file is None:
        if data is None or data.mask_file is None:
            raise ValueError(
                "mask_file must be passed or specified in the Data object."
            )
        else:
            mask_file = data.mask_file

    if parcellation_file is None:
        if data is None or data.parcellation_file is None:
            raise ValueError(
                "parcellation_file must be passed or specified in the Data object."
            )
        else:
            parcellation_file = data.parcellation_file

    # Directories
    spectra_dir = output_dir + "/spectra"
    networks_dir = output_dir + "/networks"
    os.makedirs(networks_dir, exist_ok=True)

    # Load spectra
    f = load(f"{spectra_dir}/f.npy")
    psd = load(f"{spectra_dir}/psd.npy")
    coh = load(f"{spectra_dir}/coh.npy")
    if Path(f"{spectra_dir}/w.npy").exists():
        w = load(f"{spectra_dir}/w.npy")
    else:
        w = None

    # Calculate group average
    gpsd = np.average(psd, axis=0, weights=w)
    gcoh = np.average(coh, axis=0, weights=w)

    # Calculate average PSD across channels and the standard error
    p = np.mean(gpsd, axis=-2)
    e = np.std(gpsd, axis=-2) / np.sqrt(gpsd.shape[-2])

    # Plot PSDs
    from osl_dynamics.utils import plotting

    n_states = gpsd.shape[0]
    for i in range(n_states):
        fig, ax = plotting.plot_line(
            [f],
            [p[i]],
            errors=[[p[i] - e[i]], [p[i] + e[i]]],
            labels=[f"State {i + 1}"],
            x_range=[f[0], f[-1]],
            y_range=[p.min() - 0.1 * p.max(), 1.2 * p.max()],
            x_label="Frequency (Hz)",
            y_label="PSD (a.u.)",
        )
        if frequency_range is not None:
            ax.axvspan(frequency_range[0], frequency_range[1], alpha=0.25, color="gray")
        plotting.save(fig, filename=f"{networks_dir}/psd_{i}.png")

    # Calculate power maps from the group-level PSDs
    from osl_dynamics.analysis import power

    gp = power.variance_from_spectra(f, gpsd, frequency_range=frequency_range)

    # Save power maps
    default_power_save_kwargs = {
        "mask_file": mask_file,
        "parcellation_file": parcellation_file,
        "filename": f"{networks_dir}/pow_.png",
        "subtract_mean": True,
    }
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, power_save_kwargs
    )
    _logger.info(f"Using power_save_kwargs: {power_save_kwargs}")
    power.save(gp, **power_save_kwargs)

    # Calculate coherence networks from group-level spectra
    from osl_dynamics.analysis import connectivity

    gc = connectivity.mean_coherence_from_spectra(
        f, gcoh, frequency_range=frequency_range
    )

    # Threshold
    gc = connectivity.threshold(gc, percentile=percentile, subtract_mean=True)

    # Save coherence networks
    default_conn_save_kwargs = {
        "parcellation_file": parcellation_file,
        "filename": f"{networks_dir}/coh_.png",
        "plot_kwargs": {"edge_cmap": "Reds"},
    }
    conn_save_kwargs = override_dict_defaults(
        default_conn_save_kwargs, conn_save_kwargs
    )
    _logger.info(f"Using conn_save_kwargs: {conn_save_kwargs}")
    connectivity.save(gc, **conn_save_kwargs)


def plot_group_nnmf_tde_hmm_networks(
    data,
    output_dir,
    nnmf_file,
    mask_file=None,
    parcellation_file=None,
    component=0,
    percentile=97,
    power_save_kwargs={},
    conn_save_kwargs={},
):
    """Plot group-level TDE-HMM networks using a NNMF component to integrate
    the spectra.

    This function will:

    1. Plot state PSDs.
    2. Plot the power maps.
    3. Plot coherence networks.

    This function expects spectra have already been calculated and are in:

    - :code:`<output_dir>/spectra`, which contains multitaper spectra.

    This function will create:

    - :code:`<output_dir>/networks`, which contains plots of the networks.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    nnmf_file : str
        Path relative to :code:`output_dir` for a npy file (with the output of
        `analysis.spectral.decompose_spectra
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis.spectral.decompose_spectra>`_)
        containing the NNMF components.
    mask_file : str
        Mask file used to preprocess the training data. Optional. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str
        Parcellation file used to parcellate the training data. Optional. If
        :code:`None`, we use :code:`data.parcellation_file`.
    component : int
        NNMF component to plot. Defaults to the first component.
    percentile : float
        Percentile for thresholding the coherence networks. Default is 97, which
        corresponds to the top 3% of edges (relative to the mean across states).
    plot_save_kwargs : dict
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'mask_file': mask_file,
             'parcellation_file': parcellation_file,
             'component': component,
             'filename': '<output_dir>/networks/pow_.png',
             'subtract_mean': True}
    conn_save_kwargs : dict
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_.
        Defaults to::

            {'parcellation_file': parcellation_file,
             'component': component,
             'filename': '<output_dir>/networks/coh_.png',
             'plot_kwargs': {'edge_cmap': 'Reds'}}
    """
    # Validation
    if mask_file is None:
        if data is None or data.mask_file is None:
            raise ValueError(
                "mask_file must be passed or specified in the Data object."
            )
        else:
            mask_file = data.mask_file

    if parcellation_file is None:
        if data is None or data.parcellation_file is None:
            raise ValueError(
                "parcellation_file must be passed or specified in the Data object."
            )
        else:
            parcellation_file = data.parcellation_file

    # Directories
    spectra_dir = output_dir + "/spectra"
    networks_dir = output_dir + "/networks"
    os.makedirs(networks_dir, exist_ok=True)

    # Load the NNMF components
    nnmf_file = output_dir + "/" + nnmf_file
    if Path(nnmf_file).exists():
        nnmf = load(nnmf_file)
    else:
        raise ValueError(f"{nnmf_file} not found.")

    # Load spectra
    f = load(f"{spectra_dir}/f.npy")
    psd = load(f"{spectra_dir}/psd.npy")
    coh = load(f"{spectra_dir}/coh.npy")
    if Path(f"{spectra_dir}/w.npy").exists():
        w = load(f"{spectra_dir}/w.npy")
    else:
        w = None

    # Plot the NNMF components
    from osl_dynamics.utils import plotting

    n_components = nnmf.shape[0]
    plotting.plot_line(
        [f] * n_components,
        nnmf,
        labels=[f"Component {i}" for i in range(n_components)],
        x_label="Frequency (Hz)",
        y_label="Weighting",
        filename=f"{networks_dir}/nnmf.png",
    )

    # Calculate group average
    gpsd = np.average(psd, axis=0, weights=w)
    gcoh = np.average(coh, axis=0, weights=w)

    # Calculate average PSD across channels and the standard error
    p = np.mean(gpsd, axis=-2)
    e = np.std(gpsd, axis=-2) / np.sqrt(gpsd.shape[-2])

    # Plot PSDs
    n_states = gpsd.shape[0]
    for i in range(n_states):
        fig, ax = plotting.plot_line(
            [f],
            [p[i]],
            errors=[[p[i] - e[i]], [p[i] + e[i]]],
            labels=[f"State {i + 1}"],
            x_range=[f[0], f[-1]],
            y_range=[p.min() - 0.1 * p.max(), 1.2 * p.max()],
            x_label="Frequency (Hz)",
            y_label="PSD (a.u.)",
        )
        plotting.save(fig, filename=f"{networks_dir}/psd_{i}.png")

    # Calculate power maps from the group-level PSDs
    from osl_dynamics.analysis import power

    gp = power.variance_from_spectra(f, gpsd, nnmf)

    # Save power maps
    default_power_save_kwargs = {
        "mask_file": mask_file,
        "parcellation_file": parcellation_file,
        "component": component,
        "filename": f"{networks_dir}/pow_.png",
        "subtract_mean": True,
    }
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, power_save_kwargs
    )
    _logger.info(f"Using power_save_kwargs: {power_save_kwargs}")
    power.save(gp, **power_save_kwargs)

    # Calculate coherence networks from group-level spectra
    from osl_dynamics.analysis import connectivity

    gc = connectivity.mean_coherence_from_spectra(f, gcoh, nnmf)

    # Threshold
    gc = connectivity.threshold(gc, percentile=percentile, subtract_mean=True)

    # Save coherence networks
    default_conn_save_kwargs = {
        "parcellation_file": parcellation_file,
        "component": component,
        "filename": f"{networks_dir}/coh_.png",
        "plot_kwargs": {"edge_cmap": "Reds"},
    }
    conn_save_kwargs = override_dict_defaults(
        default_conn_save_kwargs, conn_save_kwargs
    )
    _logger.info(f"Using conn_save_kwargs: {conn_save_kwargs}")
    connectivity.save(gc, **conn_save_kwargs)


def plot_group_tde_dynemo_networks(
    data,
    output_dir,
    mask_file=None,
    parcellation_file=None,
    frequency_range=None,
    percentile=97,
    power_save_kwargs={},
    conn_save_kwargs={},
):
    """Plot group-level TDE-DyNeMo networks for a specified frequency band.

    This function will:

    1. Plot mode PSDs.
    2. Plot the power maps.
    3. Plot coherence networks.

    This function expects spectra have already been calculated and are in:

    - :code:`<output_dir>/spectra`, which contains regression spectra.

    This function will create:

    - :code:`<output_dir>/networks`, which contains plots of the networks.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    mask_file : str
        Mask file used to preprocess the training data. Optional. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str
        Parcellation file used to parcellate the training data. Optional. If
        :code:`None`, we use :code:`data.parcellation_file`.
    frequency_range : list
        List of length 2 containing the minimum and maximum frequency to integrate
        spectra over. Optional, defaults to the full frequency range.
    percentile : float
        Percentile for thresholding the coherence networks. Default is 97, which
        corresponds to the top 3% of edges (relative to the mean across states).
    plot_save_kwargs : dict
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'mask_file': mask_file,
             'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/pow_.png',
             'subtract_mean': True}
    conn_save_kwargs : dict
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/analysis/connectivity/index.html#osl_dynamics.analysis.connectivity.save>`_.
        Defaults to::

            {'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/coh_.png',
             'plot_kwargs': {'edge_cmap': 'Reds'}}
    """
    # Validation
    if mask_file is None:
        if data is None or data.mask_file is None:
            raise ValueError(
                "mask_file must be passed or specified in the Data object."
            )
        else:
            mask_file = data.mask_file

    if parcellation_file is None:
        if data is None or data.parcellation_file is None:
            raise ValueError(
                "parcellation_file must be passed or specified in the Data object."
            )
        else:
            parcellation_file = data.parcellation_file

    # Directories
    spectra_dir = output_dir + "/spectra"
    networks_dir = output_dir + "/networks"
    os.makedirs(networks_dir, exist_ok=True)

    # Load spectra
    f = load(f"{spectra_dir}/f.npy")
    psd = load(f"{spectra_dir}/psd.npy")
    coh = load(f"{spectra_dir}/coh.npy")
    if Path(f"{spectra_dir}/w.npy").exists():
        w = load(f"{spectra_dir}/w.npy")
    else:
        w = None

    # Only keep the regression coefficients
    psd = psd[:, 0]

    # Calculate group average
    gpsd = np.average(psd, axis=0, weights=w)
    gcoh = np.average(coh, axis=0, weights=w)

    # Calculate average PSD across channels and the standard error
    p = np.mean(gpsd, axis=-2)
    e = np.std(gpsd, axis=-2) / np.sqrt(gpsd.shape[-2])

    # Plot PSDs
    from osl_dynamics.utils import plotting

    n_modes = gpsd.shape[0]
    for i in range(n_modes):
        fig, ax = plotting.plot_line(
            [f],
            [p[i]],
            errors=[[p[i] - e[i]], [p[i] + e[i]]],
            labels=[f"Mode {i + 1}"],
            x_range=[f[0], f[-1]],
            y_range=[p.min() - 0.1 * p.max(), 1.4 * p.max()],
            x_label="Frequency (Hz)",
            y_label="PSD (a.u.)",
        )
        if frequency_range is not None:
            ax.axvspan(frequency_range[0], frequency_range[1], alpha=0.25, color="gray")
        plotting.save(fig, filename=f"{networks_dir}/psd_{i}.png")

    # Calculate power maps from the group-level PSDs
    from osl_dynamics.analysis import power

    gp = power.variance_from_spectra(f, gpsd, frequency_range=frequency_range)

    # Save power maps
    default_power_save_kwargs = {
        "mask_file": mask_file,
        "parcellation_file": parcellation_file,
        "filename": f"{networks_dir}/pow_.png",
        "subtract_mean": True,
    }
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, power_save_kwargs
    )
    _logger.info(f"Using power_save_kwargs: {power_save_kwargs}")
    power.save(gp, **power_save_kwargs)

    # Calculate coherence networks from group-level spectra
    from osl_dynamics.analysis import connectivity

    gc = connectivity.mean_coherence_from_spectra(
        f, gcoh, frequency_range=frequency_range
    )

    # Threshold
    gc = connectivity.threshold(gc, percentile=percentile, subtract_mean=True)

    # Save coherence networks
    default_conn_save_kwargs = {
        "parcellation_file": parcellation_file,
        "filename": f"{networks_dir}/coh_.png",
        "plot_kwargs": {"edge_cmap": "Reds"},
    }
    conn_save_kwargs = override_dict_defaults(
        default_conn_save_kwargs, conn_save_kwargs
    )
    _logger.info(f"Using conn_save_kwargs: {conn_save_kwargs}")
    connectivity.save(gc, **conn_save_kwargs)


def plot_alpha(
    data, output_dir, subject=0, normalize=False, sampling_frequency=None, kwargs={}
):
    """Plot inferred alphas.

    This is a wrapper for `utils.plotting.plot_alpha
    <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/utils/plotting/index.html#osl_dynamics.utils.plotting.plot_alpha>`_.

    This function expects a model has been trained and the following directory to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/alphas`, which contains plots of the inferred alphas.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    subject : int
        Index for subject to plot. If 'all' is passed we create a separate plot
        for each subject.
    normalize : bool
        Should we also plot the alphas normalized using the trace of the inferred
        covariance matrices? Optional. Useful if we are plotting the inferred alphas
        from DyNeMo.
    sampling_frequency : float
        Sampling frequency in Hz. Optional. If :code:`None`, we see if it is
        present in :code:`data.sampling_frequency`.
    kwargs : dict
        Keyword arguments to pass to `utils.plotting.plot_alpha
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/utils/plotting/index.html#osl_dynamics.utils.plotting.plot_alpha>`_.
        Optional. Defaults to::

            {'sampling_frequency': data.sampling_frequency,
             'filename': '<output_dir>/alphas/alpha_<subject>.png'}
    """
    if sampling_frequency is None and data is not None:
        sampling_frequency = data.sampling_frequency

    # Directories
    inf_params_dir = output_dir + "/inf_params"
    alphas_dir = output_dir + "/alphas"
    os.makedirs(alphas_dir, exist_ok=True)

    # Load inferred alphas
    alp = load(f"{inf_params_dir}/alp.pkl")
    if isinstance(alp, np.ndarray):
        alp = [alp]

    # Plot
    from osl_dynamics.utils import plotting

    default_kwargs = {
        "sampling_frequency": sampling_frequency,
        "filename": f"{alphas_dir}/alpha_*.png",
    }
    kwargs = override_dict_defaults(default_kwargs, kwargs)
    _logger.info(f"Using kwargs: {kwargs}")

    if subject == "all":
        for i in range(len(alp)):
            kwargs["filename"] = f"{alphas_dir}/alpha_{i}.png"
            plotting.plot_alpha(alp[i], **kwargs)
    else:
        kwargs["filename"] = f"{alphas_dir}/alpha_{subject}.png"
        plotting.plot_alpha(alp[subject], **kwargs)

    if normalize:
        # Calculate normalised alphas
        covs = load(f"{inf_params_dir}/covs.npy")
        traces = np.trace(covs, axis1=1, axis2=2)
        norm_alp = [a * traces[np.newaxis, :] for a in alp]
        norm_alp = [na / np.sum(na, axis=1, keepdims=True) for na in norm_alp]

        # Plot
        if subject == "all":
            for i in range(len(alp)):
                kwargs["filename"] = f"{alphas_dir}/norm_alpha_{i}.png"
                plotting.plot_alpha(norm_alp[i], **kwargs)
        else:
            kwargs["filename"] = f"{alphas_dir}/norm_alpha_{subject}.png"
            plotting.plot_alpha(norm_alp[subject], **kwargs)


def calc_gmm_alpha(data, output_dir, kwargs={}):
    """Binarize inferred alphas using a two-component GMM.

    This function expects a model has been trained and the following directory to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create the following file:

    - :code:`<output_dir>/inf_params/gmm_alp.pkl`, which contains the binarized alphas.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    kwargs : dict
        Keyword arguments to pass to `inference.modes.gmm_time_courses
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/inference/modes/index.html#osl_dynamics.inference.modes.gmm_time_courses>`_.
    """
    inf_params_dir = output_dir + "/inf_params"

    # Load inferred alphas
    alp_file = f"{inf_params_dir}/alp.pkl"
    if not Path(alp_file).exists():
        raise ValueError(f"{alp_file} missing.")
    alp = load(alp_file)

    # Binarise using a two-component GMM
    from osl_dynamics.inference import modes

    _logger.info(f"Using kwargs: {kwargs}")
    gmm_alp = modes.gmm_time_courses(alp, **kwargs)
    save(f"{inf_params_dir}/gmm_alp.pkl", gmm_alp)


def plot_summary_stats(data, output_dir, use_gmm_alpha=False, sampling_frequency=None):
    """Plot summary statistics as violin plots.

    This function will plot the distribution over subjects for the following summary
    statistics:

    - Fractional occupancy.
    - Mean lifetime (s).
    - Mean interval (s).
    - Switching rate (Hz).

    This function expects a model has been trained and the following directory to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/summary_stats`, which contains plots of the summary statistics.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    use_gmm_alpha : bool
        Should we use alphas binarised using a Gaussian mixture model?
        This function assumes :code:`calc_gmm_alpha` has been called and the file
        :code:`<output_dir>/inf_params/gmm_alp.pkl` exists.
    sampling_frequency : float
        Sampling frequency in Hz. Optional. If :code:`None`, we use
        :code:`data.sampling_frequency`.
    """
    if sampling_frequency is None:
        if data is None or data.sampling_frequency is None:
            raise ValueError(
                "sampling_frequency must be passed or specified in the Data object."
            )
        else:
            sampling_frequency = data.sampling_frequency

    # Directories
    inf_params_dir = output_dir + "/inf_params"
    summary_stats_dir = output_dir + "/summary_stats"
    os.makedirs(summary_stats_dir, exist_ok=True)

    from osl_dynamics.inference import modes

    if use_gmm_alpha:
        # Use alphas that were binarised using a GMM
        gmm_alp_file = f"{inf_params_dir}/gmm_alp.pkl"
        if Path(gmm_alp_file).exists():
            stc = load(gmm_alp_file)
        else:
            raise ValueError(f"{gmm_alp_file} missing.")

    else:
        # Load inferred alphas and hard classify
        alp = load(f"{inf_params_dir}/alp.pkl")
        if isinstance(alp, np.ndarray):
            raise ValueError(
                "We must train on multiple subjects to plot the distribution of summary statistics."
            )
        stc = modes.argmax_time_courses(alp)

    # Calculate summary stats
    fo = modes.fractional_occupancies(stc)
    lt = modes.mean_lifetimes(stc, sampling_frequency)
    intv = modes.mean_intervals(stc, sampling_frequency)
    sr = modes.switching_rates(stc, sampling_frequency)

    # Plot
    from osl_dynamics.utils import plotting

    n_states = fo.shape[1]
    x = range(1, n_states + 1)
    plotting.plot_violin(
        fo.T,
        x=x,
        x_label="State",
        y_label="Fractional Occupancy",
        filename=f"{summary_stats_dir}/fo.png",
    )
    plotting.plot_violin(
        lt.T,
        x=x,
        x_label="State",
        y_label="Mean Lifetime (s)",
        filename=f"{summary_stats_dir}/lt.png",
    )
    plotting.plot_violin(
        intv.T,
        x=x,
        x_label="State",
        y_label="Mean Interval (s)",
        filename=f"{summary_stats_dir}/intv.png",
    )
    plotting.plot_violin(
        sr.T,
        x=x,
        x_label="State",
        y_label="Switching rate (Hz)",
        filename=f"{summary_stats_dir}/sr.png",
    )


def compare_groups_hmm_summary_stats(
    data,
    output_dir,
    group2_indices,
    separate_tests=False,
    covariates=None,
    n_perm=1000,
    n_jobs=1,
    sampling_frequency=None,
):
    """Compare HMM summary statistics between two groups.

    This function expects a model has been trained and the following directory to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/group_diff`, which contains the summary statistics and plots.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    group2_indices : np.ndarray or list
        Indices indicating which subjects belong to the second group.
    separate_tests : bool
        Should we perform a maximum statistic permutation test for each summary
        statistic separately?
    covariates : str
        Path to a pickle file containing a dict with covariances. Each item in the
        dict must be the covariate name and value for each subject. The covariates
        will be loaded with::

            from osl_dynamics.utils.misc import load
            covariates = load("/path/to/file.pkl")

        Example covariates::

            covariates = {"age": [...], "sex": [...]}
    n_perm : int
        Number of permutations.
    n_jobs : int
        Number of jobs for parallel processing.
    sampling_frequency : float
        Sampling frequency in Hz. Optional. If :code:`None`, we use
        :code:`data.sampling_frequency`.
    """
    if sampling_frequency is None:
        if data is None or data.sampling_frequency is None:
            raise ValueError(
                "sampling_frequency must be passed or specified in the Data object."
            )
        else:
            sampling_frequency = data.sampling_frequency

    # Directories
    inf_params_dir = output_dir + "/inf_params"
    group_diff_dir = output_dir + "/group_diff"
    os.makedirs(group_diff_dir, exist_ok=True)

    # Get inferred state time courses
    from osl_dynamics.inference import modes

    alp = load(f"{inf_params_dir}/alp.pkl")
    stc = modes.argmax_time_courses(alp)

    # Calculate summary stats
    names = ["fo", "lt", "intv", "sr"]
    fo = modes.fractional_occupancies(stc)
    lt = modes.mean_lifetimes(stc, sampling_frequency)
    intv = modes.mean_intervals(stc, sampling_frequency)
    sr = modes.switching_rates(stc, sampling_frequency)
    sum_stats = np.swapaxes([fo, lt, intv, sr], 0, 1)

    # Save
    for i in range(4):
        save(f"{group_diff_dir}/{names[i]}.npy", sum_stats[:, i])

    # Create a vector for group assignments
    n_subjects = fo.shape[0]
    assignments = np.ones(n_subjects)
    assignments[group2_indices] += 1

    # Load covariates
    if covariates is not None:
        covariates = load(covariates)
    else:
        covariates = {}

    # Perform statistical significance testing
    from osl_dynamics.analysis import statistics

    if separate_tests:
        pvalues = []
        for i in range(4):
            # Calculate a statistical significance test for each summary stat separately
            _, p = statistics.group_diff_max_stat_perm(
                sum_stats[:, i],
                assignments,
                n_perm=n_perm,
                covariates=covariates,
                n_jobs=n_jobs,
            )
            pvalues.append(p)
            _logger.info(f"{names[i]}: {np.sum(p <  0.05)} states have p-value<0.05")
            save(f"{group_diff_dir}/{names[i]}_pvalues.npy", p)
        pvalues = np.array(pvalues)
    else:
        # Calculate a statistical significance test for all summary stats concatenated
        _, pvalues = statistics.group_diff_max_stat_perm(
            sum_stats, assignments, n_perm=n_perm, covariates=covariates, n_jobs=n_jobs
        )
        for i in range(4):
            _logger.info(
                f"{names[i]}: {np.sum(pvalues[i] < 0.05)} states have p-value<0.05"
            )
            save(f"{group_diff_dir}/{names[i]}_pvalues.npy", pvalues[i])

    # Plot
    from osl_dynamics.utils import plotting

    labels = [
        "Fractional Occupancy",
        "Mean Lifetime (s)",
        "Mean Interval (s)",
        "Switching Rate (Hz)",
    ]
    for i in range(4):
        plotting.plot_summary_stats_group_diff(
            name=labels[i],
            summary_stats=sum_stats[:, i],
            pvalues=pvalues[i],
            assignments=assignments,
            filename=f"{group_diff_dir}/{names[i]}.png",
        )
