"""Wrapper functions for use in the config API.

All of the functions in this module can be listed in the config passed to
:code:`osl_dynamics.run_pipeline`.

All wrapper functions have the structure::

    func(data, output_dir, **kwargs)

where:

- :code:`data` is an :code:`osl_dynamics.data.Data` object.
- :code:`output_dir` is the path to save output to.
- :code:`kwargs` are keyword arguments for function specific options.
"""

import os
import logging
from pathlib import Path

import numpy as np

from osl_dynamics import array_ops
from osl_dynamics.utils.misc import load, override_dict_defaults, save

_logger = logging.getLogger("osl-dynamics")


def load_data(inputs, kwargs=None, prepare=None):
    """Load and prepare data.

    Parameters
    ----------
    inputs : str
        Path to directory containing :code:`npy` files.
    kwargs : dict, optional
        Keyword arguments to pass to the `Data <https://osl-dynamics\
        .readthedocs.io/en/latest/autoapi/osl_dynamics/data/index.html\
        #osl_dynamics.data.Data>`_ class. Useful keyword arguments to pass are
        :code:`sampling_frequency`, :code:`mask_file` and
        :code:`parcellation_file`.
    prepare : dict, optional
        Methods dict to pass to the prepare method. See docstring for
        `Data <https://osl-dynamics.readthedocs.io/en/latest/autoapi\
        /osl_dynamics/data/index.html#osl_dynamics.data.Data>`_.prepare.

    Returns
    -------
    data : osl_dynamics.data.Data
        Data object.
    """
    from osl_dynamics.data import Data

    kwargs = {} if kwargs is None else kwargs
    prepare = {} if prepare is None else prepare

    data = Data(inputs, **kwargs)
    data.prepare(prepare)
    return data


def train_hmm(
    data,
    output_dir,
    config_kwargs,
    init_kwargs=None,
    fit_kwargs=None,
    save_inf_params=True,
):
    """Train a `Hidden Markov Model <https://osl-dynamics.readthedocs.io/en\
    /latest/autoapi/osl_dynamics/models/hmm/index.html>`_.

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
      This directory is only created if :code:`save_inf_params=True`.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object for training the model.
    output_dir : str
        Path to output directory.
    config_kwargs : dict
        Keyword arguments to pass to `hmm.Config <https://osl-dynamics\
        .readthedocs.io/en/latest/autoapi/osl_dynamics/models/hmm/index.html\
        #osl_dynamics.models.hmm.Config>`_. Defaults to::

            {'sequence_length': 2000,
             'batch_size': 32,
             'learning_rate': 0.01,
             'n_epochs': 20}.
    init_kwargs : dict, optional
        Keyword arguments to pass to
        :code:`Model.random_state_time_course_initialization`. Defaults to::

            {'n_init': 3, 'n_epochs': 1}.
    fit_kwargs : dict, optional
        Keyword arguments to pass to the :code:`Model.fit`. No defaults.
    save_inf_params : bool, optional
        Should we save the inferred parameters?
    """
    if data is None:
        raise ValueError("data must be passed.")

    from osl_dynamics.models import hmm

    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    # Directories
    model_dir = output_dir + "/model"

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
    init_history = model.random_state_time_course_initialization(
        data,
        **init_kwargs,
    )

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
        # Make output directory
        inf_params_dir = output_dir + "/inf_params"
        os.makedirs(inf_params_dir, exist_ok=True)

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
    init_kwargs=None,
    fit_kwargs=None,
    save_inf_params=True,
):
    """Train `DyNeMo <https://osl-dynamics.readthedocs.io/en/latest/autoapi\
    /osl_dynamics/models/dynemo/index.html>`_.

    This function will:

    1. Build a :code:`dynemo.Model` object.
    2. Initialize the parameters of the model using
       :code:`Model.random_subset_initialization`.
    3. Perform full training.
    4. Save the inferred parameters (mode mixing coefficients, means and
       covariances) if :code:`save_inf_params=True`.

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
        Keyword arguments to pass to `dynemo.Config <https://osl-dynamics\
        .readthedocs.io/en/latest/autoapi/osl_dynamics/models/dynemo\
        /index.html#osl_dynamics.models.dynemo.Config>`_. Defaults to::

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
             'lr_decay': 0.1,
             'n_epochs': 40}
    init_kwargs : dict, optional
        Keyword arguments to pass to :code:`Model.random_subset_initialization`.
        Defaults to::

            {'n_init': 5, 'n_epochs': 2, 'take': 1}.
    fit_kwargs : dict, optional
        Keyword arguments to pass to the :code:`Model.fit`.
    save_inf_params : bool, optional
        Should we save the inferred parameters?
    """

    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

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
        "lr_decay": 0.1,
        "n_epochs": 40,
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
    _logger.info(f"Using config_kwargs: {config_kwargs}")
    config = dynemo.Config(**config_kwargs)
    model = dynemo.Model(config)
    model.summary()

    # Set regularisers
    model.set_regularizers(data)

    # Initialisation
    default_init_kwargs = {"n_init": 5, "n_epochs": 2, "take": 1}
    init_kwargs = override_dict_defaults(default_init_kwargs, init_kwargs)
    _logger.info(f"Using init_kwargs: {init_kwargs}")
    init_history = model.random_subset_initialization(data, **init_kwargs)

    # Keyword arguments for the fit method
    default_fit_kwargs = {}
    fit_kwargs = override_dict_defaults(default_fit_kwargs, fit_kwargs)
    _logger.info(f"Using fit_kwargs: {fit_kwargs}")

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


def train_mdynemo(
    data,
    output_dir,
    config_kwargs,
    init_kwargs=None,
    fit_kwargs=None,
    corrs_init_kwargs=None,
    save_inf_params=True,
):
    """Train `MDyNeMo <https://osl-dynamics.readthedocs.io/en/latest/autoapi\
        /osl_dynamics/models/mdynemo/index.html>`_. This function will:
    
    1. Build an :code:`mdynemo.Model` object.
    2. Initialize the mode correlations using sliding window and KMeans.
    3. Initialize the parameters of the model using
        :code:`Model.random_subset_initialization`.
    4. Perform full training.
    5. Save the inferred parameters (mode time courses, means, stds and corrs)
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
        Keyword arguments to pass to `mdynemo.Config <https://osl-dynamics\
        .readthedocs.io/en/latest/autoapi/osl_dynamics/models/mdynemo\
        /index.html#osl_dynamics.models.mdynemo.Config>`_. Defaults to::

            {
                'n_channels': data.n_channels,
                'sequence_length': 200,
                'inference_n_units': 64,
                'inference_normalization': 'layer',
                'model_n_units': 64,
                'model_normalization': 'layer',
                'do_kl_annealing': True,
                'kl_annealing_curve': 'tanh',
                'kl_annealing_sharpness': 10,
                'n_kl_annealing_epochs': 20,
                'batch_size': 128,
                'learning_rate': 0.01,
                'lr_decay': 0.1,
                'n_epochs': 40,
            }.
    init_kwargs : dict, optional
        Keyword arguments to pass to :code:`Model.random_subset_initialization`.
        Defaults to::

            {'n_init': 5, 'n_epochs': 5, 'take': 1}.
    fit_kwargs : dict, optional
        Keyword arguments to pass to the :code:`Model.fit`.
    corrs_init_kwargs : dict, optional
        Keyword arguments to pass to the mode correlations
        initialisation. Defaults to::

            {
                'window_length': data.sampling_frequency * 2,
                'step_size': data.sampling_frequency // 25,
                'random_state': None,
                'n_init': 'auto',
                'init': 'k-means++',
            }.
    save_inf_params : bool, optional
        Should we save the inferred parameters?
    """
    if data is None:
        raise ValueError("data must be passed.")

    from osl_dynamics.models import mdynemo
    from osl_dynamics.analysis import connectivity
    from sklearn.cluster import KMeans

    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    corrs_init_kwargs = {} if corrs_init_kwargs is None else corrs_init_kwargs

    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"

    _logger.info("Building model")
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "sequence_length": 200,
        "inference_n_units": 64,
        "inference_normalization": "layer",
        "model_n_units": 64,
        "model_normalization": "layer",
        "do_kl_annealing": True,
        "kl_annealing_curve": "tanh",
        "kl_annealing_sharpness": 10,
        "n_kl_annealing_epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.01,
        "lr_decay": 0.1,
        "n_epochs": 40,
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)
    _logger.info(f"Using config_kwargs: {config_kwargs}")
    config = mdynemo.Config(**config_kwargs)
    config.pca_components = data.pca_components

    # KMeans to intialise corrs
    _logger.info("Initialising corrs")
    default_corrs_init_kwargs = {
        "window_length": data.sampling_frequency * 2,
        "step_size": data.sampling_frequency // 25,
        "random_state": None,
        "n_init": "auto",
        "init": "k-means++",
    }
    corrs_init_kwargs = override_dict_defaults(
        default_corrs_init_kwargs, corrs_init_kwargs
    )
    _logger.info(f"Using corrs_init_kwargs: {corrs_init_kwargs}")
    tv_corr = connectivity.sliding_window_connectivity(
        data.time_series(),
        window_length=corrs_init_kwargs["window_length"],
        step_size=corrs_init_kwargs["step_size"],
        conn_type="corr",
        concatenate=True,
        n_jobs=data.n_jobs,
    )
    tv_corr = np.reshape(tv_corr, (tv_corr.shape[0], -1))
    kmeans = KMeans(
        n_clusters=config.n_corr_modes,
        n_init=corrs_init_kwargs["n_init"],
        init=corrs_init_kwargs["init"],
        random_state=corrs_init_kwargs["random_state"],
    ).fit(tv_corr)
    initial_corrs = kmeans.cluster_centers_.reshape(
        config.n_corr_modes, data.n_channels, data.n_channels
    )
    initial_corrs = array_ops.cov2corr(initial_corrs)
    config.initial_corrs = (
        config.pca_components @ initial_corrs @ config.pca_components.T
    )

    model = mdynemo.Model(config)
    model.summary()

    # Initialisation
    default_init_kwargs = {"n_init": 5, "n_epochs": 5, "take": 1}
    init_kwargs = override_dict_defaults(default_init_kwargs, init_kwargs)
    _logger.info(f"Using init_kwargs: {init_kwargs}")
    init_history = model.random_subset_initialization(data, **init_kwargs)

    # Keyword arguments for the fit method
    default_fit_kwargs = {}
    fit_kwargs = override_dict_defaults(default_fit_kwargs, fit_kwargs)
    _logger.info(f"Using fit_kwargs: {fit_kwargs}")

    # Training
    history = model.fit(data, **fit_kwargs)

    # Add free energy to the history object
    history["free_energy"] = model.free_energy(data)

    # Save trained model
    model.save(model_dir)
    save(f"{model_dir}/init_history.pkl", init_history)
    save(f"{model_dir}/history.pkl", history)

    if save_inf_params:
        del model
        model = mdynemo.Model.load(model_dir)
        os.makedirs(inf_params_dir, exist_ok=True)

        # Get the inferred parameters
        alpha, beta = model.get_mode_time_courses(data)
        means, stds, corrs = model.get_means_stds_corrs()

        save(f"{inf_params_dir}/alp.pkl", alpha)
        save(f"{inf_params_dir}/bet.pkl", beta)
        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/stds.npy", stds)
        save(f"{inf_params_dir}/corrs.npy", corrs)


def train_hive(
    data,
    output_dir,
    config_kwargs,
    init_kwargs=None,
    fit_kwargs=None,
    save_inf_params=True,
):
    """ Train a `HIVE Model <https://osl-dynamics.\
    readthedocs.io/en/latest/autoapi/osl_dynamics/models/hive/index.html>`_.
    
    This function will:

    1. Build an :code:`hive.Model` object.
    2. Initialize the parameters of the HIVE model using
        :code:`Model.random_state_time_course_initialization`.
    3. Perform full training.
    4. Save the inferred parameters (state probabilities, means,
        covariances and embeddings) if :code:`save_inf_params=True`.
    
    This function will create two directories:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.
        This directory is only created if :code:`save_inf_params=True`.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object for training the model.
    output_dir : str
        Path to output directory.
    config_kwargs : dict
        Keyword arguments to pass to `hive.Config <https://osl-dynamics\
        .readthedocs.io/en/latest/autoapi/osl_dynamics/models/hive/index.html\
        #osl_dynamics.models.hive.Config>`_. Defaults to::

            {
                'sequence_length': 200,
                'spatial_embeddings_dim': 2,
                'dev_n_layers': 5,
                'dev_n_units': 32,
                'dev_activation': 'tanh',
                'dev_normalization': 'layer',
                'dev_regularizer': 'l1',
                'dev_regularizer_factor': 10,
                'batch_size': 128,
                'learning_rate': 0.005,
                'lr_decay': 0.1,
                'n_epochs': 30,
                'do_kl_annealing': True,
                'kl_annealing_curve': 'tanh',
                'kl_annealing_sharpness': 10,
                'n_kl_annealing_epochs': 15,
            }.
    init_kwargs : dict, optional
        Keyword arguments to pass to
        :code:`Model.random_state_time_course_initialization`. Defaults to::

            {'n_init': 10, 'n_epochs': 2}.
    fit_kwargs : dict, optional
        Keyword arguments to pass to the :code:`Model.fit`. No defaults.
    save_inf_params : bool, optional
        Should we save the inferred parameters?
    """
    if data is None:
        raise ValueError("data must be passed.")

    if not data.get_session_labels():
        data.add_session_labels("session_id", np.arange(data.n_sessions), "categorical")

    from osl_dynamics.models import hive

    init_kwargs = {} if init_kwargs is None else init_kwargs
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    # Directories
    model_dir = output_dir + "/model"

    _logger.info("Building model")

    # SE-HMM config
    default_config_kwargs = {
        "n_channels": data.n_channels,
        "n_sessions": data.n_sessions,
        "sequence_length": 200,
        "spatial_embeddings_dim": 2,
        "dev_n_layers": 5,
        "dev_n_units": 32,
        "dev_activation": "tanh",
        "dev_normalization": "layer",
        "dev_regularizer": "l1",
        "dev_regularizer_factor": 10,
        "batch_size": 128,
        "learning_rate": 0.005,
        "lr_decay": 0.1,
        "n_epochs": 30,
        "do_kl_annealing": True,
        "kl_annealing_curve": "tanh",
        "kl_annealing_sharpness": 10,
        "n_kl_annealing_epochs": 15,
        "session_labels": data.get_session_labels(),
    }
    config_kwargs = override_dict_defaults(default_config_kwargs, config_kwargs)

    default_init_kwargs = {"n_init": 10, "n_epochs": 2}
    init_kwargs = override_dict_defaults(default_init_kwargs, init_kwargs)
    _logger.info(f"Using init_kwargs: {init_kwargs}")

    # Initialise and train HIVE
    _logger.info(f"Using config_kwargs: {config_kwargs}")
    config = hive.Config(**config_kwargs)
    model = hive.Model(config)
    model.summary()

    # Set regularisers
    model.set_regularizers(data)

    # Set deviation initializer
    model.set_dev_parameters_initializer(data)

    # Initialise HIVE
    _logger.info(f"Using init_kwargs: {init_kwargs}")
    init_history = model.random_state_time_course_initialization(
        data,
        **init_kwargs,
    )

    # Training
    history = model.fit(data, **fit_kwargs)

    _logger.info(f"Saving model to: {model_dir}")
    model.save(model_dir)

    del model
    model = hive.Model.load(model_dir)

    # Get the variational free energy
    history["free_energy"] = model.free_energy(data)

    # Save trained model
    save(f"{model_dir}/init_history.pkl", init_history)
    save(f"{model_dir}/history.pkl", history)

    if save_inf_params:
        # Make output directory
        inf_params_dir = output_dir + "/inf_params"
        os.makedirs(inf_params_dir, exist_ok=True)

        # Get the inferred parameters
        alpha = model.get_alpha(data)
        means, covs = model.get_means_covariances()
        session_means, session_covs = model.get_session_means_covariances()
        summed_embeddings = model.get_summed_embeddings()
        embedding_weights = model.get_embedding_weights()

        # Save inferred parameters
        save(f"{inf_params_dir}/alp.pkl", alpha)
        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/covs.npy", covs)
        save(f"{inf_params_dir}/session_means.npy", session_means)
        save(f"{inf_params_dir}/session_covs.npy", session_covs)
        save(f"{inf_params_dir}/summed_embeddings.npy", summed_embeddings)
        save(f"{inf_params_dir}/embedding_weights.pkl", embedding_weights)


def get_inf_params(data, output_dir, observation_model_only=False):
    """Get inferred alphas.

    This function expects a model has already been trained and the following
    directory to exist:

    - :code:`<output_dir>/model`, which contains the trained model.

    This function will create the following directory:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    observation_model_only : bool, optional
        We we only want to get the observation model parameters?
    """
    # Make output directory
    inf_params_dir = output_dir + "/inf_params"
    os.makedirs(inf_params_dir, exist_ok=True)

    #  Load model
    from osl_dynamics.models import load

    model_dir = output_dir + "/model"
    model = load(model_dir)

    if observation_model_only:
        # Get the inferred parameters
        means, covs = model.get_means_covariances()

        # Save
        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/covs.npy", covs)
    else:
        # Get the inferred parameters
        alpha = model.get_alpha(data)
        means, covs = model.get_means_covariances()

        # Save
        save(f"{inf_params_dir}/alp.pkl", alpha)
        save(f"{inf_params_dir}/means.npy", means)
        save(f"{inf_params_dir}/covs.npy", covs)


def plot_power_maps_from_covariances(
    data,
    output_dir,
    mask_file=None,
    parcellation_file=None,
    power_save_kwargs=None,
):
    """Plot power maps calculated directly from the inferred covariances.

    This function expects a model has already been trained and the following
    directory to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will output files called :code:`covs_.png` which contain
    plots of the power map of each state/mode taken directly from the inferred
    covariance matrices. The files will be saved to
    :code:`<output_dir>/inf_params`.

    This function also expects the data to be prepared in the same script
    that this wrapper is called from.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    mask_file : str, optional
        Mask file used to preprocess the training data. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str, optional
        Parcellation file used to parcellate the training data. If
        :code:`None`, we use :code:`data.parcellation_file`.
    power_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'filename': '<inf_params_dir>/covs_.png',
             'mask_file': data.mask_file,
             'parcellation_file': data.parcellation_file,
             'plot_kwargs': {'symmetric_cbar': True}}
    """
    # Validation
    power_save_kwargs = {} if power_save_kwargs is None else power_save_kwargs

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

    if hasattr(data, "n_embeddings"):
        n_embeddings = data.n_embeddings
    else:
        n_embeddings = 1

    if hasattr(data, "pca_components"):
        pca_components = data.pca_components
    else:
        pca_components = None

    # Directories
    inf_params_dir = f"{output_dir}/inf_params"

    # Load inferred covariances
    covs = load(f"{inf_params_dir}/covs.npy")

    # Reverse the effects of preparing the data
    from osl_dynamics.analysis import modes

    covs = modes.raw_covariances(covs, n_embeddings, pca_components)

    # Save
    from osl_dynamics.analysis import power

    default_power_save_kwargs = {
        "filename": f"{inf_params_dir}/covs_.png",
        "mask_file": mask_file,
        "parcellation_file": parcellation_file,
        "plot_kwargs": {"symmetric_cbar": True},
    }
    if "plot_kwargs" in power_save_kwargs:
        power_save_kwargs["plot_kwargs"] = override_dict_defaults(
            default_power_save_kwargs["plot_kwargs"],
            power_save_kwargs["plot_kwargs"],
        )
    power_save_kwargs = override_dict_defaults(
        default_power_save_kwargs, power_save_kwargs
    )
    _logger.info(f"Using power_save_kwargs: {power_save_kwargs}")
    power.save(covs, **power_save_kwargs)


def plot_tde_covariances(data, output_dir):
    """Plot inferred covariance of the time-delay embedded data.

    This function expects a model has already been trained and the following
    directory to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will output a :code:`tde_covs.png` file containing a plot of
    the covariances in the :code:`<output_dir>/inf_params` directory.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    """
    inf_params_dir = f"{output_dir}/inf_params"

    covs = load(f"{inf_params_dir}/covs.npy")

    if hasattr(data, "pca_components"):
        if data.pca_components is not None:
            from osl_dynamics.analysis import modes

            covs = modes.reverse_pca(covs, data.pca_components)

    from osl_dynamics.utils import plotting

    plotting.plot_matrices(covs, filename=f"{inf_params_dir}/tde_covs.png")


def plot_state_psds(data, output_dir):
    """Plot state PSDs.

    This function expects multitaper spectra to have already been calculated
    and are in:

    - :code:`<output_dir>/spectra`.

    This function will output a file called :code:`psds.png` which contains
    a plot of each state PSD.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    """
    spectra_dir = f"{output_dir}/spectra"

    f = load(f"{spectra_dir}/f.npy")
    psd = load(f"{spectra_dir}/psd.npy")
    psd = np.mean(psd, axis=(0, 2))  # average over arrays and channels
    n_states = psd.shape[0]

    from osl_dynamics.utils import plotting

    plotting.plot_line(
        [f] * n_states,
        psd,
        labels=[f"State {i + 1}" for i in range(n_states)],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        x_range=[f[0], f[-1]],
        filename=f"{spectra_dir}/psds.png",
    )


def dual_estimation(data, output_dir, n_jobs=1):
    """Dual estimation for session-specific observation model parameters.

    This function expects a model has already been trained and the following
    directories to exist:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create the following directory:

    - :code:`<output_dir>/dual_estimates`, which contains the session-specific
      means and covariances.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    n_jobs : int, optional
        Number of jobs to run in parallel.
    """
    if data is None:
        raise ValueError("data must be passed.")

    # Directories
    model_dir = f"{output_dir}/model"
    inf_params_dir = f"{output_dir}/inf_params"
    dual_estimates_dir = f"{output_dir}/dual_estimates"
    os.makedirs(dual_estimates_dir, exist_ok=True)

    #  Load model
    from osl_dynamics import models

    model = models.load(model_dir)

    # Load the inferred state probabilities
    alpha = load(f"{inf_params_dir}/alp.pkl")

    # Dual estimation
    means, covs = model.dual_estimation(data, alpha=alpha, n_jobs=n_jobs)

    # Save
    save(f"{dual_estimates_dir}/means.npy", means)
    save(f"{dual_estimates_dir}/covs.npy", covs)


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
        Keyword arguments to pass to `analysis.spectral.multitaper_spectra
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/spectral/index.html#osl_dynamics.analysis.spectral\
        .multitaper_spectra>`_. Defaults to::

            {'sampling_frequency': data.sampling_frequency,
             'keepdims': True}
    nnmf_components : int, optional
        Number of non-negative matrix factorization (NNMF) components to fit to
        the stacked session-specific coherence spectra.
    """
    if data is None:
        raise ValueError("data must be passed.")

    sampling_frequency = kwargs.pop("sampling_frequency", None)
    if sampling_frequency is None and data.sampling_frequency is None:
        raise ValueError(
            "sampling_frequency must be passed or specified in the Data object."
        )
    else:
        sampling_frequency = data.sampling_frequency

    default_kwargs = {
        "sampling_frequency": sampling_frequency,
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

    spectra = spectral.multitaper_spectra(data=data, alpha=alpha, **kwargs)

    # Unpack spectra and save
    return_weights = kwargs.pop("return_weights", False)
    if return_weights:
        f, psd, coh, w = spectra
        save(f"{spectra_dir}/f.npy", f)
        save(f"{spectra_dir}/psd.npy", psd)
        save(f"{spectra_dir}/coh.npy", coh)
        save(f"{spectra_dir}/w.npy", w)
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
    from osl_dynamics.analysis import spectral

    spectra_dir = output_dir + "/spectra"
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
        Keyword arguments to pass to `analysis.spectral.regress_spectra
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/spectral/index.html#osl_dynamics.analysis.spectral\
        .regression_spectra>`_. Defaults to::

            {'sampling_frequency': data.sampling_frequency,
             'window_length': 4 * sampling_frequency,
             'step_size': 20,
             'n_sub_windows': 8,
             'return_coef_int': True,
             'keepdims': True}
    """
    if data is None:
        raise ValueError("data must be passed.")

    sampling_frequency = kwargs.pop("sampling_frequency", None)
    if sampling_frequency is None and data.sampling_frequency is None:
        raise ValueError(
            "sampling_frequency must be passed or specified in the Data object."
        )
    else:
        sampling_frequency = data.sampling_frequency

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

    spectra = spectral.regression_spectra(data=data, alpha=alpha, **kwargs)

    # Unpack spectra and save
    return_weights = kwargs.pop("return_weights", False)
    if return_weights:
        f, psd, coh, w = spectra
        save(f"{spectra_dir}/f.npy", f)
        save(f"{spectra_dir}/psd.npy", psd)
        save(f"{spectra_dir}/coh.npy", coh)
        save(f"{spectra_dir}/w.npy", w)
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
    power_save_kwargs=None,
    conn_save_kwargs=None,
):
    """Plot group-level amplitude envelope networks.

    This function expects a model has been trained and the following directory
    to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/networks`, which contains plots of the networks.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    mask_file : str, optional
        Mask file used to preprocess the training data. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str, optional
        Parcellation file used to parcellate the training data. If
        :code:`None`, we use :code:`data.parcellation_file`.
    aec_abs : bool, optional
        Should we take the absolute value of the amplitude envelope
        correlations?
    power_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'filename': '<output_dir>/networks/mean_.png',
             'mask_file': data.mask_file,
             'parcellation_file': data.parcellation_file,
             'plot_kwargs': {'symmetric_cbar': True}}
    conn_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/connectivity/index.html#osl_dynamics.analysis.connectivity\
        .save>`_. Defaults to::

            {'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/aec_.png',
             'threshold': 0.97}
    """
    power_save_kwargs = {} if power_save_kwargs is None else power_save_kwargs
    conn_save_kwargs = {} if conn_save_kwargs is None else conn_save_kwargs

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
        "plot_kwargs": {"symmetric_cbar": True},
    }
    if "plot_kwargs" in power_save_kwargs:
        power_save_kwargs["plot_kwargs"] = override_dict_defaults(
            default_power_save_kwargs["plot_kwargs"],
            power_save_kwargs["plot_kwargs"],
        )
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
    power_save_kwargs=None,
    conn_save_kwargs=None,
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
    mask_file : str, optional
        Mask file used to preprocess the training data. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str, optional
        Parcellation file used to parcellate the training data. If
        :code:`None`, we use :code:`data.parcellation_file`.
    frequency_range : list, optional
        List of length 2 containing the minimum and maximum frequency to
        integrate spectra over. Defaults to the full frequency range.
    percentile : float, optional
        Percentile for thresholding the coherence networks. Default is 97, which
        corresponds to the top 3% of edges (relative to the mean across states).
    power_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'mask_file': mask_file,
             'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/pow_.png',
             'subtract_mean': True,
             'plot_kwargs': {'symmetric_cbar': True}}
    conn_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/connectivity/index.html#osl_dynamics.analysis.connectivity\
        .save>`_. Defaults to::

            {'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/coh_.png',
             'plot_kwargs': {'edge_cmap': 'Reds'}}
    """
    power_save_kwargs = {} if power_save_kwargs is None else power_save_kwargs
    conn_save_kwargs = {} if conn_save_kwargs is None else conn_save_kwargs

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
            ax.axvspan(
                frequency_range[0],
                frequency_range[1],
                alpha=0.25,
                color="gray",
            )
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
        "plot_kwargs": {"symmetric_cbar": True},
    }
    if "plot_kwargs" in power_save_kwargs:
        power_save_kwargs["plot_kwargs"] = override_dict_defaults(
            default_power_save_kwargs["plot_kwargs"],
            power_save_kwargs["plot_kwargs"],
        )
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
    power_save_kwargs=None,
    conn_save_kwargs=None,
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
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/spectral/index.html#osl_dynamics.analysis.spectral\
        .decompose_spectra>`_) containing the NNMF components.
    mask_file : str, optional
        Mask file used to preprocess the training data. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str, optional
        Parcellation file used to parcellate the training data. If
        :code:`None`, we use :code:`data.parcellation_file`.
    component : int, optional
        NNMF component to plot. Defaults to the first component.
    percentile : float, optional
        Percentile for thresholding the coherence networks. Default is 97, which
        corresponds to the top 3% of edges (relative to the mean across states).
    power_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'mask_file': mask_file,
             'parcellation_file': parcellation_file,
             'component': component,
             'filename': '<output_dir>/networks/pow_.png',
             'subtract_mean': True,
             'plot_kwargs': {'symmetric_cbar': True}}
    conn_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/connectivity/index.html#osl_dynamics.analysis.connectivity\
        .save>`_. Defaults to::

            {'parcellation_file': parcellation_file,
             'component': component,
             'filename': '<output_dir>/networks/coh_.png',
             'plot_kwargs': {'edge_cmap': 'Reds'}}
    """
    power_save_kwargs = {} if power_save_kwargs is None else power_save_kwargs
    conn_save_kwargs = {} if conn_save_kwargs is None else conn_save_kwargs

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
        "plot_kwargs": {"symmetric_cbar": True},
    }
    if "plot_kwargs" in power_save_kwargs:
        power_save_kwargs["plot_kwargs"] = override_dict_defaults(
            default_power_save_kwargs["plot_kwargs"],
            power_save_kwargs["plot_kwargs"],
        )
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
    power_save_kwargs=None,
    conn_save_kwargs=None,
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
    mask_file : str, optional
        Mask file used to preprocess the training data. If :code:`None`,
        we use :code:`data.mask_file`.
    parcellation_file : str, optional
        Parcellation file used to parcellate the training data. If
        :code:`None`, we use :code:`data.parcellation_file`.
    frequency_range : list, optional
        List of length 2 containing the minimum and maximum frequency to
        integrate spectra over. Defaults to the full frequency range.
    percentile : float, optional
        Percentile for thresholding the coherence networks. Default is 97, which
        corresponds to the top 3% of edges (relative to the mean across states).
    plot_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.power.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/power/index.html#osl_dynamics.analysis.power.save>`_.
        Defaults to::

            {'mask_file': mask_file,
             'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/pow_.png',
             'subtract_mean': True,
             'plot_kwargs': {'symmetric_cbar': True}}
    conn_save_kwargs : dict, optional
        Keyword arguments to pass to `analysis.connectivity.save
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /analysis/connectivity/index.html#osl_dynamics.analysis.connectivity\
        .save>`_. Defaults to::

            {'parcellation_file': parcellation_file,
             'filename': '<output_dir>/networks/coh_.png',
             'plot_kwargs': {'edge_cmap': 'Reds'}}
    """
    power_save_kwargs = {} if power_save_kwargs is None else power_save_kwargs
    conn_save_kwargs = {} if conn_save_kwargs is None else conn_save_kwargs

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
            ax.axvspan(
                frequency_range[0],
                frequency_range[1],
                alpha=0.25,
                color="gray",
            )
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
        "plot_kwargs": {"symmetric_cbar": True},
    }
    if "plot_kwargs" in power_save_kwargs:
        power_save_kwargs["plot_kwargs"] = override_dict_defaults(
            default_power_save_kwargs["plot_kwargs"],
            power_save_kwargs["plot_kwargs"],
        )
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
    data,
    output_dir,
    session=0,
    normalize=False,
    sampling_frequency=None,
    kwargs=None,
):
    """Plot inferred alphas.

    This is a wrapper for `utils.plotting.plot_alpha
    <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/utils\
    /plotting/index.html#osl_dynamics.utils.plotting.plot_alpha>`_.

    This function expects a model has been trained and the following directory
    to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/alphas`, which contains plots of the inferred alphas.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    session : int, optional
        Index for session to plot. If 'all' is passed we create a separate plot
        for each session.
    normalize : bool, optional
        Should we also plot the alphas normalized using the trace of the
        inferred covariance matrices? Useful if we are plotting the inferred
        alphas from DyNeMo.
    sampling_frequency : float, optional
        Sampling frequency in Hz. If :code:`None`, we see if it is
        present in :code:`data.sampling_frequency`.
    kwargs : dict, optional
        Keyword arguments to pass to `utils.plotting.plot_alpha
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /utils/plotting/index.html#osl_dynamics.utils.plotting.plot_alpha>`_.
        Defaults to::

            {'sampling_frequency': data.sampling_frequency,
             'filename': '<output_dir>/alphas/alpha_*.png'}
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

    if session == "all":
        for i in range(len(alp)):
            kwargs["filename"] = f"{alphas_dir}/alpha_{i}.png"
            plotting.plot_alpha(alp[i], **kwargs)
    else:
        kwargs["filename"] = f"{alphas_dir}/alpha_{session}.png"
        plotting.plot_alpha(alp[session], **kwargs)

    if normalize:
        from osl_dynamics.inference import modes

        # Calculate normalised alphas
        covs = load(f"{inf_params_dir}/covs.npy")
        norm_alp = modes.reweight_alphas(alp, covs)

        # Plot
        if session == "all":
            for i in range(len(alp)):
                kwargs["filename"] = f"{alphas_dir}/norm_alpha_{i}.png"
                plotting.plot_alpha(norm_alp[i], **kwargs)
        else:
            kwargs["filename"] = f"{alphas_dir}/norm_alpha_{session}.png"
            plotting.plot_alpha(norm_alp[session], **kwargs)


def calc_gmm_alpha(data, output_dir, kwargs=None):
    """Binarize inferred alphas using a two-component GMM.

    This function expects a model has been trained and the following directory
    to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create the following file:

    - :code:`<output_dir>/inf_params/gmm_alp.pkl`, which contains the binarized
      alphas.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    kwargs : dict, optional
        Keyword arguments to pass to `inference.modes.gmm_time_courses
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
        /inference/modes/index.html#osl_dynamics.inference.modes\
        .gmm_time_courses>`_.
    """
    kwargs = {} if kwargs is None else kwargs
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


def plot_hmm_network_summary_stats(
    data,
    output_dir,
    use_gmm_alpha=False,
    sampling_frequency=None,
    sns_kwargs=None,
):
    """Plot HMM summary statistics for networks as violin plots.

    This function will plot the distribution over sessions for the following
    summary statistics:

    - Fractional occupancy.
    - Mean lifetime (s).
    - Mean interval (s).
    - Switching rate (Hz).

    This function expects a model has been trained and the following directory
    to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/summary_stats`, which contains plots of the summary
      statistics.

    The :code:`<output_dir>/summary_stats` directory will also contain numpy
    files with the summary statistics.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    use_gmm_alpha : bool, optional
        Should we use alphas binarised using a Gaussian mixture model?
        This function assumes :code:`calc_gmm_alpha` has been called and the
        file :code:`<output_dir>/inf_params/gmm_alp.pkl` exists.
    sampling_frequency : float, optional
        Sampling frequency in Hz. If :code:`None`, we use
        :code:`data.sampling_frequency`.
    sns_kwargs : dict, optional
        Arguments to pass to :code:`sns.violinplot()`.
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
                "We must train on multiple sessions to plot the distribution "
                "of summary statistics."
            )
        stc = modes.argmax_time_courses(alp)

    # Calculate summary stats
    fo = modes.fractional_occupancies(stc)
    lt = modes.mean_lifetimes(stc, sampling_frequency)
    intv = modes.mean_intervals(stc, sampling_frequency)
    sr = modes.switching_rates(stc, sampling_frequency)

    # Save summary stats
    save(f"{summary_stats_dir}/fo.npy", fo)
    save(f"{summary_stats_dir}/lt.npy", lt)
    save(f"{summary_stats_dir}/intv.npy", intv)
    save(f"{summary_stats_dir}/sr.npy", sr)

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
        sns_kwargs=sns_kwargs,
    )
    plotting.plot_violin(
        lt.T,
        x=x,
        x_label="State",
        y_label="Mean Lifetime (s)",
        filename=f"{summary_stats_dir}/lt.png",
        sns_kwargs=sns_kwargs,
    )
    plotting.plot_violin(
        intv.T,
        x=x,
        x_label="State",
        y_label="Mean Interval (s)",
        filename=f"{summary_stats_dir}/intv.png",
        sns_kwargs=sns_kwargs,
    )
    plotting.plot_violin(
        sr.T,
        x=x,
        x_label="State",
        y_label="Switching rate (Hz)",
        filename=f"{summary_stats_dir}/sr.png",
        sns_kwargs=sns_kwargs,
    )


def plot_dynemo_network_summary_stats(data, output_dir):
    """Plot DyNeMo summary statistics for networks as violin plots.

    This function will plot the distribution over sessions for the following
    summary statistics:

    - Mean (renormalised) mixing coefficients.
    - Standard deviation of (renormalised) mixing coefficients.

    This function expects a model has been trained and the following directories
    to exist:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/summary_stats`, which contains plots of the summary
      statistics.

    The :code:`<output_dir>/summary_stats` directory will also contain numpy
    files with the summary statistics.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    """
    # Directories
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"
    summary_stats_dir = output_dir + "/summary_stats"
    os.makedirs(summary_stats_dir, exist_ok=True)

    # Load inferred parameters
    alp = load(f"{inf_params_dir}/alp.pkl")
    if isinstance(alp, np.ndarray):
        raise ValueError(
            "We must train on multiple sessions to plot the distribution "
            "of summary statistics."
        )

    # Get the config used to create the model
    from osl_dynamics.models.mod_base import ModelBase

    config, _ = ModelBase.load_config(model_dir)

    # Renormalise (only if we are learning covariances)
    from osl_dynamics.inference import modes

    if config["learn_covariances"]:
        covs = load(f"{inf_params_dir}/covs.npy")
        alp = modes.reweight_alphas(alp, covs)

    # Calculate summary stats
    alp_mean = np.array([np.mean(a, axis=0) for a in alp])
    alp_std = np.array([np.std(a, axis=0) for a in alp])
    alp_corr = np.array([np.corrcoef(a, rowvar=False) for a in alp])
    for c in alp_corr:
        np.fill_diagonal(c, 0)  # remove diagonal to see the off-diagonals better

    # Save summary stats
    save(f"{summary_stats_dir}/alp_mean.npy", alp_mean)
    save(f"{summary_stats_dir}/alp_std.npy", alp_std)
    save(f"{summary_stats_dir}/alp_corr.npy", alp_corr)

    # Plot
    from osl_dynamics.utils import plotting

    n_modes = alp_mean.shape[1]
    x = range(1, n_modes + 1)
    plotting.plot_violin(
        alp_mean.T,
        x=x,
        x_label="Mode",
        y_label="Mean",
        filename=f"{summary_stats_dir}/alp_mean.png",
    )
    plotting.plot_violin(
        alp_std.T,
        x=x,
        x_label="Mode",
        y_label="Standard Deviation",
        filename=f"{summary_stats_dir}/alp_std.png",
    )
    plotting.plot_matrices(
        np.mean(alp_corr, axis=0), filename=f"{summary_stats_dir}/alp_corr.png"
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

    This function expects a model has been trained and the following directory
    to exist:

    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/group_diff`, which contains the summary statistics
      and plots.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    group2_indices : np.ndarray or list
        Indices indicating which sessions belong to the second group.
    separate_tests : bool, optional
        Should we perform a maximum statistic permutation test for each summary
        statistic separately?
    covariates : str, optional
        Path to a pickle file containing a :code:`dict` with covariances. Each
        item in the :code:`dict` must be the covariate name and value for each
        session. The covariates will be loaded with::

            from osl_dynamics.utils.misc import load
            covariates = load("/path/to/file.pkl")

        Example covariates::

            covariates = {"age": [...], "sex": [...]}
    n_perm : int, optional
        Number of permutations.
    n_jobs : int, optional
        Number of jobs for parallel processing.
    sampling_frequency : float, optional
        Sampling frequency in Hz. If :code:`None`, we use
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
    n_sessions = fo.shape[0]
    assignments = np.ones(n_sessions)
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
            # Calculate a statistical significance test for each
            # summary stat separately
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
        # Calculate a statistical significance test for all
        # summary stats concatenated
        _, pvalues = statistics.group_diff_max_stat_perm(
            sum_stats,
            assignments,
            n_perm=n_perm,
            covariates=covariates,
            n_jobs=n_jobs,
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


def plot_burst_summary_stats(data, output_dir, sampling_frequency=None):
    """Plot burst summary statistics as violin plots.

    This function will plot the distribution over sessions for the following
    summary statistics:

    - Mean lifetime (s).
    - Mean interval (s).
    - Burst count (Hz).
    - Mean amplitude (a.u.).

    This function expects a model has been trained and the following
    directories to exist:

    - :code:`<output_dir>/model`, which contains the trained model.
    - :code:`<output_dir>/inf_params`, which contains the inferred parameters.

    This function will create:

    - :code:`<output_dir>/summary_stats`, which contains plots of the summary
      statistics.

    The :code:`<output_dir>/summary_stats` directory will also contain numpy
    files with the summary statistics.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        Data object.
    output_dir : str
        Path to output directory.
    sampling_frequency : float, optional
        Sampling frequency in Hz. If :code:`None`, we use
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
    model_dir = output_dir + "/model"
    inf_params_dir = output_dir + "/inf_params"
    summary_stats_dir = output_dir + "/summary_stats"
    os.makedirs(summary_stats_dir, exist_ok=True)

    from osl_dynamics.inference import modes

    # Load state time course
    alp = load(f"{inf_params_dir}/alp.pkl")
    stc = modes.argmax_time_courses(alp)

    # Get the config used to create the model
    from osl_dynamics.models.mod_base import ModelBase

    model_config, _ = ModelBase.load_config(model_dir)

    # Get unprepared data (i.e. the data before calling Data.prepare)
    # We also trim the data to account for the data points lost to
    # time embedding or applying a sliding window
    data = data.trim_time_series(
        sequence_length=model_config["sequence_length"], prepared=False
    )

    # Calculate summary stats
    lt = modes.mean_lifetimes(stc, sampling_frequency)
    intv = modes.mean_intervals(stc, sampling_frequency)
    bc = modes.switching_rates(stc, sampling_frequency)
    amp = modes.mean_amplitudes(stc, data)

    # Save summary stats
    save(f"{summary_stats_dir}/lt.npy", lt)
    save(f"{summary_stats_dir}/intv.npy", intv)
    save(f"{summary_stats_dir}/bc.npy", bc)
    save(f"{summary_stats_dir}/amp.npy", amp)

    from osl_dynamics.utils import plotting

    # Plot
    n_states = lt.shape[1]
    plotting.plot_violin(
        lt.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Lifetime (s)",
        filename=f"{summary_stats_dir}/fo.png",
    )
    plotting.plot_violin(
        intv.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Interval (s)",
        filename=f"{summary_stats_dir}/intv.png",
    )
    plotting.plot_violin(
        bc.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Burst Count (Hz)",
        filename=f"{summary_stats_dir}/bc.png",
    )
    plotting.plot_violin(
        amp.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Amplitude (a.u.)",
        filename=f"{summary_stats_dir}/amp.png",
    )
