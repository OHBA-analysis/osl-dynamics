"""Helpful functions related to observation models.

"""


import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference import regularizers
from osl_dynamics.inference.initializers import WeightInitializer


def get_observation_model_parameter(model, layer_name):
    """Get the parameter of an observation model layer.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model.
    layer_name : str
        Name of the layer of the observation model parameter.

    Returns
    -------
    obs_parameter : np.ndarray
        The observation model parameter.
    """
    available_layers = ["means", "covs", "stds", "fcs", "group_means", "group_covs"]
    if layer_name not in available_layers:
        raise ValueError(
            f"Layer name {layer_name} not in available layers {available_layers}."
        )
    obs_layer = model.get_layer(layer_name)
    obs_parameter = obs_layer(1)
    return obs_parameter.numpy()


def set_observation_model_parameter(
    model,
    obs_parameter,
    layer_name,
    update_initializer=True,
    diagonal_covariances=False,
):
    """Set the value of an observation model parameter.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model.
    obs_parameter : np.ndarray
        The value of the observation model parameter to set.
    layer_name : str
        Layer name of the observation model parameter.
    update_initializer : bool, optional
        Whether to update the initializer of the layer.
    diagonal_covariances : bool, optional
        Whether the covariances are diagonal.
        Ignored if :code:`layer_name` is not :code:`"covs"`.
    """
    available_layers = ["means", "covs", "stds", "fcs", "group_means", "group_covs"]
    if layer_name not in available_layers:
        raise ValueError(
            f"Layer name {layer_name} not in available layers {available_layers}."
        )

    obs_parameter = obs_parameter.astype(np.float32)

    if layer_name == "stds" or (layer_name == "covs" and diagonal_covariances):
        if obs_parameter.ndim == 3:
            # Only keep the diagonal as a vector
            obs_parameter = np.diagonal(obs_parameter, axis1=1, axis2=2)

    obs_layer = model.get_layer(layer_name)
    learnable_tensor_layer = obs_layer.layers[0]

    if layer_name not in ["means", "group_means"]:
        obs_parameter = obs_layer.bijector.inverse(obs_parameter)

    learnable_tensor_layer.tensor.assign(obs_parameter)

    if update_initializer:
        learnable_tensor_layer.tensor_initializer = WeightInitializer(obs_parameter)


def set_means_regularizer(model, training_dataset, layer_name="means"):
    """Set the means regularizer based on training data.

    A multivariate normal prior is applied to the mean vectors with
    :code:`mu=0`, :code:`sigma=diag((range/2)**2)`.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model.
    training_dataset : osl_dynamics.data.Data
        The training dataset.
    layer_name : str, optional
        Layer name of the means. Can be :code:`"means"` or
        :code:`"group_means"`.
    """
    n_channels = dtf.get_n_channels(training_dataset)
    range_ = dtf.get_range(training_dataset)

    mu = np.zeros(n_channels, dtype=np.float32)
    sigma = np.diag((range_ / 2) ** 2)

    means_layer = model.get_layer(layer_name)
    learnable_tensor_layer = means_layer.layers[0]
    learnable_tensor_layer.regularizer = regularizers.MultivariateNormal(
        mu,
        sigma,
    )


def set_covariances_regularizer(
    model,
    training_dataset,
    epsilon,
    diagonal=False,
    layer_name="covs",
):
    """Set the covariances regularizer based on training data.

    If config.diagonal_covariances is True, a log-normal prior is applied to
    the diagonal of the covariance matrices with :code:`mu=0`,
    :code:`sigma=sqrt(log(2*range))`. Otherwise, an inverse Wishart prior is
    applied to the covariance matrices with :code:`nu=n_channels-1+0.1`,
    :code:`psi=diag(1/range)`.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model.
    training_dataset : osl_dynamics.data.Data
        The training dataset.
    epsilon : float
        Error added to the covariance matrices.
    diagonal : bool, optional
        Whether the covariances are diagonal.
    layer_name : str, optional
        Layer name of the covariances. Can be :code:`"covs"` or
        :code:`"group_covs"`.
    """
    n_channels = dtf.get_n_channels(training_dataset)
    range_ = dtf.get_range(training_dataset)

    covs_layer = model.get_layer(layer_name)
    if diagonal:
        mu = np.zeros([n_channels], dtype=np.float32)
        sigma = np.sqrt(np.log(2 * range_))
        learnable_tensor_layer = covs_layer.layers[0]
        learnable_tensor_layer.regularizer = regularizers.LogNormal(
            mu,
            sigma,
            epsilon,
        )

    else:
        nu = n_channels - 1 + 0.1
        psi = np.diag(range_)
        learnable_tensor_layer = covs_layer.layers[0]
        learnable_tensor_layer.regularizer = regularizers.InverseWishart(
            nu, psi, epsilon
        )


def set_stds_regularizer(model, training_dataset, epsilon):
    """Set the standard deviations regularizer based on training data.

    A log-normal prior is applied to the standard deviations with :code:`mu=0`,
    :code:`sigma=sqrt(log(2*range))`.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model.
    training_dataset : osl_dynamics.data.Data
        The training dataset.
    epsilon : float
        Error added to the standard deviations.
    """
    n_channels = dtf.get_n_channels(training_dataset)
    range_ = dtf.get_range(training_dataset)

    mu = np.zeros([n_channels], dtype=np.float32)
    sigma = np.sqrt(np.log(2 * range_))

    stds_layer = model.get_layer("stds")
    learnable_tensor_layer = stds_layer.layers[0]
    learnable_tensor_layer.regularizer = regularizers.LogNormal(
        mu,
        sigma,
        epsilon,
    )


def set_fcs_regularizer(model, training_dataset, epsilon):
    """Set the FCS regularizer based on training data.

    A marginal inverse Wishart prior is applied to the functional connectivities
    with :code:`nu=n_channels-1+0.1`.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model.
    training_dataset : osl_dynamics.data.Data
        The training dataset.
    epsilon : float
        Error added to the functional connectivities.
    """
    n_channels = dtf.get_n_channels(training_dataset)

    nu = n_channels - 1 + 0.1

    fcs_layer = model.get_layer("fcs")
    learnable_tensor_layer = fcs_layer.layers[0]
    learnable_tensor_layer.regularizer = regularizers.MarginalInverseWishart(
        nu,
        epsilon,
        n_channels,
    )


def get_subject_embeddings(model):
    """Get the subject embeddings.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`sehmm` or :code:`sedynemo`.

    Returns
    -------
    subject_embeddings : np.ndarray
        The subject embeddings. Shape is (n_subjects, subject_embeddings_dim).
    """
    subject_embeddings_layer = model.get_layer("subject_embeddings")
    n_subjects = subject_embeddings_layer.input_dim
    return subject_embeddings_layer(np.arange(n_subjects)).numpy()


def get_means_mode_embeddings(model):
    """Get the means mode embeddings.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`sehmm` or :code:`sedynemo`.

    Returns
    -------
    means_mode_embeddings : np.ndarray
        The means mode embeddings. Shape is (n_modes, mode_embeddings_dim).
    """
    group_means = get_observation_model_parameter(model, "group_means")
    means_mode_embeddings_layer = model.get_layer("means_mode_embeddings")
    means_mode_embeddings = means_mode_embeddings_layer(group_means)
    return means_mode_embeddings.numpy()


def get_covs_mode_embeddings(model):
    """Get the covariances mode embeddings.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`sehmm` or :code:`sedynemo`.

    Returns
    -------
    covs_mode_embeddings : np.ndarray
        The covariances mode embeddings.
        Shape is (n_modes, mode_embeddings_dim).
    """
    cholesky_bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])
    group_covs = get_observation_model_parameter(model, "group_covs")
    covs_mode_embeddings_layer = model.get_layer("covs_mode_embeddings")
    covs_mode_embeddings = covs_mode_embeddings_layer(
        cholesky_bijector.inverse(group_covs)
    )
    return covs_mode_embeddings.numpy()


def get_mode_embeddings(model, map):
    """Wrapper for getting the mode embeddings for the means and covariances."""
    if map == "means":
        return get_means_mode_embeddings(model)
    elif map == "covs":
        return get_covs_mode_embeddings(model)
    else:
        raise ValueError("map must be either 'means' or 'covs'")


def get_concatenated_embeddings(model, map, subject_embeddings=None):
    """Get the concatenated embeddings.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`sehmm` or :code:`sedynemo`.
    map : str
        The map to use. Either :code:`"means"` or :code:`"covs"`.
    subject_embeddings : np.ndarray, optional
        Input subject embeddings. If :code:`None`, they are retrieved from
        the model. Shape is (n_subjects, subject_embeddings_dim).

    Returns
    -------
    concat_embeddings : np.ndarray
        The concatenated embeddings. Shape is (n_subjects, n_modes,
        subject_embeddings_dim + mode_embeddings_dim).
    """
    if subject_embeddings is None:
        subject_embeddings = get_subject_embeddings(model)
    if map == "means":
        mode_embeddings = get_means_mode_embeddings(model)
        concat_embeddings_layer = model.get_layer("means_concat_embeddings")
    elif map == "covs":
        mode_embeddings = get_covs_mode_embeddings(model)
        concat_embeddings_layer = model.get_layer("covs_concat_embeddings")
    else:
        raise ValueError("map must be either 'means' or 'covs'")
    concat_embeddings = concat_embeddings_layer([subject_embeddings, mode_embeddings])
    return concat_embeddings.numpy()


def get_means_dev_mag_parameters(model):
    """Get the means deviation magnitude parameters.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`sehmm` or :code:`sedynemo`.

    Returns
    -------
    means_dev_mag_inf_alpha : np.ndarray
        The means deviation magnitude alpha parameters.
        Shape is (n_subjects, n_modes, 1).
    means_dev_mag_inf_beta : np.ndarray
        The means deviation magnitude beta parameters.
        Shape is (n_subjects, n_modes, 1).
    """
    means_dev_mag_inf_alpha_input_layer = model.get_layer(
        "means_dev_mag_inf_alpha_input"
    )
    means_dev_mag_inf_alpha_layer = model.get_layer("means_dev_mag_inf_alpha")
    means_dev_mag_inf_beta_input_layer = model.get_layer("means_dev_mag_inf_beta_input")
    means_dev_mag_inf_beta_layer = model.get_layer("means_dev_mag_inf_beta")

    means_dev_mag_inf_alpha_input = means_dev_mag_inf_alpha_input_layer(1)
    means_dev_mag_inf_alpha = means_dev_mag_inf_alpha_layer(
        means_dev_mag_inf_alpha_input
    )

    means_dev_mag_inf_beta_input = means_dev_mag_inf_beta_input_layer(1)
    means_dev_mag_inf_beta = means_dev_mag_inf_beta_layer(means_dev_mag_inf_beta_input)
    return means_dev_mag_inf_alpha.numpy(), means_dev_mag_inf_beta.numpy()


def get_covs_dev_mag_parameters(model):
    """Get the covariances deviation magnitude parameters.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`sehmm` or :code:`sedynemo`.

    Returns
    -------
    covs_dev_mag_inf_alpha : np.ndarray
        The covariances deviation magnitude alpha parameters.
        Shape is (n_subjects, n_modes, 1).
    covs_dev_mag_inf_beta : np.ndarray
        The covariances deviation magnitude beta parameters.
        Shape is (n_subjects, n_modes, 1).
    """
    covs_dev_mag_inf_alpha_input_layer = model.get_layer("covs_dev_mag_inf_alpha_input")
    covs_dev_mag_inf_alpha_layer = model.get_layer("covs_dev_mag_inf_alpha")
    covs_dev_mag_inf_beta_input_layer = model.get_layer("covs_dev_mag_inf_beta_input")
    covs_dev_mag_inf_beta_layer = model.get_layer("covs_dev_mag_inf_beta")

    covs_dev_mag_inf_alpha_input = covs_dev_mag_inf_alpha_input_layer(1)
    covs_dev_mag_inf_alpha = covs_dev_mag_inf_alpha_layer(covs_dev_mag_inf_alpha_input)

    covs_dev_mag_inf_beta_input = covs_dev_mag_inf_beta_input_layer(1)
    covs_dev_mag_inf_beta = covs_dev_mag_inf_beta_layer(covs_dev_mag_inf_beta_input)
    return covs_dev_mag_inf_alpha.numpy(), covs_dev_mag_inf_beta.numpy()


def get_dev_mag_parameters(model, map):
    """Wrapper for getting the deviance magnitude parameters for the means
    and covariances."""
    if map == "means":
        return get_means_dev_mag_parameters(model)
    elif map == "covs":
        return get_covs_dev_mag_parameters(model)
    else:
        raise ValueError("map must be either 'means' or 'covs'")


def get_dev_mag(model, map):
    """Getting the deviance magnitude.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`sehmm` or :code:`sedynemo`.
    map : str
        The map. Must be either :code:`'means'` or :code:`'covs'`.

    Returns
    -------
    dev_mag : np.ndarray
        The deviance magnitude. Shape is (n_subjects, n_modes, 1).
    """
    if map == "means":
        alpha, beta = get_means_dev_mag_parameters(model)
        dev_mag_layer = model.get_layer("means_dev_mag")
    elif map == "covs":
        alpha, beta = get_covs_dev_mag_parameters(model)
        dev_mag_layer = model.get_layer("covs_dev_mag")
    else:
        raise ValueError("map must be either 'means' or 'covs'")
    dev_mag = dev_mag_layer([alpha, beta])
    return dev_mag.numpy()


def get_dev_map(model, map, subject_embeddings=None):
    """Get the deviance map.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`sehmm` or :code:`sedynemo`.
    map : str
        The map to use. Either :code:`"means"` or :code:`"covs"`.
    subject_embeddings : np.ndarray, optional
        Input subject embeddings. If :code:`None`, they are retrieved from
        the model. Shape is (n_subjects, subject_embeddings_dim).

    Returns
    -------
    dev_map : np.ndarray
        The deviance map.
        If :code:`map="means"`, shape is (n_subjects, n_modes, n_channels).
        If :code:`map="covs"`, shape is (n_subjects, n_modes,
        n_channels * (n_channels + 1) // 2).
    """
    concat_embeddings = get_concatenated_embeddings(
        model,
        map,
        subject_embeddings,
    )
    if map == "means":
        dev_map_input_layer = model.get_layer("means_dev_map_input")
        dev_map_layer = model.get_layer("means_dev_map")
        norm_dev_map_layer = model.get_layer("norm_means_dev_map")
    elif map == "covs":
        dev_map_input_layer = model.get_layer("covs_dev_map_input")
        dev_map_layer = model.get_layer("covs_dev_map")
        norm_dev_map_layer = model.get_layer("norm_covs_dev_map")
    else:
        raise ValueError("map must be either 'means' or 'covs'")
    dev_map_input = dev_map_input_layer(concat_embeddings)
    dev_map = dev_map_layer(dev_map_input)
    norm_dev_map = norm_dev_map_layer(dev_map)
    return norm_dev_map.numpy()


def get_subject_dev(
    model,
    learn_means,
    learn_covariances,
    subject_embeddings=None,
    n_neighbours=2,
):
    """Get the subject deviation.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`sehmm` or :code:`sedynemo`.
    learn_means : bool
        Whether the mean is learnt.
    learn_covariances : bool
        Whether the covariances are learnt.
    subject_embeddings : np.ndarray, optional
        Input subject embeddings. Shape is (n_subjects, subject_embeddings_dim).
        If :code:`None`, then the subject embeddings are retrieved from the
        model.
    n_neighbours : int, optional
        The number of nearest neighbours if :code:`subject_embedding` is not
        :code:`None`.

    Returns
    -------
    means_dev : np.ndarray
        The means deviation. Shape is (n_subjects, n_modes, n_channels).
    covs_dev : np.ndarray
        The covariances deviation.
        Shape is (n_subjects, n_modes, n_channels * (n_channels + 1) // 2).
    """
    means_dev_layer = model.get_layer("means_dev")
    covs_dev_layer = model.get_layer("covs_dev")

    if subject_embeddings is not None:
        nearest_neighbours = get_nearest_neighbours(
            model, subject_embeddings, n_neighbours
        )

    if learn_means:
        means_dev_mag = get_dev_mag(model, "means")
        if subject_embeddings is not None:
            means_dev_mag = np.mean(
                tf.gather(means_dev_mag, nearest_neighbours, axis=0),
                axis=1,
            )
        means_dev_map = get_dev_map(
            model=model, map="means", subject_embeddings=subject_embeddings
        )
        means_dev = means_dev_layer([means_dev_mag, means_dev_map])
    else:
        means_dev = means_dev_layer(1)

    if learn_covariances:
        covs_dev_mag = get_dev_mag(model, "covs")
        if subject_embeddings is not None:
            covs_dev_mag = np.mean(
                tf.gather(covs_dev_mag, nearest_neighbours, axis=0),
                axis=1,
            )
        covs_dev_map = get_dev_map(
            model=model, map="covs", subject_embeddings=subject_embeddings
        )
        covs_dev = covs_dev_layer([covs_dev_mag, covs_dev_map])
    else:
        covs_dev = covs_dev_layer(1)

    return means_dev.numpy(), covs_dev.numpy()


def get_subject_means_covariances(
    model,
    learn_means,
    learn_covariances,
    subject_embeddings=None,
    n_neighbours=2,
):
    """Get the subject means and covariances.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`sehmm` or :code:`sedynemo`.
    learn_means : bool
        Whether the mean is learnt.
    learn_covariances : bool
        Whether the covariances are learnt.
    subject_embeddings : np.ndarray, optional
        Input subject embeddings. Shape is (n_subjects, subject_embeddings_dim).
        If None, then the subject embeddings are retrieved from the model.
    n_neighbours : int, optional
        The number of nearest neighbours if :code:`subject_embedding` is not
        :code:`None`.

    Returns
    -------
    mu : np.ndarray
        The subject_means. Shape is (n_subjects, n_modes, n_channels).
    D : np.ndarray
        The subject_covariances.
        Shape is (n_subjects, n_modes, n_channels, n_channels).
    """
    group_means = get_observation_model_parameter(model, "group_means")
    group_covs = get_observation_model_parameter(model, "group_covs")
    means_dev, covs_dev = get_subject_dev(
        model, learn_means, learn_covariances, subject_embeddings, n_neighbours
    )

    subject_means_layer = model.get_layer("subject_means")
    subject_covs_layer = model.get_layer("subject_covs")

    mu = subject_means_layer([group_means, means_dev])
    D = subject_covs_layer([group_covs, covs_dev])
    return mu.numpy(), D.numpy()


def get_nearest_neighbours(model, subject_embeddings, n_neighbours):
    """Get the indices of the nearest neighours in the subject embedding space.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`sehmm` or :code:`sedynemo`.
    subject_embeddings : np.ndarray
        Input subject embeddings. Shape is (n_subjects, subject_embeddings_dim).
    n_neighbours : int
        The number of nearest neighbours.

    Returns
    -------
    nearest_neighbours : np.ndarray
        The indices of the nearest neighbours.
        Shape is (n_subjects, n_neighbours).
    """
    model_subject_embeddings = get_subject_embeddings(model)
    distances = np.linalg.norm(
        np.expand_dims(subject_embeddings, axis=1)
        - np.expand_dims(model_subject_embeddings, axis=0),
        axis=-1,
    )

    # Sort distances and get indices of nearest neighbours
    sorted_distances = np.argsort(distances, axis=1)
    nearest_neighbours = sorted_distances[:, :n_neighbours]
    return nearest_neighbours
