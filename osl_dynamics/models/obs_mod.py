"""Helpful functions related to observation models.

"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference import regularizers
from osl_dynamics.inference.initializers import (
    WeightInitializer,
    RandomWeightInitializer,
)


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
    available_layers = [
        "means",
        "covs",
        "stds",
        "fcs",
        "group_means",
        "group_covs",
        "log_rates",
    ]
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
    available_layers = [
        "means",
        "covs",
        "stds",
        "fcs",
        "group_means",
        "group_covs",
        "log_rates",
    ]
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

    if layer_name not in ["means", "group_means", "log_rates"]:
        obs_parameter = obs_layer.bijector.inverse(obs_parameter)

    learnable_tensor_layer.tensor.assign(obs_parameter)

    if update_initializer:
        learnable_tensor_layer.tensor_initializer = WeightInitializer(obs_parameter)


def set_dev_parameters_initializer(
    model, training_dataset, learn_means, learn_covariances
):
    """Set the deviance parameters initializer based on training data.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    training_dataset : osl_dynamics.data.Data
        The training dataset.
    learn_means : bool
        Whether the mean is learnt.
    learn_covariances : bool
        Whether the covariances are learnt.
    """
    time_series = training_dataset.time_series()
    n_channels = training_dataset.n_channels
    if isinstance(time_series, np.ndarray):
        time_series = [time_series]
    if learn_means:
        static_means = np.array([np.mean(t, axis=0) for t in time_series])
        static_means_dev = np.abs(static_means - np.mean(static_means, axis=0))
        static_means_dev_mean = np.mean(static_means_dev, axis=1)
        static_means_dev_var = static_means_dev_mean / 5

        means_alpha = tfp.math.softplus_inverse(
            np.square(static_means_dev_mean) / static_means_dev_var
        )[..., None, None]
        means_beta = tfp.math.softplus_inverse(
            static_means_dev_mean / static_means_dev_var
        )[..., None, None]

        means_alpha_layer = model.get_layer("means_dev_mag_inf_alpha_input")
        means_beta_layer = model.get_layer("means_dev_mag_inf_beta_input")

        means_alpha_layer.tensor_initializer = RandomWeightInitializer(means_alpha, 0.1)
        means_beta_layer.tensor_initializer = RandomWeightInitializer(means_beta, 0.1)

    if learn_covariances:
        static_cov = np.array([np.cov(t, rowvar=False) for t in time_series])
        static_cov_chol = np.linalg.cholesky(static_cov)[
            :,
            np.tril_indices(n_channels)[0],
            np.tril_indices(n_channels)[1],
        ]
        static_cov_chol_dev = np.abs(static_cov_chol - np.mean(static_cov_chol, axis=0))
        static_cov_chol_dev_mean = np.mean(static_cov_chol_dev, axis=1)
        static_cov_chol_dev_var = static_cov_chol_dev_mean / 5

        covs_alpha = tfp.math.softplus_inverse(
            np.square(static_cov_chol_dev_mean) / static_cov_chol_dev_var
        )[..., None, None]
        covs_beta = tfp.math.softplus_inverse(
            static_cov_chol_dev_mean / static_cov_chol_dev_var
        )[..., None, None]

        covs_alpha_layer = model.get_layer("covs_dev_mag_inf_alpha_input")
        covs_beta_layer = model.get_layer("covs_dev_mag_inf_beta_input")

        covs_alpha_layer.tensor_initializer = RandomWeightInitializer(covs_alpha, 0.1)
        covs_beta_layer.tensor_initializer = RandomWeightInitializer(covs_beta, 0.1)


def set_embeddings_initializer(model, embeddings):
    """Set the embeddings initializer.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    embeddings : np.ndarray
        The embeddings. Shape is (n_sessions, embeddings_dim).
    """
    embeddings_layer = model.get_layer("embeddings")
    embeddings_layer.embeddings_initializer = WeightInitializer(embeddings)


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
    :code:`psi=diag(1/range)`.x

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


def get_embeddings(model):
    """Get the embeddings.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.

    Returns
    -------
    embeddings : np.ndarray
        The embeddings. Shape is (n_sessions, embeddings_dim).
    """
    embeddings_layer = model.get_layer("embeddings")
    n_sessions = embeddings_layer.input_dim
    return embeddings_layer(np.arange(n_sessions)).numpy()


def get_means_spatial_embeddings(model):
    """Get the means spatial embeddings.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.

    Returns
    -------
    means_spatial_embeddings : np.ndarray
        The means spatial embeddings. Shape is (n_states, spatial_embeddings_dim).
    """
    group_means = get_observation_model_parameter(model, "group_means")
    means_spatial_embeddings_layer = model.get_layer("means_spatial_embeddings")
    means_spatial_embeddings = means_spatial_embeddings_layer(group_means)
    return means_spatial_embeddings.numpy()


def get_covs_spatial_embeddings(model):
    """Get the covariances spatial embeddings.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.

    Returns
    -------
    covs_spatial_embeddings : np.ndarray
        The covariances spatial embeddings.
        Shape is (n_states, spatial_embeddings_dim).
    """
    cholesky_bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])
    group_covs = get_observation_model_parameter(model, "group_covs")
    covs_spatial_embeddings_layer = model.get_layer("covs_spatial_embeddings")
    covs_spatial_embeddings = covs_spatial_embeddings_layer(
        cholesky_bijector.inverse(group_covs)
    )
    return covs_spatial_embeddings.numpy()


def get_spatial_embeddings(model, map):
    """Wrapper for getting the spatial embeddings for the means and covariances."""
    if map == "means":
        return get_means_spatial_embeddings(model)
    elif map == "covs":
        return get_covs_spatial_embeddings(model)
    else:
        raise ValueError("map must be either 'means' or 'covs'")


def get_concatenated_embeddings(model, map, embeddings=None):
    """Get the concatenated embeddings.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    map : str
        The map to use. Either :code:`"means"` or :code:`"covs"`.
    embeddings : np.ndarray, optional
        Input embeddings. If :code:`None`, they are retrieved from
        the model. Shape is (n_sessions, embeddings_dim).

    Returns
    -------
    concat_embeddings : np.ndarray
        The concatenated embeddings. Shape is (n_sessions, n_states,
        embeddings_dim + spatial_embeddings_dim).
    """
    if embeddings is None:
        embeddings = get_embeddings(model)
    if map == "means":
        spatial_embeddings = get_means_spatial_embeddings(model)
        concat_embeddings_layer = model.get_layer("means_concat_embeddings")
    elif map == "covs":
        spatial_embeddings = get_covs_spatial_embeddings(model)
        concat_embeddings_layer = model.get_layer("covs_concat_embeddings")
    else:
        raise ValueError("map must be either 'means' or 'covs'")
    concat_embeddings = concat_embeddings_layer([embeddings, spatial_embeddings])
    return concat_embeddings.numpy()


def get_means_dev_mag_parameters(model):
    """Get the means deviation magnitude parameters.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.

    Returns
    -------
    means_dev_mag_inf_alpha : np.ndarray
        The means deviation magnitude alpha parameters.
        Shape is (n_sessions, n_states, 1).
    means_dev_mag_inf_beta : np.ndarray
        The means deviation magnitude beta parameters.
        Shape is (n_sessions, n_states, 1).
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
        The model. * must be :code:`hive` or :code:`dive`.

    Returns
    -------
    covs_dev_mag_inf_alpha : np.ndarray
        The covariances deviation magnitude alpha parameters.
        Shape is (n_sessions, n_states, 1).
    covs_dev_mag_inf_beta : np.ndarray
        The covariances deviation magnitude beta parameters.
        Shape is (n_sessions, n_states, 1).
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
        The model. * must be :code:`hive` or :code:`dive`.
    map : str
        The map. Must be either :code:`'means'` or :code:`'covs'`.

    Returns
    -------
    dev_mag : np.ndarray
        The deviance magnitude. Shape is (n_sessions, n_states, 1).
    """
    if map == "means":
        alpha, beta = get_means_dev_mag_parameters(model)
        dev_mag_layer = model.get_layer("means_dev_mag")
    elif map == "covs":
        alpha, beta = get_covs_dev_mag_parameters(model)
        dev_mag_layer = model.get_layer("covs_dev_mag")
    else:
        raise ValueError("map must be either 'means' or 'covs'")
    n_sessions = alpha.shape[0]
    dev_mag = dev_mag_layer([alpha, beta, np.arange(n_sessions)[:, None]])
    return dev_mag.numpy()


def get_dev_map(model, map, embeddings=None):
    """Get the deviance map.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    map : str
        The map to use. Either :code:`"means"` or :code:`"covs"`.
    embeddings : np.ndarray, optional
        Input embeddings. If :code:`None`, they are retrieved from
        the model. Shape is (n_sessions, embeddings_dim).

    Returns
    -------
    dev_map : np.ndarray
        The deviance map.
        If :code:`map="means"`, shape is (n_sessions, n_states, n_channels).
        If :code:`map="covs"`, shape is (n_sessions, n_states,
        n_channels * (n_channels + 1) // 2).
    """
    concat_embeddings = get_concatenated_embeddings(
        model,
        map,
        embeddings,
    )
    if map == "means":
        dev_decoder_layer = model.get_layer("means_dev_decoder")
        dev_map_layer = model.get_layer("means_dev_map")
        norm_dev_map_layer = model.get_layer("norm_means_dev_map")
    elif map == "covs":
        dev_decoder_layer = model.get_layer("covs_dev_decoder")
        dev_map_layer = model.get_layer("covs_dev_map")
        norm_dev_map_layer = model.get_layer("norm_covs_dev_map")
    else:
        raise ValueError("map must be either 'means' or 'covs'")
    dev_decoder = dev_decoder_layer(concat_embeddings)
    dev_map = dev_map_layer(dev_decoder)
    norm_dev_map = norm_dev_map_layer(dev_map)
    return norm_dev_map.numpy()


def get_session_dev(
    model,
    learn_means,
    learn_covariances,
    embeddings=None,
    n_neighbours=2,
):
    """Get the session deviation.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    learn_means : bool
        Whether the mean is learnt.
    learn_covariances : bool
        Whether the covariances are learnt.
    embeddings : np.ndarray, optional
        Input embeddings. Shape is (n_sessions, embeddings_dim).
        If :code:`None`, then the embeddings are retrieved from the model.
    n_neighbours : int, optional
        The number of nearest neighbours if :code:`embedding` is not
        :code:`None`.

    Returns
    -------
    means_dev : np.ndarray
        The means deviation. Shape is (n_sessions, n_states, n_channels).
    covs_dev : np.ndarray
        The covariances deviation.
        Shape is (n_sessions, n_states, n_channels * (n_channels + 1) // 2).
    """
    means_dev_layer = model.get_layer("means_dev")
    covs_dev_layer = model.get_layer("covs_dev")

    if embeddings is not None:
        nearest_neighbours = get_nearest_neighbours(model, embeddings, n_neighbours)

    if learn_means:
        means_dev_mag = get_dev_mag(model, "means")
        if embeddings is not None:
            means_dev_mag = np.mean(
                tf.gather(means_dev_mag, nearest_neighbours, axis=0),
                axis=1,
            )
        means_dev_map = get_dev_map(model=model, map="means", embeddings=embeddings)
        means_dev = means_dev_layer([means_dev_mag, means_dev_map])
    else:
        means_dev = means_dev_layer(1)

    if learn_covariances:
        covs_dev_mag = get_dev_mag(model, "covs")
        if embeddings is not None:
            covs_dev_mag = np.mean(
                tf.gather(covs_dev_mag, nearest_neighbours, axis=0),
                axis=1,
            )
        covs_dev_map = get_dev_map(model=model, map="covs", embeddings=embeddings)
        covs_dev = covs_dev_layer([covs_dev_mag, covs_dev_map])
    else:
        covs_dev = covs_dev_layer(1)

    return means_dev.numpy(), covs_dev.numpy()


def get_session_means_covariances(
    model,
    learn_means,
    learn_covariances,
    embeddings=None,
    n_neighbours=2,
):
    """Get the session means and covariances.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    learn_means : bool
        Whether the mean is learnt.
    learn_covariances : bool
        Whether the covariances are learnt.
    embeddings : np.ndarray, optional
        Input embeddings. Shape is (n_sessions, embeddings_dim).
        If None, then the embeddings are retrieved from the model.
    n_neighbours : int, optional
        The number of nearest neighbours if :code:`+embedding` is not
        :code:`None`.

    Returns
    -------
    mu : np.ndarray
        The session means. Shape is (n_sessions, n_states, n_channels).
    D : np.ndarray
        The session covariances.
        Shape is (n_sessions, n_states, n_channels, n_channels).
    """
    group_means = get_observation_model_parameter(model, "group_means")
    group_covs = get_observation_model_parameter(model, "group_covs")
    means_dev, covs_dev = get_session_dev(
        model, learn_means, learn_covariances, embeddings, n_neighbours
    )

    session_means_layer = model.get_layer("session_means")
    session_covs_layer = model.get_layer("session_covs")

    mu = session_means_layer([group_means, means_dev])
    D = session_covs_layer([group_covs, covs_dev])
    return mu.numpy(), D.numpy()


def get_nearest_neighbours(model, embeddings, n_neighbours):
    """Get the indices of the nearest neighours in the embedding space.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    embeddings : np.ndarray
        Input embeddings. Shape is (n_sessions, embeddings_dim).
    n_neighbours : int
        The number of nearest neighbours.

    Returns
    -------
    nearest_neighbours : np.ndarray
        The indices of the nearest neighbours.
        Shape is (n_sessions, n_neighbours).
    """
    model_embeddings = get_embeddings(model)
    distances = np.linalg.norm(
        np.expand_dims(embeddings, axis=1) - np.expand_dims(model_embeddings, axis=0),
        axis=-1,
    )

    # Sort distances and get indices of nearest neighbours
    sorted_distances = np.argsort(distances, axis=1)
    nearest_neighbours = sorted_distances[:, :n_neighbours]
    return nearest_neighbours
