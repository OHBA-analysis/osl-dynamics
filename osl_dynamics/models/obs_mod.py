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
        "corrs",
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
        "corrs",
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
    training_dataset : tf.data.Dataset
        The training dataset.
    learn_means : bool
        Whether the mean is learnt.
    learn_covariances : bool
        Whether the covariances are learnt.
    """
    time_series = []
    for d in training_dataset:
        subject_data = []
        for batch in d:
            subject_data.append(np.concatenate(batch["data"]))
        time_series.append(np.concatenate(subject_data))

    n_channels = time_series[0].shape[1]
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


def set_embeddings_initializer(model, initial_embeddings):
    """Set the embeddings initializer.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    initial_embeddings : dict
        The initial_embeddings dictionary. {name: value}
    """

    # Helper function to set a single layer's initializer
    def _set_embeddings_initializer(layer_name, value):
        embedding_layer = model.get_layer(layer_name)
        embedding_layer.embedding_layer.embeddings_initializer = WeightInitializer(
            value
        )

    for k, v in initial_embeddings.items():
        _set_embeddings_initializer(f"{k}_embeddings", v)


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


def set_corrs_regularizer(model, training_dataset, epsilon):
    """Set the correlations regularizer based on training data.

    A marginal inverse Wishart prior is applied to the correlations
    with :code:`nu=n_channels-1+0.1`.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model.
    training_dataset : osl_dynamics.data.Data
        The training dataset.
    epsilon : float
        Error added to the correlations.
    """
    n_channels = dtf.get_n_channels(training_dataset)

    nu = n_channels - 1 + 0.1

    corrs_layer = model.get_layer("corrs")
    learnable_tensor_layer = corrs_layer.layers[0]
    learnable_tensor_layer.regularizer = regularizers.MarginalInverseWishart(
        nu,
        epsilon,
        n_channels,
    )


def get_embedding_weights(model, session_labels):
    """Get the weights of the embedding layers.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    session_labels : List[osl_dynamics.data.SessionLabel]
        List of session labels.

    Returns
    -------
    embedding_weights : dict
        The weights of the embedding layers.
    """
    embedding_weights = dict()
    for session_label in session_labels:
        label_name = session_label.name
        label_type = session_label.label_type
        embeddings_layer = model.get_layer(f"{label_name}_embeddings")
        if label_type == "categorical":
            embedding_weights[label_name] = embeddings_layer.embeddings.numpy()
        else:
            embedding_weights[label_name] = [
                embeddings_layer.kernel.numpy(),
                embeddings_layer.bias.numpy(),
            ]

    return embedding_weights


def get_session_embeddings(model, session_labels):
    """Get the embeddings for each session.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    session_labels : List[osl_dynamics.data.SessionLabel]
        List of session labels.

    Returns
    -------
    embeddings : dict
        The embeddings for each session label.
    """
    embeddings = dict()
    for session_label in session_labels:
        label_name = session_label.name
        label_values = session_label.values
        embeddings_layer = model.get_layer(f"{label_name}_embeddings")
        embeddings[label_name] = embeddings_layer(label_values)

    return embeddings


def get_summed_embeddings(model, session_labels):
    """Get the summed embeddings for each session.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    session_labels : List[osl_dynamics.data.SessionLabel]
        List of session labels.

    Returns
    -------
    summed_embeddings : np.ndarray
        The summed embeddings. Shape is (n_sessions, embeddings_dim).
    """
    embeddings = get_session_embeddings(model, session_labels)
    summed_embeddings = 0
    for _, embedding in embeddings.items():
        summed_embeddings += embedding
    return summed_embeddings.numpy()


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


def get_spatial_embeddings(model, param):
    """Wrapper for getting the spatial embeddings for the means and covariances."""
    if param == "means":
        return get_means_spatial_embeddings(model)
    elif param == "covs":
        return get_covs_spatial_embeddings(model)
    else:
        raise ValueError("param must be either 'means' or 'covs'")


def get_concatenated_embeddings(model, param, session_labels):
    """Get the concatenated embeddings.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    param : str
        The param to use. Either :code:`"means"` or :code:`"covs"`.
    embeddings : np.ndarray, optional
        Input embeddings. If :code:`None`, they are retrieved from
        the model. Shape is (n_sessions, embeddings_dim).

    Returns
    -------
    concat_embeddings : np.ndarray
        The concatenated embeddings. Shape is (n_sessions, n_states,
        embeddings_dim + spatial_embeddings_dim).
    """
    embeddings = get_summed_embeddings(model, session_labels)
    if param == "means":
        spatial_embeddings = get_means_spatial_embeddings(model)
        concat_embeddings_layer = model.get_layer("means_concat_embeddings")
    elif param == "covs":
        spatial_embeddings = get_covs_spatial_embeddings(model)
        concat_embeddings_layer = model.get_layer("covs_concat_embeddings")
    else:
        raise ValueError("param must be either 'means' or 'covs'")
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


def get_dev_mag_parameters(model, param):
    """Wrapper for getting the deviance magnitude parameters for the means
    and covariances."""
    if param == "means":
        return get_means_dev_mag_parameters(model)
    elif param == "covs":
        return get_covs_dev_mag_parameters(model)
    else:
        raise ValueError("param must be either 'means' or 'covs'")


def get_dev_mag(model, param):
    """Getting the deviance magnitude.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    param : str
        The param. Must be either :code:`'means'` or :code:`'covs'`.

    Returns
    -------
    dev_mag : np.ndarray
        The deviance magnitude. Shape is (n_sessions, n_states, 1).
    """
    if param == "means":
        alpha, beta = get_means_dev_mag_parameters(model)
        dev_mag_layer = model.get_layer("means_dev_mag")
    elif param == "covs":
        alpha, beta = get_covs_dev_mag_parameters(model)
        dev_mag_layer = model.get_layer("covs_dev_mag")
    else:
        raise ValueError("param must be either 'means' or 'covs'")

    n_sessions = alpha.shape[0]
    dev_mag = dev_mag_layer([alpha, beta, np.arange(n_sessions)[..., None]])
    return dev_mag.numpy()


def get_dev_map(model, param, session_labels):
    """Get the deviance map.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    param : str
        The param to use. Either :code:`"means"` or :code:`"covs"`.
    embeddings : np.ndarray, optional
        Input embeddings. If :code:`None`, they are retrieved from
        the model. Shape is (n_sessions, embeddings_dim).

    Returns
    -------
    dev_map : np.ndarray
        The deviance map.
        If :code:`param="means"`, shape is (n_sessions, n_states, n_channels).
        If :code:`param="covs"`, shape is (n_sessions, n_states,
        n_channels * (n_channels + 1) // 2).
    """
    concat_embeddings = get_concatenated_embeddings(model, param, session_labels)
    if param == "means":
        dev_decoder_layer = model.get_layer("means_dev_decoder")
        dev_map_layer = model.get_layer("means_dev_map")
        norm_dev_map_layer = model.get_layer("norm_means_dev_map")
    elif param == "covs":
        dev_decoder_layer = model.get_layer("covs_dev_decoder")
        dev_map_layer = model.get_layer("covs_dev_map")
        norm_dev_map_layer = model.get_layer("norm_covs_dev_map")
    else:
        raise ValueError("param must be either 'means' or 'covs'")
    dev_decoder = dev_decoder_layer(concat_embeddings)
    dev_map = dev_map_layer(dev_decoder)
    norm_dev_map = norm_dev_map_layer(dev_map)
    return norm_dev_map.numpy()


def get_session_dev(
    model,
    learn_means,
    learn_covariances,
    session_labels,
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
    session_labels : List[osl_dynamics.data.SessionLabel]
        List of session labels.

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

    if learn_means:
        means_dev_mag = get_dev_mag(model, "means")
        means_dev_map = get_dev_map(model, "means", session_labels)
        means_dev = means_dev_layer([means_dev_mag, means_dev_map])
    else:
        means_dev = means_dev_layer(1)

    if learn_covariances:
        covs_dev_mag = get_dev_mag(model, "covs")
        covs_dev_map = get_dev_map(model, "covs", session_labels)
        covs_dev = covs_dev_layer([covs_dev_mag, covs_dev_map])
    else:
        covs_dev = covs_dev_layer(1)

    return means_dev.numpy(), covs_dev.numpy()


def get_session_means_covariances(
    model,
    learn_means,
    learn_covariances,
    session_labels,
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
    session_labels : List[osl_dynamics.data.SessionLabel]
        List of session labels.

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
        model, learn_means, learn_covariances, session_labels
    )

    session_means_layer = model.get_layer("session_means")
    session_covs_layer = model.get_layer("session_covs")

    mu = session_means_layer([group_means, means_dev])
    D = session_covs_layer([group_covs, covs_dev])
    return mu.numpy(), D.numpy()


def generate_covariances(model, session_labels):
    """Generate covariances from the generative model.

    Parameters
    ----------
    model : osl_dynamics.models.*.Model.model
        The model. * must be :code:`hive` or :code:`dive`.
    session_labels : List[osl_dynamics.data.SessionLabel]
        List of session labels.

    Returns
    -------
    covs : np.ndarray
        The covariances. Shape is (n_sessions, n_states, n_channels, n_channels)
        or (n_states, n_channels, n_channels).
    """

    dev_map = get_dev_map(model, "covs", session_labels)
    concat_embeddings = get_concatenated_embeddings(model, "covs", session_labels)

    covs_dev_decoder_layer = model.get_layer("covs_dev_decoder")
    dev_mag_mod_layer = model.get_layer("covs_dev_mag_mod_beta")
    dev_mag_mod = 1 / dev_mag_mod_layer(covs_dev_decoder_layer(concat_embeddings))

    # Generate deviations
    dev_layer = model.get_layer("covs_dev")
    dev = dev_layer([dev_mag_mod, dev_map])

    # Generate covariances
    group_covs = get_observation_model_parameter(model, "group_covs")
    covs_layer = model.get_layer("session_covs")
    covs = np.squeeze(covs_layer([group_covs, dev]).numpy())

    return covs
