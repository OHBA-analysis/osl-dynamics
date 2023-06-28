"""Helpful functions related to observation models.

"""


import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb

import osl_dynamics.data.tf as dtf
from osl_dynamics.inference import regularizers
from osl_dynamics.inference.initializers import WeightInitializer
from osl_dynamics.inference.layers import add_epsilon


def get_means(model, layer_name="means"):
    means_layer = model.get_layer(layer_name)
    means = means_layer(1)
    return means.numpy()


def get_covariances(model, layer_name="covs"):
    covs_layer = model.get_layer(layer_name)
    covs = covs_layer(1)
    return covs.numpy()


def get_means_covariances(model):
    means = get_means(model)
    covs = get_covariances(model)
    return means, covs


def set_means(model, means, update_initializer=True, layer_name="means"):
    means = means.astype(np.float32)
    means_layer = model.get_layer(layer_name)
    learnable_tensor_layer = means_layer.layers[0]
    learnable_tensor_layer.tensor.assign(means)
    if update_initializer:
        learnable_tensor_layer.tensor_initializer = WeightInitializer(means)


def set_covariances(
    model, covariances, diagonal=False, update_initializer=True, layer_name="covs"
):
    covariances = covariances.astype(np.float32)
    covs_layer = model.get_layer(layer_name)
    learnable_tensor_layer = covs_layer.layers[0]

    if diagonal:
        if covariances.ndim == 3:
            # Only keep the diagonal as a vector
            covariances = np.diagonal(covariances, axis1=1, axis2=2)
        diagonals = covs_layer.bijector.inverse(covariances)
        learnable_tensor_layer.tensor.assign(diagonals)
        if update_initializer:
            learnable_tensor_layer.tensor_initializer = WeightInitializer(diagonals)

    else:
        flattened_cholesky_factors = covs_layer.bijector.inverse(covariances)
        learnable_tensor_layer.tensor.assign(flattened_cholesky_factors)
        if update_initializer:
            learnable_tensor_layer.tensor_initializer = WeightInitializer(
                flattened_cholesky_factors
            )


def set_means_regularizer(model, training_dataset, layer_name="means"):
    n_batches = dtf.get_n_batches(training_dataset)
    n_channels = dtf.get_n_channels(training_dataset)
    range_ = dtf.get_range(training_dataset)

    mu = np.zeros(n_channels, dtype=np.float32)
    sigma = np.diag((range_ / 2) ** 2)

    means_layer = model.get_layer(layer_name)
    learnable_tensor_layer = means_layer.layers[0]
    learnable_tensor_layer.regularizer = regularizers.MultivariateNormal(
        mu, sigma, n_batches
    )


def set_covariances_regularizer(
    model,
    training_dataset,
    epsilon,
    diagonal=False,
    layer_name="covs",
):
    n_batches = dtf.get_n_batches(training_dataset)
    n_channels = dtf.get_n_channels(training_dataset)
    range_ = dtf.get_range(training_dataset)

    covs_layer = model.get_layer(layer_name)
    if diagonal:
        mu = np.zeros([n_channels], dtype=np.float32)
        sigma = np.sqrt(np.log(2 * range_))
        learnable_tensor_layer = covs_layer.layers[0]
        learnable_tensor_layer.regularizer = regularizers.LogNormal(
            mu, sigma, epsilon, n_batches
        )

    else:
        nu = n_channels - 1 + 0.1
        psi = np.diag(range_)
        learnable_tensor_layer = covs_layer.layers[0]
        learnable_tensor_layer.regularizer = regularizers.InverseWishart(
            nu, psi, epsilon, n_batches
        )


def get_means_stds_fcs(model):
    means_layer = model.get_layer("means")
    stds_layer = model.get_layer("stds")
    fcs_layer = model.get_layer("fcs")

    means = means_layer(1)
    stds = add_epsilon(
        stds_layer(1),
        stds_layer.epsilon,
        diag=True,
    )
    fcs = add_epsilon(
        fcs_layer(1),
        fcs_layer.epsilon,
        diag=True,
    )
    return means.numpy(), stds.numpy(), fcs.numpy()


def set_means_stds_fcs(model, means, stds, fcs, update_initializer=True):
    if stds.ndim == 3:
        # Only keep the diagonal as a vector
        stds = np.diagonal(stds, axis1=1, axis2=2)

    means = means.astype(np.float32)
    stds = stds.astype(np.float32)
    fcs = fcs.astype(np.float32)

    # Get layers
    means_layer = model.get_layer("means")
    stds_layer = model.get_layer("stds")
    fcs_layer = model.get_layer("fcs")

    # Transform the matrices to layer weights
    diagonals = stds_layer.bijector.inverse(stds)
    flattened_cholesky_factors = fcs_layer.bijector.inverse(fcs)

    # Set values
    means_layer.vectors_layer.tensor.assign(means)
    stds_layer.diagonals_layer.tensor.assign(diagonals)
    fcs_layer.flattened_cholesky_factors_layer.tensor.assign(flattened_cholesky_factors)

    # Update initialisers
    if update_initializer:
        means_layer.vectors_layer.tensor_initializer = WeightInitializer(means)
        stds_layer.diagonals_layer.tensor_initializer = WeightInitializer(diagonals)
        fcs_layer.flattened_cholesky_factors_layer.tensor_initializer = (
            WeightInitializer(flattened_cholesky_factors)
        )


def set_stds_regularizer(model, training_dataset, epsilon):
    n_batches = dtf.get_n_batches(training_dataset)
    n_channels = dtf.get_n_channels(training_dataset)
    range_ = dtf.get_range(training_dataset)

    mu = np.zeros([n_channels], dtype=np.float32)
    sigma = np.sqrt(np.log(2 * range_))

    stds_layer = model.get_layer("stds")
    learnable_tensor_layer = stds_layer.layers[0]
    learnable_tensor_layer.regularizer = regularizers.LogNormal(
        mu, sigma, epsilon, n_batches
    )


def set_fcs_regularizer(model, training_dataset, epsilon):
    n_batches = dtf.get_n_batches(training_dataset)
    n_channels = dtf.get_n_channels(training_dataset)

    nu = n_channels - 1 + 0.1

    fcs_layer = model.get_layer("fcs")
    learnable_tensor_layer = fcs_layer.layers[0]
    learnable_tensor_layer.regularizer = regularizers.MarginalInverseWishart(
        nu,
        epsilon,
        n_channels,
        n_batches,
    )


def get_group_means_covariances(model):
    """Wrapper for getting the group level means and covariances."""
    group_means = get_means(model, "group_means")
    group_covariances = get_covariances(model, "group_covs")
    return group_means, group_covariances


def get_subject_embeddings(model):
    subject_embeddings_layer = model.get_layer("subject_embeddings")
    n_subjects = subject_embeddings_layer.input_dim
    return subject_embeddings_layer(np.arange(n_subjects)).numpy()


def get_mode_embeddings(model, map):
    """Wrapper for getting the mode embeddings for the means and covariances."""
    if map == "means":
        return _get_means_mode_embeddings(model)
    elif map == "covs":
        return _get_covs_mode_embeddings(model)
    else:
        raise ValueError("map must be either 'means' or 'covs'")


def get_concatenated_embeddings(model, map, subject_embeddings=None):
    """Getting the concatenated embeddings for the means and covariances."""
    if subject_embeddings is None:
        subject_embeddings = get_subject_embeddings(model)
    if map == "means":
        mode_embeddings = _get_means_mode_embeddings(model)
        concat_embeddings_layer = model.get_layer("means_concat_embeddings")
    elif map == "covs":
        mode_embeddings = _get_covs_mode_embeddings(model)
        concat_embeddings_layer = model.get_layer("covs_concat_embeddings")
    else:
        raise ValueError("map must be either 'means' or 'covs'")
    concat_embeddings = concat_embeddings_layer([subject_embeddings, mode_embeddings])
    return concat_embeddings.numpy()


def get_dev_mag_parameters(model, map):
    """Wrapper for getting the deviance magnitude parameters
    for the means and covariances."""
    if map == "means":
        return _get_means_dev_mag_parameters(model)
    elif map == "covs":
        return _get_covs_dev_mag_parameters(model)
    else:
        raise ValueError("map must be either 'means' or 'covs'")


def get_dev_mag(model, map):
    """Getting the deviance magnitude for the means and covariances."""
    if map == "means":
        alpha, beta = _get_means_dev_mag_parameters(model)
        dev_mag_layer = model.get_layer("means_dev_mag")
    elif map == "covs":
        alpha, beta = _get_covs_dev_mag_parameters(model)
        dev_mag_layer = model.get_layer("covs_dev_mag")
    else:
        raise ValueError("map must be either 'means' or 'covs'")
    dev_mag = dev_mag_layer([alpha, beta])
    return dev_mag.numpy()


def get_dev_map(model, map, subject_embeddings=None):
    """Getting the deviance map for the means and covariances."""
    concat_embeddings = get_concatenated_embeddings(model, map, subject_embeddings)
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
    dev_map_input = dev_map_input_layer([np.zeros(1), concat_embeddings])
    dev_map = dev_map_layer(dev_map_input)
    norm_dev_map = norm_dev_map_layer(dev_map)
    return norm_dev_map.numpy()


def get_subject_dev(
    model, learn_means, learn_covariances, subject_embeddings=None, n_neighbours=2
):
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
        means_dev_map = get_dev_map(model, "means", subject_embeddings)
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
        covs_dev_map = get_dev_map(model, "covs", subject_embeddings)
        covs_dev = covs_dev_layer([covs_dev_mag, covs_dev_map])
    else:
        covs_dev = covs_dev_layer(1)

    return means_dev.numpy(), covs_dev.numpy()


def get_subject_means_covariances(
    model, learn_means, learn_covariances, subject_embeddings=None, n_neighbours=2
):
    group_means, group_covs = get_group_means_covariances(model)
    means_dev, covs_dev = get_subject_dev(
        model, learn_means, learn_covariances, subject_embeddings, n_neighbours
    )

    subject_means_layer = model.get_layer("subject_means")
    subject_covs_layer = model.get_layer("subject_covs")

    mu = subject_means_layer([group_means, means_dev])
    D = subject_covs_layer([group_covs, covs_dev])
    return mu.numpy(), D.numpy()


def _get_means_mode_embeddings(model):
    group_means, _ = get_group_means_covariances(model)
    means_mode_embeddings_layer = model.get_layer("means_mode_embeddings")
    means_mode_embeddings = means_mode_embeddings_layer(group_means)
    return means_mode_embeddings.numpy()


def _get_covs_mode_embeddings(model):
    cholesky_bijector = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.FillScaleTriL()])
    _, group_covs = get_group_means_covariances(model)
    covs_mode_embeddings_layer = model.get_layer("covs_mode_embeddings")
    covs_mode_embeddings = covs_mode_embeddings_layer(
        cholesky_bijector.inverse(group_covs)
    )
    return covs_mode_embeddings.numpy()


def _get_means_dev_mag_parameters(model):
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


def _get_covs_dev_mag_parameters(model):
    covs_dev_mag_inf_alpha_input_layer = model.get_layer("covs_dev_mag_inf_alpha_input")
    covs_dev_mag_inf_alpha_layer = model.get_layer("covs_dev_mag_inf_alpha")
    covs_dev_mag_inf_beta_input_layer = model.get_layer("covs_dev_mag_inf_beta_input")
    covs_dev_mag_inf_beta_layer = model.get_layer("covs_dev_mag_inf_beta")

    covs_dev_mag_inf_alpha_input = covs_dev_mag_inf_alpha_input_layer(1)
    covs_dev_mag_inf_alpha = covs_dev_mag_inf_alpha_layer(covs_dev_mag_inf_alpha_input)

    covs_dev_mag_inf_beta_input = covs_dev_mag_inf_beta_input_layer(1)
    covs_dev_mag_inf_beta = covs_dev_mag_inf_beta_layer(covs_dev_mag_inf_beta_input)
    return covs_dev_mag_inf_alpha.numpy(), covs_dev_mag_inf_beta.numpy()


def set_bayesian_kl_scaling(model, n_batches, learn_means, learn_covariances):
    if learn_means:
        means_dev_mag_kl_loss_layer = model.get_layer("means_dev_mag_kl_loss")
        means_dev_mag_kl_loss_layer.n_batches = n_batches

    if learn_covariances:
        covs_dev_mag_kl_loss_layer = model.get_layer("covs_dev_mag_kl_loss")
        covs_dev_mag_kl_loss_layer.n_batches = n_batches


def get_nearest_neighbours(model, subject_embeddings, n_neighbours):
    """Get nearest neighbours for each subject in the embedding space."""
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


def set_dev_mlp_reg_scaling(model, n_batches, learn_means, learn_covariances):
    if learn_means:
        model.get_layer("means_dev_map_input").n_batches = n_batches
        model.get_layer("means_dev_mag_mod_beta_input").n_batches = n_batches
    if learn_covariances:
        model.get_layer("covs_dev_map_input").n_batches = n_batches
        model.get_layer("covs_dev_mag_mod_beta_input").n_batches = n_batches
