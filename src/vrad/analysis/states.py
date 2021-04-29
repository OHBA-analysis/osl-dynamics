from typing import Union

import numpy as np


def _acf_from_te_cov(te_covariances: np.ndarray, n_embeddings: int) -> np.ndarray:
    """Get the autocorrelation function from the covariance matrix of time embedded data.

    Parameters
    ----------
    te_covariances : np.ndarray
        Time embedded state covariance matrices.
        Shape is (n_states, n_te_channels, n_te_channels).
    n_embeddings : int
        Number of time embeddings.

    Returns
    -------
    np.ndarray
        Autocorrelation function. Shape is (n_states, n_acf)
    """
    # Number of data points in the autocorrelation function
    n_acf = 2 * n_embeddings - 1

    # Number of states
    n_states = len(te_covariances)

    # Number of channels in the original covariance matrix
    n_raw_data_channels = te_covariances[0].shape[-1] // n_embeddings

    # Get the autocorrelation function
    autocorrelation_function = np.empty(
        [n_states, n_raw_data_channels, n_raw_data_channels, n_acf]
    )
    for i in range(n_states):
        for j in range(n_raw_data_channels):
            for k in range(n_raw_data_channels):
                # Auto/cross-correlation between channel j and channel k
                # of state i
                autocorrelation_function_jk = te_covariances[
                    i,
                    j * n_embeddings : (j + 1) * n_embeddings,
                    k * n_embeddings : (k + 1) * n_embeddings,
                ]

                # Take elements from the first row and column
                autocorrelation_function[i, j, k] = np.concatenate(
                    [
                        autocorrelation_function_jk[0, n_embeddings // 2 + 1 :][::-1],
                        autocorrelation_function_jk[:, n_embeddings // 2],
                        autocorrelation_function_jk[-1, : n_embeddings // 2][::-1],
                    ]
                )

    return autocorrelation_function


def autocorrelation_functions(
    state_covariances: Union[list, np.ndarray],
    n_embeddings: int,
    pca_components: np.ndarray,
) -> np.ndarray:
    """Gets the autocorrelation function from the state covariance matrices.

    An autocorrelation function is calculated for each state for each subject.

    Parameters
    ----------
    state_covariances : np.ndarray
        State covariance matrices.
        Shape is (n_subjects, n_states, n_channels, n_channels).
        These must be subject specific covariances.
    n_embeddings : int
        Number of embeddings applied to the training data.
    pca_components : np.ndarray
        Components used for dimensionality reduction.
        Shape must be (n_te_channels, n_pca_components).

    Returns
    -------
    np.ndarray
        Autocorrelation functions.
        Shape is (n_subjects, n_states, n_channels, n_channels, n_acf)
        or (n_states, n_channels, n_channels, n_acf).
    """
    # Get covariance of time embedded data
    te_covs = reverse_pca(state_covariances, pca_components)

    # Take elements from the time embedded covariances that
    # correspond to the autocorrelation function
    autocorrelation_functions = []
    for n in range(len(te_covs)):
        autocorrelation_functions.append(_acf_from_te_cov(te_covs[n], n_embeddings))

    return np.squeeze(autocorrelation_functions)


def raw_covariances(
    state_covariances: Union[list, np.ndarray],
    n_embeddings: int,
    pca_components: np.ndarray,
) -> np.ndarray:
    """Covariance matrix of the raw channels.

    PCA and standardization is reversed to give you to the covariance
    matrix for the raw channels.

    Parameters
    ----------
    state_covariances : np.ndarray
        State covariance matrices.
        Shape is (n_subjects, n_states, n_channels, n_channels).
        These must be subject specific covariances.
    n_embeddings : int
        Number of embeddings applied to the training data.
    pca_components : np.ndarray
        Components used for dimensionality reduction.
        Shape must be (n_te_channels, n_pca_components).

    Returns
    -------
    np.ndarray
        The variance for each channel, state and subject.
        Shape is (n_subjects, n_states, n_channels, n_channels) or
        (n_states, n_channels, n_channels).
    """
    # Get covariance of time embedded data.
    te_covs = reverse_pca(state_covariances, pca_components)

    # Get elements corresponding to zero-lag.
    raw_covs = te_covs[
        :, :, n_embeddings // 2 :: n_embeddings, n_embeddings // 2 :: n_embeddings
    ]

    return np.squeeze(raw_covs)


def raw_covariances_mean(
    state_covariances: Union[list, np.ndarray],
    n_embeddings: int,
    pca_components: np.ndarray,
) -> np.ndarray:
    """Covariance matrix of the raw channels.

    PCA and standardization is reversed to give you to the covariance
    matrix for the raw channels. Produces equivalent results to raw_covariances, but
    uses a block mean for each parcel, rather than taking the zero-lag elements.

    Parameters
    ----------
    state_covariances : np.ndarray
        State covariance matrices.
        Shape is (n_subjects, n_states, n_channels, n_channels).
        These must be subject specific covariances.
    n_embeddings : int
        Number of embeddings applied to the training data.
    pca_components : np.ndarray
        Components used for dimensionality reduction.
        Shape must be (n_te_channels, n_pca_components).

    Returns
    -------
    np.ndarray
        The variance for each channel, state and subject.
        Shape is (n_subjects, n_states, n_channels, n_channels) or
        (n_states, n_channels, n_channels).
    """
    # Get covariance of time embedded data.
    te_covs = reverse_pca(state_covariances, pca_components)

    # Get block means.
    n_parcels = te_covs.shape[-1] // n_embeddings
    raw_covs = te_covs.reshape(
        -1, n_parcels, n_embeddings, n_parcels, n_embeddings
    ).mean(axis=(2, 4))

    return np.squeeze(raw_covs)


def reverse_pca(covariances: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
    """Reverses the effect of PCA on a covariance matrix.

    Parameters
    ----------
    covariances : np.ndarray
        State covariance matrices.
        Shape is (n_subjects, n_states, n_channels, n_channels).
        These must be subject specific covariances.
    pca_components : np.ndarray
        Components used for dimensionality reduction.
        Shape must be (n_te_channels, n_pca_components).

    Returns
    -------
    np.ndarray
        Covariance matrix of the time embedded data.
    """

    # Validation
    covariances = np.array(covariances)
    if covariances.ndim == 3:
        covariances = covariances[None]
    if covariances.ndim != 4:
        raise ValueError(
            "covariances must be shape (n_states, n_channels, n_channels) or"
            + " (n_subjects, n_states, n_channels, n_channels)."
        )

    n_subjects, n_states = covariances.shape[:2]
    te_covs = pca_components @ covariances @ pca_components.T

    return te_covs
