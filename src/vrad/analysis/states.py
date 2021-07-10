"""Functions to manipulate state data for analysis.

"""

import numpy as np
from vrad import array_ops


def autocorrelation_functions(
    state_covariances: np.ndarray,
    n_embeddings: int,
    pca_components: np.ndarray,
) -> np.ndarray:
    """Auto/cross-correlation function from the state covariance matrices.

    Parameters
    ----------
    state_covariances : np.ndarray
        State covariance matrices.
    n_embeddings : int
        Number of embeddings applied to the training data.
    pca_components : np.ndarray
        PCA components used for dimensionality reduction.

    Returns
    -------
    np.ndarray
        Auto/cross-correlation functions.
    """

    # Validation
    error_message = (
        "state_covariances must be of shape (n_channels, n_channels) or "
        + "(n_states, n_channels, n_channels) or "
        + "(n_subjects, n_states, n_channels, n_channels)."
    )
    state_covariances = array_ops.validate(
        state_covariances,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message=error_message,
    )

    # Get covariance of time embedded data
    te_covs = reverse_pca(state_covariances, pca_components)

    # Dimensions
    n_subjects = te_covs.shape[0]
    n_states = te_covs.shape[1]
    n_raw_channels = te_covs.shape[-1] // n_embeddings
    n_acf = 2 * n_embeddings - 1

    # Take elements from the time embedded covariances that
    # correspond to the auto/cross-correlation function
    acfs = np.empty([n_subjects, n_states, n_raw_channels, n_raw_channels, n_acf])
    for i in range(n_subjects):
        for j in range(n_states):
            for k in range(n_raw_channels):
                for l in range(n_raw_channels):
                    # Auto/cross-correlation between channel k and channel l
                    # of state j
                    block = te_covs[
                        i,
                        j,
                        k * n_embeddings : (k + 1) * n_embeddings,
                        l * n_embeddings : (l + 1) * n_embeddings,
                    ]

                    # Take elements that correspond to the auto/cross-correlation
                    # function
                    acfs[i, j, k, l] = np.concatenate(
                        [
                            block[0, n_embeddings // 2 + 1 :][::-1],
                            block[:, n_embeddings // 2],
                            block[-1, : n_embeddings // 2][::-1],
                        ]
                    )

    return np.squeeze(acfs)


def raw_covariances(
    state_covariances: np.ndarray,
    n_embeddings: int,
    pca_components: np.ndarray,
    zero_lag: bool = False,
) -> np.ndarray:
    """Covariance matrix of the raw channels.

    PCA and time embedding is reversed to give you to the covariance matrix
    of the raw channels.

    Parameters
    ----------
    state_covariances : np.ndarray
        State covariance matrices.
    n_embeddings : int
        Number of embeddings applied to the training data.
    pca_components : np.ndarray
        PCA components used for dimensionality reduction.
    zero_lag : bool
        Should we return just the zero-lag elements? Otherwise, we return
        the mean over time lags. Optional, default is False.

    Returns
    -------
    np.ndarray
        Covariance matrix for raw channels.
    """

    # Validation
    error_message = (
        "state_covariances must be of shape (n_channels, n_channels) or "
        + "(n_states, n_channels, n_channels) or "
        + "(n_subjects, n_states, n_channels, n_channels)."
    )
    state_covariances = array_ops.validate(
        state_covariances,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message=error_message,
    )

    # Get covariance of time embedded data
    te_covs = reverse_pca(state_covariances, pca_components)

    if zero_lag:
        # Return the zero-lag elements only
        raw_covs = te_covs[
            :, :, n_embeddings // 2 :: n_embeddings, n_embeddings // 2 :: n_embeddings
        ]

    else:
        # Return block means
        n_subjects = te_covs.shape[0]
        n_states = te_covs.shape[1]
        n_parcels = te_covs.shape[-1] // n_embeddings

        n_parcels = te_covs.shape[-1] // n_embeddings
        block_te = te_covs.reshape(
            n_subjects, n_states, n_parcels, n_embeddings, n_parcels, n_embeddings
        )
        block_diagonal = block_te.diagonal(0, 2, 4)
        diagonal_means = block_diagonal.diagonal(0, 2, 3).mean(3)

        raw_covs = block_te.mean((3, 5))
        raw_covs[:, :, np.arange(n_parcels), np.arange(n_parcels)] = diagonal_means

    return np.squeeze(raw_covs)


def reverse_pca(covariances: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
    """Reverses the effect of PCA on covariance matrices.

    Parameters
    ----------
    covariances : np.ndarray
        Covariance matrices.
    pca_components : np.ndarray
        PCA components used for dimensionality reduction.

    Returns
    -------
    np.ndarray
        Covariance matrix of the time embedded data.
    """
    if covariances.shape[-1] != pca_components.shape[-1]:
        raise ValueError(
            "Covariance matrix and PCA components have incompatible shapes: "
            + f"covariances.shape={covariances.shape}, "
            + f"pca_components.shape={pca_components.shape}."
        )

    return pca_components @ covariances @ pca_components.T
