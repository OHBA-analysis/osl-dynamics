"""Functions to manipulate mode data for analysis.

"""

import numpy as np
from dynemo import array_ops


def autocorrelation_functions(
    mode_covariances: np.ndarray,
    n_embeddings: int,
    pca_components: np.ndarray,
) -> np.ndarray:
    """Auto/cross-correlation function from the mode covariance matrices.

    Parameters
    ----------
    mode_covariances : np.ndarray
        Mode covariance matrices.
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
        "mode_covariances must be of shape (n_channels, n_channels) or "
        + "(n_modes, n_channels, n_channels) or "
        + "(n_subjects, n_modes, n_channels, n_channels)."
    )
    mode_covariances = array_ops.validate(
        mode_covariances,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message=error_message,
    )

    # Get covariance of time embedded data
    te_covs = reverse_pca(mode_covariances, pca_components)

    # Dimensions
    n_subjects = te_covs.shape[0]
    n_modes = te_covs.shape[1]
    n_parcels = te_covs.shape[-1] // n_embeddings
    n_acf = 2 * n_embeddings - 1

    # Take mean of elements from the time embedded covariances that
    # correspond to the auto/cross-correlation function
    blocks = te_covs.reshape(
        n_subjects, n_modes, n_parcels, n_embeddings, n_parcels, n_embeddings
    )
    acfs = np.empty([n_subjects, n_modes, n_parcels, n_parcels, n_acf])
    for i in range(n_acf):
        acfs[:, :, :, :, i] = np.mean(
            np.diagonal(blocks, offset=i - n_embeddings + 1, axis1=3, axis2=5), axis=-1
        )

    return np.squeeze(acfs)


def raw_covariances(
    mode_covariances: np.ndarray,
    n_embeddings: int,
    pca_components: np.ndarray,
    zero_lag: bool = False,
) -> np.ndarray:
    """Covariance matrix of the raw channels.

    PCA and time embedding is reversed to give you to the covariance matrix
    of the raw channels.

    Parameters
    ----------
    mode_covariances : np.ndarray
        Mode covariance matrices.
    n_embeddings : int
        Number of embeddings applied to the training data.
    pca_components : np.ndarray
        PCA components used for dimensionality reduction.
    zero_lag : bool
        Should we return just the zero-lag elements? Otherwise, we return
        the mean over time lags.

    Returns
    -------
    np.ndarray
        Covariance matrix for raw channels.
    """

    # Validation
    error_message = (
        "mode_covariances must be of shape (n_channels, n_channels) or "
        + "(n_modes, n_channels, n_channels) or "
        + "(n_subjects, n_modes, n_channels, n_channels)."
    )
    mode_covariances = array_ops.validate(
        mode_covariances,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message=error_message,
    )

    # Get covariance of time embedded data
    te_covs = reverse_pca(mode_covariances, pca_components)

    if zero_lag:
        # Return the zero-lag elements only
        raw_covs = te_covs[
            :, :, n_embeddings // 2 :: n_embeddings, n_embeddings // 2 :: n_embeddings
        ]

    else:
        # Return block means
        n_subjects = te_covs.shape[0]
        n_modes = te_covs.shape[1]
        n_parcels = te_covs.shape[-1] // n_embeddings

        n_parcels = te_covs.shape[-1] // n_embeddings
        blocks = te_covs.reshape(
            n_subjects, n_modes, n_parcels, n_embeddings, n_parcels, n_embeddings
        )
        block_diagonal = blocks.diagonal(0, 2, 4)
        diagonal_means = block_diagonal.diagonal(0, 2, 3).mean(3)

        raw_covs = blocks.mean((3, 5))
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
