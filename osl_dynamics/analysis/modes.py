"""Functions to manipulate and analyse inferred mode/state data.

"""

import numpy as np

from osl_dynamics import array_ops


def autocorrelation_functions(
    mode_covariances,
    n_embeddings,
    pca_components,
):
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
    acfs : np.ndarray
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
    mode_covariances,
    n_embeddings,
    pca_components,
    zero_lag=False,
):
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
    raw_covs : np.ndarray
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


def reverse_pca(covariances, pca_components):
    """Reverses the effect of PCA on covariance matrices.

    Parameters
    ----------
    covariances : np.ndarray
        Covariance matrices.
    pca_components : np.ndarray
        PCA components used for dimensionality reduction.

    Returns
    -------
    covariances : np.ndarray
        Covariance matrix of the time embedded data.
    """
    if covariances.shape[-1] != pca_components.shape[-1]:
        raise ValueError(
            "Covariance matrix and PCA components have incompatible shapes: "
            + f"covariances.shape={covariances.shape}, "
            + f"pca_components.shape={pca_components.shape}."
        )

    return pca_components @ covariances @ pca_components.T


def partial_covariances(data, alpha):
    """Calculate partial covariances.

    Returns the multiple regression parameters estimates, pcovs, of the state
    time courses regressed onto the data from each channel. The regression is
    done separately for each channel. I.e. pcovs is the estimate of the
    (n_modes, n_channels) matrix, Beta. We fit the regression Y_i = X @ Beta_i + e,
    where:

    - Y_i is (n_samples, 1) the data amplitude/envelope/power/abs time course at channel i.
    - X is (n_samples, n_modes) matrix of the variance normalised mode time courses (i.e. alpha).
    - Beta_i is (n_modes, 1) vector of multiple regression parameters for channel i.
    - e is the error.

    NOTE: state time courses are variance normalised so that all amplitude info goes
    into Beta (i.e. pcovs).

    Parameters
    ----------
    data : np.ndarray or list of np.ndarray
        Training data for each subject.
    alpha : np.ndarray or list of np.ndarray
        State/mode time courses for each subject.

    Returns
    -------
    pcovs : np.ndarray
        Matrix of partial covariance (multiple regression parameter estimates).
        Shape is (n_modes, n_channels).
    """
    if type(data) != type(alpha):
        raise ValueError(
            "data and alpha must be the same type: numpy arrays or lists of numpy arrays."
        )
    if isinstance(data, np.ndarray):
        data = [data]
        alpha = [alpha]
    for i in range(len(data)):
        if data[i].shape[0] != alpha[i].shape[0]:
            raise ValueError("Difference number of samples in data and alpha.")

    pcovs = []
    for X, a in zip(data, alpha):
        # Variance normalise state/mode time courses
        a /= np.std(a, axis=0, keepdims=True)

        # Do multiple regression of alpha onto data
        pcovs.append(np.linalg.pinv(a) @ X)

    return np.squeeze(pcovs)
