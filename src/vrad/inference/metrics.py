"""Metrics to analyse model performance.

"""

import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion
from vrad.utils.decorators import transpose


@transpose("state_time_course_1", 0, "state_time_course_2", 1)
def confusion_matrix(
    state_time_course_1: np.ndarray, state_time_course_2: np.ndarray
) -> np.ndarray:
    """Calculate the confusion matrix of two state time courses.

    For two state-time-courses, calculate the confusion matrix (i.e. the
    disagreement between the state selection for each sample). If either sequence is
    two dimensional, it will first have argmax(axis=1) applied to it. The produces the
    expected result for a one-hot encoded sequence but other inputs are not guaranteed
    to behave.

    This function is a wrapper for sklearn.metrics.confusion_matrix.

    Parameters
    ----------
    state_time_course_1: numpy.ndarray
    state_time_course_2: numpy.ndarray

    Returns
    -------
    confusion_matrix: numpy.ndarray
    """
    if state_time_course_1.ndim == 2:
        state_time_course_1 = state_time_course_1.argmax(axis=1)
    if state_time_course_2.ndim == 2:
        state_time_course_2 = state_time_course_2.argmax(axis=1)
    if not ((state_time_course_1.ndim == 1) and (state_time_course_2.ndim == 1)):
        raise ValueError("Both state time courses must be either 1D or 2D.")

    return sklearn_confusion(state_time_course_1, state_time_course_2)


def dice_coefficient_1d(sequence_1: np.ndarray, sequence_2: np.ndarray) -> float:
    """Calculate the Dice coefficient of a discrete array

    Given two sequences containing a number of discrete elements (i.e. a
    categorical variable), calculate the Dice coefficient of those sequences.

    The Dice coefficient is 2 times the number of equal elements (equivalent to
    true-positives) divided by the sum of the total number of elements.

    Parameters
    ----------
    sequence_1 : numpy.ndarray
        A sequence containing discrete elements.
    sequence_2 : numpy.ndarray
        A sequence containing discrete elements.

    Returns
    -------
    dice_coefficient : float
        The Dice coefficient of the two sequences.

    Raises
    ------
    ValueError
        If either sequence is not one dimensional.
    """
    if (sequence_1.ndim, sequence_2.ndim) != (1, 1):
        raise ValueError(
            f"sequences must be 1D: {(sequence_1.ndim, sequence_2.ndim)} != (1, 1)."
        )
    if (sequence_1.dtype, sequence_2.dtype) != (int, int):
        raise TypeError("Both sequences must be integer (categorical).")

    return 2 * ((sequence_1 == sequence_2).sum()) / (len(sequence_1) + len(sequence_2))


@transpose(0, 1, "sequence_1", "sequence_2")
def dice_coefficient(sequence_1: np.ndarray, sequence_2: np.ndarray) -> float:
    """Wrapper method for `dice_coefficient`.

    If passed a one-dimensional array, it will be sent straight to `dice_coefficient`.
    Given a two-dimensional array, it will perform an argmax calculation on each sample.
    The default axis for this is zero, i.e. each row represents a sample.

    Parameters
    ----------
    sequence_1 : numpy.ndarray
        A sequence containing either 1D discrete or 2D continuous data.
    sequence_2 : numpy.ndarray
        A sequence containing either 1D discrete or 2D continuous data.
    axis_1 : int
        For a 2D sequence_1, the axis on which to perform argmax. Default is 0.
    axis_2 : int
        For a 2D sequence_2, the axis on which to perform argmax. Default is 0.

    Returns
    -------
    dice_coefficient : float
        The Dice coefficient of the two sequences.

    See Also
    --------
    dice_coefficient_1D : Dice coefficient of 1D categorical sequences.
    """
    if (len(sequence_1.shape) not in [1, 2]) or (len(sequence_2.shape) not in [1, 2]):
        raise ValueError("Both sequences must be either 1D or 2D")
    if (len(sequence_1.shape) == 1) and (len(sequence_2.shape) == 1):
        return dice_coefficient_1d(sequence_1, sequence_2)
    if len(sequence_1.shape) == 2:
        sequence_1 = sequence_1.argmax(axis=1)
    if len(sequence_2.shape) == 2:
        sequence_2 = sequence_2.argmax(axis=1)
    return dice_coefficient_1d(sequence_1, sequence_2)


def log_likelihood(
    time_series: np.ndarray,
    state_mixing_factors: np.ndarray,
    covariances: np.ndarray,
    means: np.ndarray = None,
) -> float:
    """Calculates the log likelihood.

    The log likelihood is calculated as:

    c - 0.5 * log(det(sigma)) - 0.5 * [(x - mu)^T sigma^-1 (x - mu)]

    where x is a single observation, mu is the mean vector, sigma is the
    covariance matrix and c is a constant.

    Parameters
    ----------
    time_series : np.ndarray
        Data time series.
    state_mixing_factors : np.ndarary
        Times series containing the state mixing factors alpha_t.
    covariances : np.ndarray
        Covariance matrix for each state.
    means : np.ndarray
        Mean vector for each state.

    Returns
    -------
    nll: float
        The negative of the log likelihood.
    """
    if means is None:
        means = np.zeros([covariances.shape[0], covariances.shape[1]])

    # Negative log likelihood
    nll = 0

    # Loop through the data
    for i in range(state_mixing_factors.shape[0]):
        x = time_series[i]
        alpha = state_mixing_factors[i]

        # State mixing
        mu = np.sum(alpha[..., np.newaxis] * means, axis=0)
        sigma = np.sum(alpha[..., np.newaxis, np.newaxis] * covariances, axis=0)

        # Calculate second term: -0.5 * log(|sigma|)
        sign, logdet = np.linalg.slogdet(sigma)
        second_term = -0.5 * sign * logdet

        # Calculate third term: -0.5 * [(x - mu)^T sigma^-1 (x - mu)]
        inv_sigma = np.linalg.inv(sigma + 1e-8 * np.eye(sigma.shape[-1]))
        x_minus_mu = x[..., np.newaxis] - mu[..., np.newaxis]
        x_minus_mu_T = np.transpose(x_minus_mu)
        third_term = -0.5 * x_minus_mu_T @ inv_sigma @ x_minus_mu

        # Calculate the negative log likelihood
        # We ignore the first term which is a constant
        nll -= np.squeeze(second_term + third_term)

    return nll
