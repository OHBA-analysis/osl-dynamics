"""Metrics to analyse model performance.

"""
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.linalg import eigvalsh
from sklearn.metrics import confusion_matrix as sklearn_confusion
from tqdm import trange
from vrad.inference.states import state_lifetimes
from vrad.utils.decorators import transpose


def alpha_correlation(alpha_1: np.ndarray, alpha_2: np.ndarray) -> np.ndarray:
    """Calculates the correlation between states of two alpha time series.

    Parameters
    ----------
    alpha_1 : np.ndarray
        First alpha time series. Shape is (n_samples, n_states).
    alpha_2 : np.ndarray
        Second alpha time series. Shape is (n_samples, n_states).

    Returns
    -------
    np.ndarray
        Correlation of each state in the corresponding alphas.
        Shape is (n_states,).
    """
    if alpha_1.shape[1] != alpha_2.shape[1]:
        raise ValueError(
            "alpha_1 and alpha_2 shapes are incomptible. "
            + f"alpha_1.shape={alpha_1.shape}, alpha_2.shape={alpha_2.shape}."
        )
    n_states = alpha_1.shape[1]
    corr = np.corrcoef(alpha_1, alpha_2, rowvar=False)
    corr = np.diagonal(corr[:n_states, n_states:])
    return corr


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
    numpy.ndarray
        Confusion matrix
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
    float
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
    float
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


def fractional_occupancies(state_time_course: np.ndarray) -> np.ndarray:
    """Calculates the fractional occupancy.

    Parameters
    ----------
    state_time_course : np.ndarray
        State time course. Shape is (n_samples, n_states).

    Returns
    -------
    np.ndarray
        The fractional occupancy of each state.
    """
    return np.sum(state_time_course, axis=0) / state_time_course.shape[0]


def frobenius_norm(A: np.ndarray, B: np.ndarray) -> float:
    """Calculates the frobenius norm of the difference of two matrices.

    The Frobenius norm is calculated as sqrt( Sum_ij abs(a_ij - b_ij)^2 ).

    Parameters
    ----------
    A : np.ndarray
        First matrix. Shape must be (n_states, n_channels, n_channels) or
        (n_channels, n_channels).
    B : np.ndarray
        Second matrix. Shape must be (n_states, n_channels, n_channels) or
        (n_channels, n_channels).

    Returns
    -------
    float
        The Frobenius norm of the difference of A and B. If A and B are
        stacked matrices, we sum the Frobenius norm of each sub-matrix.
    """
    if A.ndim == 2 and B.ndim == 2:
        norm = np.linalg.norm(A - B, ord="fro")
    elif A.ndim == 3 and B.ndim == 3:
        norm = np.linalg.norm(A - B, ord="fro", axis=(1, 2))
        norm = np.sum(norm)
    else:
        raise ValueError(
            f"Shape of A and/or B is incorrect. A.shape={A.shape}, B.shape={B.shape}."
        )
    return norm


def log_likelihood(
    time_series: np.ndarray,
    alpha: np.ndarray,
    covariances: np.ndarray,
    sequence_length: int,
    means: np.ndarray = None,
):
    """Calculate the negative log-likelihood.

    We calculate the negative log-likelihood using a tensorflow implementation by
    recursively calling tf_nll because it's quicker.

    Parameters
    ----------
    time_series : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    alpha : np.ndarray
        Inferred state mixing factors. Shape must be (n_samples, n_states).
    covariances : np.ndarray
        Inferred state covariances. Shape must be (n_states, n_channels, n_channels).
    means : np.ndarray
        Inferred mean vectors. Shape must be (n_states, n_channels).
    sequence_length : int
        Length of time series to recursively pass to tf_nll.
    """
    # Validation
    if time_series.shape[0] != alpha.shape[0]:
        raise ValueError("time_series and alpha must have the same length.")

    if time_series.shape[0] % sequence_length != 0:
        raise ValueError("time_series and alpha must be divisible by sequence_length.")

    time_series = time_series.astype(np.float32)
    alpha = alpha.astype(np.float32)
    covariances = covariances.astype(np.float32)
    if means is not None:
        means = means.astype(np.float32)

    #  Convert means and covariances to tensors
    if means is None:
        m = tf.zeros([covariances.shape[0], covariances.shape[1]])
    else:
        m = tf.constant(means)
    C = tf.constant(covariances)

    # Number times to call tf_nll
    n_calls = time_series.shape[0] // sequence_length

    nll = []
    for i in trange(n_calls, desc="Calculating log-likelihood", ncols=98):

        # Convert data to tensors
        x = tf.constant(time_series[i * sequence_length : (i + 1) * sequence_length])
        a = tf.constant(alpha[i * sequence_length : (i + 1) * sequence_length])

        # Calculate the negative log-likelihood for each sequence
        nll.append(tf_nll(x, a, m, C))

    # Return the sum for all sequences
    return np.sum(nll)


@tf.function
def tf_nll(x: tf.constant, alpha: tf.constant, mu: tf.constant, D: tf.constant):
    """Calculates the negative log likelihood using a tensorflow implementation.

    Parameters
    ----------
    x : tf.constant
        Time series data. Shape must be (sequence_length, n_channels).
    alpha : tf.constant
        State mixing factors. Shape must be (sequence_length, n_states).
    mu : tf.constant
        State mean vectors. Shape must be (n_states, n_channels).
    D : tf.constant
        State covariances. Shape must be (n_states, n_channels, n_channels).
    """
    # Calculate the mean: m = Sum_j alpha_jt mu_j
    alpha = tf.expand_dims(alpha, axis=-1)
    mu = tf.expand_dims(mu, axis=0)
    m = tf.reduce_sum(tf.multiply(alpha, mu), axis=1)

    # Calculate the covariance: C = Sum_j alpha_jt D_j
    alpha = tf.expand_dims(alpha, axis=-1)
    D = tf.expand_dims(D, axis=0)
    C = tf.reduce_sum(tf.multiply(alpha, D), axis=1)

    # Calculate the log-likelihood at each time point
    mvn = tfp.distributions.MultivariateNormalTriL(
        loc=m,
        scale_tril=tf.linalg.cholesky(C + 1e-6 * tf.eye(C.shape[-1])),
    )
    ll = mvn.log_prob(x)

    # Sum over time and return the negative log-likelihood
    return -tf.reduce_sum(ll, axis=0)


def lifetime_statistics(state_time_course: np.ndarray) -> Tuple:
    """Calculate statistics of the lifetime distribution of each state.

    Parameters
    ----------
    state_time_course : np.ndarray
        State time course. Shape is (n_samples, n_states).

    Returns
    -------
    means : np.ndarray
        Mean lifetime of each state.
    std : np.ndarray
        Standard deviation of each state.
    """
    lifetimes = state_lifetimes(state_time_course)
    mean = np.array([np.mean(lt) for lt in lifetimes])
    std = np.array([np.std(lt) for lt in lifetimes])
    return mean, std


def state_covariance_correlations(
    state_covariances: np.ndarray, remove_diagonal: bool = True
) -> np.ndarray:
    """Calculate the correlation between elements of the state covariances.

    Parameters
    ----------
    state_covariances : np.ndarray
        State covariances matrices.
        Shape must be (n_states, n_channels, n_channels).

    Returns
    -------
    np.ndarray
        Correlation between elements of each state covariance.
        Shape is (n_states, n_states).
    """
    n_states = state_covariances.shape[0]
    state_covariances = state_covariances.reshape(n_states, -1)
    correlations = np.corrcoef(state_covariances)
    correlations -= np.eye(n_states)
    return correlations


def riemannian_distance(M1: np.ndarray, M2: np.ndarray) -> float:
    """Calculate the Riemannian distance between two matrices.

    The Riemannian distance is defined as: d = (sum log(eig(M_1 * M_2))) ^ 0.5

    Parameters
    ----------
    M1 : np.ndarray
    M2 : np.ndarray

    Returns
    -------
    np.ndarray
    """
    d = np.sqrt(np.sum((np.log(eigvalsh(M1, M2)) ** 2)))
    return d


def state_covariance_riemannian_distances(state_covariances: np.ndarray) -> np.ndarray:
    """Calculate the Riemannian distance between state covariances.

    Parameters
    ----------
    state_covariances : np.ndarray
        State covariances. Shape must be (n_states, n_channels, n_channels).

    Returns
    -------
    np.ndarray
        Matrix containing the Riemannian distances between states.
        Shape is (n_states, n_states).
    """
    n_states = state_covariances.shape[0]
    riemannian_distances = np.empty([n_states, n_states])
    for i in range(n_states):
        for j in range(n_states):
            riemannian_distances[i][j] = riemannian_distance(
                state_covariances[i], state_covariances[j]
            )
    return riemannian_distances
