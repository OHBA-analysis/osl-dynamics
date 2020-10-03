import logging
from typing import Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import trange
from vrad.simulation import VRADSimulation
from vrad.utils.decorators import transpose

_logger = logging.getLogger("VRAD")


def trim_trials(
    epoched_time_series: np.ndarray,
    trial_start: int = None,
    trial_cutoff: int = None,
    trial_skip: int = None,
):
    """Remove trials from input data.

    If given as a three dimensional input with axes (channels x trials x time),
    remove trials by slicing and stepping.

    Parameters
    ----------
    epoched_time_series : numpy.ndarray
        The epoched time series which will have trials removed.
    trial_start : int
        The first trial to keep.
    trial_cutoff : int
        The last trial to keep.
    trial_skip : int
        How many steps to take between selected trials.

    """
    if epoched_time_series.ndim == 3:
        return epoched_time_series[:, trial_start:trial_cutoff, ::trial_skip]
    else:
        _logger.warning(
            f"Array is not 3D (ndim = {epoched_time_series.ndim}). Can't trim trials."
        )


def standardize(time_series: np.ndarray, discontinuities: np.ndarray) -> np.ndarray:
    """Standardizes time series data.

    Returns a time series standardized over continuous segments of data.
    """
    for i in range(len(discontinuities)):
        start = sum(discontinuities[:i])
        end = sum(discontinuities[: i + 1])
        time_series[start:end] = scale(time_series[start:end], axis=0)
    return time_series


@transpose
def scale(time_series: np.ndarray, axis: int = 0) -> np.ndarray:
    """Scales a time series.

    Returns a time series with zero mean and unit variance.
    """
    time_series -= time_series.mean(axis=axis)
    time_series /= time_series.std(axis=axis)
    return time_series


@transpose
def pca(
    time_series: np.ndarray,
    n_components: Union[int, float] = 1,
    whiten: bool = True,
    random_state: int = None,
):
    """Perform PCA on time_series.

    Wrapper for sklearn.decomposition.PCA.

    Parameters
    ----------
    n_components: float or int
        If >1, number of components to be kept in PCA. If <1, the amount of
        variance to be explained by PCA. If equal to 1, no PCA applied.

    Returns
    -------

    """
    if time_series.ndim != 2:
        raise ValueError("time_series must be a 2D array")

    if n_components == 1:
        _logger.info("n_components of 1 was passed. Skipping PCA.")

    else:
        pca_from_variance = PCA(
            n_components=n_components, whiten=whiten, random_state=random_state
        )
        time_series = pca_from_variance.fit_transform(time_series)
        if 0 < n_components < 1:
            print(
                f"{pca_from_variance.n_components_} components are required to "
                f"explain {n_components * 100}% of the variance "
            )
    return time_series


def trials_to_continuous(trials_time_course: np.ndarray) -> np.ndarray:
    """Given trial data, return a continuous time series.

    With data input in the form (channels x trials x time), reshape the array to
    create a (time x channels) array.

    Parameters
    ----------
    trials_time_course: numpy.ndarray
        A (channels x trials x time) time course.

    Returns
    -------
    concatenated: numpy.ndarray
        The (time x channels) array created by concatenating trial data.

    """
    if trials_time_course.ndim == 2:
        _logger.warning(
            "A 2D time series was passed. Assuming it doesn't need to "
            "be concatenated."
        )
        if trials_time_course.shape[1] > trials_time_course.shape[0]:
            trials_time_course = trials_time_course.T
        return trials_time_course

    if trials_time_course.ndim != 3:
        raise ValueError(
            f"trials_time_course has {trials_time_course.ndim}"
            f" dimensions. It should have 3."
        )
    concatenated = np.concatenate(
        np.transpose(trials_time_course, axes=[2, 0, 1]), axis=1
    )
    if concatenated.shape[1] > concatenated.shape[0]:
        concatenated = concatenated.T
        _logger.warning(
            "Assuming longer axis to be time and transposing. Check your inputs to be "
            "sure."
        )

    return concatenated


@transpose
def time_embed(
    time_series: np.ndarray,
    discontinuities: np.ndarray,
    n_embeddings: int,
    output_file=None,
) -> np.ndarray:
    """Performs time embedding."""

    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    # Unpack shape of the original data
    n_samples, n_channels = time_series.shape

    # If an output file hasn't been passed we create a numpy array for the
    # time embedded data
    if output_file is None:
        time_embedded_series = np.empty(
            [
                n_samples - (n_embeddings + 1) * len(discontinuties),
                n_channels * (n_embeddings + 2),
            ]
        )
    else:
        time_embedded_series = output_file

    # Loop through continuous segments of data
    for i in range(len(discontinuities)):
        n_segment = discontinuities[i]
        start = sum(discontinuities[:i])
        end = sum(discontinuities[: i + 1])
        original_time_series = time_series[start:end]

        # Generate time embedded data
        time_embedded_segment = np.empty(
            [n_segment - (n_embeddings + 1), n_channels * (n_embeddings + 2)]
        )
        for j in range(n_channels):
            for k in range(n_embeddings + 2):
                time_embedded_segment[
                    :, j * (n_embeddings + 2) + k
                ] = original_time_series[n_embeddings + 1 - k : n_segment - k, j]

        # Fill the final time embedded series array
        time_embedded_series[
            start - (n_embeddings + 1) * i : end - (n_embeddings + 1) * (i + 1)
        ] = time_embedded_segment

    return time_embedded_series


def num_batches(arr, sequence_length: int, step_size: int = None):
    step_size = step_size or sequence_length
    final_slice_start = arr.shape[0] - sequence_length + 1
    index = np.arange(0, final_slice_start, step_size)[:, None] + np.arange(
        sequence_length
    )
    return len(index)
