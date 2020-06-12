import logging
from typing import Union

import numpy as np
from sklearn.decomposition import PCA
from taser.utils.decorators import transpose


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
        logging.warning(
            f"Array is not 3D (ndim = {epoched_time_series.ndim}). Can't trim trials."
        )


def standardize(
    time_series: np.ndarray,
    n_components: Union[float, int] = 0.9,
    pre_scale: bool = True,
    do_pca: Union[bool, str] = True,
    post_scale: bool = True,
):
    """Function for scaling and performing PCA on time_series.

    Wraps scale and pca.

    Parameters
    ----------
    n_components: int or float
        If >1, number of components to be kept in PCA. If <1, the amount of
        variance to be explained by PCA. Passed to Data.pca.
    pre_scale: bool
        If True (default) scale data to have mean of zero and standard
        deviation of one before PCA is applied.
    do_pca: bool or str
        If True perform PCA on time_series. n_components = 1 is equivalent to
        False. If PCA has already been performed on time_series, set to "force".
        This is a safety check to make sure PCA isn't accidentally run twice.
    post_scale: bool
        If True (default) scale data to have mean of zero and standard
        deviation of one after PCA is applied.
    """
    if pre_scale:
        time_series = scale(time_series=time_series)
    if do_pca:
        time_series = pca(time_series=time_series, n_components=n_components)
    if post_scale:
        time_series = scale(time_series=time_series)
    return time_series


@transpose
def scale(time_series: np.ndarray):
    """Scale time_series to have mean zero and standard deviation 1.

    """
    return (time_series - time_series.mean(axis=0)) / time_series.std(axis=0)


@transpose
def pca(time_series, n_components: Union[int, float] = 1):
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
        logging.info("n_components of 1 was passed. Skipping PCA.")

    else:
        pca_from_variance = PCA(n_components=n_components)
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
        logging.warning(
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
        logging.warning(
            "Assuming longer axis to be time and transposing. Check your inputs to be "
            "sure."
        )

    return concatenated
