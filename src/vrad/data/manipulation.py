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
def scale(time_series: np.ndarray) -> np.ndarray:
    """Scale time_series to have mean zero and standard deviation 1.

    """
    time_series -= time_series.mean(axis=0)
    time_series /= time_series.std(axis=0)
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
    n_embeddings: int,
    random_seed: int = None,
    output_file=None,
):
    """Performs time embedding. This function reproduces the OSL function embedx.
    """
    n_samples, n_channels = time_series.shape
    lags = range(-n_embeddings // 2, n_embeddings // 2 + 1)

    # Generate time embedded dataset
    if output_file is None:
        time_embedded_series = np.empty([n_samples, n_channels * len(lags)])
    else:
        time_embedded_series = output_file
    for i in range(n_channels):
        for j in range(len(lags)):
            time_embedded_series[:, i * (n_embeddings + 1) + j] = np.roll(
                time_series[:, i], lags[j]
            )

    # Calculate the standard deviation of the first 500 time points
    sigma = np.std(time_series[:500])

    # Setup random number generator
    rng = np.random.default_rng(random_seed)

    # We fill the values we don't have all the time lags for with Gaussian
    # random numbers
    for i in range(n_channels * len(lags)):
        time_embedded_series[: lags[-1], i] = rng.normal(0, sigma, lags[-1])
        time_embedded_series[lags[0] :, i] = rng.normal(0, sigma, abs(lags[0]))

    return time_embedded_series


def covariance(time_series: np.ndarray, weighted: bool = False) -> np.ndarray:
    if weighted:
        weight = time_series.shape[0] - 1
    else:
        weight = 1.0
    return weight * np.cov(time_series, rowvar=False)


def eigen_decomposition(
    covariance: np.ndarray, n_components: int
) -> (np.ndarray, np.ndarray):
    # Calculate the eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort the eigenvalues into desending order
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Only keep the first n_components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]

    return eigenvalues, eigenvectors


def prepare(
    subjects,
    n_embeddings: int,
    n_pca_components: int,
    whiten: bool,
    random_seed: int = None,
    return_pca_object: bool = False,
):
    """Prepares subject data by time embeddings and performing PCA.

    Follows the data preparation done in the OSL script teh_groupinference_parcels.m
    """

    for subject in subjects:
        # Perform time embedding
        subject.time_embed(n_embeddings, random_seed=random_seed)

        # Rescale (z-transform) the time series
        subject.scaler = StandardScaler()
        subject.time_series = subject.scaler.fit_transform(subject.time_series)

    pca_object = PCA(n_pca_components, svd_solver="full", whiten=whiten)
    for subject in subjects:
        pca_object.fit(subject.time_series)
    for subject in subjects:
        subject.time_series = pca_object.transform(subject.time_series)

    if return_pca_object:
        return subjects, pca_object

    return subjects


def inferred_states_to_source_space(inferred_covariances, state_time_course, meg_data):
    """Take VRAD results and transform them to source space."""

    n_states = inferred_covariances.shape[0]
    n_source_channels = meg_data[0].original_shape[1]
    n_embeddings = meg_data.pca.components_.shape[1] // n_source_channels
    n_samples = state_time_course.shape[0]

    state_indexer = np.eye(n_states, dtype=bool)

    state_covariances = np.empty((n_states, n_source_channels, n_source_channels))
    state_recon_time_series = np.empty((n_states, n_samples, n_source_channels))

    for state in trange(n_states, desc="Recovering source space"):
        inferred_state_covariance = (
            inferred_covariances * state_indexer[state][:, None, None]
        )

        state_sim = VRADSimulation(inferred_state_covariance, state_time_course)
        state_sim.simulate()

        time_embedded_state_sim = meg_data.pca.inverse_transform(state_sim.time_series)
        state_recon_time_series[state] = time_embedded_state_sim.reshape(
            n_samples, n_source_channels, -1
        )[..., (n_embeddings + 1) // 2]
        state_covariances[state] = np.cov(state_recon_time_series[state].T)

    return state_recon_time_series, state_covariances
