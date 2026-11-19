"""Functions to calculate static properties of time series data."""

import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import numpy as np
from sklearn.covariance import LedoitWolf

from osl_dynamics.analysis import spectral
from osl_dynamics.utils import array_ops
from osl_dynamics.utils.progress import parallel_map

if TYPE_CHECKING:
    from osl_dynamics.data import Data

_logger = logging.getLogger("osl-dynamics")


def functional_connectivity(
    data: Union[np.ndarray, List[np.ndarray]],
    conn_type: Union[str, Callable] = "corr",
    demean: bool = True,
    window_length: Optional[int] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Calculate functional connectivity networks.

    This function uses the `LedoitWolf <https://scikit-learn.org/stable\
    /modules/generated/sklearn.covariance.LedoitWolf.html>`_ estimator
    to calculate covariance. If another measure is requested it is calculated
    from this covariance matrix.

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape must be (n_sessions, n_samples, n_channels)
        or (n_samples, n_channels).
    conn_type : str or func, optional
        What measure should we use? Must be one of:

        - :code:`"cov"`: covariance.
        - :code:`"pcov"`: partial covariance.
        - :code:`"corr"`: Pearson correlation.
        - :code:`"pcorr"`: partial correlation.
        - :code:`"prec"`: precision (inverse of the covariance matrix).

        Or can be a function that accepts a (n_samples, n_channels) array and
        returns a (n_channels, n_channels) array.
    demean : bool, optional
        Should we demean the data before calculating the functional connectivity?
    window_length : int, optional
        Window length (in samples) to split the data and average the functional
        connectivity network for. If None, no averaging is done. Windows are
        non-overlapping.
    n_jobs : int, optional
        Number of jobs.

    Returns
    -------
    fc : np.ndarray
        Functional connectivity network.
        Shape is (n_sessions, n_channels, n_channels) or (n_channels, n_channels).
    """

    # Validation
    if isinstance(conn_type, str):
        allowed_conn_types = ["cov", "pcov", "corr", "pcorr", "prec"]
        if conn_type not in allowed_conn_types:
            raise ValueError(
                f"conn_type must be one of: {allowed_conn_types}, got: {conn_type}."
            )
    elif not callable(conn_type):
        raise TypeError("conn_type must be a str, list or callable function.")

    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            data = list(data)
        else:
            data = [data]

    def _calc_cov(x):
        # Function to calculate covariance
        lw = LedoitWolf(assume_centered=(not demean), store_precision=False)

        # Use all data to calculate the covariance
        if window_length is None:
            lw.fit(x)
            return lw.covariance_

        # Validation
        L = int(window_length)
        T = x.shape[0]
        n_windows = T // L
        if n_windows == 0:
            raise ValueError("window_length is larger than number of samples.")

        # Average the covariance for multiple windows
        covs = []
        for w in range(n_windows):
            start = w * L
            end = start + L
            chunk = x[start:end]
            lw.fit(chunk)
            covs.append(lw.covariance_)

        return np.mean(covs, axis=0)

    # Covariance
    if conn_type == "cov":
        measure = _calc_cov

    # Partial covariance
    elif conn_type == "pcov":

        def measure(x):
            cov = _calc_cov(x)
            return array_ops.cov2partialcov(cov)

    # Correlation
    elif conn_type == "corr":

        def measure(x):
            cov = _calc_cov(x)
            return array_ops.cov2corr(cov)

    # Partial correlation
    elif conn_type == "pcorr":

        def measure(x):
            cov = _calc_cov(x)
            return array_ops.cov2partialcorr(cov)

    # Precision
    elif conn_type == "prec":

        def measure(x):
            cov = _calc_cov(x)
            return np.linalg.inv(cov)

    # User specified function
    else:
        measure = conn_type

    # Calculate networks
    results = parallel_map(
        measure, data, desc="Calculating FC", n_jobs=n_jobs, label="session"
    )

    return np.squeeze(results)


def welch_spectra(
    data: Union[np.ndarray, list, "Data"],
    sampling_frequency: Optional[float] = None,
    *,
    window_length: Optional[int] = None,
    step_size: Optional[int] = None,
    frequency_range: Optional[List[float]] = None,
    standardize: bool = True,
    averaging: str = "mean",
    n_jobs: int = 1,
    cross_spectra: bool = True,
) -> spectral.Spectra:
    """Calculate static spectra using Welch's method.

    The scaling for the power spectra matches SciPy
    (:code:`scipy.signal.welch`).

    Parameters
    ----------
    data : np.ndarray, list or osl_dynamics.data.Data
        Time series data. Must have shape (n_sessions, n_samples,
        n_channels) or (n_samples, n_channels). If a Data object is passed,
        we use the data before it was prepared.
    sampling_frequency : float, optional
        Sampling frequency in Hz. Defaults to
        :code:`data.sampling_frequency` for a Data object.
    window_length : int, optional
        Length of the data segment to use to calculate spectra.
        If None, we use :code:`2 * sampling_frequency`.
    step_size : int, optional
        Step size for shifting the window. If None, we use
        :code:`window_length // 2`.
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    standardize : bool, optional
        Should we standardize the data before calculating the spectra?
    averaging : str, optional
        Method used to average periodograms.
        Must be :code:`'mean'` or :code:`'median'`.
    n_jobs : int, optional
        Number of parallel jobs.
    cross_spectra : bool, optional
        Should we calculate the cross spectra (needed for the coherence)?

    Returns
    -------
    spectra : osl_dynamics.analysis.spectral.Spectra
        Spectra for each session.
    """
    data, _, sampling_frequency = spectral._resolve_data(data, None, sampling_frequency)
    if window_length is None:
        window_length = 2 * sampling_frequency
    if step_size is None:
        step_size = int(window_length) // 2

    return spectral._state_spectra(
        method="welch",
        data=data,
        sampling_frequency=sampling_frequency,
        alpha=None,
        window_length=window_length,
        step_size=step_size,
        frequency_range=frequency_range,
        standardize=standardize,
        averaging=averaging,
        cross_spectra=cross_spectra,
        n_jobs=n_jobs,
    )


def multitaper_spectra(
    data: Union[np.ndarray, list, "Data"],
    sampling_frequency: Optional[float] = None,
    *,
    window_length: Optional[int] = None,
    time_half_bandwidth: float = 4,
    n_tapers: int = 7,
    frequency_range: Optional[List[float]] = None,
    standardize: bool = True,
    averaging: str = "mean",
    n_jobs: int = 1,
    step_size: Optional[int] = None,
    cross_spectra: bool = True,
) -> spectral.Spectra:
    """Calculate static spectra using a multitaper.

    Parameters
    ----------
    data : np.ndarray, list or osl_dynamics.data.Data
        Time series data. Must have shape (n_sessions, n_samples,
        n_channels) or (n_samples, n_channels). If a Data object is passed,
        we use the data before it was prepared.
    sampling_frequency : float, optional
        Sampling frequency in Hz. Defaults to
        :code:`data.sampling_frequency` for a Data object.
    window_length : int, optional
        Length of the data segment to use to calculate the multitaper.
        If None, we use :code:`2 * sampling_frequency`.
    time_half_bandwidth : float, optional
        Parameter to control the resolution of the spectra.
    n_tapers : int, optional
        Number of tapers.
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    standardize : bool, optional
        Should we standardize the data before calculating the spectra?
    averaging : str, optional
        Method used to average periodograms.
        Must be :code:`'mean'` or :code:`'median'`.
    n_jobs : int, optional
        Number of parallel jobs.
    step_size : int, optional
        Step size for shifting the window. If None, we use
        :code:`window_length` (no overlap).
    cross_spectra : bool, optional
        Should we calculate the cross spectra (needed for the coherence)?

    Returns
    -------
    spectra : osl_dynamics.analysis.spectral.Spectra
        Spectra for each session.
    """
    data, _, sampling_frequency = spectral._resolve_data(data, None, sampling_frequency)
    if window_length is None:
        window_length = 2 * sampling_frequency
    if step_size is None:
        step_size = int(window_length)

    return spectral._state_spectra(
        method="multitaper",
        data=data,
        sampling_frequency=sampling_frequency,
        alpha=None,
        window_length=window_length,
        step_size=step_size,
        frequency_range=frequency_range,
        standardize=standardize,
        averaging=averaging,
        cross_spectra=cross_spectra,
        n_jobs=n_jobs,
        time_half_bandwidth=time_half_bandwidth,
        n_tapers=n_tapers,
    )
