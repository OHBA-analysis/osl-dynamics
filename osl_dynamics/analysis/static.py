"""Functions to calculate static properties of time series data."""

import logging

import numpy as np
from sklearn.covariance import LedoitWolf
from pqdm.threads import pqdm
from tqdm.auto import trange

from osl_dynamics.analysis import spectral
from osl_dynamics.utils import array_ops

_logger = logging.getLogger("osl-dynamics")


def functional_connectivity(
    data, conn_type="corr", demean=True, window_length=None, n_jobs=1
):
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
    if len(data) == 1:
        # Don't need to parallelise
        results = [measure(data[0])]

    elif n_jobs == 1:
        # We have multiple arrays but we're running in serial
        results = []
        for i in trange(len(data), desc="Calculating FC"):
            results.append(measure(data[i]))

    else:
        # Calculate in parallel
        _logger.info("Calculating FC")
        results = pqdm(data, measure, n_jobs=n_jobs)

    return np.squeeze(results)


def welch_spectra(
    data,
    sampling_frequency,
    window_length=None,
    step_size=None,
    frequency_range=None,
    standardize=True,
    averaging="mean",
    calc_cpsd=False,
    calc_coh=False,
    return_weights=False,
    keepdims=False,
    n_jobs=1,
):
    """Calculate spectra using Welch's method.

    Wrapper for `spectral.welch_spectra <https://osl-dynamics.readthedocs\
    .io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html\
    #osl_dynamics.analysis.spectral.welch_spectra>`_ assuming only one
    state is active for all time points.

    Parameters
    ----------
    data : np.ndarray or list
        Time series data. Must have shape (n_sessions, n_samples,
        n_channels) or (n_samples, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
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
    calc_cpsd : bool, optional
        Should we return the cross spectra for :code:`psd`?
        If True, we force :code:`calc_coh` to False.
    calc_coh : bool, optional
        Should we also return the coherence spectra?
    return_weights : bool, optional
        Should we return the weights for session-specific spectra?
        This is useful for calculating a group average.
    keepdims : bool, optional
        Should we enforce a (n_sessions, ...) array is returned
        for :code:`psd` and :code:`coh`? If :code:`False`, we remove any
        dimension of length 1.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freq,).
    psd : np.ndarray
        Power spectra for each session. Shape is (n_sessions, n_channels,
        n_freq) if :code:`calc_cpsd=False`, otherwise the shape is (n_sessions,
        n_channels, n_channels, n_freq). Any axis of length 1 is removed if
        :code:`keepdims=False`.
    coh : np.ndarray
        Coherences. Shape is (n_sessions, n_channels, n_channels, n_freq).
        Any axis of length 1 is removed if :code:`keepdims=False`.
        Only returned is :code:`calc_coh=True`.
    w : np.ndarray
        Weighting for session-specific spectra. Only returned if
        :code:`return_weights=True`. Shape is (n_sessions,).
    """
    return spectral.welch_spectra(
        data=data,
        sampling_frequency=sampling_frequency,
        window_length=window_length,
        step_size=step_size,
        frequency_range=frequency_range,
        standardize=standardize,
        averaging=averaging,
        calc_cpsd=calc_cpsd,
        calc_coh=calc_coh,
        return_weights=return_weights,
        keepdims=keepdims,
        n_jobs=n_jobs,
    )


def multitaper_spectra(
    data,
    sampling_frequency,
    window_length=None,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=None,
    standardize=True,
    averaging="mean",
    calc_cpsd=False,
    calc_coh=False,
    return_weights=False,
    keepdims=False,
    n_jobs=1,
):
    """Calculate multitaper spectra.

    Wrapper for `spectral.multitaper_spectra <https://osl-dynamics.readthedocs\
    .io/en/latest/autoapi/osl_dynamics/analysis/spectral/index.html\
    #osl_dynamics.analysis.spectral.multitaper_spectra>`_ assuming only one
    state is active for all time points.

    Parameters
    ----------
    data : np.ndarray or list
        Time series data. Must have shape (n_sessions, n_samples,
        n_channels) or (n_samples, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    window_length : int, optional
        Length of the data segment to use to calculate the multitaper.
    time_half_bandwidth : float, optional
        Parameter to control the resolution of the spectra.
    n_tapers : int, optional
        Number of tapers.
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    standardize : bool, optional
        Should we standardize the data before calculating the multitaper?
    averaging : str, optional
        Method used to average periodograms.
        Must be :code:`'mean'` or :code:`'median'`.
    calc_cpsd : bool, optional
        Should we return the cross spectra for :code:`psd`?
        If True, we force :code:`calc_coh` to False.
    calc_coh : bool, optional
        Should we also return the coherence spectra?
    return_weights : bool, optional
        Should we return the weights for session-specific PSDs?
        Useful for calculating the group average PSD.
    keepdims : bool, optional
        Should we enforce a (n_sessions, ...) array is returned for
        :code:`psd` and :code:`coh`? If :code:`False`, we remove any dimension
        of length 1.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freq,).
    psd : np.ndarray
        Power spectra for each session. Shape is (n_sessions, n_channels,
        n_freq) if :code:`calc_cpsd=False`, otherwise the shape is (n_sessions,
        n_channels, n_channels, n_freq). Any axis of length 1 is removed if
        :code:`keepdims=False`.
    coh : np.ndarray
        Coherences. Shape is (n_sessions, n_channels, n_channels, n_freq).
        Any axis of length 1 is removed if :code:`keepdims=False`.
        Only returned is :code:`calc_coh=True`.
    w : np.ndarray
        Weighting for session-specific spectra. Only returned if
        :code:`return_weights=True`. Shape is (n_sessions,).
    """
    return spectral.multitaper_spectra(
        data=data,
        sampling_frequency=sampling_frequency,
        window_length=window_length,
        time_half_bandwidth=time_half_bandwidth,
        n_tapers=n_tapers,
        frequency_range=frequency_range,
        standardize=standardize,
        averaging=averaging,
        calc_cpsd=calc_cpsd,
        calc_coh=calc_coh,
        return_weights=return_weights,
        keepdims=keepdims,
        n_jobs=n_jobs,
    )
