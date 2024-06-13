"""Functions to calculate static properties of time series data.

"""

import logging

import numpy as np

from osl_dynamics.analysis import spectral

_logger = logging.getLogger("osl-dynamics")


def functional_connectivity(data, conn_type="corr"):
    """Calculate functional connectivity (Pearson correlation).

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape must be (n_sessions, n_samples, n_channels)
        or (n_samples, n_channels).
    conn_type : str, optional
        What metric should we use? :code:`"corr"` (Pearson correlation) or
        :code:`"cov"` (covariance).

    Returns
    -------
    fc : np.ndarray
        Functional connectivity. Shape is (n_sessions, n_channels, n_channels)
        or (n_channels, n_channels).
    """
    if conn_type not in ["corr", "cov"]:
        raise ValueError(f"conn_type must be 'corr' or 'cov', got: {conn_type}")

    if conn_type == "cov":
        metric = np.cov
    else:
        metric = np.corrcoef

    if isinstance(data, np.ndarray):
        data = [data]
    fc = [metric(d, rowvar=False) for d in data]
    return np.squeeze(fc)


def welch_spectra(
    data,
    sampling_frequency,
    window_length=None,
    step_size=None,
    frequency_range=None,
    return_weights=False,
    standardize=True,
    calc_coh=False,
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
        Power spectra. Shape is (n_sessions, n_channels, n_freq).
        Any axis of length 1 is removed if :code:`keepdims=False`.
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
        Length of the data segement to use to calculate the multitaper.
    time_half_bandwidth : float, optional
        Parameter to control the resolution of the spectra.
    n_tapers : int, optional
        Number of tapers.
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    standardize : bool, optional
        Should we standardize the data before calculating the multitaper?
    calc_coh : bool, optional
        Should we also return the coherence spectra?
    return_weights : bool, optional
        Should we return the weights for session-specific PSDs?
        Useful for calculating the group average PSD.
    keepdims : bool, optional
        Should we enforce a (n_sessions, n_states, ...) array is returned for
        :code:`psd` and :code:`coh`? If :code:`False`, we remove any dimension
        of length 1.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freq,).
    psd : np.ndarray
        Power spectra. Shape is (n_sessions, n_channels, n_freq).
        Any axis of length 1 is removed if :code:`keepdims=False`.
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
        calc_coh=calc_coh,
        return_weights=return_weights,
        keepdims=keepdims,
        n_jobs=n_jobs,
    )
