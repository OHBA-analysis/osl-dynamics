"""Functions to calculate static properties of time series data.

Note
----
This module is used in the following tutorials:

- `Static Spectral Analysis <https://osl-dynamics.readthedocs.io/en/latest\
  /tutorials_build/static_spectra_analysis.html>`_.
- `Static Power Analysis <https://osl-dynamics.readthedocs.io/en/latest\
  /tutorials_build/static_power_analysis.html>`_
- `Static AEC Analysis <https://osl-dynamics.readthedocs.io/en/latest\
  /tutorials_build/static_aec_analysis.html>`_.
"""

import logging

import numpy as np
from pqdm.processes import pqdm

from osl_dynamics.analysis import spectral

_logger = logging.getLogger("osl-dynamics")


def functional_connectivity(data, conn_type="corr"):
    """Calculate functional connectivity (Pearson correlation).

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape must be (n_subjects, n_samples, n_channels)
        or (n_samples, n_channels).
    conn_type : str, optional
        What metric should we use? :code:`"corr"` (Pearson correlation) or
        :code:`"cov"` (covariance).

    Returns
    -------
    fc : np.ndarray
        Functional connectivity. Shape is (n_subjects, n_channels, n_channels)
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


def power_spectra(
    data,
    window_length,
    sampling_frequency=1.0,
    frequency_range=None,
    step_size=None,
    return_weights=False,
    standardize=True,
    calc_coh=False,
    n_jobs=1,
):
    """Calculate static power spectra.

    This function uses Welch's method to calculate the spectra.

    Parameters
    ----------
    data : np.ndarray
        Data to calculate the spectrogram for. Shape must be
        (n_subjects, n_samples, n_channels) or (n_samples, n_channels).
    window_length : int
        Number of data points to use when calculating the periodogram.
    sampling_frequency : float, optional
        Sampling frequency in Hz.
    step_size : int, optional
        Step size for shifting the window. By default we will use
        :code:`step_size=window_length // 2`.
    return_weights : bool, optional
        Should we return the weights for subject-specific PSDs?
        Useful for calculating the group average PSD.
    standardize : bool, optional
        Should we standardise the data?
    calc_coh : bool, optional
        Should we also return the coherence spectra?
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    f : np.ndarray
        Frequency axis. Shape is (n_freq,).
    psd : np.ndarray
        Power spectral density. Shape is (n_subjects, n_channels, n_freq).
    coh : np.ndarray
        Coherence spectra. Shape is (n_subjects, n_channels, n_channels,
        n_freq). Only returned is :code:`calc_coh=True`.
    weights : np.ndarray
        Weight for each subject-specific PSD. Only returned if
        :code:`return_weights=True`.
    """

    # Validation
    if isinstance(data, np.ndarray):
        data = [data]

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    if step_size is None:
        step_size = window_length // 2

    # Standardise
    if standardize:
        data = [(d - np.mean(d, axis=0)) / np.std(d, axis=0) for d in data]

    if len(data) == 1:
        # We only have one subject so we don't need to parallelise
        # the calculation
        results = spectral.spectrogram(
            data=data[0],
            window_length=window_length,
            sampling_frequency=sampling_frequency,
            frequency_range=frequency_range,
            calc_cpsd=calc_coh,
            step_size=step_size,
            n_sub_windows=1,
            print_progress_bar=True,
        )
        results = [results]

    else:
        # Create arguments to pass to a function in parallel
        args = []
        for d in data:
            args.append(
                [
                    d,
                    window_length,
                    sampling_frequency,
                    frequency_range,
                    calc_coh,
                    step_size,
                    1,
                    False,
                ]
            )

        # Calculate power spectra
        _logger.info("Calculating spectra")
        results = pqdm(
            args,
            spectral.spectrogram,
            n_jobs=n_jobs,
            argument_type="args",
        )

    # Unpack results
    psd = []
    for result in results:
        _, f, p = result
        psd.append(np.mean(p, axis=0))

    # Weights for calculating the group average PSD
    n_samples = [d.shape[0] for d in data]
    weights = np.array(n_samples) / np.sum(n_samples)

    if calc_coh:
        psd = np.array(psd)
        n_subjects = psd.shape[0]
        n_channels = data[0].shape[1]
        n_freq = psd.shape[2]

        # Build an n_channels by n_channels cross spectra matrix
        m, n = np.triu_indices(n_channels)
        cpsd = np.empty(
            [n_subjects, n_channels, n_channels, n_freq], dtype=np.complex64
        )
        for i in range(n_subjects):
            cpsd[i, m, n] = psd[i]
            cpsd[i, n, m] = np.conj(psd[i])

        # Calculate coherences
        coh = spectral.coherence_spectra(cpsd)

        # The PSD is the diagonal of the cross spectra matrix
        psd = cpsd[:, range(n_channels), range(n_channels)].real

        if return_weights:
            return f, np.squeeze(psd), np.squeeze(coh), weights
        else:
            return f, np.squeeze(psd), np.squeeze(coh)

    else:
        if return_weights:
            return f, np.squeeze(psd), weights
        else:
            return f, np.squeeze(psd)


def multitaper_spectra(
    data,
    sampling_frequency,
    time_half_bandwidth,
    n_tapers,
    segment_length=None,
    frequency_range=None,
    return_weights=False,
    standardize=True,
    calc_coh=False,
    n_jobs=1,
):
    """Calculate static power spectra using a multitaper.

    Parameters
    ----------
    data : np.ndarray or list
        Raw time series data. Must have shape (n_subjects, n_samples,
        n_channels) or (n_samples, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    time_half_bandwidth : float
        Parameter to control the resolution of the spectra.
    n_tapers : int
        Number of tapers to use when calculating the multitaper.
    segment_length : int, optional
        Length of the data segement to use to calculate the multitaper.
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    return_weights : bool, optional
        Should we return the weights for subject-specific PSDs?
        Useful for calculating the group average PSD.
    standardize : bool, optional
        Should we standardize the data before calculating the multitaper?
    calc_coh : bool, optional
        Should we also return the coherence spectra?
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    frequencies : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freq,).
    power_spectra : np.ndarray
        Power spectra for each state. Shape is (n_subjects, n_channels, n_freq).
    coherences : np.ndarray
        Coherences for each state.
        Shape is (n_subjects, n_channels, n_channels, n_freq).
        Only returned is :code:`calc_coh=True`.
    weights : np.ndarray
        Weight for each subject-specific PSD. Only returned if
        :code:`return_weights=True`. Shape is (n_subjects,).
    """
    if isinstance(data, np.ndarray):
        if data.ndim != 3:
            data = [data]
    alpha = [np.ones([d.shape[0], 1], dtype=np.float32) for d in data]
    return spectral.multitaper_spectra(
        data,
        alpha,
        sampling_frequency,
        time_half_bandwidth,
        n_tapers,
        segment_length,
        frequency_range,
        return_weights,
        standardize,
        calc_coh,
        n_jobs,
    )
