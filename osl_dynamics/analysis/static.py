"""Functions to calculate static properties of time series data.

"""

import numpy as np
from pqdm.processes import pqdm

from osl_dynamics.analysis import spectral


def functional_connectivity(data, conn_type="corr"):
    """Calculate functional connectivity (Pearson correlation).

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape must be (n_subjects, n_samples, n_channels)
        or (n_samples, n_channels).
    conn_type : str
        What metric should we use? "corr" (Pearson correlation) or "cov" (covariance).

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
        (n_subjects,  n_samples, n_channels) or (n_samples, n_channels).
    window_length : int
        Number of data points to use when calculating the periodogram.
    sampling_frequency : float
        Sampling frequency in Hz.
    step_size : int
        Step size for shifting the window. By default we will use half
        the window length.
    standardize : bool
        Should we standardise the data?
    calc_coh : bool
        Should we also return the coherence spectra?
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    f : np.ndarray
        Frequency axis. Shape is (n_freq,).
    psd : np.ndarray
        Power spectral density. Shape is (n_subjects, n_channels, n_freq).
    coh : np.ndarray
        Coherence spectra. Shape is (n_subjects, n_channels, n_channels, n_freq).
        Only returned is calc_coh=True.
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
        # We only have one subject so we don't need to parallelise the calculation
        results = spectral.spectrogram(
            data=d,
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
        print("Calculating power spectra")
        results = pqdm(
            args, spectral.spectrogram, n_jobs=n_jobs, argument_type="args", ncols=98
        )

    # Unpack results
    psd = []
    for result in results:
        _, f, p = result
        psd.append(np.mean(p, axis=0))

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

        return f, np.squeeze(psd), np.squeeze(coh)

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
    n_jobs=1,
):
    """Calculate static power spectra using a multitaper.

    Parameters
    ----------
    data : np.ndarray or list
        Raw time series data. Must have shape (n_subjects, n_samples, n_channels)
        or (n_samples, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    time_half_bandwidth : float
        Parameter to control the resolution of the spectra.
    n_tapers : int
        Number of tapers to use when calculating the multitaper.
    segment_length : int
        Length of the data segement to use to calculate the multitaper.
    frequency_range : list
        Minimum and maximum frequency to keep.
    return_weights : bool
        Should we return the weights for subject-specific PSDs?
        Useful for calculating the group average PSD.
    standardize : bool
        Should we standardize the data before calculating the multitaper?
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    frequencies : np.ndarray
        Frequencies of the power spectra and coherences.
        Shape is (n_freq,).
    power_spectra : np.ndarray
        Power spectra for each state.
        Shape is (n_subjects, n_channels, n_freq).
    coherences : np.ndarray
        Coherences for each state.
        Shape is (n_subjects, n_channels, n_channels, n_freq).
    weights : np.ndarray
        Weight for each subject-specific PSD. Only returned if return_weights=True.
        Shape is (n_subjects,).
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
        n_jobs,
    )
