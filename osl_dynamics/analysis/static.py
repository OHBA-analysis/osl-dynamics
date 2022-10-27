"""Functions to calculate static properties of time series data.

"""

import numpy as np

from osl_dynamics.analysis.spectral import spectrogram, coherence_spectra


def functional_connectivity(data, conn_type="corr"):
    """Calculate functional connectivity (Pearson correlation).

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape must be (n_subjects, n_samples, n_channels)
        or (n_samples, n_channels).
    conn_type : str
        What metric should we use?"corr" or "cov".

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
    standardize=False,
    calc_coh=False,
):
    """Calculate static power spectra.

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

    Returns
    -------
    f : np.ndarray
        Frequency axis. Shape is (n_f,).
    psd : np.ndarray
        Power spectral density. Shape is (n_subjects, n_channels, n_f).
    coh : np.ndarray
        Coherence spectra. Shape is (n_subjects, n_channels, n_channels, n_f).
        Only returned is calc_coh=True.
    """

    # Validation
    if isinstance(data, np.ndarray):
        data = [data]

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    if step_size is None:
        step_size = window_length // 2

    # List to hold the static PSD/coherence for each subject
    psd = []
    if calc_coh:
        coh = []

    # Loop over subjects
    for i, d in enumerate(data):
        if len(data) > 1:
            print(f"Subject {i}")

        # Standardisation
        if standardize:
            d = np.copy(d)
            d -= np.mean(d, axis=0)
            d /= np.std(d, axis=0)

        # Calculate spectrogram (time-varying PSD)
        _, f, p = spectrogram(
            data=d,
            window_length=window_length,
            sampling_frequency=sampling_frequency,
            frequency_range=frequency_range,
            calc_cpsd=calc_coh,
            step_size=step_size,
            n_sub_windows=1,
        )

        # Average over time
        psd.append(np.mean(p, axis=0))

    if calc_coh:
        psd = np.array(psd)
        n_subjects = psd.shape[0]
        n_channels = data[0].shape[1]
        n_f = psd.shape[2]

        # Build an n_channels by n_channels cross spectra matrix
        m, n = np.triu_indices(n_channels)
        cpsd = np.empty([n_subjects, n_channels, n_channels, n_f], dtype=np.complex64)
        for i in range(n_subjects):
            cpsd[i, m, n] = psd[i]
            cpsd[i, n, m] = np.conj(psd[i])

        # Calculate coherences
        coh = coherence_spectra(cpsd)

        # The PSD is the diagonal of the cross spectra matrix
        psd = cpsd[:, range(n_channels), range(n_channels)].real

        return f, np.squeeze(psd), np.squeeze(coh)

    else:
        return f, np.squeeze(psd)
