"""Functions to perform spectral analysis of a fit.

"""

import logging

import numpy as np
from scipy.signal.windows import dpss
from sklearn.decomposition import non_negative_factorization
from tqdm import trange
from vrad.analysis.functions import fourier_transform, nextpow2, validate_array
from vrad.analysis.time_series import get_state_time_series
from vrad.data.manipulation import scale

_logger = logging.getLogger("VRAD")


def multitaper(
    data,
    sampling_frequency,
    nfft=None,
    tapers=None,
    time_half_bandwidth=None,
    n_tapers=None,
    args_range=None,
):
    """Calculates a power (or cross) spectral density using the multitaper method."""

    # Transpose the data so that it is [n_channels, n_samples]
    data = np.transpose(data)

    # Number of channels and length of each signal
    n_channels, n_samples = data.shape

    # Number of FFT data points to calculate
    if nfft is None:
        nfft = max(256, 2 ** nextpow2(n_samples))

    # If tapers are not passed we generate them here
    if tapers is None:

        # Check the time half width bandwidth and number of tapers has been passed
        if time_half_bandwidth is None or n_tapers is None:
            raise ValueError("time_half_bandwidth and n_tapers must be passed.")

        # Calculate tapers
        tapers = dpss(n_samples, NW=time_half_bandwidth, Kmax=n_tapers)
        tapers *= np.sqrt(sampling_frequency)

    else:
        # Get number of tapers from the tapers passed
        n_tapers = len(tapers)

    # Multiply the data by the tapers
    data = data[np.newaxis, :, :] * tapers[:, np.newaxis, :]

    # Calculate the FFT, X, which has shape [n_tapers, n_channels, n_f]
    X = fourier_transform(data, sampling_frequency, nfft=nfft, args_range=args_range)

    # Number of frequency bins in the FFT
    n_f = X.shape[-1]

    # Calculate the periodogram with each taper
    P = np.zeros([n_channels, n_channels, n_f], dtype=np.complex_)
    for i in range(n_tapers):
        for j in range(n_channels):
            for k in range(j, n_channels):
                P[j, k] += np.conjugate(X[i, j]) * X[i, k]
                if i == n_tapers - 1 and k != j:
                    P[k, j] = np.conjugate(P[j, k])

    return P


def state_spectra(
    data,
    state_probabilities,
    sampling_frequency,
    time_half_bandwidth,
    n_tapers,
    segment_length=None,
    frequency_range=None,
    n_embeddings=None,
):
    """Calculates spectra for inferred states.

    This include power spectra and coherence.
    Follows the same procedure as the OSL function HMM-MAR/spectral/hmmspectamt.m
    """

    # Validation
    if type(data) != type(state_probabilities):
        raise ValueError(
            f"data is type {type(data)} and state_probabilities is type "
            + f"{type(state_probabilities)}. They must both be lists or numpy arrays."
        )

    if isinstance(data, list):
        # A list of subject data has been pass so let's concatenate
        data = np.concatenate(data, axis=0)

    if isinstance(state_probabilities, list):
        # A list of subject state probabilities has been pass so let's concatenate
        state_probabilities = np.concatenate(state_probabilities, axis=0)

    if data.ndim != 2:
        raise ValueError(
            "data must have shape (n_samples, n_states) "
            + "or (n_subjects, n_samples, n_states)."
        )

    if state_probabilities.ndim != 2:
        raise ValueError(
            "state_probabilities must have shape (n_samples, n_states) "
            + "or (n_subjects, n_samples, n_states)."
        )

    if segment_length is None:
        segment_length = 2 * sampling_frequency

    elif segment_length != 2 * sampling_frequency:
        _logger.warning("segment_length is recommended to be 2*sampling_frequency.")

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    # Standardise (z-transform) the data
    data = scale(data, axis=0)

    # Use the state probabilities to get a time series for each state
    state_time_series = get_state_time_series(data, state_probabilities, n_embeddings)

    # Number of subjects, states, samples and channels
    n_states, n_samples, n_channels = state_time_series.shape

    # Number of FFT data points to calculate
    nfft = max(256, 2 ** nextpow2(segment_length))

    # Calculate the argments to keep for the given frequency range
    frequencies = np.arange(0, sampling_frequency / 2, sampling_frequency / nfft)
    f_min_arg = np.argwhere(frequencies > frequency_range[0])[0, 0]
    f_max_arg = np.argwhere(frequencies < frequency_range[1])[-1, 0]
    frequencies = frequencies[f_min_arg : f_max_arg + 1]
    args_range = [f_min_arg, f_max_arg + 1]

    # Number of frequency bins
    n_f = args_range[1] - args_range[0]

    # Calculate tapers so we can estimate spectra with the multitaper method
    tapers = dpss(segment_length, NW=time_half_bandwidth, Kmax=n_tapers)
    tapers *= np.sqrt(sampling_frequency)

    # We will calculate the spectrum for several non-overlapping segments
    # of the time series and return the average over these segments.

    # Number of segments in the time series
    n_segments = round(n_samples / segment_length)

    # Power spectra for each state
    power_spectra = np.zeros([n_states, n_channels, n_channels, n_f], dtype=np.complex_)

    print("Calculating power spectra")
    for i in range(n_states):
        for j in trange(n_segments, desc=f"State {i}", ncols=98):

            # Time series for state i and segment j
            time_series_segment = state_time_series[
                i, j * segment_length : (j + 1) * segment_length
            ]

            # If we're missing samples we pad with zeros either side of the data
            if time_series_segment.shape[0] != segment_length:
                n_zeros = segment_length - time_series_segment.shape[0]
                n_padding = n_zeros // 2
                time_series_segment = np.pad(time_series_segment, n_padding)[
                    :, n_padding:-n_padding
                ]

            # Calculate the power (and cross) spectrum using the multitaper method
            power_spectra[i] += multitaper(
                time_series_segment,
                sampling_frequency,
                nfft=nfft,
                tapers=tapers,
                args_range=args_range,
            )

    # Normalise the power spectra
    sum_probabilities = np.sum(state_probabilities ** 2, axis=0)[
        ..., np.newaxis, np.newaxis, np.newaxis
    ]
    power_spectra *= n_samples / (sum_probabilities * n_tapers * n_segments)

    # Coherences for each state
    coherences = np.empty([n_states, n_channels, n_channels, n_f])

    print("Calculating coherences")
    for i in range(n_states):
        for j in range(n_channels):
            for k in range(n_channels):
                coherences[i, j, k] = abs(
                    power_spectra[i, j, k]
                    / np.sqrt(power_spectra[i, j, j] * power_spectra[i, k, k])
                )

    return frequencies, np.squeeze(power_spectra), np.squeeze(coherences)


def decompose_spectra(
    coherences, n_components, max_iter=10000, random_state=None, verbose=0,
):
    """Performs spectral decomposition using coherences.

    Uses non-negative matrix factorization to decompose spectra.
    Follows the same procedure as the OSL funciton HMM-MAR/spectral/spectdecompose.m

    Return the spectral components.
    """
    print("Performing spectral decomposition")

    # Validation
    error_message = (
        "coherences must be a numpy array with shape (n_channels, n_channels), "
        + "(n_states, n_channels, n_channels) or (n_subjects, n_states, "
        + "n_channels, n_channels)."
    )
    coherences = validate_array(
        coherences,
        correct_dimensionality=5,
        allow_dimensions=[2, 3, 4],
        error_message=error_message,
    )

    # Number of subjects, states, channels and frequency bins
    n_subjects, n_states, n_channels, n_channels, n_f = coherences.shape

    # Indices of the upper triangle of the [n_channels, n_channels, n_f] sub-array
    i, j = np.triu_indices(n_channels, 1)

    # Concatenate coherences for each subject and state and only keep the upper triangle
    coherences = coherences[:, :, i, j].reshape(-1, n_f)

    # Perform non-negative matrix factorisation
    _, components, _ = non_negative_factorization(
        coherences,
        n_components=n_components,
        init=None,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )

    # Order the weights and components in ascending frequency
    order = np.argsort(components.argmax(axis=1))
    components = components[order]

    return components
