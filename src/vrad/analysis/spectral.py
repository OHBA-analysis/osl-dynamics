"""Functions to perform spectral analysis of a fit.

"""

import logging
import numpy as np
from scipy.signal.windows import dpss
from sklearn.decomposition import non_negative_factorization
from tqdm import trange

from vrad.analysis.functions import fourier_transform, nextpow2, residuals_gaussian_fit
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
):
    """Calculates spectra for inferred states.

    This include power spectra and coherence.
    Follows the same procedure as the OSL function HMM-MAR/spectral/hmmspectamt.m
    """

    # Validation
    if data.ndim == 2:
        # Data for one subject has been passed, so we add one dimension
        data = data[np.newaxis, :, :]

    if data.ndim != 3:
        raise ValueError(
            "data must have shape (n_samples, n_channels) or "
            + "(n_subjects, n_samples, n_channels)."
        )

    if state_probabilities.ndim == 2:
        # Data for one subject has been passed, so we add one dimension
        state_probabilities = state_probabilities[np.newaxis, :, :]

    if state_probabilities.ndim != 3:
        raise ValueError(
            "state_probabilities must have shape (n_samples, n_states) or "
            + "(n_subjects, n_samples, n_states)."
        )

    if segment_length is None:
        segment_length = 2 * sampling_frequency
    elif segment_length != 2 * sampling_frequency:
        _logger.warning("segment_length is recommended to be 2*sampling_frequency.")

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    # Standardise (z-transform) the data
    data = scale(data, axis=1)

    # Use the state probabilities to get a time series for each state
    state_time_series = get_state_time_series(data, state_probabilities)

    # Number of subjects, states, samples and channels
    n_subjects, n_states, n_samples, n_channels = state_time_series.shape

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
    power_spectra = np.zeros(
        [n_subjects, n_states, n_channels, n_channels, n_f], dtype=np.complex_
    )

    print("Calculating power spectra")
    for i in range(n_subjects):
        for j in range(n_states):
            for k in trange(n_segments, desc=f"Subject {i}, state {j}"):

                # Time series for state j and segment k
                time_series_segment = state_time_series[
                    i, j, k * segment_length : (k + 1) * segment_length
                ]

                # If we're missing samples we pad with zeros either side of the data
                if time_series_segment.shape[0] != segment_length:
                    n_zeros = segment_length - time_series_segment.shape[0]
                    n_padding = n_zeros // 2
                    time_series_segment = np.pad(time_series_segment, n_padding)[
                        :, n_padding:-n_padding
                    ]

                # Calculate the power (and cross) spectrum using the multitaper method
                power_spectra[i, j] += multitaper(
                    time_series_segment,
                    sampling_frequency,
                    nfft=nfft,
                    tapers=tapers,
                    args_range=args_range,
                )

    # Normalise the power spectra
    sum_probabilities = np.sum(state_probabilities ** 2, axis=1)[
        :, :, np.newaxis, np.newaxis, np.newaxis
    ]
    power_spectra *= n_samples / (sum_probabilities * n_tapers * n_segments)

    # Coherences for each state
    coherences = np.empty([n_subjects, n_states, n_channels, n_channels, n_f])

    print("Calculating coherences")
    for i in range(n_subjects):
        for j in range(n_states):
            for k in range(n_channels):
                for l in range(n_channels):
                    coherences[i, j, k, l] = abs(
                        power_spectra[i, j, k, l]
                        / np.sqrt(power_spectra[i, j, k, k] * power_spectra[i, j, l, l])
                    )

    return frequencies, np.squeeze(power_spectra), np.squeeze(coherences)


def decompose_spectra(
    spectra,
    n_components,
    n_iter=100,
    init="random",
    max_iter=2000,
    random_state=None,
    verbose=0,
):
    """Performs spectral decomposition.

    Uses non-negative matrix factorization to decompose spectra.
    A power spectrum or coherence spectrum can be passed.
    Follows the same procedure as the OSL funciton HMM-MAR/spectral/spectdecompose.m
    """
    print("Performing spectral decomposition")

    # Validation
    if spectra.ndim == 3:
        # The spectra for one subject and one state has been passed, so we
        # add two dimensions
        spectra = spectra[np.newaxis, np.newaxis, :, :, :]

    if spectra.ndim == 4:
        # The spectra for one subject has been passed, so we add one dimension
        spectra = spectra[np.newaxis, :, :, :]

    if spectra.ndim != 5:
        raise ValueError(
            "a 3D numpy array [n_channels, n_channels, n_frequency_bins] or "
            + "4D numpy array [n_states, n_channels, n_channels, n_frequency_bins] "
            + "must be passed for spectra."
        )

    # Number of subjects, states, channels and frequency bins
    n_subjects, n_states, n_channels, n_channels, n_f = spectra.shape

    # Indices of the upper triangle of the [n_channels, n_channels, n_f] sub-array
    i, j = np.triu_indices(n_channels, 1)

    # Concatenate spectra for each subject and state and only keep the upper triangle
    spectra = spectra[:, :, i, j].reshape(-1, n_f)

    # Perform full procedure n_iter times
    best_residuals_squared = np.Inf
    for i in trange(n_iter, desc="Iterating"):

        # Perform non-negative matrix factorisation
        weights, components, _ = non_negative_factorization(
            spectra,
            n_components=n_components,
            init=init,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
        )

        # Fit a Gaussian to each component and calculate residuals
        # This is done to find spectral components which have a single peak
        residuals_squared = residuals_gaussian_fit(components)

        # Keep the best factorisation
        if residuals_squared < best_residuals_squared:
            best_residuals_squared = residuals_squared
            best_weights = weights
            best_components = components

    # Order the weights and components in ascending frequency
    order = np.argsort(best_components.argmax(axis=1))
    best_weights = best_weights[:, order]
    best_components = best_components[order]

    return best_weights, best_components
