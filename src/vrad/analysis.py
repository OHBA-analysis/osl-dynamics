"""Functions to analyse results of a fit.

"""

import numpy as np
from scipy.signal.windows import dpss
from tqdm import trange

from vrad.data.manipulation import scale


def get_state_time_series(data, state_probabilities):
    """Returns the data for when a state is on."""

    # Make sure the data and state time courses have the same length
    n_samples = min(data.shape[0], state_probabilities.shape[0])
    data = data[:n_samples]
    state_probabilities = state_probabilities[:n_samples]

    # Number of states and channels
    n_states = state_probabilities.shape[1]
    n_channels = data.shape[1]

    # Get the corresponding time series for when a state is on
    state_time_series = np.empty([n_states, n_samples, n_channels])
    for i in range(n_states):
        state_time_series[i] = data * state_probabilities[:, i, np.newaxis]

    return state_time_series


def nextpow2(x):
    """Returns the smallest power of two that is greater than or equal to the
    absolute value of x.
    """
    res = np.ceil(np.log2(x))
    return res.astype("int")


def fourier_transform(data, sampling_frequency, nfft=None, args_range=None):
    """Calculates a Fast Fourier Transform (FFT)."""

    # Number of data points
    n = data.shape[-1]

    # Number of FFT data points to calculate
    if nfft is None:
        nfft = max(256, 2 ** nextpow2(n))

    # Calculate the FFT
    X = np.fft.fft(data, nfft) / sampling_frequency

    # Only keep the desired frequency range
    if args_range is not None:
        X = X[:, :, args_range[0] : args_range[1]]

    return X


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
    segment_length,
    time_half_bandwidth,
    n_tapers,
    frequency_range=None,
):
    """Calculates spectra for inferred states.

    Follows the same procedure are the OSL function HMM-MAR/spectral/hmmspectamt.m
    """

    # Validation
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError(
            "a 1D numpy array [n_samples] or 2D numpy array [n_samples, n_channels] "
            + "must be passed."
        )
    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    # Standardise (z-transform) the data
    data = scale(data)

    # Use the state probabilities to get a time series for each state
    state_time_series = get_state_time_series(data, state_probabilities)

    # Number of states, samples and channels
    n_states, n_samples, n_channels = state_time_series.shape

    # Number of FFT data points to calculate
    nfft = max(256, 2 ** nextpow2(segment_length))

    # Calculate the argments to keep for the given frequency range
    f = np.arange(0, sampling_frequency / 2, sampling_frequency / nfft)
    f_min_arg = np.argwhere(f > frequency_range[0])[0, 0]
    f_max_arg = np.argwhere(f < frequency_range[1])[-1, 0]
    f = f[f_min_arg:f_max_arg]
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

    # Spectra for each state and segment
    P = np.zeros([n_states, n_channels, n_channels, n_f], dtype=np.complex_)

    print("Calculating spectra:")
    for i in range(n_states):

        print(f"State {i}")
        for j in trange(n_segments, desc="Segments"):

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

            # Calculate the spectrum using the multitaper method
            P[i] += multitaper(
                time_series_segment,
                sampling_frequency,
                nfft=nfft,
                tapers=tapers,
                args_range=args_range,
            )

    # Normalise the spectra
    sum_probabilities = np.sum(state_probabilities ** 2, axis=0)[
        :, np.newaxis, np.newaxis, np.newaxis
    ]
    P *= n_samples / (sum_probabilities * n_tapers * n_segments)

    return f, P
