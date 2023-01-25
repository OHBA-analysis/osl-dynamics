"""Functions to perform spectral analysis.

"""

import warnings

import numpy as np
from scipy.signal.windows import dpss, hann
from sklearn.decomposition import non_negative_factorization
from tqdm import trange
from pqdm.processes import pqdm

from osl_dynamics import array_ops
from osl_dynamics.analysis import regression


def nextpow2(x):
    """Next power of 2.

    Parameters
    ----------
    x : int
        Any integer.

    Returns
    -------
    res : int
        The smallest power of two that is greater than or equal to the absolute
        value of x.
    """
    res = np.ceil(np.log2(x))
    return res.astype("int")


def get_state_time_series(data, alpha):
    """Calculate a time series for each state.

    Time series calculated as the raw time series (of preprocessed data) multiplied
    by the state probability or time course.

    Parameters
    ----------
    data : np.ndarray
        Raw data time series with shape (n_samples, n_channels).
    alpha : np.ndarray
        Mode mixing factors alpha_t with shape (n_samples, n_states).

    Returns
    -------
    state_time_series : np.ndarray
        Time series for each state. Shape is (n_states, n_samples, n_channels).

    """
    # Make sure the data and state time courses have the same length
    if data.shape[0] != alpha.shape[0]:
        raise ValueError(
            "data and alpha have different lengths:"
            + f"data.shape[0]={data.shape[0]},"
            + f"alpha.shape[0]={alpha.shape[0]}"
        )

    # Number of samples, channels and states
    n_samples = data.shape[0]
    n_channels = data.shape[1]
    n_states = alpha.shape[1]

    # Get the corresponding time series for when a state is on
    state_time_series = np.empty([n_states, n_samples, n_channels], dtype=np.float32)
    for i in range(n_states):
        state_time_series[i] = data * alpha[:, i, np.newaxis]

    return state_time_series


def window_mean(data, window_length, step_size=1, n_sub_windows=1):
    """Applies a windowing function to a time series and takes the mean.

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape is (n_samples, n_modes).
    window_length : int
        Number of data points in a window.
    step_size : int
        Step size for shifting the window.
    n_sub_windows : int
        Should we split the window into a set of sub-windows and average each sub-window.

    Returns
    -------
    a : np.ndarray
        Mean for each window.
    """

    # Number of samples and modes
    n_samples = data.shape[0]
    n_modes = data.shape[1]

    # First pad the data
    data = np.pad(data, window_length // 2)[
        :, window_length // 2 : window_length // 2 + n_modes
    ]

    # Window to apply to the data
    window = hann(window_length // n_sub_windows)

    # Indices of time points to calculate a periodogram for
    time_indices = range(0, n_samples, step_size)
    n_windows = n_samples // step_size

    # Array to hold mean of data multiplied by the windowing function
    a = np.empty([n_windows, n_modes], dtype=np.float32)
    for i in range(n_windows):

        # Alpha in the window
        j = time_indices[i]
        a_window = data[j : j + window_length]

        # Calculate data for the sub-window by taking the mean
        # over time after applying the windowing function
        a_sub_window = np.empty([n_sub_windows, n_modes], dtype=np.float32)
        for k in range(n_sub_windows):
            a_sub_window[k] = np.mean(
                a_window[
                    k
                    * window_length
                    // n_sub_windows : (k + 1)
                    * window_length
                    // n_sub_windows
                ]
                * window[..., np.newaxis],
                axis=0,
            )

        # Average data for each sub-window
        a[i] = np.mean(a_sub_window, axis=0)

    return a


def coherence_spectra(power_spectra, print_message=True):
    """Calculates coherences from (cross) power spectral densities.

    Parameters
    ----------
    power_spectra : np.ndarray
        Power spectra. Shape is (n_modes, n_channels, n_channels, n_freq).
    print_message : bool
        Should we print a message to screen?

    Returns
    -------
    coherences : np.ndarray
        Coherence spectra for each mode.
        Shape is (n_modes, n_channels, n_channels, n_freq).
    """
    n_modes, n_channels, n_channels, n_freq = power_spectra.shape

    if print_message:
        print("Calculating coherences")

    coherences = np.empty([n_modes, n_channels, n_channels, n_freq])
    for i in range(n_modes):
        for j in range(n_channels):
            for k in range(n_channels):
                coherences[i, j, k] = abs(power_spectra[i, j, k]) / np.sqrt(
                    power_spectra[i, j, j].real * power_spectra[i, k, k].real
                )

    # Zero nan values
    return np.nan_to_num(coherences)


def decompose_spectra(
    coherences,
    n_components,
    max_iter=50000,
    random_state=None,
    verbose=0,
):
    """Performs spectral decomposition using coherences.

    Uses non-negative matrix factorization to decompose spectra.
    Follows the same procedure as the OSL function HMM-MAR/spectral/spectdecompose.m

    Parameters
    ----------
    coherences : np.ndarray
        Coherences spectra.
    n_components : int
        Number of spectral components to fit.
    max_iter : int
        Maximum number of iterations in sklearn's non_negative_factorization.
    random_state : int
        Seed for the random number generator.
    verbose : int
        Show verbose? (1) yes, (0) no.

    Returns
    -------
    components : np.ndarray
        Spectral components. Shape is (n_components, n_freq).
    """
    print("Performing spectral decomposition")

    # Validation
    error_message = (
        "coherences must be a numpy array with shape "
        + "(n_channels, n_channels, n_freq), "
        + "(n_modes, n_channels, n_channels, n_freq) or "
        + "(n_subjects, n_modes, n_channels, n_channels, n_freq)."
    )
    coherences = array_ops.validate(
        coherences,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    # Number of subjects, modes, channels and frequency bins
    n_subjects, n_modes, n_channels, n_channels, n_freq = coherences.shape

    # Indices of the upper triangle of the [n_channels, n_channels, n_freq] sub-array
    i, j = np.triu_indices(n_channels, 1)

    # Concatenate coherences for each subject and mode and only keep the upper triangle
    coherences = coherences[:, :, i, j].reshape(-1, n_freq)

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


def fourier_transform(
    data,
    nfft,
    args_range=None,
    one_side=False,
):
    """Calculates a Fast Fourier Transform (FFT).

    Parameters
    ----------
    data : np.ndarray
        Data with shape (n_samples, n_channels) to FFT.
    nfft : int
        Number of points in the FFT.
    args_range : list
        Minimum and maximum indices of the FFT to keep.
    one_side : bool
        Should we return a one-sided FFT?

    Returns
    -------
    X : np.ndarray
        FFT data.
    """

    # Calculate the FFT
    X = np.fft.fft(data, nfft)

    # Only keep the postive frequency side
    if one_side:
        X = X[..., : X.shape[-1] // 2]

    # Only keep the desired frequency range
    if args_range is not None:
        X = X[..., args_range[0] : args_range[1]]

    return X


def get_frequency_args_range(frequencies, frequency_range):
    """Get min/max indices of a range in a frequency axis.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis.
    frequency_range : list of len 2
        Min/max frequency.

    Returns
    -------
    args_range : list of len 2
        Min/max index.
    """
    f_min_arg = np.argwhere(frequencies >= frequency_range[0])[0, 0]
    f_max_arg = np.argwhere(frequencies <= frequency_range[1])[-1, 0]
    if f_max_arg <= f_min_arg:
        raise ValueError("Cannot select requested frequency range.")
    args_range = [f_min_arg, f_max_arg + 1]
    return args_range


def mar_spectra(coeffs, covs, sampling_frequency, n_freq=512):
    """Calculates cross power spectral densities from MAR model parameters.

    Parameters
    ----------
    coeffs : np.ndarray
        MAR coefficients. Shape must be (n_modes, n_channels, n_channels) or
        (n_channels, n_channels) or (n_channels,).
    covs : np.ndarray
        MAR covariances. Shape must be (n_modes, n_channels, n_channels) or
        (n_channels, n_channels).
    sampling_frequency : float
        Sampling frequency in Hertz.
    n_freq : int
        Number of frequency bins in the cross power spectral density to calculate.

    Returns
    -------
    f : np.ndarray
        Frequency axis. Shape is (n_freq,).
    P : np.ndarray
        Cross power spectral densities. Shape is (n_freq, n_modes, n_channels,
        n_channels) or (n_freq, n_channels, n_channels).
    """
    # Validation
    if covs.shape[-1] != covs.shape[-2]:
        if covs.ndim == 2:
            covs = [np.diag(c) for c in covs]
        else:
            covs = np.diag(covs)
    error_message = (
        "covs must be a numpy array with shape "
        + "(n_modes, n_channels, n_channels), "
        + "(n_channels, n_channels) or (n_channels,)."
    )
    covs = array_ops.validate(
        covs,
        correct_dimensionality=3,
        allow_dimensions=[2],
        error_message=error_message,
    )
    error_message = (
        "coeffs must be a numpy array with shape "
        + "(n_modes, n_lags, n_channels, n_channels), "
        + "(n_lags, n_channels, n_channels)."
    )
    coeffs = array_ops.validate(
        coeffs,
        correct_dimensionality=4,
        allow_dimensions=[3],
        error_message=error_message,
    )

    n_modes = coeffs.shape[0]
    n_lags = coeffs.shape[1]
    n_channels = coeffs.shape[-1]

    # Frequencies to evaluate the PSD at
    f = np.arange(0, sampling_frequency / 2, sampling_frequency / (2 * n_freq))

    # z-transform of the coefficients
    A = np.zeros([n_freq, n_modes, n_channels, n_channels], dtype=np.complex64)
    for i in range(n_freq):
        for l in range(n_lags):
            z = np.exp(-1j * (l + 1) * 2 * np.pi * f[i] / sampling_frequency)
            A[i] += coeffs[:, l] * z

    # Transfer function
    I = np.identity(n_channels)[np.newaxis, np.newaxis, ...]
    e = 1e-6 * I
    H = np.linalg.inv(I - A + e)

    # Cross PSDs
    P = H @ covs[np.newaxis, ...] @ np.transpose(np.conj(H), axes=[0, 1, 3, 2])

    return f, np.squeeze(P)


def mode_covariance_spectra(
    autocorrelation_function,
    sampling_frequency,
    nfft=64,
    frequency_range=None,
):
    """Calculates spectra from the autocorrelation function.

    The power spectrum of each mode is calculated as the Fourier transform of
    the auto-correlation function. Coherences are calculated from the power spectra.

    Parameters
    ----------
    autocorrelation_function : np.ndarray
        Mode autocorrelation functions.
        Shape must be (n_modes, n_channels, n_channels, n_acf).
    sampling_frequency : float
        Frequency at which the data was sampled (Hz).
    nfft : int
        Number of data points in the FFT. The auto-correlation function will only
        have 2 * (n_embeddings + 2) - 1 data points. We pad the auto-correlation
        function with zeros to have nfft data points if the number of data points
        in the auto-correlation function is less than nfft. Default is 64.
    frequency_range : list
        Minimum and maximum frequency to keep (Hz).

    Returns
    -------
    frequencies : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freq,).
    power_spectra : np.ndarray
        Power (or cross) spectra calculated for each mode. Shape is (n_modes,
        n_channels, n_channels, n_freq).
    coherences : np.ndarray
        Coherences calculated for each mode. Shape is (n_modes, n_channels,
        n_channels, n_freq).
    """
    print("Calculating power spectra")

    # Validation
    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    # Number of data points in the autocorrelation function and FFT
    n_acf = autocorrelation_function.shape[-1]
    nfft = max(nfft, 2 ** nextpow2(n_acf))

    # Calculate the argments to keep for the given frequency range
    frequencies = np.arange(0, sampling_frequency / 2, sampling_frequency / nfft)
    args_range = get_frequency_args_range(frequencies, frequency_range)
    frequencies = frequencies[args_range[0] : args_range[1]]

    # Calculate cross power spectra as the Fourier transform of the
    # auto/cross-correlation function
    power_spectra = abs(fourier_transform(autocorrelation_function, nfft, args_range))

    # Normalise the power spectra
    power_spectra /= nfft**2

    # Coherences for each mode
    coherences = coherence_spectra(power_spectra)

    return frequencies, np.squeeze(power_spectra), np.squeeze(coherences)


def multitaper(
    data,
    sampling_frequency,
    nfft=None,
    tapers=None,
    time_half_bandwidth=None,
    n_tapers=None,
    args_range=None,
):
    """Calculates a power (or cross) spectral density using the multitaper method.

    Parameters
    ----------
    data : np.ndarray
        Data with shape (n_samples, n_channels) to calculate a multitaper for.
    sampling_frequency : float
        Frequency used to sample the data (Hz).
    nfft : int
        Number of points in the FFT.
    tapers : np.ndarray
        Taper functions.
    time_half_bandwidth : float
        Parameter to control the resolution of the multitaper.
    n_tapers : int
        Number of tapers.
    args_range : list
        Minimum and maximum indices of the multitaper to keep.

    Returns
    -------
    P : np.ndarray
        Power (or cross) spectral density with shape (n_channels, n_channels, n_freq).
    """

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

    # Calculate the FFT, X, which has shape [n_tapers, n_channels, n_freq]
    X = fourier_transform(data, nfft, args_range)
    X /= sampling_frequency

    # Number of frequency bins in the FFT
    n_freq = X.shape[-1]

    # Calculate the periodogram with each taper
    P = np.zeros([n_channels, n_channels, n_freq], dtype=np.complex64)
    for i in range(n_tapers):
        for j in range(n_channels):
            for k in range(j, n_channels):
                P[j, k] += np.conjugate(X[i, j]) * X[i, k]
                if i == n_tapers - 1 and k != j:
                    P[k, j] = np.conjugate(P[j, k])

    return P


def single_multitaper_spectra(
    data,
    alpha,
    tapers,
    n_freq,
    sampling_frequency,
    nfft,
    args_range,
    parallel,
):
    """Calculate a multitaper spectrum for a single subject.

    Parameters
    ----------
    data : np.ndarray or list
        Raw time series data. Must have shape (n_samples, n_channels).
    alpha : np.ndarray or list
        Inferred state mixing factors. Must have shape (n_samples, n_states).
    tapers : np.ndarray
        Tapers for apply to each data segments.
        Shape must be (n_tapers, segment_length).
    n_freq : int
        Number of frequency bins.
    sampling_frequency : float
        Sampling frequency in Hz.
    nfft : int
        Number of data points to use in the FFT.
    args_range : list
        Minimum and maximum indices of the multitaper to keep.
    parallel : bool
        Is this function being called in parallel? Only affects whether
        a progress bar is displayed or not.

    Returns
    -------
    p : np.ndarray
        Power spectra. Shape is (n_states, n_channels, n_freq).
    c : np.ndarray
        Coherence spectra. Shape is (n_states, n_channels, n_channels, n_freq).
    """

    # Use the state time course to get a time series for each state
    state_time_series = get_state_time_series(data, alpha)

    # Number of subjects, states, samples and channels
    n_states, n_samples, n_channels = state_time_series.shape

    # Number of tapers and segment length
    n_tapers, segment_length = tapers.shape

    # We will calculate the spectrum for several non-overlapping segments
    # of the time series and return the average over these segments.

    # Number of segments in the time series
    n_segments = round(n_samples / segment_length)

    # Power spectra for each state
    p = np.zeros([n_states, n_channels, n_channels, n_freq], dtype=np.complex64)
    for i in range(n_states):
        if parallel:
            iterator = range(n_segments)
        else:
            if i == 0:
                print("Calculating spectra:")
            iterator = trange(n_segments, desc=f"Mode {i}", ncols=98)

        for j in iterator:
            # Time series for state i and segment j
            time_series_segment = state_time_series[
                i, j * segment_length : (j + 1) * segment_length
            ]

            # If we're missing samples we pad with zeros either side of the data
            if time_series_segment.shape[0] != segment_length:
                n_zeros = segment_length - time_series_segment.shape[0]
                n_padding = round(n_zeros / 2)
                time_series_segment = np.pad(time_series_segment, n_padding)[
                    :segment_length, n_padding:-n_padding
                ]
                if time_series_segment.shape[0] == segment_length - 1:
                    time_series_segment = np.append(
                        time_series_segment,
                        np.zeros([1, time_series_segment.shape[1]], dtype=np.float32),
                        axis=0,
                    )

            # Calculate the power (and cross) spectrum using the multitaper method
            p[i] += multitaper(
                time_series_segment,
                sampling_frequency,
                nfft=nfft,
                tapers=tapers,
                args_range=args_range,
            )

    # Normalise the power spectra
    # NOTE: We should be normalising using sum alpha instead of sum
    # alpha^2, but this makes a small difference, so we left it like
    # this for consistency with the HMM-MAR toolbox.
    sum_alpha = np.sum(alpha**2, axis=0)[..., np.newaxis, np.newaxis, np.newaxis]
    p *= n_samples / (sum_alpha * n_tapers * n_segments)

    # Coherences for each state
    c = coherence_spectra(p, print_message=False)

    # Only need to keep the diagonal of the power spectra matrix
    p = p[:, range(n_channels), range(n_channels)].real

    return p, c


def multitaper_spectra(
    data,
    alpha,
    sampling_frequency,
    time_half_bandwidth,
    n_tapers,
    segment_length=None,
    frequency_range=None,
    return_weights=False,
    standardize=True,
    n_jobs=1,
):
    """Calculates spectra for inferred states using a multitaper.

    This includes power and coherence spectra.
    Follows the same procedure as the OSL function HMM-MAR/spectral/hmmspectamt.m

    Parameters
    ----------
    data : np.ndarray or list
        Raw time series data. Must have shape (n_subjects, n_samples, n_channels)
        or (n_samples, n_channels).
    alpha : np.ndarray or list
        Inferred state time course. Must have shape (n_subjects, n_samples, n_states)
        or (n_samples, n_states).
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
        Shape is (n_subjects, n_states, n_channels, n_freq).
    coherences : np.ndarray
        Coherences for each state.
        Shape is (n_subjects, n_states, n_channels, n_channels, n_freq).
    weights : np.ndarray
        Weight for each subject-specific PSD. Only returned if return_weights=True.
        Shape is (n_subjects,).
    """

    # Validation
    if (isinstance(data, list) != isinstance(alpha, list)) or (
        isinstance(data, np.ndarray) != isinstance(alpha, np.ndarray)
    ):
        raise ValueError(
            f"data is type {type(data)} and alpha is type "
            + f"{type(alpha)}. They must both be lists or numpy arrays."
        )

    if isinstance(data, list):
        # Check data and state time course for the same number of subjects
        # has been passed
        if len(data) != len(alpha):
            raise ValueError(
                "A different number of subjects has been passed for "
                + f"data and alpha: len(data)={len(data)}, "
                + f"len(alpha)={len(alpha)}."
            )

        # Check the number of samples in data and alpha
        for i in range(len(alpha)):
            if alpha[i].shape[0] != data[i].shape[0]:
                raise ValueError("items in data and alpha must have the same shape.")

    if isinstance(data, np.ndarray):
        if alpha.shape[0] != data.shape[0]:
            raise ValueError("data and alpha must have the same shape.")

        if data.ndim == 2:
            data = [data]
            alpha = [alpha]

    if segment_length is None:
        segment_length = 2 * sampling_frequency

    elif segment_length != 2 * sampling_frequency:
        warnings.warn(
            "segment_length is recommended to be 2 * sampling_frequency.",
            RuntimeWarning,
        )

    segment_length = int(segment_length)

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    # Number of subjects
    n_subjects = len(data)

    # Number of FFT data points to calculate
    nfft = max(256, 2 ** nextpow2(segment_length))

    # Calculate the argments to keep for the given frequency range
    frequencies = np.arange(0, sampling_frequency / 2, sampling_frequency / nfft)
    args_range = get_frequency_args_range(frequencies, frequency_range)
    frequencies = frequencies[args_range[0] : args_range[1]]

    # Number of frequency bins
    n_freq = args_range[1] - args_range[0]

    # Standardise before calculating the multitaper
    if standardize:
        data = [(d - np.mean(d, axis=0)) / np.std(d, axis=0) for d in data]

    # Calculate tapers so we can estimate spectra with the multitaper method
    tapers = dpss(segment_length, NW=time_half_bandwidth, Kmax=n_tapers)
    tapers *= np.sqrt(sampling_frequency)

    if n_subjects == 1:
        # We only have one subject so we don't need to parallelise the
        # calculation
        results = single_multitaper_spectra(
            data[0],
            alpha[0],
            tapers,
            n_freq,
            sampling_frequency,
            nfft,
            args_range,
            parallel=False,
        )
        results = [results]

    elif n_jobs == 1:
        # We have multiple subjects but we're running in serial
        results = []
        for n in range(n_subjects):
            results.append(
                single_multitaper_spectra(
                    data[n],
                    alpha[n],
                    tapers,
                    n_freq,
                    sampling_frequency,
                    nfft,
                    args_range,
                    parallel=False,
                )
            )

    else:
        # Create arguments to pass to single_multitaper_spectra, which will
        # calculate spectra for each subject in parallel
        args = []
        for n in range(n_subjects):
            args.append(
                [
                    data[n],
                    alpha[n],
                    tapers,
                    n_freq,
                    sampling_frequency,
                    nfft,
                    args_range,
                    True,
                ]
            )

        # Calculate spectra in parallel
        print("Calculating spectra:")
        results = pqdm(
            args,
            single_multitaper_spectra,
            n_jobs=n_jobs,
            argument_type="args",
            ncols=98,
        )

    # Unpack the results
    power_spectra = []
    coherences = []
    for result in results:
        p, c = result
        power_spectra.append(p)
        coherences.append(c)

    # Weights for calculating the group average PSD
    n_samples = [d.shape[0] for d in data]
    weights = np.array(n_samples) / np.sum(n_samples)

    if return_weights:
        return frequencies, np.squeeze(power_spectra), np.squeeze(coherences), weights
    else:
        return frequencies, np.squeeze(power_spectra), np.squeeze(coherences)


def single_regression_spectra(
    data,
    alpha,
    window_length,
    sampling_frequency,
    frequency_range,
    calc_cpsd,
    step_size,
    n_sub_windows,
    parallel,
):
    """Calculate cross-spectra for a single subject using the regression method.

    Parameters
    ----------
    data : np.ndarray
        Data to calculate the spectrogram for. Shape must be (n_samples, n_channels).
    alpha : np.ndarray
        Inferred mode mixing factors. Shape must be (n_samples, n_modes).
    window_length : int
        Number of data points to use when calculating the periodogram.
    sampling_frequency : float
        Sampling frequency in Hz.
    calc_cpsd : bool
        Should we calculate cross spectra?
    step_size : int
        Step size for shifting the window.
    n_sub_windows : int
        Should we split the window into a set of sub-windows and average the
        spectra for each sub-window.
    parallel : bool
        Is this function being called in parallel? Only affects whether
        a progress bar is displayed or not.

    Returns
    -------
    t : np.ndarray
        Time axis.
    f : np.ndarray
        Frequency axis.
    coefs : np.ndarray
        Regression coefficients.
    intercept : np.ndarray
        Regression intercept.
    """
    print_message = not parallel
    t, f, p = spectrogram(
        data,
        window_length,
        sampling_frequency,
        frequency_range,
        calc_cpsd=calc_cpsd,
        step_size=step_size,
        n_sub_windows=n_sub_windows,
        print_progress_bar=print_message,
    )
    a = window_mean(
        alpha,
        window_length,
        step_size=step_size,
        n_sub_windows=n_sub_windows,
    )
    coefs, intercept = regression.linear(
        a,
        p,
        fit_intercept=True,
        normalize=True,
        print_message=print_message,
    )
    return t, f, coefs, intercept


def regression_spectra(
    data,
    alpha,
    window_length,
    sampling_frequency=1.0,
    frequency_range=None,
    n_embeddings=None,
    psd_only=False,
    step_size=1,
    n_sub_windows=1,
    return_weights=False,
    return_coef_int=False,
    standardize=True,
    n_jobs=1,
):
    """Calculates the PSD of each mode by regressing a time-varying PSD with alpha.

    Parameters
    ----------
    data : np.ndarray or list
        Data to calculate a time-varying PSD for. Shape must be (n_subjects,
        n_samples, n_channels) or (n_samples, n_channels).
    alpha : np.ndarray
        Inferred mode mixing factors. Shape must be (n_subjects, n_samples,
        n_modes) or (n_samples, n_modes).
    window_length : int
        Number samples to use in the window to calculate a PSD.
    sampling_frequency : float
        Sampling_frequency in Hz.
    frequency_range : list
        Minimum and maximum frequency to keep.
    n_embeddings : int
        Number of time embeddings applied when inferring alpha.
    psd_only : bool
        Should we only calculate the PSD?
    step_size : int
        Step size for shifting the window.
    n_sub_windows : int
        We split the window into a number of sub-windows and average the
        spectra for each sub-window. window_length must be divisible by
        n_sub_windows.
    return_weights : bool
        Should we return the weights for subject-specific PSDs?
        Useful for calculating the group average PSD.
    return_coef_int : bool
        Should we return the regression coefficients and intercept
        separately for the PSDs?
    standardize : bool
        Should we standardize the data before calculating the spectrogram?
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    f : np.ndarray
        Frequency axis.
    psd : np.ndarray
        Mode PSDs. A numpy array with shape (n_subjects, 2, n_modes, n_channels, n_freq)
        where the first axis is the coefficients/intercept if return_coef_int=True,
        otherwise shape is (n_subjects, n_modes, n_channels, n_freq).
    coh : np.ndarray
        Mode coherences.
    w : np.ndarray
        Weight for each subject-specific PSD. Only returned if return_weights=True.
    """

    # Validation
    if isinstance(data, list):
        if not isinstance(alpha, list):
            raise ValueError(
                "data and alpha must both be lists or both be numpy arrays."
            )

    if isinstance(data, np.ndarray):
        if not isinstance(alpha, np.ndarray):
            raise ValueError(
                "data and alpha must both be lists or both be numpy arrays."
            )
        if data.ndim == 2:
            data = [data]
            alpha = [alpha]

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    if window_length % n_sub_windows != 0:
        raise ValueError("window_length must be divisible by n_sub_windows.")

    # Do we calculate cross spectral densities?
    calc_cpsd = not psd_only

    # Number of subjects
    n_subjects = len(data)

    # Standardise before calculating the spectrogram
    if standardize:
        data = [(d - np.mean(d, axis=0)) / np.std(d, axis=0) for d in data]

    # Remove data points not in alpha due to time embedding the training data
    if n_embeddings is not None:
        data = [d[n_embeddings // 2 : -(n_embeddings // 2)] for d in data]

    # Remove the data points lost due to separating into sequences
    data = [d[: a.shape[0]] for d, a in zip(data, alpha)]

    if n_subjects == 1:
        # We only have one subject so we don't need to parallelise the
        # calculation
        results = single_regression_spectra(
            data[0],
            alpha[0],
            window_length,
            sampling_frequency,
            frequency_range,
            calc_cpsd,
            step_size,
            n_sub_windows,
            parallel=False,
        )
        results = [results]

    elif n_jobs == 1:
        # We have multiple subjects but we're running in serial
        results = []
        for n in range(n_subjects):
            results.append(
                single_regression_spectra(
                    data[n],
                    alpha[n],
                    window_length,
                    sampling_frequency,
                    frequency_range,
                    calc_cpsd,
                    step_size,
                    n_sub_windows,
                    parallel=False,
                )
            )
    else:
        # Create arguments to pass to single_regression_spectra, which will
        # calculate spectra for each subject in parallel
        args = []
        for n in range(n_subjects):
            args.append(
                [
                    data[n],
                    alpha[n],
                    window_length,
                    sampling_frequency,
                    frequency_range,
                    calc_cpsd,
                    step_size,
                    n_sub_windows,
                    True,
                ]
            )

        # Calculate a time-varying PSD and regress to get the mode PSDs
        print("Calculating power spectra")
        results = pqdm(
            args,
            single_regression_spectra,
            n_jobs=n_jobs,
            argument_type="args",
            ncols=98,
        )

    # Unpack results
    Pj = []
    for result in results:
        t, f, coefs, intercept = result
        Pj.append([coefs, [intercept] * coefs.shape[0]])
    Pj = np.array(Pj)

    # Weights for calculating the group average PSD
    n_samples = [d.shape[0] for d in data]
    weights = np.array(n_samples) / np.sum(n_samples)

    if psd_only:
        if not return_coef_int:
            # Sum coefficients and intercept
            Pj = np.sum(Pj, axis=1)

        if return_weights:
            return f, np.squeeze(Pj), weights
        else:
            return f, np.squeeze(Pj)

    # Number of channels and freqency bins
    n_channels = data[0].shape[1]
    n_modes = alpha[0].shape[1]
    n_freq = Pj.shape[-1]

    # Indices of the upper triangle of an n_channels by n_channels array
    m, n = np.triu_indices(n_channels)

    # Create a n_channels by n_channels array
    P = np.empty(
        [n_subjects, 2, n_modes, n_channels, n_channels, n_freq], dtype=np.complex64
    )
    for i in range(n_subjects):
        for j in range(2):  # j=0 is the coefficients and j=1 is the intercepts
            P[i, j][:, m, n] = Pj[i, j]
            P[i, j][:, n, m] = np.conj(Pj[i, j])

    # PSDs and coherences for each mode
    psd = np.empty([n_subjects, 2, n_modes, n_channels, n_freq], dtype=np.float32)
    coh = np.empty(
        [n_subjects, n_modes, n_channels, n_channels, n_freq], dtype=np.float32
    )
    for i in range(n_subjects):
        # PSDs
        p = P[i, :, :, range(n_channels), range(n_channels)].real
        psd[i] = np.rollaxis(p, axis=0, start=3)

        # Coherences
        p = np.sum(P[i], axis=0)  # sum coefs and intercept
        coh[i] = coherence_spectra(p, print_message=False)

    if not return_coef_int:
        # Sum coefficients and intercept
        psd = np.sum(psd, axis=1)

    if return_weights:
        return f, np.squeeze(psd), np.squeeze(coh), weights
    else:
        return f, np.squeeze(psd), np.squeeze(coh)


def spectrogram(
    data,
    window_length,
    sampling_frequency=1.0,
    frequency_range=None,
    calc_cpsd=True,
    step_size=1,
    n_sub_windows=1,
    print_progress_bar=True,
):
    """Calculates a spectogram.

    The data is segmented into overlapping windows which are then used to calculate
    a periodogram.

    Parameters
    ----------
    data : np.ndarray
        Data to calculate the spectrogram for. Shape must be (n_samples, n_channels).
    window_length : int
        Number of data points to use when calculating the periodogram.
    sampling_frequency : float
        Sampling frequency in Hz.
    calc_cpsd : bool
        Should we calculate cross spectra?
    step_size : int
        Step size for shifting the window.
    n_sub_windows : int
        Should we split the window into a set of sub-windows and average the
        spectra for each sub-window.
    print_progress_bar : bool
        Should we print a progress bar?

    Returns
    -------
    t : np.ndarray
        Time axis.
    f : np.ndarray
        Frequency axis.
    P : np.ndarray
        Spectrogram.
    """

    # Number of samples, channels and modes
    n_samples = data.shape[0]
    n_channels = data.shape[1]

    # First pad the data so we have enough data points to estimate the periodogram
    # for time points at the start/end of the data
    data = np.pad(data, window_length // 2)[
        :, window_length // 2 : window_length // 2 + n_channels
    ]

    # Window to apply to the data before calculating the Fourier transform
    window = hann(window_length // n_sub_windows)

    # Number of data points in the FFT
    nfft = max(256, 2 ** nextpow2(window_length // n_sub_windows))

    # Time and frequency axis
    t = np.arange(n_samples) / sampling_frequency
    f = np.arange(nfft // 2) * sampling_frequency / nfft

    # Only keep a particular frequency range
    args_range = get_frequency_args_range(f, frequency_range)
    f = f[args_range[0] : args_range[1]]

    # Number of frequency bins
    n_freq = args_range[1] - args_range[0]

    # Indices of an upper triangle of an n_channels by n_channels array
    m, n = np.triu_indices(n_channels)

    # Indices of time points to calculate a periodogram for
    time_indices = range(0, n_samples, step_size)
    n_psds = n_samples // step_size

    if calc_cpsd:
        # Calculate cross periodograms for each segment of the data
        P = np.empty(
            [n_psds, n_channels * (n_channels + 1) // 2, n_freq], dtype=np.complex64
        )
        XY_sub_window = np.empty(
            [n_sub_windows, n_channels * (n_channels + 1) // 2, n_freq],
            dtype=np.complex64,
        )
        if print_progress_bar:
            iterator = trange(n_psds, desc="Calculating spectrogram", ncols=98)
        else:
            iterator = range(n_psds)
        for i in iterator:
            # Data in the window
            j = time_indices[i]
            x_window = data[j : j + window_length].T

            for k in range(n_sub_windows):
                # Data in the sub-window with the windowing function applied
                x_sub_window = (
                    x_window[
                        :,
                        k
                        * window_length
                        // n_sub_windows : (k + 1)
                        * window_length
                        // n_sub_windows,
                    ]
                    * window[np.newaxis, ...]
                )

                # Calculate cross spectra for the sub-window
                X = fourier_transform(x_sub_window, nfft, args_range)
                XY = X[:, np.newaxis, :] * np.conj(X)[np.newaxis, :, :]
                XY_sub_window[k] = XY[m, n]

            # Average the cross spectra for each sub-window
            P[i] = np.mean(XY_sub_window, axis=0)

    else:
        # Calculate the periodogram for each segment of the data
        P = np.empty([n_psds, n_channels, n_freq], dtype=np.float32)
        XX_sub_window = np.empty([n_sub_windows, n_channels, n_freq], dtype=np.float32)
        if print_progress_bar:
            iterator = trange(n_psds, desc="Calculating spectrogram", ncols=98)
        else:
            iterator = range(n_psds)
        for i in iterator:
            # Data in the window
            j = time_indices[i]
            x_window = data[j : j + window_length].T

            for k in range(n_sub_windows):
                # Data in the sub-window with the windowing function applied
                x_sub_window = (
                    x_window[
                        :,
                        k
                        * window_length
                        // n_sub_windows : (k + 1)
                        * window_length
                        // n_sub_windows,
                    ]
                    * window[np.newaxis, ...]
                )

                # Calculate PSD for the sub-window
                X = fourier_transform(x_sub_window, nfft, args_range)
                XX_sub_window[k] = np.real(X * np.conj(X))

            # Average the cross spectra for each sub-window
            P[i] = np.mean(XX_sub_window, axis=0)

    # Scaling for the periodograms
    P /= sampling_frequency * np.sum(window**2)

    return t, f, P
