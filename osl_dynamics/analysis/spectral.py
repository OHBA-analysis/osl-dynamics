"""Functions to perform spectral analysis.

"""

import logging

import numpy as np
from scipy import signal
from sklearn.decomposition import non_negative_factorization
from pqdm.processes import pqdm
from tqdm.auto import trange

from osl_dynamics import array_ops
from osl_dynamics.analysis import regression
from osl_dynamics.inference import modes
from osl_dynamics.utils.misc import nextpow2

_logger = logging.getLogger("osl-dynamics")


def wavelet(
    data,
    sampling_frequency,
    w=5,
    standardize=True,
    time_range=None,
    frequency_range=None,
    df=0.5,
):
    """Calculate a wavelet transform.

    This function uses a `scipy.signal.morlet2 <https://docs.scipy.org/doc\
    /scipy/reference/generated/scipy.signal.morlet2.html>`_ window to calculate
    the wavelet transform.

    Parameters
    ----------
    data : np.ndarray
        1D time series data. Shape must be (n_samples,).
    sampling_frequency : float
        Sampling frequency in Hz.
    w : float, optional
        :code:`w` parameter to pass to `scipy.signal.morlet2
        <https://docs.scipy.org/doc/scipy/reference/generated\
        /scipy.signal.morlet2.html>`_.
    standardize : bool, optional
        Should we standardize the data before calculating the wavelet?
    time_range : list, optional
        Start time and end time to plot in seconds.
        Default is the full time axis of the data.
    frequency_range : list of length 2, optional
        Start and end frequency to plot in Hz.
        Default is :code:`[1, sampling_frequency / 2]`.
    df : float, optional
        Frequency resolution in Hz.

    Returns
    -------
    t : np.ndarray
        1D numpy array for the time axis.
    f : np.ndarray
        1D numpy array for the frequency axis.
    wt : np.ndarray
        2D numpy array (frequency, time) containing the wavelet transform.
    """
    # Validation
    if np.array(data).ndim != 1:
        raise ValueError("data must be a 1D numpy array.")

    if time_range is None:
        time_range = [0, data.shape[0] / sampling_frequency]
    if time_range[0] is None:
        time_range[0] = 0
    if time_range[1] is None:
        time_range[1] = data.shape[0] / sampling_frequency

    if frequency_range is None:
        frequency_range = [1, sampling_frequency / 2]
    if frequency_range[0] is None:
        frequency_range[0] = 1
    if frequency_range[1] is None:
        frequency_range[1] = sampling_frequency / 2

    # Standardize the data
    if standardize:
        data = (data - np.mean(data)) / np.std(data)

    # Keep selected time points
    start_index = int(time_range[0] * sampling_frequency)
    end_index = int(time_range[1] * sampling_frequency)
    data = data[start_index:end_index]

    # Time axis (s)
    t = np.arange(time_range[0], time_range[1], 1 / sampling_frequency)

    # Frequency axis (Hz)
    f = np.arange(frequency_range[0], frequency_range[1], df)

    # Calculate the width for each Morlet window based on the frequency
    widths = w * sampling_frequency / (2 * f * np.pi)

    # Calculate wavelet transform
    wt = signal.cwt(data=data, wavelet=signal.morlet2, widths=widths, w=w)
    wt = abs(wt)

    return t, f, wt


def welch_spectra(
    data,
    sampling_frequency,
    alpha=None,
    window_length=None,
    step_size=None,
    frequency_range=None,
    standardize=True,
    calc_coh=True,
    return_weights=False,
    keepdims=False,
    n_jobs=1,
):
    """Calculates spectra for inferred states using Welch's method.

    The scaling for the power spectra calculated by this function
    matches SciPy (:code:`scipy.signal.welch`).

    Parameters
    ----------
    data : np.ndarray or list
        Time series data. Must have shape (n_sessions, n_samples,
        n_channels) or (n_samples, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    alpha : np.ndarray or list, optional
        Inferred state probability time course. Must have shape
        (n_sessions, n_samples, n_states) or (n_samples, n_states).
    window_length : int, optional
        Length of the data segment to use to calculate spectra.
        If None, we use :code:`2 * sampling_frequency`.
    step_size : int, optional
        Step size for shifting the window.
        Defaults to :code:`window_length // 2` (50% overlap).
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
        Should we enforce a (n_sessions, n_states, ...) array is returned
        for :code:`psd` and :code:`coh`? If :code:`False`, we remove any
        dimension of length 1.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freq,).
    psd : np.ndarray
        Power spectra for each session and state. Shape is (n_sessions,
        n_states, n_channels, n_freq). Any axis of length 1 is removed if
        :code:`keepdims=False`.
    coh : np.ndarray
        Coherences for each session and state. Shape is (n_sessions, n_states,
        n_channels, n_channels, n_freq). Any axis of length 1 is removed if
        :code:`keepdims=False`. Only returned is :code:`calc_coh=True`.
    w : np.ndarray
        Weighting for session-specific spectra. Only returned if
        :code:`return_weights=True`. Shape is (n_sessions,).
    """

    # Validation
    if alpha is not None:
        if (isinstance(data, list) != isinstance(alpha, list)) or (
            isinstance(data, np.ndarray) != isinstance(alpha, np.ndarray)
        ):
            raise ValueError(
                f"data is type {type(data)} and alpha is type "
                f"{type(alpha)}. They must both be lists or numpy arrays."
            )

        if isinstance(data, list):
            # Check data and state time course for the same number of arrays
            # has been passed
            if len(data) != len(alpha):
                raise ValueError(
                    "A different number of arrays has been passed for "
                    f"data and alpha: len(data)={len(data)}, "
                    f"len(alpha)={len(alpha)}."
                )

            # Check the number of samples in data and alpha
            for i in range(len(alpha)):
                if alpha[i].shape[0] != data[i].shape[0]:
                    raise ValueError(
                        "items in data and alpha must have the number of samples."
                    )

    elif isinstance(data, list):
        alpha = np.array([None] * len(data))

    if isinstance(data, np.ndarray):
        if alpha is not None and alpha.shape[0] != data.shape[0]:
            raise ValueError("data and alpha must have the number of samples.")

        if data.ndim == 2:
            data = [data]
            alpha = [alpha]

    if window_length is None:
        window_length = 2 * sampling_frequency
    window_length = int(window_length)

    if step_size is None:
        step_size = window_length // 2
    step_size = int(step_size)

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    # Create keyword arguments to pass to the main function
    # that calculates spectra
    kwargs = []
    for d, a in zip(data, alpha):
        kwargs.append(
            {
                "data": d,
                "sampling_frequency": sampling_frequency,
                "alpha": a,
                "window_length": window_length,
                "step_size": step_size,
                "frequency_range": frequency_range,
                "standardize": standardize,
                "calc_coh": calc_coh,
                "keepdims": True,
            }
        )

    # Main calculation of spectra
    if len(data) == 1:
        # We only have one array so we don't need to parallelise
        # the calculation
        _logger.info("Calculating spectra")
        results = [_welch(**kwargs[0])]

    elif n_jobs == 1:
        # We have multiple arrays but we're running in serial
        results = []
        for i in trange(len(data), desc="Calculating spectra"):
            results.append(_welch(**kwargs[i]))

    else:
        # Calculate spectra in parallel
        _logger.info("Calculating spectra")
        results = pqdm(kwargs, _welch, n_jobs=n_jobs, argument_type="kwargs")

    # Unpack the results
    if calc_coh:
        psd = []
        coh = []
        for result in results:
            f, p, c = result
            psd.append(p)
            coh.append(c)
    else:
        psd = []
        for result in results:
            f, p = result
            psd.append(p)

    if not keepdims:
        # Remove any axes that are of length 1
        psd = np.squeeze(psd)
        if calc_coh:
            coh = np.squeeze(coh)

    # Weights for calculating a group-average spectrum
    n_samples = [d.shape[0] for d in data]
    w = np.array(n_samples) / np.sum(n_samples)

    if calc_coh:
        if return_weights:
            return f, psd, coh, w
        else:
            return f, psd, coh
    else:
        if return_weights:
            return f, psd, w
        else:
            return f, psd


def multitaper_spectra(
    data,
    sampling_frequency,
    alpha=None,
    window_length=None,
    step_size=None,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=None,
    standardize=True,
    calc_cpsd=False,
    calc_coh=True,
    return_weights=False,
    keepdims=False,
    n_jobs=1,
):
    """Calculates spectra for inferred states using a multitaper.

    Parameters
    ----------
    data : np.ndarray or list
        Time series data. Must have shape (n_sessions, n_samples,
        n_channels) or (n_samples, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    alpha : np.ndarray or list, optional
        Inferred state probability time course. Must have shape
        (n_sessions, n_samples, n_states) or (n_samples, n_states).
    window_length : int, optional
        Length of the data segment to use to calculate spectra.
    step_size : int, optional
        Step size for shifting the window.
        Defaults to :code:`window_length` (no overlap).
    time_half_bandwidth : float, optional
        Parameter to control the resolution of the spectra.
    n_tapers : int, optional
        Number of tapers.
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    standardize : bool, optional
        Should we standardize the data before calculating the spectra?
    calc_cpsd : bool, optional
        Should we return the cross spectra for :code:`psd`?
        If True, we force :code:`calc_coh` to False.
    calc_coh : bool, optional
        Should we also return the coherence spectra?
    return_weights : bool, optional
        Should we return the weights for session-specific spectra?
        This is useful for calculating a group average.
    keepdims : bool, optional
        Should we enforce a (n_sessions, n_states, ...) array is returned
        for :code:`psd` and :code:`coh`? If :code:`False`, we remove any
        dimension of length 1.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freq,).
    psd : np.ndarray
        Power spectra for each session and state. Shape is (n_sessions,
        n_states, n_channels, n_freq) if :code:`calc_cpsd=False`, otherwise
        the shape is (n_sessions, n_states, n_channels, n_channels, n_freq).
        Any axis of length 1 is removed if :code:`keepdims=False`.
    coh : np.ndarray
        Coherences for each session and state. Shape is (n_sessions, n_states,
        n_channels, n_channels, n_freq). Any axis of length 1 is removed if
        :code:`keepdims=False`. Only returned is :code:`calc_coh=True`.
    w : np.ndarray
        Weighting for session-specific spectra. Only returned if
        :code:`return_weights=True`. Shape is (n_sessions,).
    """

    # Validation
    if alpha is not None:
        if (isinstance(data, list) != isinstance(alpha, list)) or (
            isinstance(data, np.ndarray) != isinstance(alpha, np.ndarray)
        ):
            raise ValueError(
                f"data is type {type(data)} and alpha is type "
                f"{type(alpha)}. They must both be lists or numpy arrays."
            )

        if isinstance(data, list):
            # Check data and state time course for the same number of arrays
            # has been passed
            if len(data) != len(alpha):
                raise ValueError(
                    "A different number of arrays has been passed for "
                    f"data and alpha: len(data)={len(data)}, "
                    f"len(alpha)={len(alpha)}."
                )

            # Check the number of samples in data and alpha
            for i in range(len(alpha)):
                if alpha[i].shape[0] != data[i].shape[0]:
                    raise ValueError(
                        "items in data and alpha must have the same number of samples."
                    )

    elif isinstance(data, list):
        alpha = np.array([None] * len(data))

    if isinstance(data, np.ndarray):
        if alpha is not None and alpha.shape[0] != data.shape[0]:
            raise ValueError("data and alpha must have the number of samples.")

        if data.ndim == 2:
            data = [data]
            alpha = [alpha]

    if window_length is None:
        window_length = 2 * sampling_frequency
    window_length = int(window_length)

    if step_size is None:
        step_size = window_length
    step_size = int(step_size)

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    if calc_cpsd:
        calc_coh = False

    # Create keyword arguments to pass to the main function
    # that calculates spectra
    kwargs = []
    for d, a in zip(data, alpha):
        kwargs.append(
            {
                "data": d,
                "sampling_frequency": sampling_frequency,
                "alpha": a,
                "window_length": window_length,
                "step_size": step_size,
                "time_half_bandwidth": time_half_bandwidth,
                "n_tapers": n_tapers,
                "frequency_range": frequency_range,
                "standardize": standardize,
                "calc_cpsd": calc_cpsd,
                "calc_coh": calc_coh,
                "keepdims": True,
            }
        )

    # Main calculation of spectra
    if len(data) == 1:
        # We only have one array so we don't need to parallelise
        # the calculation
        _logger.info("Calculating spectra")
        results = [_multitaper(**kwargs[0])]

    elif n_jobs == 1:
        # We have multiple arrays but we're running in serial
        results = []
        for i in trange(len(data), desc="Calculating spectra"):
            results.append(_multitaper(**kwargs[i]))

    else:
        # Calculate spectra in parallel
        _logger.info("Calculating spectra")
        results = pqdm(
            kwargs,
            _multitaper,
            n_jobs=n_jobs,
            argument_type="kwargs",
        )

    # Unpack the results
    if calc_coh:
        psd = []
        coh = []
        for result in results:
            f, p, c = result
            psd.append(p)
            coh.append(c)
    else:
        psd = []
        for result in results:
            f, p = result
            psd.append(p)

    if not keepdims:
        # Remove any axes that are of length 1
        psd = np.squeeze(psd)
        if calc_coh:
            coh = np.squeeze(coh)

    # Weights for calculating a group-average spectrum
    n_samples = [d.shape[0] for d in data]
    w = np.array(n_samples) / np.sum(n_samples)

    if calc_coh:
        if return_weights:
            return f, psd, coh, w
        else:
            return f, psd, coh
    else:
        if return_weights:
            return f, psd, w
        else:
            return f, psd


def regression_spectra(
    data,
    alpha,
    sampling_frequency,
    window_length,
    step_size,
    n_sub_windows=1,
    frequency_range=None,
    standardize=True,
    calc_coh=True,
    return_weights=False,
    return_coef_int=False,
    keepdims=False,
    n_jobs=1,
):
    """Calculates mode-specific spectra by regressing a spectrogram with alpha.

    We use `_welch_spectrogram <https://osl-dynamics.readthedocs.io/en\
    /latest/autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics\
    .analysis.spectral._welch_spectrogram>`_ to calculate the spectrogram.

    Parameters
    ----------
    data : np.ndarray or list
        Data to calculate a spectrogram for. Shape must be
        (n_sessions, n_samples, n_channels) or (n_samples, n_channels).
    alpha : np.ndarray or list
        Inferred mode mixing factors. Shape must be
        (n_sessions, n_samples, n_modes) or (n_samples, n_modes).
    sampling_frequency : float
        Sampling_frequency in Hz.
    window_length : int
        Number samples to use in the window to calculate a spectrum.
    step_size : int
        Step size for shifting the window.
    n_sub_windows : int, optional
        We split the window into a number of sub-windows and average the
        spectra for each sub-window. window_length must be divisible by
        :code:`n_sub_windows`.
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    standardize : bool, optional
        Should we standardize the data before calculating the spectrogram?
    calc_coh : bool, optional
        Should we also return the coherence spectra?
    return_weights : bool, optional
        Should we return the weights for session-specific spectra?
        Useful for calculating the group average spectra.
    return_coef_int : bool, optional
        Should we return the regression coefficients and intercept
        separately for the mode PSDs?
    keepdims : bool, optional
        Should we enforce a (n_sessions, n_states, ...) array is returned for
        :code:`psd` and :code:`coh`? If :code:`False`, we remove any dimension
        of length 1.
    n_jobs : int, optional
        Number of parallel jobs.

    Returns
    -------
    f : np.ndarray
        Frequency axis. Shape is (n_freq).
    psd : np.ndarray
        Mode PSDs. A numpy array with shape (n_sessions, 2, n_modes, n_channels,
        n_freq) where :code:`axis=1` is the coefficients/intercept if
        :code:`return_coef_int=True`, otherwise shape is (n_sessions, n_modes,
        n_channels, n_freq). Any axis of length 1 is removed if
        :code:`keepdims=False`.
    coh : np.ndarray
        Mode coherences. Shape is (n_sessions, n_modes, n_channels, n_channels,
        n_freq). Any axis of length 1 is removed if :code:`keepdims=False`.
    w : np.ndarray
        Weight for each session-specific PSD. Shape is (n_sessions,).
        Only returned if :code:`return_weights=True`.
    """

    # Validation
    if (isinstance(data, list) != isinstance(alpha, list)) or (
        isinstance(data, np.ndarray) != isinstance(alpha, np.ndarray)
    ):
        raise ValueError(
            f"data is type {type(data)} and alpha is type "
            f"{type(alpha)}. They must both be lists or numpy arrays."
        )

    if isinstance(data, list):
        # Check data and state time course for the same number of arrays
        # has been passed
        if len(data) != len(alpha):
            raise ValueError(
                "A different number of arrays has been passed for "
                f"data and alpha: len(data)={len(data)}, "
                f"len(alpha)={len(alpha)}."
            )

        # Check the number of samples in data and alpha
        for i in range(len(alpha)):
            if alpha[i].shape[0] != data[i].shape[0]:
                raise ValueError(
                    "items in data and alpha must have the number of samples."
                )

    if isinstance(data, np.ndarray):
        if alpha.shape[0] != data.shape[0]:
            raise ValueError("data and alpha must have the number of samples.")

        if data.ndim == 2:
            data = [data]
            alpha = [alpha]

    # Ensure data and alpha are float32
    data = [d.astype(np.float32) for d in data]
    alpha = [a.astype(np.float32) for a in alpha]

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    if window_length % n_sub_windows != 0:
        raise ValueError("window_length must be divisible by n_sub_windows.")

    # Standardise before calculating the spectrogram
    if standardize:
        data = [(d - np.mean(d, axis=0)) / np.std(d, axis=0) for d in data]

    # Create keyword arguments to pass to the main function
    # that calculates spectra
    kwargs = []
    for d, a in zip(data, alpha):
        kwargs.append(
            {
                "data": d,
                "alpha": a,
                "sampling_frequency": sampling_frequency,
                "window_length": window_length,
                "step_size": step_size,
                "frequency_range": frequency_range,
                "calc_cpsd": calc_coh,
                "n_sub_windows": n_sub_windows,
            }
        )

    # Main calculation of spectra
    if len(data) == 1:
        # We only have one array so we don't need to parallelise the
        # calculation
        _logger.info("Calculating spectra")
        results = [_regress_welch_spectrogram(**kwargs[0])]

    elif n_jobs == 1:
        # We have multiple arrays but we're running in serial
        results = []
        for i, kws in enumerate(kwargs):
            _logger.info(f"Calculating spectra {i}")
            results.append(_regress_welch_spectrogram(**kws))
    else:
        # Calculate a time-varying PSD and regress to get the mode PSDs
        _logger.info("Calculating spectra")
        results = pqdm(
            kwargs,
            _regress_welch_spectrogram,
            n_jobs=n_jobs,
            argument_type="kwargs",
        )

    # Unpack results
    Pj = []
    for result in results:
        f, coefs, intercept = result
        Pj.append([coefs, [intercept] * coefs.shape[0]])
    Pj = np.array(Pj)

    # Weights for calculating a group-average spectrum
    n_samples = [d.shape[0] for d in data]
    w = np.array(n_samples) / np.sum(n_samples)

    if not calc_coh:
        if not return_coef_int:
            # Sum coefficients and intercept
            Pj = np.sum(Pj, axis=1)

        if return_weights:
            return f, np.squeeze(Pj), w
        else:
            return f, np.squeeze(Pj)

    # Number of channels and freqency bins
    n_channels = data[0].shape[1]
    n_modes = alpha[0].shape[1]
    n_freq = Pj.shape[-1]

    # Indices of the upper triangle of an n_channels by n_channels array
    m, n = np.triu_indices(n_channels)

    # Number of arrays
    n_sessions = len(data)

    # Create a n_channels by n_channels array
    P = np.empty(
        [n_sessions, 2, n_modes, n_channels, n_channels, n_freq],
        dtype=np.complex64,
    )
    for i in range(n_sessions):
        for j in range(2):  # j=0 is the coefficients and j=1 is the intercepts
            P[i, j][:, m, n] = Pj[i, j]
            P[i, j][:, n, m] = np.conj(Pj[i, j])

    # PSDs and coherences for each mode
    psd = np.empty(
        [n_sessions, 2, n_modes, n_channels, n_freq],
        dtype=np.float32,
    )
    coh = np.empty(
        [n_sessions, n_modes, n_channels, n_channels, n_freq],
        dtype=np.float32,
    )
    for i in range(n_sessions):
        # PSDs
        p = P[i, :, :, range(n_channels), range(n_channels)].real
        psd[i] = np.rollaxis(p, axis=0, start=3)

        # Coherences
        p = np.sum(P[i], axis=0)  # sum coefs and intercept
        coh[i] = coherence_spectra(p, keepdims=True)

    if not return_coef_int:
        # Sum coefficients and intercept
        psd = np.sum(psd, axis=1)

    if not keepdims:
        # Remove any axes that are of length 1
        psd = np.squeeze(psd)
        coh = np.squeeze(coh)

    if return_weights:
        return f, psd, coh, w
    else:
        return f, psd, coh


def rescale_regression_coefs(
    psd,
    alpha,
    window_length,
    step_size,
    n_sub_windows=1,
):
    """Rescales regression coefficients with maximum regressor values.

    Parameters
    ----------
    psd : np.ndarray
        Mode PSDs. Shape must be (n_sessions, 2, n_modes, n_channels, n_freq).
    alpha : np.ndarray or list
        Inferred mode mixing coefficients. Shape must be (n_sessions, n_samples,
        n_modes) or (n_samples, n_modes).
    window_length : int
        Number samples used as the window to calcualte a PSD.
    step_size : int
        Step size used for shifting the window.
    n_sub_windows : int, optional
        Number of sub-windows used to average the spectra. :code:`window_length`
        must be divisible by :code:`n_sub_windows`.

    Returns
    -------
    psd : np.ndarray
        Mode PSDs rescaled with maximum regressor values.
        Shape is identical to the input.
    """

    # Validation
    if psd.ndim == 4:
        psd = psd[np.newaxis, ...]

    if psd.ndim != 5:
        raise ValueError(
            "psd must have shape (n_sessions, 2, n_modes, n_channels, n_freqs)."
        )

    if isinstance(alpha, np.ndarray):
        alpha = [alpha]

    if len(psd) != len(alpha):
        raise ValueError(
            "A different number of arrays has been passed for "
            f"psd and alpha: len(psd)={len(psd)}, "
            f"len(alpha)={len(alpha)}."
        )

    for i in range(len(alpha)):
        if alpha[i].shape[1] != psd[i].shape[1]:
            raise ValueError("psd and alpha must have same number of modes.")

    # Prevent in-place operation
    psd = psd.copy()

    # Number of arrays
    n_sessions = len(alpha)

    # Window alphas to match the windowing of STFT
    for i, a in enumerate(alpha):
        alpha[i] = _window_mean(
            a,
            "hann",
            window_length,
            step_size=step_size,
            n_sub_windows=n_sub_windows,
        )

    # Normalize the alphas
    alpha = [(a - np.mean(a, axis=0)) / np.std(a, axis=0) for a in alpha]

    # Get maximum regressor values
    max_alpha = [np.max(a, axis=0) for a in alpha]

    # Rescale the regression coefficients
    for n in range(n_sessions):
        psd[n, 0] *= np.expand_dims(max_alpha[n], axis=(1, 2))

    return psd


def coherence_spectra(cpsd, keepdims=False):
    """Calculates coherences from cross power spectral densities.

    Parameters
    ----------
    cpsd : np.ndarray
        Cross power spectra.
        Shape is (n_channels, n_channels, n_freq) or
        (n_modes, n_channels, n_channels, n_freq).
    keepdims: bool, optional
        Should we squeeze any axis of length 1?

    Returns
    -------
    coh : np.ndarray
        Coherence spectra. Shape is (..., n_channels, n_channels, n_freq).
    """
    error_message = (
        "cpsd must be a numpy array with shape "
        "(n_channels, n_channels, n_freq) or "
        "(n_modes, n_channels, n_channels, n_freq)."
    )
    cpsd = array_ops.validate(
        cpsd,
        correct_dimensionality=4,
        allow_dimensions=[3],
        error_message=error_message,
    )

    n_modes, n_channels, n_channels, n_freq = cpsd.shape
    coh = np.empty(
        [n_modes, n_channels, n_channels, n_freq],
        dtype=np.float32,
    )
    for i in range(n_modes):
        for j in range(n_channels):
            for k in range(n_channels):
                coh[i, j, k] = abs(cpsd[i, j, k]) / np.sqrt(
                    cpsd[i, j, j].real * cpsd[i, k, k].real
                )

    # Zero nan values
    coh = np.nan_to_num(coh)

    if not keepdims:
        coh = np.squeeze(coh)

    return coh


def decompose_spectra(
    coherences,
    n_components,
    max_iter=50000,
    random_state=None,
    verbose=0,
):
    """Uses non-negative matrix factorization to decompose spectra.

    Parameters
    ----------
    coherences : np.ndarray
        Coherences spectra.
        Shape must be (..., n_channels, n_channels, n_freq,).
    n_components : int
        Number of spectral components to fit.
    max_iter : int, optional
        Maximum number of iterations in
        `sklearn.decomposion.non_negative_factorization
        <https://scikit-learn.org/stable/modules/generated/sklearn\
        .decomposition.non_negative_factorization.html>`_.
    random_state : int, optional
        Seed for the random number generator.
    verbose : int, optional
        Show verbose? (:code:`1`) yes, (:code:`0`) no.

    Returns
    -------
    components : np.ndarray
        Spectral components. Shape is (n_components, n_freq).
    """
    _logger.info("Performing spectral decomposition")

    # Validation
    error_message = (
        "coherences must be a numpy array with shape "
        "(n_channels, n_channels, n_freq), "
        "(n_modes, n_channels, n_channels, n_freq) or "
        "(n_sessions, n_modes, n_channels, n_channels, n_freq)."
    )
    coherences = array_ops.validate(
        coherences,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    # Number of arrays, modes, channels and frequency bins
    n_sessions, n_modes, n_channels, n_channels, n_freq = coherences.shape

    # Indices of the upper triangle of the
    # [n_channels, n_channels, n_freq] sub-array
    i, j = np.triu_indices(n_channels, 1)

    # Concatenate coherences for each array and mode
    # and only keep the upper triangle
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
        Sampling frequency in Hz.
    n_freq : int, optional
        Number of frequency bins in the cross power spectral density to
        calculate.

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
        "(n_modes, n_channels, n_channels), "
        "(n_channels, n_channels) or (n_channels,)."
    )
    covs = array_ops.validate(
        covs,
        correct_dimensionality=3,
        allow_dimensions=[2],
        error_message=error_message,
    )
    error_message = (
        "coeffs must be a numpy array with shape "
        "(n_modes, n_lags, n_channels, n_channels), "
        "(n_lags, n_channels, n_channels)."
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


def autocorr_to_spectra(
    autocorr_func,
    sampling_frequency,
    nfft=64,
    frequency_range=None,
):
    """Calculates spectra from the autocorrelation function.

    The power spectrum of each mode is calculated as the Fourier transform of
    the auto-correlation function. Coherences are calculated from the power
    spectra.

    Parameters
    ----------
    autocorr_func : np.ndarray
        Autocorrelation functions. Shape must be (n_channels, n_channels,
        n_lags) or (n_modes, n_channels, n_channels, n_lags).
    sampling_frequency : float
        Sampling frequency in Hz.
    nfft : int, optional
        Number of data points in the FFT. The auto-correlation function will
        only have :code:`2 * (n_embeddings + 2) - 1` data points. We pad the
        auto-correlation function with zeros to have :code:`nfft` data points
        if the number of data points in the auto-correlation function is less
        than :code:`nfft`.
    frequency_range : list, optional
        Minimum and maximum frequency to keep (Hz).

    Returns
    -------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freq,).
    psd : np.ndarray
        Power (or cross) spectra calculated for each mode. Shape is
        (n_channels, n_channels, n_freq) or (n_modes, n_channels, n_channels,
        n_freq).
    coh : np.ndarray
        Coherences calculated for each mode. Shape is (n_channels, n_channels,
        n_freq) or (n_modes, n_channels, n_channels, n_freq).
    """
    # Validation
    error_message = (
        "autocorrelation_functions must be of shape (n_channels, n_channels, "
        "n_lags) or (n_modes, n_channels, n_channels, n_lags)."
    )
    autocorr_func = array_ops.validate(
        autocorr_func,
        correct_dimensionality=4,
        allow_dimensions=[3],
        error_message=error_message,
    )
    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    _logger.info("Calculating power spectra")

    # Number of data points in the autocorrelation function and FFT
    n_lags = autocorr_func.shape[-1]
    nfft = max(nfft, 2 ** nextpow2(n_lags))

    # Calculate the argments to keep for the given frequency range
    f = np.arange(
        0,
        sampling_frequency / 2,
        sampling_frequency / nfft,
    )
    [min_arg, max_arg] = get_frequency_args_range(f, frequency_range)
    f = f[min_arg:max_arg]

    # Calculate cross power spectra as the Fourier transform of the
    # auto/cross-correlation function
    psd = np.fft.fft(autocorr_func, nfft)
    psd = psd[..., min_arg:max_arg]
    psd = abs(psd)

    # Normalise the power spectra
    psd /= nfft**2

    # Coherences for each mode
    coh = coherence_spectra(psd, keepdims=True)

    return f, np.squeeze(psd), np.squeeze(coh)


def get_frequency_args_range(frequencies, frequency_range):
    """Get min/max indices of a range in a frequency axis.

    Parameters
    ----------
    frequencies : np.ndarray
        1D numpy array containing a Frequency axis.
    frequency_range : list of len 2
        Minimum and maximum frequency.

    Returns
    -------
    args_range : list of len 2
        Indices for minimum and maximum frequency.
    """
    f_min_arg = np.argwhere(frequencies >= frequency_range[0])[0, 0]
    f_max_arg = np.argwhere(frequencies <= frequency_range[1])[-1, 0]
    if f_max_arg <= f_min_arg:
        raise ValueError("Cannot select requested frequency range.")
    return [f_min_arg, f_max_arg]


def _welch_spectrogram(
    data,
    sampling_frequency,
    window_length=None,
    step_size=None,
    frequency_range=None,
    calc_cpsd=True,
    n_sub_windows=1,
):
    """Calculates a spectogram (time-varying power spectral density)
    using Welch's method.

    Steps:

    1. Segment data into overlapping windows.
    2. Multiply each window by a Hann tapering function.
    3. Calculate a periodogram for each window.

    We use the same scaling for the power spectra as SciPy (i.e. when
    you use :code:`scipy.signal.periodogram(..., scaling="density")`).

    Parameters
    ----------
    data : np.ndarray
        Data to calculate the spectrogram for.
        Shape must be (n_samples, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    window_length : int, optional
        Number of data points to use when calculating the periodogram.
        Defaults to :code:`2 * sampling_frequency`.
    step_size : int, optional
        Step size for shifting the window.
        Defaults to :code:`window_length // 2` (50% overlap).
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    calc_cpsd : bool, optional
        Should we calculate cross spectra?
    n_sub_windows : int, optional
        We split the window into a number of sub-windows and average the
        spectra for each sub-window. window_length must be divisible by
        :code:`n_sub_windows`.

    Returns
    -------
    t : np.ndarray
        Time axis.
    f : np.ndarray
        Frequency axis.
    P : np.ndarray
        Spectrogram. 3D numpy array with shape (time, channels *
        (channels + 1) / 2, freq) if :code:`calc_cpsd=True`, otherwise
        it is (time, channels, freq).
    """
    n_samples = data.shape[0]
    n_channels = data.shape[1]

    # Validation
    if window_length is None:
        window_length = 2 * sampling_frequency
    window_length = int(window_length)

    if step_size is None:
        step_size = window_length // 2
    step_size = int(step_size)

    # First pad the data so we have enough data points to estimate the
    # periodogram for time points at the start/end of the data
    data = np.pad(data, window_length // 2)[
        :, window_length // 2 : window_length // 2 + n_channels
    ]

    # Window to apply to the data before calculating the Fourier transform
    window = signal.get_window("hann", (window_length // n_sub_windows))

    # Number of data points in the FFT
    nfft = max(256, window_length // n_sub_windows)

    # Time and frequency axis
    t = np.arange(n_samples) / sampling_frequency
    f = np.arange(nfft // 2) * sampling_frequency / nfft

    # Only keep a particular frequency range
    [min_arg, max_arg] = get_frequency_args_range(f, frequency_range)
    f = f[min_arg:max_arg]

    # Number of frequency bins
    n_freq = max_arg - min_arg

    # Indices of an upper triangle of an n_channels by n_channels array
    m, n = np.triu_indices(n_channels)

    # Indices of time points to calculate a periodogram for
    time_indices = range(0, n_samples, step_size)
    n_windows = n_samples // step_size

    if calc_cpsd:
        # Calculate cross periodograms for each segment of the data
        P = np.empty(
            [n_windows, n_channels * (n_channels + 1) // 2, n_freq],
            dtype=np.complex64,
        )
        XY_sub_window = np.empty(
            [n_sub_windows, n_channels * (n_channels + 1) // 2, n_freq],
            dtype=np.complex64,
        )
        for i in range(n_windows):
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
                X = np.fft.fft(x_sub_window, nfft)
                X = X[..., min_arg:max_arg]
                XY = np.conj(X)[np.newaxis, :, :] * X[:, np.newaxis, :]
                XY_sub_window[k] = XY[m, n]

            # Average the cross spectra for each sub-window
            P[i] = np.mean(XY_sub_window, axis=0)

    else:
        # Calculate the periodogram for each segment of the data
        P = np.empty([n_windows, n_channels, n_freq], dtype=np.float32)
        XX_sub_window = np.empty(
            [n_sub_windows, n_channels, n_freq],
            dtype=np.float32,
        )
        for i in range(n_windows):
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
                X = np.fft.fft(x_sub_window, nfft)
                X = X[..., min_arg:max_arg]
                XX_sub_window[k] = np.real(np.conj(X) * X)

            # Average the cross spectra for each sub-window
            P[i] = np.mean(XX_sub_window, axis=0)

    # Scaling for the periodograms (we use the same scaling as SciPy)
    P *= 2 / (sampling_frequency * np.sum(window**2))

    return t, f, P


def _welch(
    data,
    sampling_frequency,
    alpha=None,
    window_length=None,
    step_size=None,
    frequency_range=None,
    standardize=True,
    calc_coh=False,
    keepdims=False,
):
    """Calculate a (cross) power spectrum using Welch's method.

    This function first calculates a spectrogram using
    `_welch_spectrogram <https://osl-dynamics.readthedocs.io/en/latest\
    /autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis\
    .spectral._welch_spectrogram>`_ then takes the time average.

    The scaling for the power spectra calculated by this function
    matches SciPy (:code:`scipy.signal.welch`).

    Parameters
    ----------
    data : np.ndarray
        Time series data. Must have shape (n_samples, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    alpha : np.ndarray, optional
        Inferred state probability time course.
        Must have shape (n_samples, n_states).
    window_length : int, optional
        Length of the data segment to use to calculate spectra.
        If None, we use :code:`2 * sampling_frequency`.
    step_size : int, optional
        Step size for shifting the window.
        Defaults to :code:`window_length // 2` (50% overlap).
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    standardize : bool, optional
        Should we standardize the data before calculating the spectra?
    calc_coh : bool, optional
        Should we calculate the coherence between channels?
    keepdims : bool, optional
        Should we squeeze any axes of length 1?

    Returns
    -------
    f : np.ndarray
        Frequencies of the power spectra and coherences.
        Shape is (n_freq,).
    psd : np.ndarray
        Power spectra. Shape is (..., n_channels, n_freq).
    coh : np.ndarray
        Coherences spectra.
        Shape is (..., n_channels, n_channels, n_freq).
        Only returned is :code:`calc_coh=True`.
    """
    if alpha is None:
        alpha = np.ones([data.shape[0], 1], dtype=np.float32)

    n_samples = data.shape[0]
    n_channels = data.shape[-1]
    n_states = alpha.shape[-1]

    # Indices for the upper triangle of a channels by channels matrix
    m, n = np.triu_indices(n_channels)

    # Standardise before calculating spectra
    if standardize:
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Rescaling for the PSD to account for how much time each
    # state was active. Note, we using the same scaling as the
    # the multitaper calculation in the HMM-MAR toolbox
    sum_alpha2 = np.sum(alpha**2, axis=0)
    scaling = n_samples / sum_alpha2

    psd = []
    coh = []
    for i in range(n_states):
        # Calculate spectrogram for this state's data
        x = data * alpha[..., i][..., np.newaxis]
        _, f, p = _welch_spectrogram(
            data=x,
            sampling_frequency=sampling_frequency,
            window_length=window_length,
            step_size=step_size,
            frequency_range=frequency_range,
            calc_cpsd=calc_coh,
        )

        # Average over the time dimension and rescale
        p = np.mean(p, axis=0)
        p *= scaling[i]

        if calc_coh:
            # Create a channels by channels matrix for cross PSDs
            n_freq = p.shape[-1]
            cpsd = np.empty(
                [n_channels, n_channels, n_freq],
                dtype=np.complex64,
            )
            cpsd[m, n] = p
            cpsd[n, m] = p

            # Unpack PSDs
            psd.append(cpsd[range(n_channels), range(n_channels)].real)

            # Calculate coherence
            coh.append(coherence_spectra(cpsd))
        else:
            psd.append(p)

    # Check for nans
    if np.isnan(psd).any():
        psd = np.nan_to_num(psd)  # zero out nan values
        _logger.warn(
            "PSD contains NaN values. This may indicate a potentially "
            "poor HMM fit. You should consider running the model again "
            "or selecting the model run with the lowest free energy after "
            "training multiple times."
        )

    if not keepdims:
        # Squeeze any axes of length 1
        psd = np.squeeze(psd)
        if calc_coh:
            coh = np.squeeze(coh)

    if calc_coh:
        return f, psd, coh
    else:
        return f, psd


def _multitaper_spectrogram(
    data,
    sampling_frequency,
    window_length=None,
    step_size=None,
    frequency_range=None,
    calc_cpsd=True,
    n_sub_windows=1,
    time_half_bandwidth=4,
    n_tapers=7,
):
    """Calculates a spectogram (time-varying power spectral density)
    using a multitaper.

    Steps:

    1. Segment data into overlapping windows.
    2. Multiply each window by a number of tapering functions.
    3. Calculate a periodogram for each tapered window.
    4. Average over tapers.

    Parameters
    ----------
    data : np.ndarray
        Data to calculate the spectrogram for.
        Shape must be (n_samples, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    window_length : int, optional
        Number of data points to use when calculating the periodogram.
        Defaults to :code:`2 * sampling_frequency`.
    step_size : int, optional
        Step size for shifting the window.
        Defaults to :code:`window_length` (no overlap).
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    calc_cpsd : bool, optional
        Should we calculate cross spectra?
    n_sub_windows : int, optional
        We split the window into a number of sub-windows and average the
        spectra for each sub-window. window_length must be divisible by
        :code:`n_sub_windows`.
    time_half_bandwidth : float, optional
        Parameter to control the resolution of the spectra.
    n_tapers : int, optional
        Number of tapers.

    Returns
    -------
    t : np.ndarray
        Time axis.
    f : np.ndarray
        Frequency axis.
    P : np.ndarray
        Spectrogram. 3D numpy array with shape (time, channels *
        (channels + 1) / 2, freq) if :code:`calc_cpsd=True`, otherwise
        it is (time, channels, freq).
    """
    n_samples = data.shape[0]
    n_channels = data.shape[1]

    # Validation
    if window_length is None:
        window_length = 2 * sampling_frequency
    window_length = int(window_length)

    if step_size is None:
        step_size = window_length
    step_size = int(step_size)

    # First pad the data so we have enough data points to estimate the
    # periodogram for time points at the start/end of the data
    data = np.pad(data, window_length // 2)[
        :, window_length // 2 : window_length // 2 + n_channels
    ]

    # Tapering windows to apply to the data before calculating the
    # Fourier transform
    tapers = signal.windows.dpss(
        window_length,
        NW=time_half_bandwidth,
        Kmax=n_tapers,
    )

    # Number of data points in the FFT
    nfft = window_length // n_sub_windows

    # Time and frequency axis
    t = np.arange(n_samples) / sampling_frequency
    f = np.arange(nfft // 2) * sampling_frequency / nfft

    # Only keep a particular frequency range
    [min_arg, max_arg] = get_frequency_args_range(f, frequency_range)
    f = f[min_arg:max_arg]

    # Number of frequency bins
    n_freq = max_arg - min_arg

    # Indices of an upper triangle of an n_channels by n_channels array
    m, n = np.triu_indices(n_channels)

    # Indices of time points to calculate a periodogram for
    time_indices = range(0, n_samples, step_size)
    n_windows = n_samples // step_size

    if calc_cpsd:
        # Calculate cross PSDs for each segment of the data
        P = np.empty(
            [n_windows, n_channels * (n_channels + 1) // 2, n_freq],
            dtype=np.complex64,
        )
        XY_sub_window = np.empty(
            [n_sub_windows, n_channels * (n_channels + 1) // 2, n_freq],
            dtype=np.complex64,
        )
        for i in range(n_windows):
            # Data in the window
            j = time_indices[i]
            x_window = data[j : j + window_length].T

            for k in range(n_sub_windows):
                # Data in the sub-window
                x_sub_window = x_window[
                    :,
                    k
                    * window_length
                    // n_sub_windows : (k + 1)
                    * window_length
                    // n_sub_windows,
                ]

                # Apply tapers
                x_sub_window = x_sub_window[np.newaxis, :, :] * tapers[:, np.newaxis, :]

                # Fourier transform
                X = np.fft.fft(x_sub_window, nfft)
                X = X[..., min_arg:max_arg]

                # Calculate cross spectra for the sub-window
                XY = np.conj(X)[:, :, np.newaxis, :] * X[:, np.newaxis, :, :]
                XY_sub_window[k] = np.mean(XY[:, m, n], axis=0)

            # Average the cross spectra for each sub-window
            P[i] = np.mean(XY_sub_window, axis=0)

    else:
        # Calculate PSDs for each segment of the data
        P = np.empty([n_windows, n_channels, n_freq], dtype=np.float32)
        XX_sub_window = np.empty(
            [n_sub_windows, n_channels, n_freq],
            dtype=np.float32,
        )
        for i in range(n_windows):
            # Data in the window
            j = time_indices[i]
            x_window = data[j : j + window_length].T

            for k in range(n_sub_windows):
                # Data in the sub-window
                x_sub_window = x_window[
                    :,
                    k
                    * window_length
                    // n_sub_windows : (k + 1)
                    * window_length
                    // n_sub_windows,
                ]

                # Apply tapers
                x_sub_window = x_sub_window[np.newaxis, :, :] * tapers[:, np.newaxis, :]

                # Fourier transform
                X = np.fft.fft(x_sub_window, nfft)
                X = X[..., min_arg:max_arg]

                # Calculate spectra for the sub-window
                XX = np.real(np.conj(X) * X)
                XX_sub_window[k] = np.mean(XX, axis=0)

            # Average the spectra for each sub-window
            P[i] = np.mean(XX_sub_window, axis=0)

    # Scaling for the multitapers
    P *= 2 / sampling_frequency

    return t, f, P


def _multitaper(
    data,
    sampling_frequency,
    alpha=None,
    window_length=None,
    step_size=None,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=None,
    standardize=True,
    calc_cpsd=False,
    calc_coh=True,
    keepdims=False,
):
    """Calculate a multitaper.

    This function first calculates a spectrogram using
    `_multitaper_spectrogram <https://osl-dynamics.readthedocs.io/en/latest\
    /autoapi/osl_dynamics/analysis/spectral/index.html#osl_dynamics.analysis\
    .spectral._multitaper_spectrogram>`_ then takes the time average.

    Parameters
    ----------
    data : np.ndarray
        Time series data. Must have shape (n_samples, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    alpha : np.ndarray, optional
        Inferred state probability time course.
        Must have shape (n_samples, n_states).
    window_length : int, optional
        Length of the data segment to use to calculate spectra.
        If None, we use :code:`2 * sampling_frequency`.
    step_size : int, optional
        Step size for shifting the window.
        Defaults to :code:`window_length` (no overlap).
    time_half_bandwidth : float, optional
        Parameter to control the resolution of the spectra.
    n_tapers : int, optional
        Number of tapers.
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    standardize : bool, optional
        Should we standardize the data before calculating the spectra?
    calc_cpsd : bool, optional
        Should we return the cross spectra for :code:`psd`?
    calc_coh : bool, optional
        Should we calculate the coherence between channels?
    keepdims : bool, optional
        Should we squeeze any axes of length 1?

    Returns
    -------
    f : np.ndarray
        Frequencies of the power spectra and coherences.
        Shape is (n_freq,).
    psd : np.ndarray
        Power spectra. Shape is (..., n_channels, n_freq) if
        :code:`calc_cpsd=False`, otherwise the shape is
        (..., n_channels, n_channels, n_freq).
    coh : np.ndarray
        Coherences spectra.
        Shape is (..., n_channels, n_channels, n_freq).
        Only returned is :code:`calc_coh=True`.
    """
    if alpha is None:
        alpha = np.ones([n_samples, 1], dtype=np.float32)

    n_samples = data.shape[0]
    n_channels = data.shape[-1]
    n_states = alpha.shape[-1]

    # Indices for the upper triangle of a channels by channels matrix
    m, n = np.triu_indices(n_channels)

    # Standardise before calculating spectra
    if standardize:
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Rescaling for the PSD to account for how much time each
    # state was active. Note, we using the same scaling as the
    # the multitaper calculation in the HMM-MAR toolbox
    sum_alpha2 = np.sum(alpha**2, axis=0)
    scaling = n_samples / sum_alpha2

    psd = []
    coh = []
    for i in range(n_states):
        # Calculate spectrogram for this state's data
        x = data * alpha[:, i][..., np.newaxis]
        _, f, p = _multitaper_spectrogram(
            data=x,
            sampling_frequency=sampling_frequency,
            window_length=window_length,
            step_size=step_size,
            frequency_range=frequency_range,
            calc_cpsd=calc_cpsd or calc_coh,
            time_half_bandwidth=time_half_bandwidth,
            n_tapers=n_tapers,
        )

        # Average over the time dimension and rescale
        p = np.mean(p, axis=0)
        p *= scaling[i]
        n_freq = p.shape[-1]

        if calc_coh:
            # Create a channels by channels matrix for cross PSDs
            cpsd = np.empty(
                [n_channels, n_channels, n_freq],
                dtype=np.complex64,
            )
            cpsd[m, n] = p
            cpsd[n, m] = p

            # Unpack PSDs
            psd.append(cpsd[range(n_channels), range(n_channels)].real)

            # Calculate coherence
            coh.append(coherence_spectra(cpsd))

        elif calc_cpsd:
            cpsd = np.empty(
                [n_channels, n_channels, n_freq],
                dtype=np.complex64,
            )
            cpsd[m, n] = p
            cpsd[n, m] = p
            psd.append(cpsd)

        else:
            psd.append(p)

    # Check for nans
    if np.isnan(psd).any():
        psd = np.nan_to_num(psd)  # zero out nan values
        _logger.warn(
            "PSD contains NaN values. This may indicate a potentially "
            "poor HMM fit. You should consider running the model again "
            "or selecting the model run with the lowest free energy after "
            "training multiple times."
        )

    if not keepdims:
        # Squeeze any axes of length 1
        psd = np.squeeze(psd)
        if calc_coh:
            coh = np.squeeze(coh)

    if calc_coh:
        return f, psd, coh
    else:
        return f, psd


def _regress_welch_spectrogram(
    data,
    alpha,
    sampling_frequency,
    window_length,
    step_size,
    n_sub_windows,
    frequency_range,
    calc_cpsd,
):
    """Calculate regression spectra for a single array.

    Parameters
    ----------
    data : np.ndarray
        Data to calculate a spectrogram for.
        Shape must be (n_samples, n_channels).
    alpha : np.ndarray
        Inferred mode mixing factors. Shape must be (n_samples, n_modes).
    sampling_frequency : float
        Sampling_frequency in Hz.
    window_length : int
        Number samples to use in the window to calculate a PSD.
    step_size : int
        Step size for shifting the window.
    n_sub_windows : int, optional
        We split the window into a number of sub-windows and average the
        spectra for each sub-window. window_length must be divisible by
        :code:`n_sub_windows`.
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    calc_cpsd : bool, optional
        Should we calculate the cross spectra?

    Returns
    -------
    f : np.ndarray
        Frequency axis. Shape is (n_freq,).
    coefs : np.ndarray
        Regression coefficients. Shape is
        (n_modes, n_channels * (n_channels + 1) / 2, n_freq) if
        :code:`calc_cpsd=True`, otherwise it is (n_modes, n_channels, n_freq).
    intercept : np.ndarray
        Intercept. Shape is (n_channels * (n_channels + 1) / 2, n_freq)
        if :code:`calc_cpsd=True`, otherwise it is (n_channels, n_freq).
    """
    _, f, p = _welch_spectrogram(
        data=data,
        sampling_frequency=sampling_frequency,
        window_length=window_length,
        step_size=step_size,
        frequency_range=frequency_range,
        calc_cpsd=calc_cpsd,
        n_sub_windows=n_sub_windows,
    )
    a = _window_mean(alpha, "hann", window_length, step_size, n_sub_windows)
    coefs, intercept = regression.linear(
        a,
        p,
        fit_intercept=True,
        normalize=True,
        log_message=False,
    )
    return f, coefs, intercept


def _window_mean(alpha, window, window_length, step_size, n_sub_windows):
    """Applies a windowing function to a time series and takes the mean.

    Parameters
    ----------
    alpha : np.ndarray
        Time series data. Shape is (n_samples, n_modes).
    window : str
        Name of the windowing function.
    window_length : int
        Number of data points in a window.
    step_size : int
        Step size for shifting the window.
    n_sub_windows : int
        We split the window into a number of sub-windows and average
        across all sub-windows. This is the number of sub-windows.

    Returns
    -------
    mean_window_alpha : np.ndarray
        Mean for each window.
    """
    n_samples = alpha.shape[0]
    n_modes = alpha.shape[1]

    # Pad alphas
    alpha = np.pad(alpha, window_length // 2)[
        :, window_length // 2 : window_length // 2 + n_modes
    ]

    # Window to apply to alpha
    window = signal.get_window(window, window_length // n_sub_windows)

    # Indices of time points to calculate a periodogram for
    time_indices = range(0, n_samples, step_size)
    n_windows = n_samples // step_size

    # Array to hold mean of alpha multiplied by the windowing function
    mean_window_alpha = np.empty([n_windows, n_modes], dtype=np.float32)
    for i in range(n_windows):
        # Alpha in the window
        j = time_indices[i]
        a_window = alpha[j : j + window_length]

        # Calculate alpha for the sub-window by taking the mean
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

        # Average alpha for each sub-window
        mean_window_alpha[i] = np.mean(a_sub_window, axis=0)

    return mean_window_alpha
