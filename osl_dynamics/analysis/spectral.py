"""Functions to perform spectral analysis."""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.signal._spectral_py import _median_bias
from sklearn.decomposition import non_negative_factorization

from osl_dynamics.utils import array_ops
from osl_dynamics.utils.misc import nextpow2
from osl_dynamics.utils.progress import parallel_map

_logger = logging.getLogger("osl-dynamics")


def wavelet(
    data: np.ndarray,
    sampling_frequency: float,
    w: float = 5,
    standardize: bool = True,
    time_range: Optional[List] = None,
    frequency_range: Optional[List] = None,
    df: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate a wavelet transform.

    This function uses a Morlet2 window to calculate a wavelet transform.

    Parameters
    ----------
    data : np.ndarray
        1D time series data. Shape must be (n_samples,).
    sampling_frequency : float
        Sampling frequency in Hz.
    w : float, optional
        Number of oscillations in the Gaussian envelope.
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
    dt = 1 / sampling_frequency
    t = np.arange(time_range[0], time_range[1], dt)

    # Frequency axis (Hz)
    f = np.arange(frequency_range[0], frequency_range[1], df)

    # Calculate the width for each Morlet window based on the frequency
    widths = w * sampling_frequency / (2 * f * np.pi)

    def _morlet2(M, s):
        x = np.arange(0, M) - (M - 1.0) / 2
        x = x / s
        sinusoidal = np.exp(1j * w * x)
        gaussian = np.exp(-0.5 * x**2)
        normalization = np.pi ** (-0.25) * np.sqrt(1 / s)
        return sinusoidal * gaussian * normalization

    def _cwt(data, widths):
        output = np.empty((len(widths), len(data)), dtype=np.complex64)
        for i, width in enumerate(widths):
            M = np.min([10 * width, len(data)])
            wavelet_data = np.conj(_morlet2(M, width)[::-1])
            output[i] = np.convolve(data, wavelet_data, mode="same")
        return output

    # Calculate wavelet transform
    wt = _cwt(data, widths)
    wt = abs(wt)

    return t, f, wt


def welch_spectra(
    data: Union[np.ndarray, list],
    sampling_frequency: float,
    alpha: Optional[Union[np.ndarray, list]] = None,
    window_length: Optional[int] = None,
    step_size: Optional[int] = None,
    frequency_range: Optional[List] = None,
    standardize: bool = True,
    averaging: str = "mean",
    calc_cpsd: bool = False,
    calc_coh: bool = True,
    return_weights: bool = False,
    keepdims: bool = False,
    n_jobs: int = 1,
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
            raise TypeError(
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
        elif alpha is None:
            # 3D array passed with no alpha
            alpha = np.array([None] * len(data))

    if window_length is None:
        window_length = 2 * sampling_frequency
    window_length = int(window_length)

    if step_size is None:
        step_size = window_length // 2
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
                "frequency_range": frequency_range,
                "standardize": standardize,
                "averaging": averaging,
                "calc_cpsd": calc_cpsd,
                "calc_coh": calc_coh,
                "keepdims": True,
            }
        )

    # Main calculation of spectra
    results = parallel_map(
        _welch,
        kwargs,
        desc="Calculating spectra",
        n_jobs=n_jobs,
        argument_type="kwargs",
        label="session",
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


def multitaper_spectra(
    data: Union[np.ndarray, list],
    sampling_frequency: float,
    alpha: Optional[Union[np.ndarray, list]] = None,
    window_length: Optional[int] = None,
    step_size: Optional[int] = None,
    time_half_bandwidth: float = 4,
    n_tapers: int = 7,
    frequency_range: Optional[List] = None,
    standardize: bool = True,
    averaging: str = "mean",
    calc_cpsd: bool = False,
    calc_coh: bool = True,
    return_weights: bool = False,
    keepdims: bool = False,
    n_jobs: int = 1,
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
            raise TypeError(
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
        elif alpha is None:
            # 3D array passed with no alpha
            alpha = np.array([None] * len(data))

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
                "averaging": averaging,
                "calc_cpsd": calc_cpsd,
                "calc_coh": calc_coh,
                "keepdims": True,
            }
        )

    # Main calculation of spectra
    results = parallel_map(
        _multitaper,
        kwargs,
        desc="Calculating spectra",
        n_jobs=n_jobs,
        argument_type="kwargs",
        label="session",
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
    data: Union[np.ndarray, list],
    alpha: Union[np.ndarray, list],
    sampling_frequency: float,
    window_length: int,
    step_size: int,
    n_sub_windows: int = 1,
    frequency_range: Optional[List] = None,
    standardize: bool = True,
    calc_coh: bool = True,
    return_weights: bool = False,
    return_coef_int: bool = False,
    return_sigma: bool = False,
    keepdims: bool = False,
    n_jobs: int = 1,
):
    r"""Calculates mode-specific spectra by regressing a spectrogram with alpha.

    We calculate a spectrogram :math:`P_t` with
    :func:`osl_dynamics.analysis.spectral._welch_spectrogram` and average
    :code:`alpha` over the same windows. Only windows that do not overlap the
    start or end of the data are used.

    If :code:`alpha` sums to one at each time point, the spectrogram is
    modeled as a mixture of mode spectra, :math:`P_t = \sum_j \alpha_{jt}
    P_j`, where :math:`P_j` is the spectrum when only mode :math:`j` is
    active. We return:

    - The **intercept**: the static (time-averaged) spectrum,
      :math:`S = \sum_j \bar{\alpha}_j P_j`.
    - The **coefficients**: :math:`\beta_j = \sigma_j (P_j - S)`, the change
      in the spectrum from the static spectrum for a one standard deviation
      increase in mode :math:`j`'s :code:`alpha`. :math:`\sigma_j` is the
      standard deviation of the windowed :code:`alpha`, averaged over
      sessions.
    - The **coherence**: the change in the cross spectrum normalized by the
      static power (change in coherence),

      .. math::
          C_{jxy} = \frac{\text{Re}(\beta_{jxy} e^{-i\phi_{xy}})}{\sqrt{S_{xx}
          S_{yy}}},

      where :math:`\phi_{xy}` is the phase of :math:`S_{xy}`. It includes
      changes in both coherence and power. The diagonal is the relative change
      in power, :math:`\beta_{jxx} / S_{xx}`.

    The mode spectrum is :math:`P_j = S + \beta_j / \sigma_j`. If :code:`alpha`
    does not sum to one (e.g. a subset of modes), we use a linear regression
    with standardized regressors and an intercept. A mode whose :code:`alpha`
    is (nearly) constant in a session has NaN coefficients for that session.

    Power maps and coherence networks should be calculated from the
    coefficients and coherence without subtracting the mean over modes. See
    the :doc:`User Guide </user_guide/spectra>` for how to interpret and plot
    the outputs.

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
    return_sigma : bool, optional
        Should we return the group-level standard deviation of the windowed
        :code:`alpha` used to scale the coefficients?
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
        Change in the cross spectrum normalized by the static power (change
        in coherence) for each mode. Shape is (n_sessions,
        n_modes, n_channels, n_channels, n_freq). Any axis of length 1 is
        removed if :code:`keepdims=False`.
    w : np.ndarray
        Weight for each session-specific PSD. Shape is (n_sessions,).
        Only returned if :code:`return_weights=True`.
    sigma : np.ndarray
        Group-level standard deviation of the windowed :code:`alpha` for each
        mode. Shape is (n_modes,). Only returned if :code:`return_sigma=True`.

    References
    ----------
    Quinn, A. J. et al. (2024). The GLM-spectrum: A multilevel framework for
    spectrum analysis with covariate and confound modelling. Imaging
    Neuroscience, 2. https://doi.org/10.1162/imag_a_00082
    """

    # Validation
    if (isinstance(data, list) != isinstance(alpha, list)) or (
        isinstance(data, np.ndarray) != isinstance(alpha, np.ndarray)
    ):
        raise TypeError(
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

    # Standardize before calculating the spectrogram
    if standardize:
        data = [(d - np.mean(d, axis=0)) / np.std(d, axis=0) for d in data]

    # Does alpha sum to a constant in every session?
    mixture = all(
        np.allclose(np.sum(a, axis=1), np.sum(a[0]), rtol=1e-4, atol=0) for a in alpha
    )

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
                "mixture": mixture,
            }
        )

    # Main calculation of spectra
    results = parallel_map(
        _regress_welch_spectrogram,
        kwargs,
        desc="Calculating spectra",
        n_jobs=n_jobs,
        argument_type="kwargs",
        label="session",
    )

    # Weights for calculating a group-average spectrum
    n_samples = [d.shape[0] for d in data]
    w = np.array(n_samples) / np.sum(n_samples)

    # Group-level standard deviation of the windowed alpha for each mode,
    # averaged over the sessions where the mode's alpha varies
    sigma = np.array([result[3] for result in results])
    used = sigma > 0
    total_weight = np.sum(w[:, np.newaxis] * used, axis=0)
    group_sigma = np.divide(
        np.sum(w[:, np.newaxis] * sigma, axis=0),
        total_weight,
        out=np.full(sigma.shape[1], np.nan),
        where=total_weight > 0,
    )

    # Unpack results, rescaling the coefficients to the change in the spectrum
    # for a one group-level standard deviation increase in a mode's alpha
    Pj = []
    for result, s, u in zip(results, sigma, used):
        f, coefs, intercept, _ = result
        # Modes that were not fitted in this session have no estimate (NaN)
        scale = np.divide(
            group_sigma, s, out=np.full_like(group_sigma, np.nan), where=u
        )
        scale = scale.reshape((-1,) + (1,) * (coefs.ndim - 1))
        coefs = (coefs * scale).astype(coefs.dtype)
        Pj.append([coefs, [intercept] * coefs.shape[0]])
    Pj = np.array(Pj)

    # Optional outputs
    extra = []
    if return_weights:
        extra.append(w)
    if return_sigma:
        extra.append(group_sigma)

    if not calc_coh:
        if not return_coef_int:
            # Sum coefficients and intercept
            Pj = np.sum(Pj, axis=1)

        if not keepdims:
            Pj = np.squeeze(Pj)

        return (f, Pj, *extra)

    # Number of channels and frequency bins
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

        # Coherence: change in the cross spectrum in the direction (phase) of
        # the static cross spectrum, normalized by the static power
        coefs, intercept = P[i, 0], P[i, 1]
        pxx = intercept[:, range(n_channels), range(n_channels)].real
        norm = np.sqrt(pxx[:, :, np.newaxis] * pxx[:, np.newaxis, :])
        abs_intercept = np.abs(intercept)
        phase = np.divide(
            intercept,
            abs_intercept,
            out=np.zeros_like(intercept),
            where=abs_intercept > 0,
        )
        coh[i] = np.real(coefs * np.conj(phase)) / norm

    if not return_coef_int:
        # Sum coefficients and intercept
        psd = np.sum(psd, axis=1)

    if not keepdims:
        # Remove any axes that are of length 1
        psd = np.squeeze(psd)
        coh = np.squeeze(coh)

    return (f, psd, coh, *extra)


def coherence_spectra(cpsd: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Calculate coherence from cross power spectral densities.

    We calculate the coherence as:

    .. math::
        C_{xy}(f) = \frac{|P_{xy}(f)|}{\sqrt{P_{xx}(f) P_{yy}(f)}}

    where :math:`x,y` are different channels.

    Parameters
    ----------
    cpsd : np.ndarray
        Cross power spectra.
        Shape is (..., n_channels, n_channels, n_freq).
    keepdims : bool, optional
        Should we squeeze any axis of length 1?

    Returns
    -------
    coh : np.ndarray
        Coherence spectra.
        Shape is (..., n_channels, n_channels, n_freq).
    """
    n_channels = cpsd.shape[-2]

    # Calculate coherency spectrum
    coh = np.empty_like(cpsd, dtype=np.float32)
    for j in range(n_channels):
        for k in range(n_channels):
            Pxy = cpsd[..., j, k, :]
            Pxx = cpsd[..., j, j, :].real
            Pyy = cpsd[..., k, k, :].real
            coh[..., j, k, :] = abs(Pxy) / np.sqrt(Pxx * Pyy)

    # Zero nan values
    coh = np.nan_to_num(coh)

    if not keepdims:
        coh = np.squeeze(coh)

    return coh


def partial_coherence_spectra(cpsd: np.ndarray, keepdims: bool = False) -> np.ndarray:
    r"""Calculate partial coherence from cross power spectral densities.

    We calculate the partial coherence as:

    .. math::
        PC_{xy}(f) = \
            \frac{|P^{-1}_{xy}(f)|}{\sqrt{P^{-1}_{xx}(f) P^{-1}_{yy}(f)}}

    where :math:`x,y` are different channels and :math:`P^{-1}(f)`
    is the inverse of the cross spectral density matrix.

    Parameters
    ----------
    cpsd : np.ndarray
        Cross power spectra.
        Shape is (..., n_channels, n_channels, n_freq).
    keepdims : bool, optional
        Should we squeeze any axis of length 1?

    Returns
    -------
    pcoh : np.ndarray
        Partial coherence spectra.
        Shape is (..., n_channels, n_channels, n_freq).
    """
    # Calculate inverse cross spectra
    # (Need to move axis for compatibility with np.linalg.pinv)
    cpsd = np.moveaxis(cpsd, -1, 0)
    icpsd = np.linalg.inv(cpsd)
    icpsd = np.moveaxis(icpsd, 0, -1)

    # The partial coherence calculation is the same as the
    # coherence calculation except we input the inverse cross
    # spectral densities
    return coherence_spectra(icpsd, keepdims=keepdims)


def partial_directed_coherence_spectra(
    f: np.ndarray,
    cpsd: np.ndarray,
    sampling_frequency: float,
    n_iterations: int = 50,
    tol: float = 1e-7,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Calculate partial directed coherence from cross power spectra.

    We calculate the partial directed coherence as:

    .. math::
        PDC_{ij}(f) = \frac{|A_{ij}(f)|}{\sqrt{\sum_m |A_{mj}(f)|^2}}

    where :math:`i,j` are different channels and :math:`A(f) = H^{-1}(f)`
    is the inverse of the transfer function, which is calculated by
    factorizing the cross spectral density matrix.

    By default this function will use all cores available.

    Parameters
    ----------
    f : np.ndarray
        Frequency axis. Shape is (n_freq,).
    cpsd : np.ndarray
        Cross power spectra.
        Shape is (..., n_channels, n_channels, n_freq).
    sampling_frequency : float
        Sampling frequency (Hz).
    n_iterations : int, optional
        Number of iterations in the Wilson factorization of the cross
        spectra.
    tol : float, optional
        Tolerance for the Wilson factorization of the cross spectra.
    keepdims : bool, optional
        Should we squeeze any axis of length 1?

    Returns
    -------
    pdc : np.ndarray
        Partial directed coherence spectra.
        Shape is (..., n_channels, n_channels, n_freq).
    """

    def _plus_operator(g):
        # Helper function for _wilson_factorization
        N = g.shape[0] // 2 + 1
        gam = np.fft.ifft(g, axis=0)
        beta0 = 0.5 * gam[0]
        gamp = gam.copy()
        gamp[0] = np.triu(beta0)
        gamp[N:] = 0
        return np.fft.fft(gamp, axis=0)

    def _wilson_factorization(
        S, freq=f, fs=sampling_frequency, Niterations=n_iterations, tol=tol
    ):
        # Algorithm for the Wilson Factorization of the spectral matrix. From:
        # https://github.com/ViniciusLima94/pyGC/blob/master/pygc/non_parametric.py
        #
        # Args:
        # - Cross spectra: S (frequencies, channels, channels)
        #
        # Returns:
        # - Transfer function: H (frequencies, channels, channels)
        m = S.shape[-1]
        N = freq.shape[0] - 1
        Sarr = np.zeros([2 * N, m, m], dtype=complex)
        Sarr[:N] = S[:N]
        Sarr[N:] = S[1:].transpose(0, 2, 1)[::-1]
        gam = np.fft.ifft(Sarr, axis=0).real
        h = np.linalg.cholesky(gam[0]).T
        psi = np.tile(h[np.newaxis, ...], (2 * N, 1, 1)).astype(complex)
        I = np.eye(m)[np.newaxis, ...]
        for _ in range(Niterations):
            inv_psi = np.linalg.inv(psi)
            g = inv_psi @ Sarr @ np.conj(inv_psi).transpose(0, 2, 1) + I
            gp = _plus_operator(g)
            psiold = psi.copy()
            psi = psi @ gp
            psierr = np.mean(np.linalg.norm(psi - psiold, 1, axis=1)) / (2 * N)
            if psierr < tol:
                break
        A = np.fft.ifft(psi, axis=0).real
        inv_A0 = np.linalg.inv(A[0])
        return psi[: N + 1] @ inv_A0

    def _calc_pdc(cpsd):
        # Calculate partial directed coherence from a single cross spectrum.
        #
        # Args:
        # - cpsd (n_channels, n_channels, n_freq) array
        #
        # Returns:
        # - pdc (n_channels, n_channels, n_freq) array
        S = np.moveaxis(cpsd, -1, 0)
        H = _wilson_factorization(S)
        A = np.linalg.inv(H)
        pdc = np.abs(A) / np.sqrt(np.sum(np.abs(A) ** 2, axis=1, keepdims=True))
        return np.moveaxis(pdc, 0, -1).astype(np.float32)

    # _calc_pdc works on 3D data, we flatten the input here
    original_shape = cpsd.shape
    flattened_cpsd = cpsd.reshape((-1, *original_shape[-3:]))

    # Create keyword arguments to pass to _calc_pdc
    kwargs = [{"cpsd": S} for S in flattened_cpsd]

    # Main calculation
    results = parallel_map(
        _calc_pdc, kwargs, desc="Calculating PDC", argument_type="kwargs"
    )

    # Put the results back into the original shape
    pdc = np.array(results).reshape(original_shape)

    if not keepdims:
        # Remove any axes that are of length 1
        pdc = np.squeeze(pdc)

    return pdc


def decompose_spectra(
    coherences: np.ndarray,
    n_components: int,
    max_iter: int = 50000,
    random_state: Optional[int] = 0,
    verbose: int = 0,
) -> np.ndarray:
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
        `sklearn.decomposition.non_negative_factorization
        <https://scikit-learn.org/stable/modules/generated/sklearn\
        .decomposition.non_negative_factorization.html>`_.
    random_state : int, optional
        Seed for the randomized SVD that initializes the factorization.
        The default makes the result reproducible.
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

    # Perform non-negative matrix factorization
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


def mar_spectra(
    coeffs: np.ndarray, covs: np.ndarray, sampling_frequency: float, n_freq: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
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
    if covs.ndim == 1:
        # A vector of diagonal elements was passed
        covs = np.diag(covs)
    elif covs.shape[-1] != covs.shape[-2]:
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

    # z-score of the coefficients
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
    autocorr_func: np.ndarray,
    sampling_frequency: float,
    nfft: int = 64,
    frequency_range: Optional[List] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Calculate the arguments to keep for the given frequency range
    f = np.arange(nfft // 2 + 1) * sampling_frequency / nfft
    [min_arg, max_arg] = get_frequency_args_range(f, frequency_range)
    f = f[min_arg : max_arg + 1]

    # Calculate cross power spectra as the Fourier transform of the
    # auto/cross-correlation function
    psd = np.fft.fft(autocorr_func, nfft)
    psd = psd[..., min_arg : max_arg + 1]
    psd = abs(psd)

    # Normalize the power spectra
    psd /= nfft**2

    # Coherences for each mode
    coh = coherence_spectra(psd, keepdims=True)

    return f, np.squeeze(psd), np.squeeze(coh)


def get_frequency_args_range(
    frequencies: np.ndarray, frequency_range: List
) -> List[int]:
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
    data: np.ndarray,
    sampling_frequency: float,
    window_length: Optional[int] = None,
    step_size: Optional[int] = None,
    frequency_range: Optional[List] = None,
    calc_cpsd: bool = True,
    n_sub_windows: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    n_samples, n_channels = data.shape

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
    f = np.arange(nfft // 2 + 1) * sampling_frequency / nfft

    # Only keep a particular frequency range
    [min_arg, max_arg] = get_frequency_args_range(f, frequency_range)
    f = f[min_arg : max_arg + 1]

    # Number of frequency bins
    n_freq = max_arg - min_arg + 1

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
                X = X[..., min_arg : max_arg + 1]
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
                X = X[..., min_arg : max_arg + 1]
                XX_sub_window[k] = np.real(np.conj(X) * X)

            # Average the cross spectra for each sub-window
            P[i] = np.mean(XX_sub_window, axis=0)

    # Scaling for the periodograms (we use the same scaling as SciPy)
    P *= 2 / (sampling_frequency * np.sum(window**2))

    return t, f, P


def _welch(
    data: np.ndarray,
    sampling_frequency: float,
    alpha: Optional[np.ndarray] = None,
    window_length: Optional[int] = None,
    step_size: Optional[int] = None,
    frequency_range: Optional[List] = None,
    standardize: bool = True,
    averaging: str = "mean",
    calc_cpsd: bool = False,
    calc_coh: bool = False,
    keepdims: bool = False,
):
    """Calculate a (cross) power spectrum using Welch's method.

    This function first calculates a spectrogram using
    :func:`osl_dynamics.analysis.spectral._welch_spectrogram`
    then takes the time average.

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
    averaging : str, optional
        Method used to average periodograms.
        Must be :code:`'mean'` or :code:`'median'`.
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
    if averaging not in ["mean", "median"]:
        raise ValueError(f"averaging must be 'mean' or 'median', got '{averaging}'.")

    n_samples, n_channels = data.shape

    if alpha is None:
        alpha = np.ones([n_samples, 1], dtype=np.float32)
    n_states = alpha.shape[-1]

    # Indices for the upper triangle of a channels by channels matrix
    m, n = np.triu_indices(n_channels)

    # Standardize before calculating spectra
    if standardize:
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Rescale the PSD to account for how much time each state was active
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
            calc_cpsd=calc_cpsd or calc_coh,
        )

        # Average over the time dimension and rescale
        if averaging == "median":
            bias = _median_bias(p.shape[0])
            if np.iscomplexobj(p):
                p = np.median(p.real, axis=0) + 1j * np.median(p.imag, axis=0)
            else:
                p = np.median(p, axis=0)
            p /= bias
        else:
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
            cpsd[n, m] = np.conj(p)

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
            cpsd[n, m] = np.conj(p)
            psd.append(cpsd)

        else:
            psd.append(p)

    # Check for nans
    if np.isnan(psd).any():
        _logger.warn("PSD contains NaN values.")
        psd = np.nan_to_num(psd)  # zero out nan values

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
    data: np.ndarray,
    sampling_frequency: float,
    window_length: Optional[int] = None,
    step_size: Optional[int] = None,
    frequency_range: Optional[List] = None,
    calc_cpsd: bool = True,
    n_sub_windows: int = 1,
    time_half_bandwidth: float = 4,
    n_tapers: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates a spectogram (time-varying power spectral density) using a multitaper.

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
    n_samples, n_channels = data.shape

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
    f = np.arange(nfft // 2 + 1) * sampling_frequency / nfft

    # Only keep a particular frequency range
    [min_arg, max_arg] = get_frequency_args_range(f, frequency_range)
    f = f[min_arg : max_arg + 1]

    # Number of frequency bins
    n_freq = max_arg - min_arg + 1

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
                X = X[..., min_arg : max_arg + 1]

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
                X = X[..., min_arg : max_arg + 1]

                # Calculate spectra for the sub-window
                XX = np.real(np.conj(X) * X)
                XX_sub_window[k] = np.mean(XX, axis=0)

            # Average the spectra for each sub-window
            P[i] = np.mean(XX_sub_window, axis=0)

    # Scaling for the multitapers
    P *= 2 / sampling_frequency

    return t, f, P


def _multitaper(
    data: np.ndarray,
    sampling_frequency: float,
    alpha: Optional[np.ndarray] = None,
    window_length: Optional[int] = None,
    step_size: Optional[int] = None,
    time_half_bandwidth: float = 4,
    n_tapers: int = 7,
    frequency_range: Optional[List] = None,
    standardize: bool = True,
    averaging: str = "mean",
    calc_cpsd: bool = False,
    calc_coh: bool = True,
    keepdims: bool = False,
):
    """Calculate a multitaper.

    This function first calculates a spectrogram using
    :func:`osl_dynamics.analysis.spectral._multitaper_spectrogram`
    then takes the time average.

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
    averaging : str, optional
        Method used to average periodograms.
        Must be :code:`'mean'` or :code:`'median'`.
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
    if averaging not in ["mean", "median"]:
        raise ValueError(f"averaging must be 'mean' or 'median', got '{averaging}'.")

    n_samples, n_channels = data.shape

    if alpha is None:
        alpha = np.ones([n_samples, 1], dtype=np.float32)
    n_states = alpha.shape[-1]

    # Indices for the upper triangle of a channels by channels matrix
    m, n = np.triu_indices(n_channels)

    # Standardize before calculating spectra
    if standardize:
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Rescale the PSD to account for how much time each state was active
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
        if averaging == "median":
            bias = _median_bias(p.shape[0])
            if np.iscomplexobj(p):
                p = np.median(p.real, axis=0) + 1j * np.median(p.imag, axis=0)
            else:
                p = np.median(p, axis=0)
            p /= bias
        else:
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
            cpsd[n, m] = np.conj(p)

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
            cpsd[n, m] = np.conj(p)
            psd.append(cpsd)

        else:
            psd.append(p)

    # Check for nans
    if np.isnan(psd).any():
        _logger.warn("PSD contains NaN values.")
        psd = np.nan_to_num(psd)  # zero out nan values

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
    data: np.ndarray,
    alpha: np.ndarray,
    sampling_frequency: float,
    window_length: int,
    step_size: int,
    n_sub_windows: int,
    frequency_range: List,
    calc_cpsd: bool,
    mixture: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    mixture : bool
        Does alpha sum to a constant? See :func:`_regress_spectrogram`.

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
    sigma : np.ndarray
        Standard deviation of the windowed alpha for each mode.
        Shape is (n_modes,).
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

    # Only use windows that do not overlap the zero padding at the start/end
    keep = _inner_windows(data.shape[0], window_length, step_size)
    if len(keep) <= a.shape[1]:
        raise ValueError(
            f"Each session must have more than n_modes={a.shape[1]} windows that "
            f"do not overlap its start or end, got {len(keep)}. Use longer "
            "sessions or a smaller window_length or step_size."
        )
    coefs, intercept, sigma = _regress_spectrogram(a[keep], p[keep], mixture)
    return f, coefs, intercept, sigma


def _regress_spectrogram(
    a: np.ndarray, p: np.ndarray, mixture: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regress a spectrogram on windowed mode time courses.

    Parameters
    ----------
    a : np.ndarray
        Windowed mode time courses. Shape must be (n_windows, n_modes).
    p : np.ndarray
        Spectrogram. Shape must be (n_windows, ...).
    mixture : bool
        Do the mode time courses sum to a constant? If so, the spectrogram is
        modeled as a mixture of mode spectra. Otherwise, we use a linear
        regression with standardized regressors and an intercept.

    Returns
    -------
    coefs : np.ndarray
        Change in the spectrum for a one standard deviation increase in each
        mode time course. NaN for modes whose time course barely varies.
        Shape is (n_modes, ...).
    intercept : np.ndarray
        Intercept. Shape is (...).
    sigma : np.ndarray
        Standard deviation of each mode time course (zero for modes that are
        not fitted). Shape is (n_modes,).
    """
    n_windows, n_modes = a.shape
    dtype = np.complex128 if np.iscomplexobj(p) else np.float64
    y = p.reshape(n_windows, -1)

    a = a.astype(np.float64)
    if mixture:
        a /= np.mean(np.sum(a, axis=1))
    sigma = np.std(a, axis=0)

    # Modes whose time course barely varies are not fitted (NaN coefficients)
    used = sigma > 1e-4 * np.max(sigma)
    sigma[~used] = 0

    # Intercept: static spectrum
    intercept = np.mean(y, axis=0, dtype=dtype)

    # Coefficients: change in the spectrum for a one standard deviation
    # increase in a mode's time course
    coefs = np.full((n_modes, y.shape[1]), np.nan, dtype=dtype)
    if np.any(used):
        if mixture:
            # Mixture of mode spectra, P_t = sum_j a_tj P_j
            pinv = np.linalg.pinv(a[:, used])
        else:
            # Linear regression with standardized regressors and an intercept
            z = (a[:, used] - np.mean(a[:, used], axis=0)) / sigma[used]
            pinv = np.linalg.pinv(z)

        # Apply the pseudo-inverse to chunks of the spectrogram to limit memory
        chunk = max(1, 2**24 // n_windows)
        for i in range(0, y.shape[1], chunk):
            fit = pinv @ y[:, i : i + chunk].astype(dtype)
            if mixture:
                fit = sigma[used, np.newaxis] * (fit - intercept[i : i + chunk])
            coefs[used, i : i + chunk] = fit

    coefs = coefs.reshape((n_modes,) + p.shape[1:]).astype(p.dtype)
    intercept = intercept.reshape(p.shape[1:]).astype(p.dtype)
    return coefs, intercept, sigma


def _inner_windows(n_samples: int, window_length: int, step_size: int) -> np.ndarray:
    """Windows of :func:`_welch_spectrogram` that do not overlap the zero padding.

    Parameters
    ----------
    n_samples : int
        Number of samples in the data.
    window_length : int
        Number of data points in a window.
    step_size : int
        Step size for shifting the window.

    Returns
    -------
    indices : np.ndarray
        Indices of the windows that only contain data.
    """
    window_length = int(window_length)
    step_size = int(step_size)
    n_windows = n_samples // step_size

    # Window i covers samples i * step_size - window_length // 2 to
    # i * step_size - window_length // 2 + window_length of the data
    start = np.arange(n_windows) * step_size - window_length // 2
    indices = np.flatnonzero((start >= 0) & (start + window_length <= n_samples))
    if len(indices) == 0:
        raise ValueError(
            f"Each session must have at least window_length={window_length} samples."
        )
    return indices


def _window_mean(
    alpha: np.ndarray,
    window: str,
    window_length: int,
    step_size: int,
    n_sub_windows: int,
) -> np.ndarray:
    """Mean of a time series in each window of :func:`_welch_spectrogram`.

    The mean in each sub-window is weighted by the square of the windowing
    function, which is how the periodogram weights each sample.

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
    n_samples, n_modes = alpha.shape

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
        # over time weighted by the squared windowing function
        a_sub_window = np.empty([n_sub_windows, n_modes], dtype=np.float32)
        for k in range(n_sub_windows):
            a_sub_window[k] = np.average(
                a_window[
                    k
                    * window_length
                    // n_sub_windows : (k + 1)
                    * window_length
                    // n_sub_windows
                ],
                weights=window**2,
                axis=0,
            )

        # Average alpha for each sub-window
        mean_window_alpha[i] = np.mean(a_sub_window, axis=0)

    return mean_window_alpha
