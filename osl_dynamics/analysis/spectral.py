"""Functions to perform spectral analysis."""

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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

    if window_length is None:
        window_length = 2 * sampling_frequency
    if step_size is None:
        step_size = int(window_length) // 2

    return _state_spectra(
        method="welch",
        data=data,
        sampling_frequency=sampling_frequency,
        alpha=alpha,
        window_length=window_length,
        step_size=step_size,
        frequency_range=frequency_range,
        standardize=standardize,
        averaging=averaging,
        calc_cpsd=calc_cpsd,
        calc_coh=calc_coh,
        return_weights=return_weights,
        keepdims=keepdims,
        n_jobs=n_jobs,
    )


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

    if window_length is None:
        window_length = 2 * sampling_frequency
    if step_size is None:
        step_size = int(window_length)

    return _state_spectra(
        method="multitaper",
        data=data,
        sampling_frequency=sampling_frequency,
        alpha=alpha,
        window_length=window_length,
        step_size=step_size,
        frequency_range=frequency_range,
        standardize=standardize,
        averaging=averaging,
        calc_cpsd=calc_cpsd,
        calc_coh=calc_coh,
        return_weights=return_weights,
        keepdims=keepdims,
        n_jobs=n_jobs,
        time_half_bandwidth=time_half_bandwidth,
        n_tapers=n_tapers,
    )


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

    We calculate a spectrogram :math:`P_t` with Welch's method (Hann-tapered
    sub-windows) and average :code:`alpha` over the same windows. Only windows that do not overlap the
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

    if alpha is None:
        raise ValueError("alpha must be passed.")
    data, alpha = _validate_data_alpha(data, alpha)

    # Ensure data and alpha are float32
    data = [d.astype(np.float32) for d in data]
    alpha = [a.astype(np.float32) for a in alpha]

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    spectrogram = _Spectrogram.create(
        method="welch",
        sampling_frequency=sampling_frequency,
        window_length=window_length,
        step_size=step_size,
        frequency_range=frequency_range,
        calc_cpsd=calc_coh,
        n_sub_windows=n_sub_windows,
    )

    # Standardize before calculating the spectrogram
    if standardize:
        data = [(d - np.mean(d, axis=0)) / np.std(d, axis=0) for d in data]

    # Does alpha sum to a constant in every session?
    mixture = all(
        np.allclose(np.sum(a, axis=1), np.sum(a[0]), rtol=1e-4, atol=0) for a in alpha
    )

    # Main calculation of spectra
    kwargs = [
        {"data": d, "alpha": a, "spectrogram": spectrogram, "mixture": mixture}
        for d, a in zip(data, alpha)
    ]
    results = parallel_map(
        _regression_spectra_session,
        kwargs,
        desc="Calculating spectra",
        n_jobs=n_jobs,
        argument_type="kwargs",
        label="session",
    )
    f = spectrogram.frequencies

    # Weights for calculating a group-average spectrum
    w = _session_weights(data)

    # Group-level standard deviation of the windowed alpha for each mode,
    # averaged over the sessions where the mode's alpha varies
    sigma = np.array([result[2] for result in results])
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
        coefs, intercept, _ = result
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


# Maximum size in bytes of the intermediate arrays used to calculate the
# spectrogram for a chunk of windows
_MAX_CHUNK_BYTES = 2**28


def _validate_data_alpha(
    data: Union[np.ndarray, list],
    alpha: Optional[Union[np.ndarray, list]],
) -> Tuple[list, list]:
    """Validate the data and alpha passed to a spectra function.

    Parameters
    ----------
    data : np.ndarray or list
        Time series data. Shape must be (n_sessions, n_samples, n_channels)
        or (n_samples, n_channels).
    alpha : np.ndarray or list, optional
        State/mode time courses. Shape must be (n_sessions, n_samples,
        n_states) or (n_samples, n_states).

    Returns
    -------
    data : list of np.ndarray
        Time series for each session.
    alpha : list
        State/mode time courses for each session. Each item is
        :code:`None` if :code:`alpha` was not passed.
    """
    if alpha is not None and (
        (isinstance(data, list) != isinstance(alpha, list))
        or (isinstance(data, np.ndarray) != isinstance(alpha, np.ndarray))
    ):
        raise TypeError(
            f"data is type {type(data)} and alpha is type "
            f"{type(alpha)}. They must both be lists or numpy arrays."
        )

    if isinstance(data, np.ndarray):
        if alpha is not None and alpha.shape[0] != data.shape[0]:
            raise ValueError("data and alpha must have the same number of samples.")
        if data.ndim == 2:
            data = [data]
            alpha = None if alpha is None else [alpha]

    if alpha is None:
        alpha = [None] * len(data)

    if len(data) != len(alpha):
        raise ValueError(
            "A different number of arrays has been passed for "
            f"data and alpha: len(data)={len(data)}, "
            f"len(alpha)={len(alpha)}."
        )
    for d, a in zip(data, alpha):
        if a is not None and a.shape[0] != d.shape[0]:
            raise ValueError(
                "items in data and alpha must have the same number of samples."
            )

    return list(data), list(alpha)


def _session_weights(data: list) -> np.ndarray:
    """Weight of each session in a group average.

    Parameters
    ----------
    data : list of np.ndarray
        Time series for each session.

    Returns
    -------
    w : np.ndarray
        Number of samples in each session divided by the total.
        Shape is (n_sessions,).
    """
    n_samples = np.array([d.shape[0] for d in data])
    return n_samples / np.sum(n_samples)


@dataclass
class _Spectrogram:
    """Settings for calculating a spectrogram.

    Windows of :code:`window_length` samples are centered every
    :code:`step_size` samples (the data is zero padded at the start/end).
    Each window is split into :code:`n_sub_windows` sub-windows, which are
    multiplied by each taper and Fourier transformed. The periodograms are
    averaged over sub-windows and tapers.

    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency in Hz.
    window_length : int
        Number of samples in a window.
    step_size : int
        Number of samples to shift the window by.
    n_sub_windows : int
        Number of sub-windows to split a window into.
    tapers : np.ndarray
        Tapers applied to each sub-window. Shape is (n_tapers, sub_length).
    nfft : int
        Number of points in the FFT.
    scale : float
        Scaling applied to the periodograms.
    frequency_range : list
        Minimum and maximum frequency to keep.
    calc_cpsd : bool
        Should we calculate cross spectra?
    conj_first : bool
        Cross spectra are :code:`conj(X_i) X_j` if :code:`True`, otherwise
        :code:`X_i conj(X_j)`.
    """

    sampling_frequency: float
    window_length: int
    step_size: int
    n_sub_windows: int
    tapers: np.ndarray
    nfft: int
    scale: float
    frequency_range: List
    calc_cpsd: bool
    conj_first: bool

    @classmethod
    def create(
        cls,
        method: str,
        sampling_frequency: float,
        window_length: int,
        step_size: int,
        frequency_range: List,
        calc_cpsd: bool,
        n_sub_windows: int = 1,
        time_half_bandwidth: float = 4,
        n_tapers: int = 7,
    ) -> "_Spectrogram":
        """Create the settings for a Welch or multitaper spectrogram.

        Parameters
        ----------
        method : str
            :code:`'welch'` (Hann taper, SciPy scaling) or
            :code:`'multitaper'` (DPSS tapers).
        sampling_frequency : float
            Sampling frequency in Hz.
        window_length : int
            Number of samples in a window.
        step_size : int
            Number of samples to shift the window by.
        frequency_range : list
            Minimum and maximum frequency to keep.
        calc_cpsd : bool
            Should we calculate cross spectra?
        n_sub_windows : int, optional
            Number of sub-windows to split a window into.
        time_half_bandwidth : float, optional
            Time half bandwidth of the DPSS tapers.
        n_tapers : int, optional
            Number of DPSS tapers.

        Returns
        -------
        spectrogram : _Spectrogram
            Spectrogram settings.
        """
        window_length = int(window_length)
        if window_length % n_sub_windows != 0:
            raise ValueError("window_length must be divisible by n_sub_windows.")
        sub_length = window_length // n_sub_windows

        if method == "welch":
            tapers = signal.get_window("hann", sub_length)[np.newaxis]
            nfft = max(256, sub_length)
            scale = 2 / (sampling_frequency * np.sum(tapers**2))
            conj_first = False
        elif method == "multitaper":
            tapers = signal.windows.dpss(
                sub_length, NW=time_half_bandwidth, Kmax=n_tapers
            )
            tapers = np.atleast_2d(tapers)
            nfft = sub_length
            scale = 2 / sampling_frequency
            conj_first = True
        else:
            raise ValueError(f"method must be 'welch' or 'multitaper', got '{method}'.")

        return cls(
            sampling_frequency=sampling_frequency,
            window_length=window_length,
            step_size=int(step_size),
            n_sub_windows=n_sub_windows,
            tapers=tapers,
            nfft=nfft,
            scale=scale,
            frequency_range=frequency_range,
            calc_cpsd=calc_cpsd,
            conj_first=conj_first,
        )

    @property
    def frequencies(self) -> np.ndarray:
        """Frequency axis. Shape is (n_freq,)."""
        f = np.arange(self.nfft // 2 + 1) * self.sampling_frequency / self.nfft
        min_arg, max_arg = get_frequency_args_range(f, self.frequency_range)
        return f[min_arg : max_arg + 1]

    def _frequency_slice(self) -> slice:
        f = np.arange(self.nfft // 2 + 1) * self.sampling_frequency / self.nfft
        min_arg, max_arg = get_frequency_args_range(f, self.frequency_range)
        return slice(min_arg, max_arg + 1)

    def n_windows(self, n_samples: int) -> int:
        """Number of windows for a time series with :code:`n_samples`."""
        return n_samples // self.step_size

    def windows(self, x: np.ndarray) -> np.ndarray:
        """Windows of a time series.

        Parameters
        ----------
        x : np.ndarray
            Time series. Shape must be (n_samples, n_channels).

        Returns
        -------
        windows : np.ndarray
            Read-only view of the windows. Shape is (n_windows, n_channels,
            window_length).
        """
        pad = self.window_length // 2
        x = np.pad(x, ((pad, pad), (0, 0)))
        windows = sliding_window_view(x, self.window_length, axis=0)
        return windows[:: self.step_size][: self.n_windows(len(x) - 2 * pad)]

    def _fourier_chunks(
        self,
        x: np.ndarray,
        window_indices: Optional[np.ndarray],
        extra_window_bytes: int,
    ) -> Iterator[np.ndarray]:
        """Fourier transform of the tapered sub-windows for chunks of windows.

        Parameters
        ----------
        x : np.ndarray
            Time series. Shape must be (n_samples, n_channels).
        window_indices : np.ndarray
            Windows to calculate. If :code:`None`, all windows are used.
        extra_window_bytes : int
            Memory per window used by the caller, to set the chunk size.

        Yields
        ------
        X : np.ndarray
            Fourier transform. Shape is (n_windows_in_chunk, n_channels,
            n_segments, n_freq), where :code:`n_segments` is the number of
            sub-windows times the number of tapers.
        """
        windows = self.windows(x)
        if window_indices is None:
            window_indices = np.arange(windows.shape[0])

        n_channels = x.shape[1]
        n_tapers, sub_length = self.tapers.shape
        n_segments = self.n_sub_windows * n_tapers
        freq_slice = self._frequency_slice()
        n_freq = freq_slice.stop - freq_slice.start

        window_bytes = 16 * n_channels * n_segments * max(sub_length, n_freq)
        chunk_size = max(1, _MAX_CHUNK_BYTES // (window_bytes + extra_window_bytes))

        for i in range(0, len(window_indices), chunk_size):
            w = windows[window_indices[i : i + chunk_size]]
            n_chunk = w.shape[0]

            # Split into sub-windows and apply the tapers
            w = w.reshape(n_chunk, n_channels, self.n_sub_windows, 1, sub_length)
            w = w * self.tapers

            # Fourier transform
            X = np.fft.rfft(w, n=self.nfft, axis=-1)[..., freq_slice]
            yield X.reshape(n_chunk, n_channels, n_segments, n_freq)

    def chunks(
        self, x: np.ndarray, window_indices: Optional[np.ndarray] = None
    ) -> Iterator[np.ndarray]:
        """Spectrogram for chunks of windows.

        Parameters
        ----------
        x : np.ndarray
            Time series. Shape must be (n_samples, n_channels).
        window_indices : np.ndarray, optional
            Windows to calculate. Defaults to all windows.

        Yields
        ------
        P : np.ndarray
            Spectrogram for a chunk of windows. Shape is (n_windows_in_chunk,
            n_channels * (n_channels + 1) / 2, n_freq) if
            :code:`calc_cpsd=True`, otherwise (n_windows_in_chunk, n_channels,
            n_freq).
        """
        n_channels = x.shape[1]
        m, n = np.triu_indices(n_channels)
        n_freq = len(self.frequencies)
        extra_bytes = 16 * n_channels**2 * n_freq if self.calc_cpsd else 0

        for X in self._fourier_chunks(x, window_indices, extra_bytes):
            n_segments = X.shape[2]
            if self.calc_cpsd:
                # Cross spectra averaged over sub-windows and tapers,
                # G[c, d] = mean_s X_c conj(X_d)
                X = np.moveaxis(X, -1, 1)
                G = X @ np.conj(np.swapaxes(X, -1, -2))
                if self.conj_first:
                    G = np.conj(G)
                P = np.moveaxis(G[..., m, n], 1, -1) / n_segments
                yield (P * self.scale).astype(np.complex64)
            else:
                P = np.mean(X.real**2 + X.imag**2, axis=2)
                yield (P * self.scale).astype(np.float32)

    def mean(self, x: np.ndarray) -> np.ndarray:
        """Spectrogram averaged over all windows.

        Parameters
        ----------
        x : np.ndarray
            Time series. Shape must be (n_samples, n_channels).

        Returns
        -------
        P : np.ndarray
            Mean spectrum. Shape is (n_channels * (n_channels + 1) / 2,
            n_freq) if :code:`calc_cpsd=True`, otherwise (n_channels, n_freq).
        """
        n_channels = x.shape[1]
        m, n = np.triu_indices(n_channels)

        # Sum over windows, sub-windows and tapers
        total = 0
        n_terms = 0
        for X in self._fourier_chunks(x, None, 0):
            n_terms += X.shape[0] * X.shape[2]
            if self.calc_cpsd:
                # G[f, c, d] = sum_{w,s} X_c conj(X_d)
                Y = np.ascontiguousarray(np.transpose(X, (3, 0, 2, 1)))
                Y = Y.reshape(X.shape[-1], -1, n_channels)
                total = total + np.conj(np.swapaxes(Y, -1, -2)) @ Y
            else:
                total = total + np.sum(X.real**2 + X.imag**2, axis=(0, 2))

        if self.calc_cpsd:
            # total[f] = X^H X, i.e. total[f, c, d] = sum conj(X_c) X_d
            G = total if self.conj_first else np.conj(total)
            P = np.moveaxis(G[:, m, n], 0, -1)
        else:
            P = total
        return P * self.scale / n_terms

    def compute(
        self, x: np.ndarray, window_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Spectrogram for all windows.

        Parameters
        ----------
        x : np.ndarray
            Time series. Shape must be (n_samples, n_channels).
        window_indices : np.ndarray, optional
            Windows to calculate. Defaults to all windows.

        Returns
        -------
        P : np.ndarray
            Spectrogram. Shape is (n_windows, n_channels * (n_channels + 1) /
            2, n_freq) if :code:`calc_cpsd=True`, otherwise (n_windows,
            n_channels, n_freq).
        """
        return np.concatenate(list(self.chunks(x, window_indices)))

    def window_mean(
        self, alpha: np.ndarray, window_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Mean of a time series in each window.

        The mean in each sub-window is weighted by the square of the first
        taper, which is how the periodogram weights each sample.

        Parameters
        ----------
        alpha : np.ndarray
            Time series. Shape must be (n_samples, n_modes).
        window_indices : np.ndarray, optional
            Windows to calculate. Defaults to all windows.

        Returns
        -------
        mean_alpha : np.ndarray
            Mean for each window. Shape is (n_windows, n_modes).
        """
        n_samples = alpha.shape[0]
        if window_indices is None:
            window_indices = np.arange(self.n_windows(n_samples))

        pad = self.window_length // 2
        alpha = np.pad(alpha.astype(np.float64), ((pad, pad), (0, 0)))

        # Weighted mean of each sub-window starting at every sample
        weights = self.tapers[0] ** 2
        weights /= np.sum(weights)
        sub_mean = np.stack(
            [np.correlate(a, weights, mode="valid") for a in alpha.T], axis=-1
        )

        # Average the sub-windows in each window
        sub_length = self.tapers.shape[1]
        starts = window_indices * self.step_size
        offsets = np.arange(self.n_sub_windows) * sub_length
        mean_alpha = np.mean(sub_mean[starts[:, np.newaxis] + offsets], axis=1)
        return mean_alpha.astype(np.float32)


def _state_spectra(
    method: str,
    data: Union[np.ndarray, list],
    sampling_frequency: float,
    alpha: Optional[Union[np.ndarray, list]],
    window_length: Optional[int],
    step_size: int,
    frequency_range: Optional[List],
    standardize: bool,
    averaging: str,
    calc_cpsd: bool,
    calc_coh: bool,
    return_weights: bool,
    keepdims: bool,
    n_jobs: int,
    time_half_bandwidth: float = 4,
    n_tapers: int = 7,
):
    """Calculate spectra for each state with Welch's method or a multitaper.

    See :func:`welch_spectra` and :func:`multitaper_spectra`.
    """
    data, alpha = _validate_data_alpha(data, alpha)

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    if calc_cpsd:
        calc_coh = False

    spectrogram = _Spectrogram.create(
        method=method,
        sampling_frequency=sampling_frequency,
        window_length=window_length,
        step_size=step_size,
        frequency_range=frequency_range,
        calc_cpsd=calc_cpsd or calc_coh,
        time_half_bandwidth=time_half_bandwidth,
        n_tapers=n_tapers,
    )

    kwargs = [
        {
            "data": d,
            "alpha": a,
            "spectrogram": spectrogram,
            "standardize": standardize,
            "averaging": averaging,
            "calc_cpsd": calc_cpsd,
            "calc_coh": calc_coh,
        }
        for d, a in zip(data, alpha)
    ]
    results = parallel_map(
        _state_spectra_session,
        kwargs,
        desc="Calculating spectra",
        n_jobs=n_jobs,
        argument_type="kwargs",
        label="session",
    )

    f = spectrogram.frequencies
    psd = [result[0] for result in results]
    if calc_coh:
        coh = [result[1] for result in results]

    if not keepdims:
        # Remove any axes that are of length 1
        psd = np.squeeze(psd)
        if calc_coh:
            coh = np.squeeze(coh)

    outputs = [f, psd]
    if calc_coh:
        outputs.append(coh)
    if return_weights:
        outputs.append(_session_weights(data))
    return tuple(outputs)


def _state_spectra_session(
    data: np.ndarray,
    alpha: Optional[np.ndarray],
    spectrogram: _Spectrogram,
    standardize: bool,
    averaging: str,
    calc_cpsd: bool,
    calc_coh: bool,
):
    """Calculate spectra for each state for a single session.

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    alpha : np.ndarray
        State probability time course. Shape must be (n_samples, n_states).
        If :code:`None`, a single state is active at all time points.
    spectrogram : _Spectrogram
        Spectrogram settings.
    standardize : bool
        Should we standardize the data before calculating the spectra?
    averaging : str
        Method used to average periodograms: :code:`'mean'` or
        :code:`'median'`.
    calc_cpsd : bool
        Should we return the cross spectra for :code:`psd`?
    calc_coh : bool
        Should we calculate the coherence between channels?

    Returns
    -------
    psd : np.ndarray
        Power spectra. Shape is (n_states, n_channels, n_freq) if
        :code:`calc_cpsd=False`, otherwise (n_states, n_channels, n_channels,
        n_freq).
    coh : np.ndarray
        Coherence. Shape is (n_states, n_channels, n_channels, n_freq).
        Only returned if :code:`calc_coh=True`.
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
        # Spectrogram of this state's data averaged over windows
        x = data * alpha[:, i][..., np.newaxis]
        if averaging == "median":
            p = spectrogram.compute(x)
            bias = _median_bias(p.shape[0])
            if np.iscomplexobj(p):
                p = np.median(p.real, axis=0) + 1j * np.median(p.imag, axis=0)
            else:
                p = np.median(p, axis=0)
            p /= bias
        else:
            p = spectrogram.mean(x)
        p = (p * scaling[i]).astype(
            np.complex64 if spectrogram.calc_cpsd else np.float32
        )
        n_freq = p.shape[-1]

        if calc_coh or calc_cpsd:
            # Create a channels by channels matrix for the cross spectra
            cpsd = np.empty([n_channels, n_channels, n_freq], dtype=np.complex64)
            cpsd[m, n] = p
            cpsd[n, m] = np.conj(p)

        if calc_coh:
            psd.append(cpsd[range(n_channels), range(n_channels)].real)
            coh.append(coherence_spectra(cpsd))
        elif calc_cpsd:
            psd.append(cpsd)
        else:
            psd.append(p)

    psd = np.array(psd)
    if np.isnan(psd).any():
        _logger.warning("PSD contains NaN values.")
        psd = np.nan_to_num(psd)  # zero out nan values

    if calc_coh:
        return psd, np.array(coh)
    return (psd,)


def _regression_spectra_session(
    data: np.ndarray,
    alpha: np.ndarray,
    spectrogram: _Spectrogram,
    mixture: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate regression spectra for a single session.

    Parameters
    ----------
    data : np.ndarray
        Data to calculate a spectrogram for.
        Shape must be (n_samples, n_channels).
    alpha : np.ndarray
        Inferred mode mixing factors. Shape must be (n_samples, n_modes).
    spectrogram : _Spectrogram
        Spectrogram settings.
    mixture : bool
        Does alpha sum to a constant? See :func:`_regress_spectrogram`.

    Returns
    -------
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
    # Only use windows that do not overlap the zero padding at the start/end
    keep = _inner_windows(
        data.shape[0], spectrogram.window_length, spectrogram.step_size
    )
    a = spectrogram.window_mean(alpha, keep)
    if len(keep) <= a.shape[1]:
        raise ValueError(
            f"Each session must have more than n_modes={a.shape[1]} windows that "
            f"do not overlap its start or end, got {len(keep)}. Use longer "
            "sessions or a smaller window_length or step_size."
        )
    return _regress_spectrogram(a, spectrogram.chunks(data, keep), mixture)


def _regress_spectrogram(
    a: np.ndarray, chunks: Iterable[np.ndarray], mixture: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regress a spectrogram on windowed mode time courses.

    Parameters
    ----------
    a : np.ndarray
        Windowed mode time courses. Shape must be (n_windows, n_modes).
    chunks : iterable of np.ndarray
        Spectrogram for consecutive chunks of the windows. Each chunk has
        shape (n_windows_in_chunk, ...).
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

    a = a.astype(np.float64)
    if mixture:
        a /= np.mean(np.sum(a, axis=1))
    sigma = np.std(a, axis=0)

    # Modes whose time course barely varies are not fitted (NaN coefficients)
    used = sigma > 1e-4 * np.max(sigma)
    sigma[~used] = 0

    if mixture:
        # Mixture of mode spectra, P_t = sum_j a_tj P_j
        pinv = np.linalg.pinv(a[:, used])
    else:
        # Linear regression with standardized regressors and an intercept
        z = (a[:, used] - np.mean(a[:, used], axis=0)) / sigma[used]
        pinv = np.linalg.pinv(z)

    # Accumulate the intercept (static spectrum) and fit over the chunks
    intercept = 0
    fit = 0
    start = 0
    for p in chunks:
        shape, dtype = p.shape[1:], p.dtype
        y = p.reshape(p.shape[0], -1)
        y = y.astype(np.complex128 if np.iscomplexobj(y) else np.float64)
        intercept = intercept + np.sum(y, axis=0)
        fit = fit + pinv[:, start : start + y.shape[0]] @ y
        start += y.shape[0]
    intercept = intercept / n_windows

    # Coefficients: change in the spectrum for a one standard deviation
    # increase in a mode's time course
    coefs = np.full((n_modes, intercept.shape[0]), np.nan, dtype=intercept.dtype)
    if mixture:
        fit = sigma[used, np.newaxis] * (fit - intercept)
    coefs[used] = fit

    coefs = coefs.reshape((n_modes,) + shape).astype(dtype)
    intercept = intercept.reshape(shape).astype(dtype)
    return coefs, intercept, sigma


def _inner_windows(n_samples: int, window_length: int, step_size: int) -> np.ndarray:
    """Windows of a spectrogram that do not overlap the zero padding.

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
