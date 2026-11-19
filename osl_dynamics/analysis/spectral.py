"""Functions to perform spectral analysis."""

import functools
import inspect
import json
import logging
import warnings
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.signal._spectral_py import _median_bias
from sklearn.decomposition import non_negative_factorization

from osl_dynamics.utils import array_ops
from osl_dynamics.utils.misc import nextpow2
from osl_dynamics.utils.progress import parallel_map

if TYPE_CHECKING:
    import matplotlib.axes

    from osl_dynamics.data import Data

_logger = logging.getLogger("osl-dynamics")


_SPECTRA_FORMAT_VERSION = 1


def _deprecate_positional(n_positional: int) -> Callable:
    """Allow arguments after the first :code:`n_positional` to be passed
    positionally, with a deprecation warning.

    Parameters
    ----------
    n_positional : int
        Number of arguments that can be passed positionally.

    Returns
    -------
    decorator : callable
        Function decorator.
    """

    def decorator(fn):
        names = list(inspect.signature(fn).parameters)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if len(args) > n_positional:
                extra = names[n_positional : len(args)]
                for name in extra:
                    if name in kwargs:
                        raise TypeError(
                            f"{fn.__name__}() got multiple values for argument "
                            f"'{name}'"
                        )
                warnings.warn(
                    f"Passing {', '.join(extra)} to {fn.__name__} positionally is "
                    "deprecated. Pass them as keyword arguments.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs.update(zip(extra, args[n_positional:]))
                args = args[:n_positional]
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def _legacy_flags(
    function: str,
    cross_spectra: Optional[bool],
    flags: dict,
    defaults: dict,
) -> Tuple[dict, bool]:
    """Handle the deprecated arguments that set the legacy output tuple.

    Parameters
    ----------
    function : str
        Name of the function the arguments were passed to.
    cross_spectra : bool
        Value of the :code:`cross_spectra` argument (:code:`None` if it was
        not passed).
    flags : dict
        Deprecated arguments (:code:`None` if not passed).
    defaults : dict
        Previous defaults of the deprecated arguments.

    Returns
    -------
    legacy : dict
        Values of the deprecated arguments used to build the legacy tuple.
    cross_spectra : bool
        Should the cross spectra be calculated?
    """
    passed = [name for name, value in flags.items() if value is not None]
    if passed:
        single = len(passed) == 1
        warnings.warn(
            f"{', '.join(passed)} {'is' if single else 'are'} deprecated in "
            f"{function}: {'it sets' if single else 'they set'} the tuple "
            "returned when the result is unpacked. Use the attributes of the "
            "returned object instead, and cross_spectra=False to only calculate "
            "the PSDs.",
            DeprecationWarning,
            stacklevel=4,
        )
    legacy = {
        name: defaults[name] if value is None else value
        for name, value in flags.items()
    }
    legacy["function"] = function

    if cross_spectra is None:
        # calc_coh=False (without calc_cpsd=True) only needs the PSDs
        cross_spectra = flags.get("calc_coh") is not False or bool(
            legacy.get("calc_cpsd")
        )
    elif not cross_spectra:
        if flags.get("calc_cpsd") or flags.get("calc_coh"):
            raise ValueError("calc_coh and calc_cpsd require cross_spectra=True.")
        legacy["calc_coh"] = False
        legacy["calc_cpsd"] = False
    return legacy, cross_spectra


def _triu(n_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of the upper triangle (including the diagonal)."""
    return np.triu_indices(n_channels)


def _psd_from_cross(x: np.ndarray, n_channels: int) -> np.ndarray:
    """PSDs (..., n_channels, n_freq) from upper-triangle cross spectra."""
    m, n = _triu(n_channels)
    return x[..., np.flatnonzero(m == n), :].real


def _full_from_triu(x: np.ndarray, n_channels: int) -> np.ndarray:
    """Hermitian (..., n_channels, n_channels, n_freq) from an upper triangle."""
    m, n = _triu(n_channels)
    full = np.empty(x.shape[:-2] + (n_channels, n_channels, x.shape[-1]), x.dtype)
    full[..., m, n, :] = x
    full[..., n, m, :] = np.conj(x)
    return full


def _symmetric_from_triu(x: np.ndarray, n_channels: int) -> np.ndarray:
    """Symmetric (..., n_channels, n_channels, n_freq) from an upper triangle."""
    m, n = _triu(n_channels)
    full = np.empty(x.shape[:-2] + (n_channels, n_channels, x.shape[-1]), x.dtype)
    full[..., m, n, :] = x
    full[..., n, m, :] = x
    return full


def _coherence_from_cross(x: np.ndarray, n_channels: int) -> np.ndarray:
    """Coherence (..., n_channels, n_channels, n_freq) from cross spectra."""
    m, n = _triu(n_channels)
    pxx = _psd_from_cross(x, n_channels)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.abs(x) / np.sqrt(pxx[..., m, :] * pxx[..., n, :])
    return _symmetric_from_triu(coh.astype(np.float32), n_channels)


def _coherence_change(
    coefs: np.ndarray, static: np.ndarray, n_channels: int
) -> np.ndarray:
    """Change in coherence for regression coefficients of the cross spectra.

    The change in the cross spectrum in the direction (phase) of the static
    cross spectrum, normalized by the static power.

    Parameters
    ----------
    coefs : np.ndarray
        Regression coefficients. Shape is (..., n_modes, n_pairs, n_freq).
    static : np.ndarray
        Static cross spectra. Shape is (..., n_pairs, n_freq).
    n_channels : int
        Number of channels.

    Returns
    -------
    coh : np.ndarray
        Change in coherence. Shape is (..., n_modes, n_channels, n_channels,
        n_freq).
    """
    m, n = _triu(n_channels)
    static = static[..., np.newaxis, :, :]
    pxx = _psd_from_cross(static, n_channels)
    norm = np.sqrt(pxx[..., m, :] * pxx[..., n, :])
    abs_static = np.abs(static)
    phase = np.divide(
        static,
        abs_static,
        out=np.zeros_like(static),
        where=abs_static > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.real(coefs * np.conj(phase)) / norm
    return _symmetric_from_triu(coh.astype(np.float32), n_channels)


def _weighted_nanmean(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean over the first axis, ignoring NaN values.

    Parameters
    ----------
    x : np.ndarray
        Array to average. Shape is (n_sessions, ...).
    weights : np.ndarray
        Weight for each session. Shape is (n_sessions,).

    Returns
    -------
    mean : np.ndarray
        Weighted mean. NaN where all sessions are NaN. Shape is (...).
    """
    w = weights.reshape((-1,) + (1,) * (x.ndim - 1))
    valid = ~np.isnan(x)
    total_weight = np.sum(w * valid, axis=0)
    total = np.sum(np.where(valid, x, 0) * w, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(total_weight > 0, total / total_weight, np.nan)
    return mean.astype(x.dtype)


def _to_json(value):
    """Convert numpy types in nested dicts/lists to JSON types."""
    if isinstance(value, dict):
        return {k: _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_to_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _frequency_weights(
    frequencies: np.ndarray,
    bands: Optional[Union[list, np.ndarray]],
    frequency_range: Optional[list],
    method: str = "mean",
) -> Tuple[np.ndarray, bool]:
    """Weights for averaging spectra over frequency bands.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis. Shape is (n_freq,).
    bands : list or np.ndarray
        Frequency ranges, e.g. :code:`[[1, 4], [4, 8]]`, or spectral
        components with shape (n_components, n_freq).
    frequency_range : list
        A single frequency range. Only one of :code:`bands` and
        :code:`frequency_range` can be passed.
    method : str, optional
        For frequency ranges: take the :code:`'mean'`, :code:`'sum'` or
        :code:`'integral'` over the range. Spectral components always give
        a weighted mean.

    Returns
    -------
    weights : np.ndarray
        Weights for each band. Shape is (n_bands, n_freq).
    single : bool
        Was a single band requested (no band axis in the output)?
    """
    if bands is not None and frequency_range is not None:
        raise ValueError("Only one of bands and frequency_range can be passed.")
    if method not in ["mean", "sum", "integral"]:
        raise ValueError(f"method must be 'mean', 'sum' or 'integral', got {method!r}.")

    n_freq = len(frequencies)
    single = bands is None
    if bands is None:
        if frequency_range is None:
            frequency_range = [frequencies[0], frequencies[-1]]
        bands = [frequency_range]
    bands = np.asarray(bands, dtype=np.float64)
    if bands.ndim != 2:
        raise ValueError(
            "bands must be a list of frequency ranges or an array of spectral "
            "components with shape (n_components, n_freq)."
        )

    if bands.shape[1] == n_freq and n_freq != 2:
        # Spectral components: weighted mean
        return bands / np.sum(bands, axis=1, keepdims=True), single

    if bands.shape[1] != 2:
        raise ValueError(
            f"bands must have shape (n_bands, 2) or (n_components, {n_freq})."
        )
    weights = np.array(
        [(frequencies >= low) & (frequencies <= high) for low, high in bands],
        dtype=np.float64,
    )
    empty = np.flatnonzero(np.sum(weights, axis=1) == 0)
    if len(empty) > 0:
        raise ValueError(
            f"No frequencies in band(s) {bands[empty].tolist()}. The frequencies "
            f"are {frequencies[0]:g}-{frequencies[-1]:g} Hz in steps of "
            f"{frequencies[1] - frequencies[0]:g} Hz."
        )
    if method == "mean":
        weights /= np.sum(weights, axis=1, keepdims=True)
    elif method == "integral":
        weights *= frequencies[1] - frequencies[0]
    return weights, single


def _threshold_networks(c: np.ndarray, percentile: float) -> np.ndarray:
    """Keep the edges with the largest absolute values.

    Parameters
    ----------
    c : np.ndarray
        Symmetric networks. Shape is (..., n_channels, n_channels).
    percentile : float
        Edges with an absolute value above this percentile (of the edges in
        each network) are kept.

    Returns
    -------
    c : np.ndarray
        Networks with the other edges (and the diagonal) set to zero.
    """
    m, n = np.triu_indices(c.shape[-1], k=1)
    edges = c[..., m, n]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        threshold = np.nanpercentile(np.abs(edges), percentile, axis=-1, keepdims=True)
    edges = np.where(np.abs(edges) > threshold, edges, 0)
    out = np.zeros_like(c)
    out[..., m, n] = edges
    out[..., n, m] = edges
    return out


class Spectra:
    """Static power spectra and cross spectra.

    Returned by :func:`osl_dynamics.analysis.static.welch_spectra`,
    :func:`osl_dynamics.analysis.static.multitaper_spectra`, and by
    :func:`welch_spectra` and :func:`multitaper_spectra` without
    :code:`alpha`.

    The cross spectra are stored (as the upper triangle) and the PSDs and
    coherences are calculated from them. The first axis of each array is the
    session, except for a group average (see :meth:`group_average`).
    Cross spectra are :math:`\\langle \\bar{X}_i X_j \\rangle`, where
    :math:`X_i` is the Fourier transform of channel :math:`i`.

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency axis. Shape is (n_freq,).
    weights : np.ndarray
        Weight of each session in a group average (the number of samples in
        a session divided by the total). Shape is (n_sessions,). :code:`None`
        for a group average.
    n_channels : int
        Number of channels.
    sampling_frequency : float
        Sampling frequency in Hz.
    method : str
        Method used to calculate the spectra.
    settings : dict
        Settings used to calculate the spectra.
    """

    _session_arrays = ("spectra",)

    def __init__(
        self,
        frequencies: np.ndarray,
        arrays: dict,
        n_channels: int,
        weights: Optional[np.ndarray],
        sampling_frequency: float,
        method: str,
        settings: dict,
        has_cross_spectra: bool,
        legacy: Optional[dict] = None,
    ):
        self.frequencies = np.asarray(frequencies)
        self._data = dict(arrays)
        self.n_channels = int(n_channels)
        self.weights = None if weights is None else np.asarray(weights)
        self.sampling_frequency = sampling_frequency
        self.method = method
        self.settings = dict(settings)
        self.has_cross_spectra = bool(has_cross_spectra)
        self._legacy = legacy

    # Shapes

    @property
    def is_group(self) -> bool:
        """Is this a group average (no session axis)?"""
        return self.weights is None

    @property
    def n_sessions(self) -> Optional[int]:
        """Number of sessions. :code:`None` for a group average."""
        return None if self.is_group else len(self.weights)

    @property
    def n_freq(self) -> int:
        """Number of frequency bins."""
        return len(self.frequencies)

    # Derived quantities

    def _require_cross_spectra(self):
        if not self.has_cross_spectra:
            raise ValueError(
                "Only PSDs were calculated. Pass cross_spectra=True to calculate "
                "the cross spectra and coherence."
            )

    def _psd(self, x: np.ndarray) -> np.ndarray:
        if self.has_cross_spectra:
            return _psd_from_cross(x, self.n_channels)
        return x

    def _cross(self, x: np.ndarray) -> np.ndarray:
        self._require_cross_spectra()
        return _full_from_triu(x, self.n_channels)

    def _coherence(self, x: np.ndarray) -> np.ndarray:
        self._require_cross_spectra()
        return _coherence_from_cross(x, self.n_channels)

    def _spectra(self) -> np.ndarray:
        return self._data["spectra"]

    @property
    def psd(self) -> np.ndarray:
        """Power spectral density.

        Shape is (n_sessions, n_channels, n_freq).
        """
        return self._psd(self._spectra())

    @property
    def cross_spectra(self) -> np.ndarray:
        """Cross spectral density.

        Shape is (n_sessions, n_channels, n_channels, n_freq).
        """
        return self._cross(self._spectra())

    @property
    def coherence(self) -> np.ndarray:
        """Coherence.

        Shape is (n_sessions, n_channels, n_channels, n_freq).
        """
        return self._coherence(self._spectra())

    # Group average

    def group_average(self) -> "Spectra":
        """Group average of the spectra.

        The cross spectra (or PSDs) are averaged over sessions with
        :code:`weights`, ignoring NaN values. Coherences are calculated from
        the group-averaged cross spectra.

        Returns
        -------
        spectra : Spectra
            Group-averaged spectra, an object of the same class without a
            session axis.
        """
        if self.is_group:
            raise ValueError("The spectra are already a group average.")
        arrays = {
            name: (
                _weighted_nanmean(value, self.weights)
                if name in self._session_arrays
                else value
            )
            for name, value in self._data.items()
        }
        return self._new(arrays, weights=None)

    def _new(self, arrays: dict, weights: Optional[np.ndarray]) -> "Spectra":
        return type(self)(
            frequencies=self.frequencies,
            arrays=arrays,
            n_channels=self.n_channels,
            weights=weights,
            sampling_frequency=self.sampling_frequency,
            method=self.method,
            settings=self.settings,
            has_cross_spectra=self.has_cross_spectra,
        )

    # Analysis

    _psd_values = ("psd",)
    _coherence_values = ("coherence",)

    def _values(self, values: Optional[str], kind: str) -> Tuple[str, np.ndarray]:
        allowed = self._psd_values if kind == "psd" else self._coherence_values
        name = allowed[0] if values is None else values
        if name not in allowed:
            raise ValueError(f"values must be one of {allowed}, got {name!r}.")
        return name, getattr(self, name)

    @property
    def _band_axis(self) -> int:
        # Position of the band axis: after the session axis
        return 0 if self.is_group else 1

    def power_maps(
        self,
        values: Optional[str] = None,
        bands: Optional[Union[list, np.ndarray]] = None,
        frequency_range: Optional[list] = None,
        method: str = "mean",
    ) -> np.ndarray:
        """Power maps: PSDs averaged over frequency bands.

        Parameters
        ----------
        values : str, optional
            Which PSDs to use. Defaults to the first of: :code:`'psd_coefs'`
            (regression spectra), :code:`'psd_change'` (state spectra, the
            change from the static PSD), :code:`'psd'` (static spectra).
            State and regression spectra also accept :code:`'psd'` and
            :code:`'static_psd'`.
        bands : list or np.ndarray, optional
            Frequency ranges, e.g. :code:`[[1, 4], [4, 8]]`, or spectral
            components from :meth:`decompose` with shape (n_components,
            n_freq).
        frequency_range : list, optional
            A single frequency range. Defaults to all frequencies.
        method : str, optional
            For frequency ranges: take the :code:`'mean'`, :code:`'sum'` or
            :code:`'integral'` of the PSD. Use :code:`'integral'` for the
            variance.

        Returns
        -------
        power_maps : np.ndarray
            Shape is (n_sessions, n_bands, n_states, n_channels). There is no
            session axis for a group average, no band axis unless
            :code:`bands` is passed and no state axis for static spectra.
        """
        _, psd = self._values(values, "psd")
        weights, single = _frequency_weights(
            self.frequencies, bands, frequency_range, method
        )
        p = np.moveaxis(psd @ weights.T, -1, self._band_axis)
        return np.squeeze(p, axis=self._band_axis) if single else p

    def coherence_networks(
        self,
        values: Optional[str] = None,
        bands: Optional[Union[list, np.ndarray]] = None,
        frequency_range: Optional[list] = None,
        percentile: Optional[float] = None,
        vectorize: bool = False,
    ) -> np.ndarray:
        """Coherence networks: coherences averaged over frequency bands.

        Parameters
        ----------
        values : str, optional
            Which coherences to use. Defaults to the first of:
            :code:`'coherence_coefs'` (regression spectra),
            :code:`'coherence_change'` (state spectra, the change from the
            static coherence), :code:`'coherence'` (static spectra).
            State and regression spectra also accept :code:`'coherence'` and
            :code:`'static_coherence'`.
        bands : list or np.ndarray, optional
            Frequency ranges, e.g. :code:`[[1, 4], [4, 8]]`, or spectral
            components from :meth:`decompose` with shape (n_components,
            n_freq).
        frequency_range : list, optional
            A single frequency range. Defaults to all frequencies.
        percentile : float, optional
            Only keep the edges with an absolute value above this percentile
            of each network, e.g. :code:`97` keeps the 3% of edges with the
            largest (absolute) values. Other edges and the diagonal are set to
            zero. Defaults to keeping all edges.
        vectorize : bool, optional
            Return the upper triangle (excluding the diagonal) of each
            network, e.g. for statistical tests.

        Returns
        -------
        networks : np.ndarray
            Shape is (n_sessions, n_bands, n_states, n_channels, n_channels),
            or (..., n_channels * (n_channels - 1) / 2) if
            :code:`vectorize=True`. There is no session axis for a group
            average, no band axis unless :code:`bands` is passed and no state
            axis for static spectra.
        """
        _, coh = self._values(values, "coherence")
        weights, single = _frequency_weights(self.frequencies, bands, frequency_range)
        c = np.moveaxis(coh @ weights.T, -1, self._band_axis)
        if percentile is not None:
            c = _threshold_networks(c, percentile)
        if vectorize:
            m, n = np.triu_indices(self.n_channels, k=1)
            c = c[..., m, n]
        return np.squeeze(c, axis=self._band_axis) if single else c

    _decompose_values = "coherence"

    def decompose(
        self, n_components: int, values: Optional[str] = None, **kwargs
    ) -> np.ndarray:
        """Spectral components from non-negative matrix factorization.

        See :func:`decompose_spectra`.

        Parameters
        ----------
        n_components : int
            Number of spectral components.
        values : str, optional
            Which (non-negative) coherences to decompose. Defaults to
            :code:`'coherence'`, or :code:`'static_coherence'` for regression
            spectra.
        **kwargs
            Keyword arguments for :func:`decompose_spectra`.

        Returns
        -------
        components : np.ndarray
            Spectral components, which can be passed as :code:`bands` to
            :meth:`power_maps` and :meth:`coherence_networks`. Shape is
            (n_components, n_freq).
        """
        values = self._decompose_values if values is None else values
        _, coh = self._values(values, "coherence")
        return decompose_spectra(coh, n_components=n_components, **kwargs)

    def plot_psd(
        self,
        values: Optional[str] = None,
        states: Optional[List[int]] = None,
        channels: Optional[List[int]] = None,
        static: bool = False,
        labels: Optional[List[str]] = None,
        frequency_range: Optional[list] = None,
        filename: Optional[str] = None,
        ax: Optional["matplotlib.axes.Axes"] = None,
    ):
        """Plot the group-average PSDs, averaged over channels.

        The shaded area is the standard error over channels.

        Parameters
        ----------
        values : str, optional
            Which PSDs to plot. See :meth:`power_maps`.
        states : list of int, optional
            States to plot. Defaults to all states.
        channels : list of int, optional
            Channels to average over. Defaults to all channels.
        static : bool, optional
            Also plot the static PSD (dashed black line).
        labels : list of str, optional
            Legend label for each state.
        frequency_range : list, optional
            Shade this frequency range.
        filename : str, optional
            Output filename. If :code:`None`, the figure is returned.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure. Only returned if :code:`filename` and :code:`ax` are
            :code:`None`.
        ax : matplotlib.axes.Axes
            Axis. Only returned if :code:`filename` and :code:`ax` are
            :code:`None`.
        """
        from osl_dynamics.utils import plotting

        if static and not isinstance(self, StateSpectra):
            raise ValueError("static=True can only be used for state spectra.")

        spectra = self if self.is_group else self.group_average()
        name, psd = spectra._values(values, "psd")
        channels = slice(None) if channels is None else list(channels)

        def _mean_and_error(p):
            p = p[..., channels, :]
            n = p.shape[-2]
            return np.mean(p, axis=-2), np.std(p, axis=-2) / np.sqrt(n)

        mean, error = _mean_and_error(psd)
        has_states = mean.ndim == 2
        mean, error = np.atleast_2d(mean), np.atleast_2d(error)
        states = range(mean.shape[0]) if states is None else states
        if labels is None:
            if has_states:
                kind = "Mode" if isinstance(self, RegressionSpectra) else "State"
                labels = [f"{kind} {i + 1}" for i in states]
            else:
                labels = ["Static"]

        create_fig = ax is None
        if create_fig:
            fig, ax = plotting.create_figure(figsize=(7, 4))

        f = self.frequencies
        for state, label in zip(states, labels):
            (line,) = ax.plot(f, mean[state], label=label)
            ax.fill_between(
                f,
                mean[state] - error[state],
                mean[state] + error[state],
                color=line.get_color(),
                alpha=0.3,
                linewidth=0,
            )
        if static:
            static_mean, _ = _mean_and_error(spectra.static_psd)
            ax.plot(f, static_mean, color="black", linestyle="--", label="Static")
        if name in ("psd_change", "psd_coefs"):
            ax.axhline(0, color="gray", linewidth=0.5)
        if frequency_range is not None:
            ax.axvspan(*frequency_range, color="gray", alpha=0.25, linewidth=0)

        ax.set_xlim(f[0], f[-1])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(
            "Change in PSD (a.u.)"
            if name in ("psd_change", "psd_coefs")
            else "PSD (a.u.)"
        )
        ax.legend()

        if filename is not None:
            plotting.save(fig if create_fig else ax.figure, filename, tight_layout=True)
        elif create_fig:
            return fig, ax

    # Saving and loading

    def save(self, filename: str) -> None:
        """Save the spectra to a :code:`.npz` file.

        Parameters
        ----------
        filename : str
            Output filename. :code:`.npz` is appended if it is missing.
        """
        metadata = {
            "class": type(self).__name__,
            "format_version": _SPECTRA_FORMAT_VERSION,
            "n_channels": self.n_channels,
            "sampling_frequency": self.sampling_frequency,
            "method": self.method,
            "settings": self.settings,
            "has_cross_spectra": self.has_cross_spectra,
        }
        arrays = {f"data_{name}": value for name, value in self._data.items()}
        if self.weights is not None:
            arrays["weights"] = self.weights
        np.savez(
            filename,
            metadata=json.dumps(_to_json(metadata)),
            frequencies=self.frequencies,
            **arrays,
        )

    @classmethod
    def load(cls, filename: str) -> "Spectra":
        """Load spectra saved with :meth:`save`.

        Parameters
        ----------
        filename : str
            Path to the :code:`.npz` file.

        Returns
        -------
        spectra : Spectra
            The spectra, with the class they were saved with.
        """
        with np.load(filename, allow_pickle=False) as f:
            metadata = json.loads(str(f["metadata"]))
            arrays = {
                key[len("data_") :]: f[key]
                for key in f.files
                if key.startswith("data_")
            }
            frequencies = f["frequencies"]
            weights = f["weights"] if "weights" in f.files else None

        if metadata["format_version"] > _SPECTRA_FORMAT_VERSION:
            raise ValueError(
                f"{filename} was saved with a newer version of osl-dynamics."
            )
        classes = {c.__name__: c for c in (Spectra, StateSpectra, RegressionSpectra)}
        spectra_cls = classes[metadata["class"]]
        if not issubclass(spectra_cls, cls):
            raise TypeError(
                f"{filename} contains {spectra_cls.__name__}, not {cls.__name__}."
            )
        return spectra_cls(
            frequencies=frequencies,
            arrays=arrays,
            n_channels=metadata["n_channels"],
            weights=weights,
            sampling_frequency=metadata["sampling_frequency"],
            method=metadata["method"],
            settings=metadata["settings"],
            has_cross_spectra=metadata["has_cross_spectra"],
        )

    # Display

    def _shape_summary(self) -> List[str]:
        sessions = "group average" if self.is_group else f"n_sessions={self.n_sessions}"
        return [sessions, f"n_channels={self.n_channels}"]

    def __repr__(self) -> str:
        f = self.frequencies
        parts = [f"method={self.method!r}"] + self._shape_summary()
        parts.append(f"frequencies={f[0]:g}-{f[-1]:g} Hz ({len(f)})")
        parts.append(f"cross_spectra={self.has_cross_spectra}")
        return f"{type(self).__name__}({', '.join(parts)})"

    # Compatibility with the tuples returned previously

    def _legacy_tuple(self) -> tuple:
        lg = self._legacy
        x = self._spectra()
        if not isinstance(self, StateSpectra):
            x = x[:, np.newaxis]
        calc_cpsd = lg["calc_cpsd"]
        calc_coh = lg["calc_coh"] and not calc_cpsd

        if calc_cpsd or calc_coh:
            cpsd = _full_from_triu(x, self.n_channels)
            if self.method == "welch":
                # The tuple for Welch's method has X_i conj(X_j)
                cpsd = np.conj(cpsd)
        if calc_coh:
            psd = cpsd[..., range(self.n_channels), range(self.n_channels), :].real
            coh = coherence_spectra(cpsd, keepdims=True)
        elif calc_cpsd:
            psd = cpsd
        else:
            psd = self._psd(x)
        psd = np.nan_to_num(psd)

        if not lg["keepdims"]:
            psd = np.squeeze(psd)
            if calc_coh:
                coh = np.squeeze(coh)

        outputs = [self.frequencies, psd]
        if calc_coh:
            outputs.append(coh)
        if lg["return_weights"]:
            outputs.append(self.weights)
        return tuple(outputs)

    def _legacy_outputs(self) -> tuple:
        if self._legacy is None:
            raise TypeError(
                f"{type(self).__name__} cannot be unpacked. Use its attributes, "
                "e.g. .frequencies and .psd."
            )
        warnings.warn(
            f"Unpacking the result of {self._legacy['function']} is deprecated "
            "and will be removed in a future release. Use the attributes of the "
            f"returned {type(self).__name__} instead, e.g. .frequencies and .psd.",
            DeprecationWarning,
            stacklevel=3,
        )
        return self._legacy_tuple()

    def __iter__(self):
        return iter(self._legacy_outputs())

    def __getitem__(self, index: int):
        return self._legacy_outputs()[index]


class StateSpectra(Spectra):
    """Spectra for each state, from Welch's method or a multitaper.

    Returned by :func:`welch_spectra` and :func:`multitaper_spectra` with
    :code:`alpha`. See :class:`Spectra` for the attributes shared by all
    spectra.

    The static spectrum is the average of the state spectra weighted by the
    fractional occupancies.

    Attributes
    ----------
    fractional_occupancies : np.ndarray
        Mean of :code:`alpha` for each state, normalized to sum to one.
        Shape is (n_sessions, n_states).
    """

    _session_arrays = ("spectra", "fractional_occupancies")

    _psd_values = ("psd_change", "psd", "static_psd")
    _coherence_values = ("coherence_change", "coherence", "static_coherence")

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self._spectra().shape[-3]

    @property
    def fractional_occupancies(self) -> np.ndarray:
        return self._data["fractional_occupancies"]

    def _static_spectra(self) -> np.ndarray:
        x = self._spectra()
        fo = self.fractional_occupancies[..., np.newaxis, np.newaxis]
        return np.sum(np.where(fo > 0, x * fo, 0), axis=-3).astype(x.dtype)

    @property
    def psd(self) -> np.ndarray:
        """Power spectral density of each state, :math:`P_j`.

        Shape is (n_sessions, n_states, n_channels, n_freq).
        """
        return self._psd(self._spectra())

    @property
    def cross_spectra(self) -> np.ndarray:
        """Cross spectral density of each state.

        Shape is (n_sessions, n_states, n_channels, n_channels, n_freq).
        """
        return self._cross(self._spectra())

    @property
    def coherence(self) -> np.ndarray:
        """Coherence of each state.

        Shape is (n_sessions, n_states, n_channels, n_channels, n_freq).
        """
        return self._coherence(self._spectra())

    @property
    def static_psd(self) -> np.ndarray:
        """Static power spectral density, :math:`S`.

        Shape is (n_sessions, n_channels, n_freq).
        """
        return self._psd(self._static_spectra())

    @property
    def static_cross_spectra(self) -> np.ndarray:
        """Static cross spectral density.

        Shape is (n_sessions, n_channels, n_channels, n_freq).
        """
        return self._cross(self._static_spectra())

    @property
    def static_coherence(self) -> np.ndarray:
        """Static coherence.

        Shape is (n_sessions, n_channels, n_channels, n_freq).
        """
        return self._coherence(self._static_spectra())

    @property
    def psd_change(self) -> np.ndarray:
        """Change in PSD from the static PSD, :math:`P_j - S`.

        Shape is (n_sessions, n_states, n_channels, n_freq).
        """
        return self.psd - self.static_psd[..., np.newaxis, :, :]

    @property
    def coherence_change(self) -> np.ndarray:
        """Change in coherence from the static coherence.

        Shape is (n_sessions, n_states, n_channels, n_channels, n_freq).
        """
        return self.coherence - self.static_coherence[..., np.newaxis, :, :, :]

    def _shape_summary(self) -> List[str]:
        summary = super()._shape_summary()
        return summary[:1] + [f"n_states={self.n_states}"] + summary[1:]


class RegressionSpectra(StateSpectra):
    """Spectra for each state/mode from regressing a spectrogram on alpha.

    Returned by :func:`regression_spectra`. See :class:`Spectra` and
    :class:`StateSpectra` for the attributes shared by all spectra and the
    :doc:`User Guide </user_guide/spectra>` for the definitions.

    Attributes
    ----------
    sigma : np.ndarray
        Group-level standard deviation of the windowed :code:`alpha`,
        :math:`\\sigma_j`. Shape is (n_modes,).
    """

    _session_arrays = ("coefs", "intercept", "fractional_occupancies")
    _decompose_values = "static_coherence"
    _psd_values = ("psd_coefs", "psd_change", "psd", "static_psd")
    _coherence_values = (
        "coherence_coefs",
        "coherence_change",
        "coherence",
        "static_coherence",
    )

    @property
    def n_states(self) -> int:
        """Number of states/modes."""
        return self._data["coefs"].shape[-3]

    @property
    def sigma(self) -> np.ndarray:
        return self._data["sigma"]

    def _spectra(self) -> np.ndarray:
        coefs = self._data["coefs"]
        sigma = self.sigma.reshape((-1, 1, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                self._static_spectra()[..., np.newaxis, :, :] + coefs / sigma
            ).astype(coefs.dtype)

    def _static_spectra(self) -> np.ndarray:
        return self._data["intercept"]

    @property
    def psd(self) -> np.ndarray:
        """Power spectral density of each state/mode, :math:`P_j = S +
        \\beta_j / \\sigma_j`.

        Shape is (n_sessions, n_modes, n_channels, n_freq).
        """
        return self._psd(self._spectra())

    @property
    def psd_coefs(self) -> np.ndarray:
        """Regression coefficients of the PSD, :math:`\\beta_j = \\sigma_j (P_j -
        S)`: the change in the PSD for a one standard deviation increase in
        a mode's :code:`alpha`.

        Shape is (n_sessions, n_modes, n_channels, n_freq).
        """
        return self._psd(self._data["coefs"])

    @property
    def cross_spectra_coefs(self) -> np.ndarray:
        """Regression coefficients of the cross spectra.

        Shape is (n_sessions, n_modes, n_channels, n_channels, n_freq).
        """
        return self._cross(self._data["coefs"])

    @property
    def coherence_coefs(self) -> np.ndarray:
        """Change in coherence for a one standard deviation increase in a
        mode's :code:`alpha`.

        The change in the cross spectrum in the direction (phase) of the
        static cross spectrum, normalized by the static power. The diagonal
        is the relative change in power.

        Shape is (n_sessions, n_modes, n_channels, n_channels, n_freq).
        """
        self._require_cross_spectra()
        return _coherence_change(
            self._data["coefs"], self._data["intercept"], self.n_channels
        )

    def _legacy_tuple(self) -> tuple:
        lg = self._legacy
        coefs = self._psd(self._data["coefs"])
        intercept = self._psd(self._data["intercept"])
        intercept = np.broadcast_to(intercept[:, np.newaxis], coefs.shape)
        psd = np.stack([coefs, intercept], axis=1)
        if not lg["return_coef_int"]:
            psd = np.sum(psd, axis=1)

        extra = []
        if lg["return_weights"]:
            extra.append(self.weights)
        if lg["return_sigma"]:
            extra.append(self.sigma)

        if not lg["calc_coh"]:
            if not lg["keepdims"]:
                psd = np.squeeze(psd)
            return (self.frequencies, psd, *extra)

        coh = self.coherence_coefs
        if not lg["keepdims"]:
            psd = np.squeeze(psd)
            coh = np.squeeze(coh)
        return (self.frequencies, psd, coh, *extra)


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


_LEGACY_STATE_DEFAULTS = {
    "calc_cpsd": False,
    "calc_coh": True,
    "return_weights": False,
    "keepdims": False,
}


@_deprecate_positional(3)
def welch_spectra(
    data: Union[np.ndarray, list, "Data"],
    sampling_frequency: Optional[float] = None,
    alpha: Optional[Union[np.ndarray, list]] = None,
    *,
    window_length: Optional[int] = None,
    step_size: Optional[int] = None,
    frequency_range: Optional[List] = None,
    standardize: bool = True,
    averaging: str = "mean",
    calc_cpsd: Optional[bool] = None,
    calc_coh: Optional[bool] = None,
    return_weights: Optional[bool] = None,
    keepdims: Optional[bool] = None,
    n_jobs: int = 1,
    cross_spectra: Optional[bool] = None,
) -> Union["Spectra", "StateSpectra"]:
    """Calculates spectra for inferred states using Welch's method.

    The scaling for the power spectra calculated by this function
    matches SciPy (:code:`scipy.signal.welch`).

    Parameters
    ----------
    data : np.ndarray, list or osl_dynamics.data.Data
        Time series data. Must have shape (n_sessions, n_samples,
        n_channels) or (n_samples, n_channels). If a Data object is passed,
        we use the data before it was prepared (e.g. before time-delay
        embedding), aligned to :code:`alpha`.
    sampling_frequency : float, optional
        Sampling frequency in Hz. Defaults to
        :code:`data.sampling_frequency` for a Data object.
    alpha : np.ndarray or list, optional
        Inferred state probability time course. Must have shape
        (n_sessions, n_samples, n_states) or (n_samples, n_states).
        If not passed, the static spectra are calculated.
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
        Deprecated. Only changes the tuple returned when the result is
        unpacked.
    calc_coh : bool, optional
        Deprecated. :code:`calc_coh=False` is the same as
        :code:`cross_spectra=False`.
    return_weights : bool, optional
        Deprecated. Only changes the tuple returned when the result is
        unpacked.
    keepdims : bool, optional
        Deprecated. Only changes the tuple returned when the result is
        unpacked.
    n_jobs : int, optional
        Number of parallel jobs.
    cross_spectra : bool, optional
        Should we calculate the cross spectra (needed for the coherence)?
        Defaults to :code:`True`.

    Returns
    -------
    spectra : StateSpectra or Spectra
        Spectra for each session and state. :class:`Spectra` if
        :code:`alpha` is not passed.
    """
    data, alpha, sampling_frequency = _resolve_data(data, alpha, sampling_frequency)
    if window_length is None:
        window_length = 2 * sampling_frequency
    if step_size is None:
        step_size = int(window_length) // 2

    legacy, cross_spectra = _legacy_flags(
        "welch_spectra",
        cross_spectra,
        dict(
            calc_cpsd=calc_cpsd,
            calc_coh=calc_coh,
            return_weights=return_weights,
            keepdims=keepdims,
        ),
        _LEGACY_STATE_DEFAULTS,
    )
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
        cross_spectra=cross_spectra,
        n_jobs=n_jobs,
        legacy=legacy,
    )


@_deprecate_positional(3)
def multitaper_spectra(
    data: Union[np.ndarray, list, "Data"],
    sampling_frequency: Optional[float] = None,
    alpha: Optional[Union[np.ndarray, list]] = None,
    *,
    window_length: Optional[int] = None,
    step_size: Optional[int] = None,
    time_half_bandwidth: float = 4,
    n_tapers: int = 7,
    frequency_range: Optional[List] = None,
    standardize: bool = True,
    averaging: str = "mean",
    calc_cpsd: Optional[bool] = None,
    calc_coh: Optional[bool] = None,
    return_weights: Optional[bool] = None,
    keepdims: Optional[bool] = None,
    n_jobs: int = 1,
    cross_spectra: Optional[bool] = None,
) -> Union["Spectra", "StateSpectra"]:
    """Calculates spectra for inferred states using a multitaper.

    Parameters
    ----------
    data : np.ndarray, list or osl_dynamics.data.Data
        Time series data. Must have shape (n_sessions, n_samples,
        n_channels) or (n_samples, n_channels). If a Data object is passed,
        we use the data before it was prepared (e.g. before time-delay
        embedding), aligned to :code:`alpha`.
    sampling_frequency : float, optional
        Sampling frequency in Hz. Defaults to
        :code:`data.sampling_frequency` for a Data object.
    alpha : np.ndarray or list, optional
        Inferred state probability time course. Must have shape
        (n_sessions, n_samples, n_states) or (n_samples, n_states).
        If not passed, the static spectra are calculated.
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
        Deprecated. Only changes the tuple returned when the result is
        unpacked.
    calc_coh : bool, optional
        Deprecated. :code:`calc_coh=False` is the same as
        :code:`cross_spectra=False`.
    return_weights : bool, optional
        Deprecated. Only changes the tuple returned when the result is
        unpacked.
    keepdims : bool, optional
        Deprecated. Only changes the tuple returned when the result is
        unpacked.
    n_jobs : int, optional
        Number of parallel jobs.
    cross_spectra : bool, optional
        Should we calculate the cross spectra (needed for the coherence)?
        Defaults to :code:`True`.

    Returns
    -------
    spectra : StateSpectra or Spectra
        Spectra for each session and state. :class:`Spectra` if
        :code:`alpha` is not passed.
    """
    data, alpha, sampling_frequency = _resolve_data(data, alpha, sampling_frequency)
    if window_length is None:
        window_length = 2 * sampling_frequency
    if step_size is None:
        step_size = int(window_length)

    legacy, cross_spectra = _legacy_flags(
        "multitaper_spectra",
        cross_spectra,
        dict(
            calc_cpsd=calc_cpsd,
            calc_coh=calc_coh,
            return_weights=return_weights,
            keepdims=keepdims,
        ),
        _LEGACY_STATE_DEFAULTS,
    )
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
        cross_spectra=cross_spectra,
        n_jobs=n_jobs,
        legacy=legacy,
        time_half_bandwidth=time_half_bandwidth,
        n_tapers=n_tapers,
    )


@_deprecate_positional(3)
def regression_spectra(
    data: Union[np.ndarray, list, "Data"],
    alpha: Union[np.ndarray, list],
    sampling_frequency: Optional[float] = None,
    *,
    window_length: Optional[int] = None,
    step_size: Optional[int] = None,
    n_sub_windows: int = 8,
    frequency_range: Optional[List] = None,
    standardize: bool = True,
    calc_coh: Optional[bool] = None,
    return_weights: Optional[bool] = None,
    return_coef_int: Optional[bool] = None,
    return_sigma: Optional[bool] = None,
    keepdims: Optional[bool] = None,
    n_jobs: int = 1,
    cross_spectra: Optional[bool] = None,
) -> "RegressionSpectra":
    r"""Calculates mode-specific spectra by regressing a spectrogram with alpha.

    We calculate a spectrogram :math:`P_t` with Welch's method (Hann-tapered
    sub-windows) and average :code:`alpha` over the same windows. Only windows
    that do not overlap the start or end of the data are used.

    If :code:`alpha` sums to one at each time point, the spectrogram is
    modeled as a mixture of mode spectra, :math:`P_t = \sum_j \alpha_{jt}
    P_j`, where :math:`P_j` is the spectrum when only mode :math:`j` is
    active. The result contains:

    - The **static spectrum** (intercept): the time-averaged spectrum,
      :math:`S = \sum_j \bar{\alpha}_j P_j`.
    - The **coefficients**: :math:`\beta_j = \sigma_j (P_j - S)`, the change
      in the spectrum from the static spectrum for a one standard deviation
      increase in mode :math:`j`'s :code:`alpha`. :math:`\sigma_j` is the
      standard deviation of the windowed :code:`alpha`, averaged over
      sessions.
    - The **change in coherence**: the change in the cross spectrum normalized
      by the static power,

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

    See the :doc:`User Guide </user_guide/spectra>` for how to interpret and
    plot the outputs.

    Parameters
    ----------
    data : np.ndarray, list or osl_dynamics.data.Data
        Data to calculate a spectrogram for. Shape must be
        (n_sessions, n_samples, n_channels) or (n_samples, n_channels). If a
        Data object is passed, we use the data before it was prepared (e.g.
        before time-delay embedding), aligned to :code:`alpha`.
    alpha : np.ndarray or list
        Inferred mode mixing factors. Shape must be
        (n_sessions, n_samples, n_modes) or (n_samples, n_modes).
    sampling_frequency : float, optional
        Sampling frequency in Hz. Defaults to
        :code:`data.sampling_frequency` for a Data object.
    window_length : int, optional
        Number samples to use in the window to calculate a spectrum.
        Defaults to 4 seconds (rounded to a multiple of
        :code:`n_sub_windows`).
    step_size : int, optional
        Step size for shifting the window. Defaults to 80 ms.
    n_sub_windows : int, optional
        We split the window into a number of sub-windows and average the
        spectra for each sub-window. window_length must be divisible by
        :code:`n_sub_windows`.
    frequency_range : list, optional
        Minimum and maximum frequency to keep.
    standardize : bool, optional
        Should we standardize the data before calculating the spectrogram?
    calc_coh : bool, optional
        Deprecated. :code:`calc_coh=False` is the same as
        :code:`cross_spectra=False`.
    return_weights : bool, optional
        Deprecated. Only changes the tuple returned when the result is
        unpacked.
    return_coef_int : bool, optional
        Deprecated. Only changes the tuple returned when the result is
        unpacked.
    return_sigma : bool, optional
        Deprecated. Only changes the tuple returned when the result is
        unpacked.
    keepdims : bool, optional
        Deprecated. Only changes the tuple returned when the result is
        unpacked.
    n_jobs : int, optional
        Number of parallel jobs.
    cross_spectra : bool, optional
        Should we calculate the cross spectra (needed for the coherence)?
        Defaults to :code:`True`.

    Returns
    -------
    spectra : RegressionSpectra
        Spectra for each session and mode.

    References
    ----------
    Quinn, A. J. et al. (2024). The GLM-spectrum: A multilevel framework for
    spectrum analysis with covariate and confound modelling. Imaging
    Neuroscience, 2. https://doi.org/10.1162/imag_a_00082
    """
    if alpha is None:
        raise ValueError("alpha must be passed.")
    data, alpha, sampling_frequency = _resolve_data(data, alpha, sampling_frequency)
    data, alpha = _validate_data_alpha(data, alpha)

    legacy, cross_spectra = _legacy_flags(
        "regression_spectra",
        cross_spectra,
        dict(
            calc_coh=calc_coh,
            return_weights=return_weights,
            return_coef_int=return_coef_int,
            return_sigma=return_sigma,
            keepdims=keepdims,
        ),
        {
            "calc_coh": True,
            "return_weights": False,
            "return_coef_int": False,
            "return_sigma": False,
            "keepdims": False,
        },
    )

    if window_length is None:
        window_length = n_sub_windows * round(4 * sampling_frequency / n_sub_windows)
    if step_size is None:
        step_size = max(1, round(0.08 * sampling_frequency))

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
        calc_cpsd=cross_spectra,
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

    # Weights for calculating a group-average spectrum
    w = _session_weights(data)

    # Group-level standard deviation of the windowed alpha for each mode,
    # averaged over the sessions where the mode's alpha varies
    sigma = np.array([result[2] for result in results])
    fractional_occupancies = np.array([result[3] for result in results])
    used = sigma > 0
    total_weight = np.sum(w[:, np.newaxis] * used, axis=0)
    group_sigma = np.divide(
        np.sum(w[:, np.newaxis] * sigma, axis=0),
        total_weight,
        out=np.full(sigma.shape[1], np.nan),
        where=total_weight > 0,
    )

    # Rescale the coefficients to the change in the spectrum for a one
    # group-level standard deviation increase in a mode's alpha
    coefs = []
    intercept = []
    for result, s, u in zip(results, sigma, used):
        c, i, _, _ = result
        # Modes that were not fitted in this session have no estimate (NaN)
        scale = np.divide(
            group_sigma, s, out=np.full_like(group_sigma, np.nan), where=u
        )
        scale = scale.reshape((-1,) + (1,) * (c.ndim - 1))
        coefs.append((c * scale).astype(c.dtype))
        intercept.append(i)

    settings = {
        "window_length": spectrogram.window_length,
        "step_size": spectrogram.step_size,
        "n_sub_windows": n_sub_windows,
        "frequency_range": frequency_range,
        "standardize": standardize,
        "mixture": mixture,
    }
    return RegressionSpectra(
        frequencies=spectrogram.frequencies,
        arrays={
            "coefs": np.array(coefs),
            "intercept": np.array(intercept),
            "sigma": group_sigma,
            "fractional_occupancies": fractional_occupancies,
        },
        n_channels=data[0].shape[1],
        weights=w,
        sampling_frequency=sampling_frequency,
        method="regression",
        settings=settings,
        has_cross_spectra=cross_spectra,
        legacy=legacy,
    )


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

    # Remove spectra with NaN values (e.g. states that are never active)
    coherences = coherences[~np.any(np.isnan(coherences), axis=1)]

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


def _resolve_data(
    data,
    alpha: Optional[Union[np.ndarray, list]],
    sampling_frequency: Optional[float],
) -> Tuple[Union[np.ndarray, list], Optional[Union[np.ndarray, list]], float]:
    """Get the time series and sampling frequency from a Data object.

    If :code:`data` is an :class:`osl_dynamics.data.Data` object, we use the
    time series before it was prepared. Each session starts after the samples
    lost to time-delay embedding or a sliding window (as in
    :code:`Data.trim_time_series`) and has the same number of samples as
    :code:`alpha`. The Data object must be prepared as it was for training,
    unless it has the same number of samples as :code:`alpha`.

    Parameters
    ----------
    data : np.ndarray, list or osl_dynamics.data.Data
        Time series data.
    alpha : np.ndarray or list
        State/mode time courses.
    sampling_frequency : float
        Sampling frequency in Hz. If :code:`None`, we use
        :code:`data.sampling_frequency`.

    Returns
    -------
    data : np.ndarray or list
        Time series data.
    alpha : np.ndarray or list
        State/mode time courses.
    sampling_frequency : float
        Sampling frequency in Hz.
    """
    is_data_object = hasattr(data, "raw_data_arrays")

    if sampling_frequency is None:
        sampling_frequency = getattr(data, "sampling_frequency", None)
        if sampling_frequency is None:
            raise ValueError(
                "sampling_frequency must be passed or specified in the Data object."
            )

    if not is_data_object:
        return data, alpha, sampling_frequency

    arrays = data.raw_data_arrays
    if alpha is None:
        return [np.asarray(x) for x in arrays], None, sampling_frequency

    if isinstance(alpha, np.ndarray):
        alpha = [alpha] if alpha.ndim == 2 else list(alpha)
    if len(alpha) != len(arrays):
        raise ValueError(
            f"The Data object has {len(arrays)} sessions, but alpha has "
            f"{len(alpha)}. Pass alpha for each session (concatenate=False)."
        )

    if data.arrays is arrays and any(len(x) != len(a) for x, a in zip(arrays, alpha)):
        raise ValueError(
            "The Data object has not been prepared and has a different number "
            "of samples to alpha, so it cannot be aligned to alpha. Prepare the "
            "Data object as it was for training, or pass "
            "data.trim_time_series(..., prepared=False)."
        )

    offset = getattr(data, "n_embeddings", 1) // 2 + getattr(data, "n_window", 1) // 2
    if any(abs(len(x) - len(y) - 2 * offset) > 2 for x, y in zip(arrays, data.arrays)):
        raise ValueError(
            "The Data object was prepared with a method that changes the number "
            "of samples other than time-delay embedding or a sliding window "
            "(e.g. downsampling), so it cannot be aligned to alpha. Pass the "
            "time series aligned to alpha instead."
        )

    time_series = []
    for i, (x, a) in enumerate(zip(arrays, alpha)):
        if offset + len(a) > len(x):
            raise ValueError(
                f"alpha for session {i} has {len(a)} samples, but the data only "
                f"has {len(x) - offset} after the samples lost to time-delay "
                "embedding/sliding windows."
            )
        time_series.append(np.asarray(x[offset : offset + len(a)]))
    return time_series, list(alpha), sampling_frequency


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
        elif method == "multitaper":
            tapers = signal.windows.dpss(
                sub_length, NW=time_half_bandwidth, Kmax=n_tapers
            )
            tapers = np.atleast_2d(tapers)
            nfft = sub_length
            scale = 2 / sampling_frequency
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
        )

    @property
    def frequencies(self) -> np.ndarray:
        """Frequency axis. Shape is (n_freq,)."""
        return self._fft_frequencies()[self._frequency_slice()]

    def _fft_frequencies(self) -> np.ndarray:
        return np.arange(self.nfft // 2 + 1) * self.sampling_frequency / self.nfft

    def _frequency_slice(self) -> slice:
        min_arg, max_arg = get_frequency_args_range(
            self._fft_frequencies(), self.frequency_range
        )
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
                # G[w, f, c, d] = sum_s conj(X_c) X_d over sub-windows and tapers
                X = np.moveaxis(X, -1, 1)
                G = np.conj(X) @ np.swapaxes(X, -1, -2)
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
                # total[f, c, d] += sum_{w,s} conj(X_c) X_d
                Y = np.ascontiguousarray(np.transpose(X, (3, 0, 2, 1)))
                Y = Y.reshape(X.shape[-1], -1, n_channels)
                total = total + np.conj(np.swapaxes(Y, -1, -2)) @ Y
            else:
                total = total + np.sum(X.real**2 + X.imag**2, axis=(0, 2))

        if self.calc_cpsd:
            P = np.moveaxis(total[:, m, n], 0, -1)
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
    window_length: int,
    step_size: int,
    frequency_range: Optional[List],
    standardize: bool,
    averaging: str,
    cross_spectra: bool,
    n_jobs: int,
    legacy: Optional[dict] = None,
    time_half_bandwidth: float = 4,
    n_tapers: int = 7,
) -> Union["Spectra", "StateSpectra"]:
    """Calculate spectra for each state with Welch's method or a multitaper.

    See :func:`welch_spectra` and :func:`multitaper_spectra`.
    """
    data, alpha = _validate_data_alpha(data, alpha)
    static = alpha[0] is None

    if frequency_range is None:
        frequency_range = [0, sampling_frequency / 2]

    spectrogram = _Spectrogram.create(
        method=method,
        sampling_frequency=sampling_frequency,
        window_length=window_length,
        step_size=step_size,
        frequency_range=frequency_range,
        calc_cpsd=cross_spectra,
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
    spectra = np.array([result[0] for result in results])
    fractional_occupancies = np.array([result[1] for result in results])

    inactive = np.argwhere(fractional_occupancies == 0)
    if len(inactive) > 0:
        _logger.warning(
            "Some states are never active, their spectra are NaN "
            f"(session, state): {inactive.tolist()}."
        )

    settings = {
        "window_length": spectrogram.window_length,
        "step_size": spectrogram.step_size,
        "frequency_range": frequency_range,
        "standardize": standardize,
        "averaging": averaging,
    }
    if method == "multitaper":
        settings["time_half_bandwidth"] = time_half_bandwidth
        settings["n_tapers"] = n_tapers

    common = dict(
        frequencies=spectrogram.frequencies,
        n_channels=data[0].shape[1],
        weights=_session_weights(data),
        sampling_frequency=sampling_frequency,
        method=method,
        settings=settings,
        has_cross_spectra=cross_spectra,
        legacy=legacy,
    )
    if static:
        return Spectra(arrays={"spectra": spectra[:, 0]}, **common)
    return StateSpectra(
        arrays={
            "spectra": spectra,
            "fractional_occupancies": fractional_occupancies,
        },
        **common,
    )


def _state_spectra_session(
    data: np.ndarray,
    alpha: Optional[np.ndarray],
    spectrogram: _Spectrogram,
    standardize: bool,
    averaging: str,
) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    spectra : np.ndarray
        Cross spectra (upper triangle) or PSDs for each state. NaN for states
        that are never active. Shape is (n_states, n_channels * (n_channels +
        1) / 2, n_freq) if :code:`spectrogram.calc_cpsd=True`, otherwise
        (n_states, n_channels, n_freq).
    fractional_occupancies : np.ndarray
        Mean of alpha for each state, normalized to sum to one.
        Shape is (n_states,).
    """
    if averaging not in ["mean", "median"]:
        raise ValueError(f"averaging must be 'mean' or 'median', got '{averaging}'.")

    n_samples = data.shape[0]
    if alpha is None:
        alpha = np.ones([n_samples, 1], dtype=np.float32)

    # Standardize before calculating spectra
    if standardize:
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Rescale the spectra to account for how much time each state was active
    sum_alpha2 = np.sum(alpha**2, axis=0, dtype=np.float64)
    scaling = np.divide(
        n_samples,
        sum_alpha2,
        out=np.full_like(sum_alpha2, np.nan),
        where=sum_alpha2 > 0,
    )
    dtype = np.complex64 if spectrogram.calc_cpsd else np.float32

    spectra = []
    for i in range(alpha.shape[1]):
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
        spectra.append((p * scaling[i]).astype(dtype))

    return np.array(spectra), _fractional_occupancies(alpha)


def _fractional_occupancies(alpha: np.ndarray) -> np.ndarray:
    """Mean of alpha for each state, normalized to sum to one.

    Parameters
    ----------
    alpha : np.ndarray
        State/mode time courses. Shape must be (n_samples, n_states).

    Returns
    -------
    fractional_occupancies : np.ndarray
        Shape is (n_states,).
    """
    mean = np.mean(alpha, axis=0, dtype=np.float64)
    return mean / np.sum(mean)


def _regression_spectra_session(
    data: np.ndarray,
    alpha: np.ndarray,
    spectrogram: _Spectrogram,
    mixture: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    fractional_occupancies : np.ndarray
        Mean of alpha for each mode, normalized to sum to one.
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
    coefs, intercept, sigma = _regress_spectrogram(
        a, spectrogram.chunks(data, keep), mixture
    )
    return coefs, intercept, sigma, _fractional_occupancies(alpha)


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
    used = sigma > max(1e-4 * np.max(sigma), 1e-8)
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
