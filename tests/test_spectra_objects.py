"""Tests for the spectra result objects."""

import warnings

import numpy as np
import pytest

from osl_dynamics.analysis import spectral, static
from spectral_cases import SAMPLING_FREQUENCY as FS
from spectral_cases import N_CHANNELS, N_STATES, simulate


@pytest.fixture(scope="module")
def sim():
    return simulate()


@pytest.fixture(scope="module")
def multitaper(sim):
    data, _, alpha_hmm, _ = sim
    return spectral.multitaper_spectra(
        data, FS, alpha=alpha_hmm, frequency_range=[1, 40]
    )


@pytest.fixture(scope="module")
def regression(sim):
    data, alpha_mix, _, _ = sim
    return spectral.regression_spectra(
        data,
        alpha_mix,
        FS,
        window_length=200,
        step_size=10,
        n_sub_windows=4,
        frequency_range=[1, 40],
    )


def test_state_spectra_shapes(multitaper):
    s = multitaper
    n_freq = len(s.frequencies)
    assert isinstance(s, spectral.StateSpectra)
    assert (s.n_sessions, s.n_states, s.n_channels) == (2, N_STATES, N_CHANNELS)
    assert s.psd.shape == (2, N_STATES, N_CHANNELS, n_freq)
    assert s.cross_spectra.shape == (2, N_STATES, N_CHANNELS, N_CHANNELS, n_freq)
    assert s.coherence.shape == (2, N_STATES, N_CHANNELS, N_CHANNELS, n_freq)
    assert s.static_psd.shape == (2, N_CHANNELS, n_freq)
    assert s.static_coherence.shape == (2, N_CHANNELS, N_CHANNELS, n_freq)
    np.testing.assert_allclose(s.weights.sum(), 1)
    np.testing.assert_allclose(s.fractional_occupancies.sum(axis=1), 1)

    # Hermitian cross spectra, unit coherence on the diagonal
    np.testing.assert_allclose(
        s.cross_spectra, np.conj(np.swapaxes(s.cross_spectra, 2, 3))
    )
    diag = s.coherence[..., range(N_CHANNELS), range(N_CHANNELS), :]
    np.testing.assert_allclose(diag, 1, rtol=1e-5)


def test_static_psd_is_occupancy_weighted(multitaper):
    s = multitaper
    fo = s.fractional_occupancies[..., np.newaxis, np.newaxis]
    np.testing.assert_allclose(s.static_psd, np.sum(fo * s.psd, axis=1), rtol=1e-5)


def test_static_spectra(sim):
    data, _, _, _ = sim
    s = static.welch_spectra(data, FS)
    assert type(s) is spectral.Spectra
    assert s.psd.shape == (2, N_CHANNELS, len(s.frequencies))
    assert s.coherence.shape == (2, N_CHANNELS, N_CHANNELS, len(s.frequencies))
    assert spectral.welch_spectra(data, FS).psd.shape == s.psd.shape


def test_regression_quantities(regression):
    s = regression
    assert isinstance(s, spectral.RegressionSpectra)
    sigma = s.sigma[:, np.newaxis, np.newaxis]
    np.testing.assert_allclose(
        s.psd, s.static_psd[:, np.newaxis] + s.psd_coefs / sigma, rtol=1e-4
    )
    assert s.coherence_coefs.shape == (
        2,
        N_STATES,
        N_CHANNELS,
        N_CHANNELS,
        len(s.frequencies),
    )
    assert not hasattr(s, "fractional_occupancies")

    # Each mode's coefficients peak at the frequency of its oscillation
    peaks = s.frequencies[s.psd_coefs.mean(axis=(0, 2)).argmax(axis=-1)]
    np.testing.assert_allclose(peaks, [6, 11, 22], atol=1.5)


@pytest.mark.parametrize("method", ["welch", "multitaper", "regression"])
def test_cross_spectra_convention(method):
    # y is x delayed by d samples, so conj(X) Y has phase -2 pi f d / fs
    rng = np.random.default_rng(0)
    d = 3
    x = rng.standard_normal(20000 + d)
    data = np.stack([x[d:], x[:-d]], axis=1).astype(np.float32)
    if method == "welch":
        s = static.welch_spectra(data, FS)
        cpsd = s.cross_spectra[0]
    elif method == "multitaper":
        s = static.multitaper_spectra(data, FS)
        cpsd = s.cross_spectra[0]
    else:
        alpha = np.ones((len(data), 1), dtype=np.float32)
        s = spectral.regression_spectra(data, alpha, FS)
        cpsd = s.static_cross_spectra[0]
    f = s.frequencies
    keep = (f > 1) & (f < 10)
    expected = np.exp(-2j * np.pi * f[keep] * d / FS)
    phase = cpsd[0, 1, keep] / np.abs(cpsd[0, 1, keep])
    np.testing.assert_allclose(phase, expected, atol=0.05)


def test_group_average(multitaper):
    s = multitaper
    g = s.group_average()
    assert g.is_group and g.n_sessions is None and type(g) is type(s)
    w = s.weights[:, np.newaxis, np.newaxis, np.newaxis]
    np.testing.assert_allclose(g.psd, np.sum(w * s.psd, axis=0), rtol=1e-5)

    # Coherence from the group-averaged cross spectra
    cpsd = np.sum(w[..., np.newaxis] * s.cross_spectra, axis=0)
    pxx = np.real(cpsd[:, range(N_CHANNELS), range(N_CHANNELS)])
    coh = np.abs(cpsd) / np.sqrt(pxx[:, :, np.newaxis] * pxx[:, np.newaxis, :])
    np.testing.assert_allclose(g.coherence, coh, rtol=1e-4)

    with pytest.raises(ValueError):
        g.group_average()
    with pytest.raises(TypeError):
        f, psd = g


def test_group_average_ignores_inactive_states(sim):
    data, _, _, alpha_missing = sim
    s = spectral.multitaper_spectra(data, FS, alpha=alpha_missing)
    assert np.all(np.isnan(s.psd[1, -1]))
    assert not np.any(np.isnan(s.static_psd))
    g = s.group_average()
    np.testing.assert_allclose(g.psd[-1], s.psd[0, -1], rtol=1e-5)

    r = spectral.regression_spectra(
        data, alpha_missing, FS, window_length=200, step_size=10, n_sub_windows=4
    )
    assert np.all(np.isnan(r.psd_coefs[1, -1]))
    np.testing.assert_allclose(
        r.group_average().psd_coefs[-1], r.psd_coefs[0, -1], rtol=1e-5
    )


@pytest.mark.parametrize("which", ["static", "multitaper", "regression", "group"])
def test_save_load(which, sim, multitaper, regression, tmp_path):
    data, _, _, _ = sim
    s = {
        "static": lambda: static.welch_spectra(data, FS, cross_spectra=False),
        "multitaper": lambda: multitaper,
        "regression": lambda: regression,
        "group": lambda: regression.group_average(),
    }[which]()
    s.save(tmp_path / "spectra.npz")
    loaded = spectral.Spectra.load(tmp_path / "spectra.npz")
    assert type(loaded) is type(s)
    assert repr(loaded) == repr(s)
    assert loaded.settings == s.settings
    np.testing.assert_array_equal(loaded.psd, s.psd)
    if s.weights is not None:
        np.testing.assert_array_equal(loaded.weights, s.weights)


def test_load_wrong_class(multitaper, tmp_path):
    multitaper.save(tmp_path / "spectra.npz")
    with pytest.raises(TypeError):
        spectral.RegressionSpectra.load(tmp_path / "spectra.npz")


def test_psd_only(sim):
    data, _, alpha_hmm, _ = sim
    s = spectral.welch_spectra(data, FS, alpha=alpha_hmm, cross_spectra=False)
    assert not s.has_cross_spectra
    assert s.psd.shape[:3] == (2, N_STATES, N_CHANNELS)
    with pytest.raises(ValueError):
        s.coherence
    with pytest.warns(DeprecationWarning):
        assert len(tuple(s)) == 2  # f, psd


def test_deprecations(sim):
    data, alpha_mix, alpha_hmm, _ = sim
    with pytest.warns(DeprecationWarning, match="positionally"):
        spectral.multitaper_spectra(data, FS, alpha_hmm, 200)
    with pytest.warns(DeprecationWarning, match="return_weights is deprecated"):
        s = spectral.multitaper_spectra(data, FS, alpha=alpha_hmm, return_weights=True)
    with pytest.warns(DeprecationWarning, match="Unpacking"):
        f, psd, coh, w = s
    with pytest.raises(ValueError):
        spectral.welch_spectra(
            data, FS, alpha=alpha_hmm, calc_coh=True, cross_spectra=False
        )

    # No warnings when the new API is used
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        s = spectral.regression_spectra(data, alpha_mix, FS, window_length=200)
        s.psd_coefs, s.coherence_coefs, s.static_psd


def test_regression_defaults(sim):
    data, alpha_mix, _, _ = sim
    s = spectral.regression_spectra(data, alpha_mix, FS)
    assert s.settings["window_length"] == 4 * FS
    assert s.settings["n_sub_windows"] == 8
    assert s.settings["step_size"] == 8


def test_repr(multitaper):
    assert repr(multitaper).startswith("StateSpectra(method='multitaper', n_sessions=2")
    assert "group average" in repr(multitaper.group_average())


def test_power_maps_match_variance_from_spectra(multitaper, regression):
    from osl_dynamics.analysis import power

    g = multitaper.group_average()
    f = g.frequencies
    for kwargs in [{}, {"frequency_range": [8, 14]}]:
        for method in ["mean", "sum", "integral"]:
            np.testing.assert_allclose(
                g.power_maps(values="psd", method=method, **kwargs),
                power.variance_from_spectra(f, g.psd, method=method, **kwargs),
                rtol=1e-5,
            )
    components = multitaper.decompose(2)
    np.testing.assert_allclose(
        g.power_maps(values="psd", bands=components),
        power.variance_from_spectra(f, g.psd, components=components),
        rtol=1e-5,
    )

    # Default values and shapes
    np.testing.assert_allclose(
        g.power_maps(),
        g.power_maps(values="psd") - g.power_maps(values="static_psd"),
        rtol=1e-4,
        atol=1e-6,
    )
    bands = [[1, 4], [4, 8], [8, 13]]
    p = multitaper.power_maps(bands=bands)
    assert p.shape == (2, 3, N_STATES, N_CHANNELS)
    np.testing.assert_allclose(p[:, 1], multitaper.power_maps(frequency_range=[4, 8]))
    np.testing.assert_allclose(
        regression.power_maps(),
        power.variance_from_spectra(regression.frequencies, regression.psd_coefs),
        rtol=1e-5,
    )


def test_coherence_networks(multitaper, regression):
    from osl_dynamics.analysis import connectivity

    g = multitaper.group_average()
    f = g.frequencies
    c = g.coherence_networks(values="coherence", frequency_range=[8, 14])
    np.testing.assert_allclose(
        c,
        connectivity.mean_coherence_from_spectra(
            f, g.coherence, frequency_range=[8, 14]
        ),
        rtol=1e-5,
    )

    # Thresholding keeps the same edges as connectivity.threshold
    c = g.coherence_networks()
    t = g.coherence_networks(percentile=90)
    expected = connectivity.threshold(c.copy(), percentile=90, absolute_value=True)
    np.testing.assert_allclose(t, expected)

    v = multitaper.coherence_networks(bands=[[1, 8], [8, 30]], vectorize=True)
    n_edges = N_CHANNELS * (N_CHANNELS - 1) // 2
    assert v.shape == (2, 2, N_STATES, n_edges)
    assert regression.coherence_networks().shape == (
        2,
        N_STATES,
        N_CHANNELS,
        N_CHANNELS,
    )


def test_static_spectra_methods(sim):
    data, _, _, _ = sim
    s = static.welch_spectra(data, FS)
    assert s.power_maps().shape == (2, N_CHANNELS)
    assert s.power_maps(bands=[[1, 4], [4, 8]]).shape == (2, 2, N_CHANNELS)
    assert s.group_average().coherence_networks().shape == (N_CHANNELS, N_CHANNELS)
    with pytest.raises(ValueError, match="values must be one of"):
        s.power_maps(values="psd_change")


def test_plot_psd(multitaper, regression, tmp_path):
    fig, ax = multitaper.plot_psd(values="psd", static=True)
    assert len(ax.get_lines()) == N_STATES + 1
    regression.plot_psd(filename=tmp_path / "psd.png", frequency_range=[8, 12])
    assert (tmp_path / "psd.png").exists()


def test_variance_from_spectra_cross_spectra_argument():
    from osl_dynamics.analysis import power

    # PSDs with n_modes == n_channels are misread as cross spectra without
    # the cross_spectra argument
    rng = np.random.default_rng(0)
    psd = rng.uniform(size=(2, N_CHANNELS, N_CHANNELS, 10))
    p = power.variance_from_spectra(None, psd, cross_spectra=False)
    np.testing.assert_allclose(p, psd.mean(axis=-1))


def test_data_object_input(sim):
    from osl_dynamics.data import Data

    data, alpha_mix, _, _ = sim
    d = Data([x.copy() for x in data], sampling_frequency=FS)
    d.tde_pca(n_embeddings=7, n_pca_components=4)

    # alpha for the trimmed data, as returned by a model
    trimmed = d.trim_time_series(sequence_length=200, prepared=False)
    alpha = [a[3 : 3 + len(t)] for a, t in zip(alpha_mix, trimmed)]

    for fn, args in [
        (spectral.multitaper_spectra, {}),
        (spectral.welch_spectra, {}),
        (spectral.regression_spectra, {"window_length": 200}),
    ]:
        expected = fn(trimmed, alpha=alpha, sampling_frequency=FS, **args)
        actual = fn(d, alpha=alpha, **args)
        np.testing.assert_array_equal(actual.psd, expected.psd)

    # Static spectra use all the raw data
    s = static.welch_spectra(d)
    np.testing.assert_array_equal(s.psd, static.welch_spectra(data, FS).psd)

    with pytest.raises(ValueError, match="sessions"):
        spectral.multitaper_spectra(d, alpha=np.concatenate(alpha))
    with pytest.raises(ValueError, match="samples"):
        spectral.multitaper_spectra(d, alpha=[np.ones((5000, 3))] * 2)
