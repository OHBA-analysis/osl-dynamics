"""Simulated data and test cases for the spectral analysis functions.

Each case calls a public function and returns a tuple of arrays. The reference
outputs are saved by :code:`generate_spectral_references.py` and compared
against by :code:`test_spectral.py`.
"""

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter1d

from osl_dynamics.analysis import connectivity, power, spectral, static

SAMPLING_FREQUENCY = 100
N_SAMPLES = (4000, 3500)
N_CHANNELS = 4
N_STATES = 3
FREQUENCIES = (6, 11, 22)


@lru_cache(maxsize=None)
def simulate():
    """Simulate data with a different oscillation for each state.

    Returns
    -------
    data : list of np.ndarray
        Time series for each session. Shape is (n_samples, n_channels).
    alpha_mix : list of np.ndarray
        Mixing coefficients (sum to one). Shape is (n_samples, n_states).
    alpha_hmm : list of np.ndarray
        One-hot state time courses. Shape is (n_samples, n_states).
    alpha_missing : list of np.ndarray
        As :code:`alpha_hmm`, but the last state is never visited in the
        second session.
    """
    rng = np.random.default_rng(42)
    spatial_maps = rng.uniform(0.2, 1.0, size=(N_STATES, N_CHANNELS))

    data, alpha_mix, alpha_hmm, alpha_missing = [], [], [], []
    for i, n in enumerate(N_SAMPLES):
        logits = gaussian_filter1d(rng.standard_normal((n, N_STATES)), 25, axis=0)
        logits *= 20
        a = np.exp(logits - logits.max(axis=1, keepdims=True))
        a /= a.sum(axis=1, keepdims=True)

        stc = np.eye(N_STATES)[a.argmax(axis=1)]
        missing = stc.copy()
        if i == 1:
            missing[:, 0] += missing[:, -1]
            missing[:, -1] = 0

        t = np.arange(n) / SAMPLING_FREQUENCY
        x = 0.5 * rng.standard_normal((n, N_CHANNELS))
        for j, f in enumerate(FREQUENCIES):
            phase = np.cumsum(0.05 * rng.standard_normal(n))
            s = np.sin(2 * np.pi * f * t + phase)
            x += (a[:, j] * s)[:, np.newaxis] * spatial_maps[j]

        data.append(x.astype(np.float32))
        alpha_mix.append(a.astype(np.float32))
        alpha_hmm.append(stc.astype(np.float32))
        alpha_missing.append(missing.astype(np.float32))

    return data, alpha_mix, alpha_hmm, alpha_missing


def _equal_length(x):
    n = min(N_SAMPLES)
    return np.array([xi[:n] for xi in x])


def _as_tuple(outputs):
    if isinstance(outputs, np.ndarray):
        return (outputs,)
    return tuple(np.asarray(o) for o in outputs)


def _spectra_cases(fn, extra_kwargs):
    data, alpha_mix, alpha_hmm, alpha_missing = simulate()
    fs = SAMPLING_FREQUENCY
    kw = extra_kwargs
    return {
        "no_alpha_2d": lambda: fn(data[0], fs, calc_coh=False, **kw),
        "no_alpha_list_coh": lambda: fn(data, fs, return_weights=True, **kw),
        "alpha_list": lambda: fn(
            data,
            fs,
            alpha=alpha_hmm,
            frequency_range=[1, 40],
            return_weights=True,
            **kw,
        ),
        "alpha_2d": lambda: fn(data[0], fs, alpha=alpha_mix[0], **kw),
        "alpha_3d_array": lambda: fn(
            _equal_length(data), fs, alpha=_equal_length(alpha_hmm), **kw
        ),
        "alpha_cpsd_keepdims": lambda: fn(
            data,
            fs,
            alpha=alpha_hmm,
            window_length=128,
            step_size=32,
            calc_cpsd=True,
            keepdims=True,
            **kw,
        ),
        "alpha_median_no_standardize": lambda: fn(
            data,
            fs,
            alpha=alpha_hmm,
            averaging="median",
            standardize=False,
            calc_coh=False,
            **kw,
        ),
        "alpha_missing_state": lambda: fn(data, fs, alpha=alpha_missing, **kw),
    }


def _welch_cases():
    cases = _spectra_cases(spectral.welch_spectra, {})
    return {f"welch/{k}": v for k, v in cases.items()}


def _multitaper_cases():
    cases = _spectra_cases(spectral.multitaper_spectra, {})
    data, _, alpha_hmm, _ = simulate()
    cases["tapers"] = lambda: spectral.multitaper_spectra(
        data,
        SAMPLING_FREQUENCY,
        alpha=alpha_hmm,
        time_half_bandwidth=2,
        n_tapers=3,
        window_length=150,
        step_size=50,
    )
    return {f"multitaper/{k}": v for k, v in cases.items()}


def _regression_cases():
    data, alpha_mix, alpha_hmm, alpha_missing = simulate()
    fs = SAMPLING_FREQUENCY
    common = {"window_length": 200, "step_size": 10, "n_sub_windows": 4}
    return {
        "regression/mixture_all_outputs": lambda: spectral.regression_spectra(
            data,
            alpha_mix,
            fs,
            frequency_range=[1, 40],
            return_coef_int=True,
            return_weights=True,
            return_sigma=True,
            **common,
        ),
        "regression/mixture_defaults": lambda: spectral.regression_spectra(
            data, alpha_mix, fs, window_length=200, step_size=20
        ),
        "regression/not_mixture": lambda: spectral.regression_spectra(
            data,
            [a[:, :2] for a in alpha_mix],
            fs,
            return_coef_int=True,
            return_sigma=True,
            **common,
        ),
        "regression/no_coh_coef_int": lambda: spectral.regression_spectra(
            data, alpha_mix, fs, calc_coh=False, return_coef_int=True, **common
        ),
        "regression/no_coh_summed": lambda: spectral.regression_spectra(
            data, alpha_mix, fs, calc_coh=False, keepdims=True, **common
        ),
        "regression/hmm_missing_state": lambda: spectral.regression_spectra(
            data,
            alpha_missing,
            fs,
            return_coef_int=True,
            return_weights=True,
            return_sigma=True,
            **common,
        ),
        "regression/2d": lambda: spectral.regression_spectra(
            data[0], alpha_hmm[0], fs, return_coef_int=True, **common
        ),
        "regression/3d_array": lambda: spectral.regression_spectra(
            _equal_length(data),
            _equal_length(alpha_mix),
            fs,
            return_coef_int=True,
            **common,
        ),
    }


def _static_cases():
    data, _, _, _ = simulate()
    fs = SAMPLING_FREQUENCY
    return {
        "static/welch": lambda: static.welch_spectra(data, fs, window_length=200),
        "static/welch_coh_weights": lambda: static.welch_spectra(
            data, fs, calc_coh=True, return_weights=True
        ),
        "static/multitaper_coh": lambda: static.multitaper_spectra(
            data, fs, calc_coh=True, frequency_range=[1, 40]
        ),
        "static/multitaper_cpsd": lambda: static.multitaper_spectra(
            data, fs, calc_cpsd=True, keepdims=True, return_weights=True
        ),
    }


def _multitaper_state_spectra():
    data, _, alpha_hmm, _ = simulate()
    return spectral.multitaper_spectra(
        data, SAMPLING_FREQUENCY, alpha=alpha_hmm, frequency_range=[1, 40]
    )


def _downstream_cases():
    def mt():
        return _multitaper_state_spectra()

    def mt_cpsd():
        data, _, alpha_hmm, _ = simulate()
        return spectral.multitaper_spectra(
            data, SAMPLING_FREQUENCY, alpha=alpha_hmm, calc_cpsd=True
        )

    def components():
        _, _, coh = mt()
        return spectral.decompose_spectra(coh, n_components=2)

    def variance(method, **kwargs):
        f, psd, _ = mt()
        return power.variance_from_spectra(f, psd, method=method, **kwargs)

    def coherence(**kwargs):
        f, _, coh = mt()
        return connectivity.mean_coherence_from_spectra(f, coh, **kwargs)

    return {
        "downstream/decompose_spectra": components,
        "downstream/coherence_spectra": lambda: spectral.coherence_spectra(
            mt_cpsd()[1]
        ),
        "downstream/variance_mean": lambda: variance("mean"),
        "downstream/variance_sum_band": lambda: variance(
            "sum", frequency_range=[8, 14]
        ),
        "downstream/variance_integral": lambda: variance("integral"),
        "downstream/variance_components": lambda: variance(
            "mean", components=components()
        ),
        "downstream/variance_cpsd": lambda: power.variance_from_spectra(*mt_cpsd()),
        "downstream/coherence_mean": lambda: coherence(),
        "downstream/coherence_band": lambda: coherence(frequency_range=[8, 14]),
        "downstream/coherence_components": lambda: coherence(components=components()),
    }


def cases():
    """All test cases.

    Returns
    -------
    cases : dict
        Maps the case name to a function that returns a tuple of arrays.
    """
    all_cases = {
        **_welch_cases(),
        **_multitaper_cases(),
        **_regression_cases(),
        **_static_cases(),
        **_downstream_cases(),
    }
    return {name: (lambda fn=fn: _as_tuple(fn())) for name, fn in all_cases.items()}
