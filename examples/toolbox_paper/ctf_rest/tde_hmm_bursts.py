"""Nottingham MRC MEGUK: TDE-HMM Burst Detection Pipeline.

In this script we train a Time-Delay Embedded Hidden Markov Model (TDE-HMM)
on a single channel (region) of source reconstructed resting-state MEG data.

The examples/toolbox_paper/ctf_rest/get_data.py script can be used
to download the training data.
"""

from sys import argv

if len(argv) != 2:
    print("Please pass the run id, e.g. python tde_hmm_bursts.py 1")
    exit()
id = argv[1]

import os
import pickle
import numpy as np
from scipy import signal

from osl_dynamics import run_pipeline
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting


def plot_wavelet(data, output_dir):
    """Plot wavelet transform of the training data."""

    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    x = data.time_series(prepared=False)  # raw data (before time-delay embedding)
    x = x[0][:, 0]  # first subject
    plotting.plot_wavelet(
        x,
        sampling_frequency=data.sampling_frequency,
        start_time=0,
        end_time=20,
        filename=f"{plots_dir}/wavelet.png",
    )


def plot_amplitude_envelopes_and_alpha(data, output_dir, n_samples):
    """Plot amplitude envelopes and inferred state probabilities."""

    inf_params_dir = f"{output_dir}/inf_params"
    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Plot state probability time course
    alp = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))
    alp = np.concatenate(alp)
    plotting.plot_alpha(
        alp,
        n_samples=n_samples,
        sampling_frequency=data.sampling_frequency,
        fig_kwargs={"figsize": (12, 3)},
        filename=f"{plots_dir}/alpha.png",
    )

    def get_amp_env(low_freq, high_freq, n_embeddings=21, sequence_length=2000):
        data.prepare_amp_env(low_freq=low_freq, high_freq=high_freq)
        x = []
        for X in data.time_series():
            X = X[n_embeddings // 2 :]
            X = X[: (X.shape[0] // sequence_length) * sequence_length]
            x.append(X)
        return np.concatenate(x)[:, 0]

    # Get amplitude envelope data for different frequency bands
    x_beta = get_amp_env(13, 30)
    x_alpha = get_amp_env(7, 13)
    x_delta_theta = get_amp_env(1, 7)
    x = np.array([x_beta, x_alpha, x_delta_theta]).T

    # Calculate the correlation of each amplitude envelope with the state probabilities
    corr = np.corrcoef(x, alp, rowvar=False)[:3, 3:]
    plotting.plot_matrices(corr, filename=f"{plots_dir}/alp_amp_env_corr.png")

    # Plot amplitude envelopes
    t = np.arange(n_samples) / data.sampling_frequency
    x_beta = x_beta[:n_samples]
    x_alpha = x_alpha[:n_samples]
    x_delta_theta = x_delta_theta[:n_samples]
    plotting.plot_line(
        [t],
        [x_beta],
        x_label="Time (s)",
        y_label="Signal (a.u.)",
        fig_kwargs={"figsize": (9, 2.5)},
        filename=f"{plots_dir}/amp_env_beta.png",
    )
    plotting.plot_line(
        [t],
        [x_alpha],
        x_label="Time (s)",
        y_label="Signal (a.u.)",
        fig_kwargs={"figsize": (9, 2.5)},
        filename=f"{plots_dir}/amp_env_alpha.png",
    )
    plotting.plot_line(
        [t],
        [x_delta_theta],
        x_label="Time (s)",
        y_label="Signal (a.u.)",
        fig_kwargs={"figsize": (9, 2.5)},
        filename=f"{plots_dir}/amp_env_delta_theta.png",
    )


def plot_psds(data, output_dir):
    """Plot state PSDs."""

    spectra_dir = f"{output_dir}/spectra"
    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    f = np.load(f"{spectra_dir}/f.npy")
    psd = np.load(f"{spectra_dir}/psd.npy")
    psd = np.squeeze(psd)  # remove the channel dimension
    psd = np.mean(psd, axis=0)  # average over subjects

    n_states = psd.shape[0]
    plotting.plot_line(
        [f] * n_states,
        psd,
        labels=[f"State {i + 1}" for i in range(n_states)],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        filename=f"{plots_dir}/psd.png",
    )


def plot_autocovariances(data, output_dir):
    """Plot inferred covariance matrices."""

    inf_params_dir = f"{output_dir}/inf_params"
    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    covs = np.load(f"{inf_params_dir}/covs.npy")
    plotting.plot_matrices(covs, filename=f"{plots_dir}/covs.png")


def plot_burst_summary_stats(data, output_dir):
    """Plot summary statistics for bursts."""

    inf_params_dir = f"{output_dir}/inf_params"
    sum_stats_dir = f"{output_dir}/summary_stats"
    plots_dir = f"{output_dir}/plots"
    os.makedirs(sum_stats_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load state time course
    alp = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))
    stc = modes.argmax_time_courses(alp)

    n_subjects = len(stc)
    n_states = stc[0].shape[1]

    # Mean lifetime
    lt = modes.mean_lifetimes(stc, sampling_frequency=data.sampling_frequency)
    np.save(f"{sum_stats_dir}/lt.npy", lt)
    fig, ax = plotting.plot_violin(
        lt.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Lifetime (s)",
        fig_kwargs={"figsize": None},
    )
    plotting.save(fig, filename=f"{plots_dir}/sum_stats_1.png")

    # Mean interval
    intv = modes.mean_intervals(stc, sampling_frequency=data.sampling_frequency)
    np.save(f"{sum_stats_dir}/intv.npy", intv)
    fig, ax = plotting.plot_violin(
        intv.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Interval (s)",
        fig_kwargs={"figsize": None},
    )
    plotting.save(fig, filename=f"{plots_dir}/sum_stats_2.png")

    # Burst count
    sr = modes.switching_rates(stc, sampling_frequency=data.sampling_frequency)
    np.save(f"{sum_stats_dir}/sr.npy", sr)
    fig, ax = plotting.plot_violin(
        sr.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Burst Count (Hz)",
        fig_kwargs={"figsize": None},
    )
    plotting.save(fig, filename=f"{plots_dir}/sum_stats_3.png")

    # Average amplitude
    amp = np.zeros([n_subjects, n_states])
    x = data.trim_time_series(n_embeddings=21, sequence_length=2000)
    for i in range(n_subjects):
        d = np.abs(signal.hilbert(x[i], axis=0))
        s = stc[i]
        for j in range(n_states):
            amp[i, j] = np.mean(d[s[:, j] == 1])
    np.save(f"{sum_stats_dir}/amp.npy", amp)
    fig, ax = plotting.plot_violin(
        amp.T,
        x=range(1, n_states + 1),
        x_label="State",
        y_label="Mean Amplitude (a.u.)",
        fig_kwargs={"figsize": None},
    )
    plotting.save(fig, filename=f"{plots_dir}/sum_stats_4.png")


config = """
    load_data:
        data_dir: training_data/bursts
        data_kwargs:
            sampling_frequency: 100
        prepare_kwargs:
            n_embeddings: 21
    plot_wavelet: {}
    train_hmm:
        config_kwargs:
            n_states: 3
            learn_means: False
            learn_covariances: True
    multitaper_spectra:
        kwargs:
            frequency_range: [1, 45]
    plot_amplitude_envelopes_and_alpha:
        n_samples: 2000
    plot_psds: {}
    plot_autocovariances: {}
    plot_burst_summary_stats: {}
"""

run_pipeline(
    config,
    output_dir=f"results/run{id}",
    extra_funcs=[
        plot_wavelet,
        plot_amplitude_envelopes_and_alpha,
        plot_psds,
        plot_autocovariances,
        plot_burst_summary_stats,
    ],
)
