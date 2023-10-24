"""CTF Rest Dataset: TDE-HMM Burst Detection Pipeline.

In this script we train a Time-Delay Embedded Hidden Markov Model (TDE-HMM)
on a single channel (region) of source reconstructed resting-state MEG data.

The examples/toolbox_paper/ctf_rest/get_data.py script can be used
to download the training data.

Functions listed in the config are defined in osl_dynamics.config_api.wrappers.
"""

from sys import argv

if len(argv) != 2:
    print("Please pass the run id, e.g. python tde_hmm_bursts.py 1")
    exit()
id = int(argv[1])

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
        time_range=[0, 20],
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
        data.prepare({
            "filter": {"low_freq": low_freq, "high_freq": high_freq, "use_raw" : True},
            "amplitude_envelope": {},
            "standardize": {},
        })
        x = data.trim_time_series(sequence_length, n_embeddings, concatenate=True)
        return x[:, 0]

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


# Full pipeline
config = """
    load_data:
        inputs: training_data/bursts
        kwargs:
            sampling_frequency: 100
        prepare:
            tde: {n_embeddings: 21}
            standardize: {}
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
    plot_state_psds: {}
    plot_tde_covariances: {}
    plot_burst_summary_stats: {}
"""

# Run analysis
run_pipeline(
    config,
    output_dir=f"results/run{id:02d}",
    extra_funcs=[plot_wavelet, plot_amplitude_envelopes_and_alpha],
)
