"""CTF Rest Dataset: TDE-HMM for Dynamic Network Analysis.

In this script we train a Time-Delay Embedded Hidden Markov Model (TDE-HMM)
on source reconstructed resting-state MEG data.

The examples/toolbox_paper/ctf_rest/get_data.py script can be used
to download the training data.

Functions listed in the config are defined in osl_dynamics.config_api.wrappers.
"""

from sys import argv

if len(argv) != 2:
    print("Please pass the run id, e.g. python tde_hmm_networks.py 1")
    exit()
id = int(argv[1])

import os
import pickle
import numpy as np
import pandas as pd

from osl_dynamics import run_pipeline
from osl_dynamics.analysis import statistics, power, connectivity
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting


def load_group_assignments():
    """Load demographic data and assign subjects to a group (young vs old)."""
    demographics = pd.read_csv("training_data/demographics.csv")
    age = demographics["age"].values
    assignments = np.ones_like(age)
    # Subjects with age > 34 are assigned to the old group
    # and age <= 34 are assigned to the young group
    for i, a in enumerate(age):
        if a > 34:
            assignments[i] += 1
    return assignments


def compare_summary_stats(data, output_dir, n_perm, n_jobs):
    """Compare summary statistics for two groups."""

    # Load group assignments
    assignments = load_group_assignments()
    group = ["Old" if a == 1 else "Young" for a in assignments]

    # Load state time course
    alp = pickle.load(open(f"{output_dir}/inf_params/alp.pkl", "rb"))
    stc = modes.argmax_time_courses(alp)

    # Calculate summary stats
    fo = modes.fractional_occupancies(stc)
    lt = modes.mean_lifetimes(stc, sampling_frequency=data.sampling_frequency)
    intv = modes.mean_intervals(stc, sampling_frequency=data.sampling_frequency)
    sr = modes.switching_rates(stc, sampling_frequency=data.sampling_frequency)
    sum_stats = np.swapaxes([fo, lt, intv, sr], 0, 1)
    n_states = fo.shape[-1]

    # Do statistical significance testing
    _, p = statistics.group_diff_max_stat_perm(
        data=sum_stats,
        assignments=assignments,
        n_perm=n_perm,
        n_jobs=n_jobs,
    )
    p = p.reshape(4, n_states)

    # Plot
    os.makedirs(f"{output_dir}/young_vs_old", exist_ok=True)
    summary_stat_names = [
        "Fractional Occupancy",
        "Mean Lifetime (s)",
        "Mean Interval (s)",
        "Switching Rate (Hz)",
    ]
    for i, name in enumerate(summary_stat_names):
        plotting.plot_summary_stats_group_diff(
            name,
            sum_stats[:, i],
            p[i],
            assignments=group,
            filename=f"{output_dir}/young_vs_old/sum_stats_{i + 1}.png",
        )


def compare_power_maps(data, output_dir, n_perm, significance_level, n_jobs):
    """Compare power maps differences between two groups."""

    # Load group assignments
    assignments = load_group_assignments()

    # Load spectra
    f = np.load(f"{output_dir}/spectra/f.npy")
    psd = np.load(f"{output_dir}/spectra/psd.npy")
    nnmf = np.load(f"{output_dir}/spectra/nnmf_2.npy")

    # Calculate power maps
    power_maps = power.variance_from_spectra(f, psd, nnmf)[:, 0]

    # Do statistical significance testing
    power_map_diff, p = statistics.group_diff_max_stat_perm(
        data=power_maps,
        assignments=assignments,
        n_perm=n_perm,
        n_jobs=n_jobs,
    )

    # Zero non-significant differences
    significant = p < significance_level
    power_map_diff[~significant] = 0

    # Plot
    os.makedirs(f"{output_dir}/young_vs_old", exist_ok=True)
    power.save(
        power_map_diff,
        mask_file=data.mask_file,
        parcellation_file=data.parcellation_file,
        plot_kwargs={"views": ["lateral"]},
        filename=f"{output_dir}/young_vs_old/pow_diff_.png",
    )


def compare_coherence_networks(data, output_dir, n_perm, significance_level, n_jobs):
    """Compare coherence network differences between two groups."""

    # Load group assignments
    assignments = load_group_assignments()

    # Load spectra
    f = np.load(f"{output_dir}/spectra/f.npy")
    coh = np.load(f"{output_dir}/spectra/coh.npy")
    nnmf = np.load(f"{output_dir}/spectra/nnmf_2.npy")

    # Calculate coherences networks
    c = connectivity.mean_coherence_from_spectra(f, coh, nnmf)[:, 0]
    n_states = c.shape[1]
    n_channels = c.shape[2]

    # Just keep the upper triangle
    m, n = np.triu_indices(c.shape[-1], k=1)
    c = c[:, :, m, n]

    # Do statistical significance testing
    c_diff, p = statistics.group_diff_max_stat_perm(
        data=c,
        assignments=assignments,
        n_perm=n_perm,
        n_jobs=n_jobs,
    )

    # Zero non-significant differences
    significant = p < significance_level
    c_diff[~significant] = 0

    # Convert back into a full matrix
    coh_diff = np.zeros([n_states, n_channels, n_channels])
    for i in range(n_states):
        coh_diff[i, m, n] = c_diff[i]
        coh_diff[i, n, m] = c_diff[i]

    # Plot
    os.makedirs(f"{output_dir}/young_vs_old", exist_ok=True)
    connectivity.save(
        coh_diff,
        parcellation_file=data.parcellation_file,
        filename=f"{output_dir}/young_vs_old/coh_diff_.png",
    )


def compare_groups(data, output_dir, n_perm, significance_level, n_jobs):
    """Compare groups."""
    compare_summary_stats(data, output_dir, n_perm, n_jobs)
    compare_power_maps(data, output_dir, n_perm, significance_level, n_jobs)
    compare_coherence_networks(data, output_dir, n_perm, significance_level, n_jobs)


# Full pipeline
config = """
    load_data:
        inputs: training_data/networks
        kwargs:
            sampling_frequency: 250
            mask_file: MNI152_T1_8mm_brain.nii.gz
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            n_jobs: 8
        prepare:
            tde_pca: {n_embeddings: 15, n_pca_components: 80}
            standardize: {}
    train_hmm:
        config_kwargs:
            n_states: 8
            learn_means: False
            learn_covariances: True
    multitaper_spectra:
        kwargs:
            frequency_range: [1, 45]
            n_jobs: 8
        nnmf_components: 2
    plot_group_nnmf_tde_hmm_networks:
        nnmf_file: spectra/nnmf_2.npy
        power_save_kwargs:
            plot_kwargs: {views: [lateral]}
    plot_alpha:
        kwargs: {n_samples: 2000}
    plot_hmm_network_summary_stats: {}
    compare_groups:
        n_perm: 1000
        significance_level: 0.05
        n_jobs: 8
"""

# Run analysis
run_pipeline(
    config,
    output_dir=f"results/run{id:02d}",
    extra_funcs=[compare_groups],
)
