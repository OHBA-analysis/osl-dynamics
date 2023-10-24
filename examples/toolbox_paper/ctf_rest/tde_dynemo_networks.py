"""CTF Rest Dataset: TDE-DyNeMo for Dynamic Network Analysis.

In this script we train a Time-Delay Embedded (TDE)-DyNeMo model on source reconstructed
resting-state MEG data.

The examples/toolbox_paper/ctf_rest/get_data.py script can be used
to download the training data.

Functions listed in the config are defined in osl_dynamics.config_api.wrappers.
"""

from sys import argv

if len(argv) != 2:
    print("Please pass the run id, e.g. python tde_dynemo_networks.py 1")
    exit()
id = int(argv[1])

import os
import numpy as np
import pandas as pd

from osl_dynamics import run_pipeline
from osl_dynamics.analysis import statistics
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

    # Load summary statistics
    alp_mean = np.load(f"{output_dir}/summary_stats/alp_mean.npy")
    alp_std = np.load(f"{output_dir}/summary_stats/alp_std.npy")
    sum_stats = np.swapaxes([alp_mean, alp_std], 0, 1)

    # Do statistical significance testing
    _, p = statistics.group_diff_max_stat_perm(
        data=sum_stats,
        assignments=assignments,
        n_perm=n_perm,
        n_jobs=n_jobs,
    )

    print("Number of significant difference in alp_mean:", np.sum(p[0] < 0.05))
    print("Number of significant difference in alp_std:", np.sum(p[1] < 0.05))

    # Plot
    os.makedirs(f"{output_dir}/young_vs_old", exist_ok=True)
    summary_stat_names = ["Mean", "Standard Deviation"]
    for i, name in enumerate(summary_stat_names):
        plotting.plot_summary_stats_group_diff(
            name,
            sum_stats[:, i],
            p[i],
            assignments=group,
            filename=f"{output_dir}/young_vs_old/sum_stats_{i + 1}.png",
        )


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
    train_dynemo:
        config_kwargs:
            n_modes: 7
            learn_means: False
            learn_covariances: True
    regression_spectra:
        kwargs:
            frequency_range: [1, 45]
            n_jobs: 8
    plot_group_tde_dynemo_networks:
        power_save_kwargs:
            plot_kwargs: {views: [lateral]}
    plot_alpha:
        normalize: True
        kwargs: {n_samples: 2000}
    plot_dynemo_network_summary_stats: {}
    compare_summary_stats:
        n_perm: 1000
        n_jobs: 4
"""

# Run analysis
run_pipeline(
    config,
    output_dir=f"results/run{id:02d}",
    extra_funcs=[compare_summary_stats],
)
