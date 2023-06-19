"""Nottingham MRC MEGUK: TDE-DyNeMo for Dynamic Network Analysis.

In this script we train a Time-Delay Embedded (TDE)-DyNeMo model on source reconstructed
resting-state MEG data.

The examples/toolbox_paper/ctf_rest/get_data.py script can be used
to download the training data.
"""

from sys import argv

if len(argv) != 2:
    print("Please pass the run id, e.g. python tde_dynemo_networks.py 1")
    exit()
id = argv[1]

from osl_dynamics import run_pipeline
from osl_dynamics.inference import tf_ops

config = """
    load_data:
        data_dir: training_data/networks
        data_kwargs:
            sampling_frequency: 250
            mask_file: MNI152_T1_8mm_brain.nii.gz
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            n_jobs: 16
        prepare_kwargs:
            n_embeddings: 15
            n_pca_components: 80
    train_dynemo:
        config_kwargs:
            n_modes: 8
            learn_means: False
            learn_covariances: True
    regression_spectra:
        kwargs:
            frequency_range: [1, 45]
            n_jobs: 16
    plot_group_tde_dynemo_networks:
        power_save_kwargs:
            plot_kwargs: {views: [lateral]}
    plot_alpha:
        normalize: True
        kwargs: {n_samples: 2000}
    calc_gmm_alpha: {}
    plot_summary_stats:
        use_gmm_alpha: True
"""
run_pipeline(config, output_dir=f"results/run{id}")
