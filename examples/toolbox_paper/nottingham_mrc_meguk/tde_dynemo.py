"""Nottingham MRC MEGUK: TDE-DyNeMo.

In this script we train a DyNeMo model on time-delay embedded data.

We will use source reconstructed resting-state MEG data. See the
examples/toolbox_paper/nottingham_mrc_meguk/get_data.py script for
how to download the training data.
"""

from osl_dynamics import run_pipeline
from osl_dynamics.inference import tf_ops

config = """
    load_data:
        data_dir: data/training_data
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
run_pipeline(config, output_dir="results/tde_dynemo")
