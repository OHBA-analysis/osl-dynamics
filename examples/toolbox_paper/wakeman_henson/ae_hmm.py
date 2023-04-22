"""Wakeman-Henson: AE-HMM.

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
            amplitude_envelope: True
            n_window: 5
    train_hmm:
        config_kwargs:
            n_states: 8
            learn_means: True
            learn_covariances: True
    calc_subject_ae_hmm_networks: {}
    plot_group_ae_networks:
        power_save_kwargs:
            plot_kwargs: {views: [lateral]}
    plot_alpha:
        kwargs: {n_samples: 2000}
    plot_summary_stats: {}
"""
run_pipeline(config, output_dir="results/ae_hmm")
