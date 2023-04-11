"""TDE-HMM.

In this script we train a Time-Delay Embedded Hidden Markov Model (TDE-HMM)
on source reconstructed resting-state MEG data and plot the inferred networks.

The examples/toolbox_paper/get_data.py script can be used to download the
training data.
"""

from osl_dynamics import run_pipeline

config = """
    load_data:
        data_dir: training_data
        data_kwargs:
            sampling_frequency: 250
            mask_file: MNI152_T1_8mm_brain.nii.gz
            parcellation_file: fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz
        prepare_kwargs:
            n_embeddings: 15
            n_pca_components: 80
    train_hmm:
        config_kwargs:
            n_states: 8
            learn_means: False
            learn_covariances: True
    multitaper_spectra:
        kwargs:
            frequency_range: [1, 45]
            n_jobs: 16
    plot_group_tde_hmm_networks: {}
"""
run_pipeline(config, output_dir="results/tde_hmm")
