"""AE-HMM.

In this script we train an Amplitude Envelope Hidden Markov Model (AE-HMM)
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
            n_jobs: 8
        prepare_kwargs:
            amplitude_envelope: True
            n_window: 5
    train_hmm:
        config_kwargs:
            n_states: 8
            learn_means: True
            learn_covariances: True
    calc_subject_ae_hmm_networks: {}
    plot_group_ae_networks: {}
"""
run_pipeline(config, output_dir="results/ae_hmm")
