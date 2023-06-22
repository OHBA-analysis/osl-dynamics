"""Elekta Task Dataset: TDE-HMM Pipeline.

In this script we train an Time-Delay Embedded Hidden Markov Model (TDE-HMM)
on source reconstructed task MEG data.

The examples/toolbox_paper/elekta_task/get_data.py script can be used
to download the training data.

Functions listed in the config are defined in osl_dynamics.config_api.wrappers.
"""

from sys import argv

if len(argv) != 2:
    print("Please pass the run id, e.g. python tde_hmm.py 1")
    exit()
id = argv[1]

from osl_dynamics import run_pipeline

config = """
    load_data:
        data_dir: training_data
        data_kwargs:
            sampling_frequency: 250
            mask_file: MNI152_T1_8mm_brain.nii.gz
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            n_jobs: 16
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
        nnmf_components: 2
    plot_group_nnmf_tde_hmm_networks:
        nnmf_file: spectra/nnmf_2.npy
        power_save_kwargs:
            plot_kwargs: {views: [lateral]}
    plot_alpha:
        kwargs: {n_samples: 2000}
    plot_hmm_network_summary_stats: {}
"""
run_pipeline(config, output_dir=f"results/run{id}")
