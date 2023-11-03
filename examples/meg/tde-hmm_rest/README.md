Example TDE-HMM Pipeline for Studying Resting-State MEG Data
------------------------------------------------------------

These script outline a common pipeline for studying MEG data using a Time-Delay Embedded Hidden Markov Model (TDE-HMM). In the pipeline we do the following:

- **1_train_hmm.py**: Train a TDE-HMM. This script is usually run multiple times.
- **2_print_free_energy.py**: Print the variational free energy of each HMM run. The run with the lowest free energy is considered the best run, which we perform subsequence analysis on.
- **3_get_inf_params.py**: Get the inferred parameters, these are the state means, covariance and the probability of each state being active at each time point.
- **4_calc_multitaper.py**: Calculate post-hoc spectra (PSD and coherence) for each state. A multitaper spectrum is calculated for each subject individually.
- **5_plot_networks.py**: This script calculates spatial power maps and coherence networks from the multitaper spectra and plots them.
- **6_calc_summary_stats.py**: This script calculates statistics that summarise the dynamics of each state. These are calculated for each subject individually.
- **7_compare_groups.py**: This script performs a group-level analysis on the subject-specific quantities (power, coherence, summary statistics) and performs statistical significance testing.

We also include a script to download example data (**0_get_data.py**) that can be used to go through these scripts.

Note, you need to activate the conda environment before running these scripts:
    
    conda activate osld
