Example TDE-DyNeMo Pipeline for Studying Resting-State MEG Data
---------------------------------------------------------------

These script outline a common pipeline for studying MEG data using DyNeMo. In the pipeline we do the following:

- **1_train_dynemo.py**: Train a TDE-DyNeMo model. This script is usually run multiple times.
- **2_print_free_energy.py**: Print the variational free energy of each DyNeMo run. The run with the lowest free energy is considered the best run, which we perform subsequence analysis on.
- **3_get_inf_params.py**: Get the inferred parameters, these are the mode means, covariance and the mode mixing coefficients.
- **4_calc_regression_spectra.py**: Calculate post-hoc spectra (PSD and coherence) for each mode. A regression spectrum is calculated for each subject individually.
- **5_calc_summary_stats.py**: This script calculates statistics that summarise the dynamics of each mode. These are calculated for each subject individually.
- **6_plot_networks.py**: This script calculates spatial power maps and coherence networks from the regression spectra and plots them.

We also include a script to download example data (**0_get_data.py**) that can be used to go through these scripts.

Note, you need to activate the conda environment before running these scripts:
    
    conda activate osld
