Example scripts for training an HMM on UK Biobank data
------------------------------------------------------

These scripts contain a typical HMM pipeline on fMRI data. We train an HMM on UK Biobank data:

- **1_find_data_files.py**: Find the files we want to train on. These files are hosted on the BMRC cluster.
- **2_train_hmm.py**: This script train the main model. Normally you want to train multiple models (runs) and only study the one with the lowest free energy. Note, the `submit_jobs.py` script can be used to submit multiple (GPU) jobs to the BMRC cluster.
- **3_print_free_energy.py**: This script prints the free energy of the each run and states the best run.
- **4_dual_estimation.py**: We're normally interested in subject-specific quantities. This script re-estimates the observation model for individual subjects.
- **5_plot_results.py**: This script plots the group-level results.

One run takes roughly 1 hour 30 minutes with 1 GPU.

These scripts can be run on BMRC. Note you need to activate the conda environment before running these scripts:
    
    conda activate osld
