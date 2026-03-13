# Cam-CAN Example Scripts

[Cam-CAN](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/) is a large publicly available MEG dataset. This directory contains scripts for training a TDE-HMM on parcellated data calculated using this dataset using [OSL](https://github.com/OHBA-analysis/osl/tree/main/examples/camcan).

Scripts:

- `1_prepare_data.py`: prepare the training dataset. We use TFRecords to store the data - this is TensorFlow's preferred data format for large datasets.
- `2_train_hmm.py`: train an HMM.
- `2_submit_jobs.py`: this script can be used to run the `2_train_hmm.py` script as a job on the Oxford BMRC cluster.
- `3_print_free_energy.py`: find the best run, i.e. the one with the lowest free energy.
- `4_get_inf_params.py`: get inferred parameters (state probability time courses and covariances).
- `5_calc_post_hoc.py`: do post-hoc analysis (calculate multitaper spectra and summary statistics).
