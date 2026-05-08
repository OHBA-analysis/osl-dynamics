CTF Rest Dataset Preprocessing and Source Reconstruction
--------------------------------------------------------

These scripts use the `osl_dynamics.meeg` subpackage. FSL is required for
the SMRI sform fix in `2_fix_smri_files.py` and for RHINO surface
extraction / coregistration.

Run in order:

1. `1_preprocess.py` — sensor-level preprocessing.
2. `2_fix_smri_files.py` — patch the sform header on the public MEGUK
   structural MRI files.
3. `3_source_reconstruct.py` — coregistration, forward model, LCMV
   beamforming and parcellation.
4. `4_sign_flip.py` — align parcel signs across sessions.
5. `5_save_data_bursts.py` / `6_save_data_networks.py` — package data for
   downstream model training.
