Elekta Task Dataset Preprocessing and Source Reconstruction
-----------------------------------------------------------

These scripts use the `osl_dynamics.meeg` subpackage. FSL is required
for RHINO surface extraction and coregistration.

Run in order:

1. `1_preprocess.py` — sensor-level preprocessing.
2. `2_source_reconstruct.py` — coregistration, forward model, LCMV
   beamforming and parcellation.
3. `3_sign_flip.py` — align parcel signs across sessions.
