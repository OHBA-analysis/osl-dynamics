"""
Cam-CAN MEG Batch Processing
============================

This tutorial shows how to process a large MEG dataset in batch using
osl-dynamics' parallel processing utilities. We use the
`Cam-CAN dataset <https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/>`_
as an example.

The pipeline mirrors the single-session tutorial
(:doc:`MEG Processing <0-1_meg_preprocessing>`) but wraps each
step in a function that is dispatched across sessions using
``osl_dynamics.meeg.parallel.run``.

Prerequisites
^^^^^^^^^^^^^

- `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki>`_ (needed for surface extraction).
- `osl-dynamics <https://github.com/OHBA-analysis/osl-dynamics>`_.

Input Data
^^^^^^^^^^

The Cam-CAN dataset contains resting-state (``rest``), sensorimotor
(``smt``) and passive stimulation (``passive``) MEG recordings from
~650 subjects. The raw data has already been MaxFiltered. Each subject
also has a structural MRI. The data is organised as::

    cc700/
    ├── meg/
    │   └── pipeline/release005/BIDSsep/
    │       ├── derivatives_rest/...
    │       ├── derivatives_smt/...
    │       └── derivatives_passive/...
    └── mri/
        └── pipeline/release004/BIDS_20190411/anat/
            └── sub-*/anat/sub-*_T1w.nii.gz

The output of this script is written to ``derivatives/``.
"""

#%%
# Step 0: Find File Paths
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# First, we locate all raw MEG files and their corresponding T1w
# structural MRIs and save them to a CSV file. This CSV acts as
# the session manifest for all subsequent steps.
#
# .. code-block:: python
#
#     from glob import glob
#     from pathlib import Path
#     import re
#     import pandas as pd
#
#     # Raw data directory
#     raw_dir = "/path/to/cc700"
#     anat_dir = f"{raw_dir}/mri/pipeline/release004/BIDS_20190411/anat"
#
#     # Find MaxFiltered MEG files for all three tasks
#     inputs = []
#     for task in ["rest", "smt", "passive"]:
#         inputs += sorted(
#             glob(
#                 f"{raw_dir}/meg/pipeline/release005/BIDSsep"
#                 f"/derivatives_{task}/aa/AA_movecomp"
#                 f"/aamod_meg_maxfilt_00002/sub-*"
#                 f"/mf2pt2_sub-*_ses-{task}_task-{task}_meg.fif"
#             )
#         )
#
#     # Build a table of sessions, keeping only those with a structural MRI
#     rows = []
#     for fpath in inputs:
#         match = re.search(r"(sub-[A-Za-z0-9]+)_ses-(\w+)_task-(\w+)_meg\.fif", fpath)
#         if match:
#             subject = match.group(1)
#             task = match.group(2)
#             mri_file = f"{anat_dir}/{subject}/anat/{subject}_T1w.nii.gz"
#             if not Path(mri_file).exists():
#                 continue
#             rows.append({
#                 "id": f"{subject}_task-{task}",
#                 "subject": subject,
#                 "task": task,
#                 "raw_file": fpath,
#                 "mri_file": mri_file,
#             })
#
#     df = pd.DataFrame(rows)
#     print(f"Found {len(df)} sessions from {df['subject'].nunique()} subjects")
#     print(f"Tasks: {df['task'].value_counts().to_dict()}")
#
#     df.to_csv("sessions.csv", index=False)
#     print("Saved sessions.csv")

#%%
# Step 1: Preprocessing
# ^^^^^^^^^^^^^^^^^^^^^
#
# We define a function that preprocesses a single session and use
# ``parallel.run`` to dispatch it across sessions.
#
# Each session is independently:
#
# - Notch filtered at 50 and 100 Hz.
# - Bandpass filtered (0.5–80 Hz, 5th-order Butterworth).
# - Downsampled to 250 Hz.
# - Checked for bad segments and channels.
# - QC plots saved.
#
# .. code-block:: python
#
#     from pathlib import Path
#     import pandas as pd
#     import mne
#     import matplotlib
#     matplotlib.use("Agg")
#     from osl_dynamics.meeg import parallel, preproc
#
#     output_dir = Path("derivatives")
#     plots_dir = Path("plots")
#     log_dir = Path("logs/1_preproc")
#
#     sessions = pd.read_csv("sessions.csv").to_dict("records")
#
#
#     def process_session(session, logger):
#         """Preprocess a single session."""
#
#         logger.log("Loading raw data...")
#         raw = mne.io.read_raw_fif(session["raw_file"], preload=True)
#
#         logger.log("Filtering and downsampling...")
#         raw = raw.notch_filter([50, 100])
#         raw = raw.filter(
#             l_freq=0.5,
#             h_freq=80,
#             method="iir",
#             iir_params={"order": 5, "ftype": "butter"},
#         )
#         raw = raw.resample(sfreq=250)
#
#         logger.log("Detecting bad segments...")
#         raw = preproc.detect_bad_segments(raw, picks="mag")
#         raw = preproc.detect_bad_segments(raw, picks="mag", mode="diff")
#         raw = preproc.detect_bad_segments(raw, picks="grad")
#         raw = preproc.detect_bad_segments(raw, picks="grad", mode="diff")
#
#         logger.log("Detecting bad channels...")
#         raw = preproc.detect_bad_channels(raw, picks="mag")
#         raw = preproc.detect_bad_channels(raw, picks="grad")
#
#         logger.log("Saving QC plots...")
#         preproc.save_qc_plots(raw, plots_dir / session["id"])
#
#         logger.log("Saving preprocessed data...")
#         preproc_out_dir = output_dir / "preprocessed"
#         preproc_out_dir.mkdir(parents=True, exist_ok=True)
#         outfile = preproc_out_dir / f"{session['id']}_preproc-raw.fif"
#         raw.save(outfile, overwrite=True)
#
#         logger.log("Done.")
#
#
#     if __name__ == "__main__":
#         parallel.run(
#             process_session,
#             items=sessions,
#             output_dir=output_dir,
#             log_dir=log_dir,
#             plots_dir=plots_dir,
#             n_workers=8,
#         )

#%%
# Step 2: Surface Extraction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Surface extraction only needs to be done once per subject (not per
# session), so we deduplicate by subject before dispatching.
#
# .. code-block:: python
#
#     from pathlib import Path
#     import pandas as pd
#     from osl_dynamics.meeg import parallel, rhino
#
#     output_dir = Path("derivatives")
#     log_dir = Path("logs/2_surfaces")
#
#     df = pd.read_csv("sessions.csv")
#     subjects = []
#     for _, row in df.drop_duplicates("subject").iterrows():
#         subjects.append({
#             "id": row["subject"],
#             "subject": row["subject"],
#             "mri_file": row["mri_file"],
#         })
#
#
#     def process_subject(subject, logger):
#         """Extract surfaces for a single subject."""
#
#         logger.log("Extracting surfaces...")
#
#         outdir = output_dir / "anat_surfaces" / subject["subject"]
#
#         rhino.extract_surfaces(
#             mri_file=subject["mri_file"],
#             outdir=str(outdir),
#             include_nose=False,
#         )
#
#         logger.log("Done.")
#
#
#     if __name__ == "__main__":
#         parallel.run(
#             process_subject,
#             items=subjects,
#             log_dir=log_dir,
#             n_workers=8,
#         )

#%%
# Step 3: Coregistration
# ^^^^^^^^^^^^^^^^^^^^^^
#
# The coregistration step also includes a dataset-specific function to
# remove stray Polhemus headshape points. You will likely need to adapt
# ``fix_headshape_points`` for your own dataset.
#
# .. code-block:: python
#
#     from pathlib import Path
#     import numpy as np
#     import pandas as pd
#     from osl_dynamics.meeg import parallel, rhino
#     from osl_dynamics.utils.filenames import OSLFilenames
#
#     output_dir = Path("derivatives")
#     plots_dir = Path("plots")
#     log_dir = Path("logs/3_coreg")
#
#     sessions = pd.read_csv("sessions.csv").to_dict("records")
#
#
#     def fix_headshape_points(fns):
#         fns = fns.coreg
#
#         hs = np.loadtxt(fns.head_headshape_file)
#         nas = np.loadtxt(fns.head_nasion_file)
#         lpa = np.loadtxt(fns.head_lpa_file)
#         rpa = np.loadtxt(fns.head_rpa_file)
#
#         # Drop nasion by 4cm and remove headshape points more than 7cm away
#         nas[2] -= 40
#         distances = np.sqrt(
#             (nas[0] - hs[0]) ** 2 + (nas[1] - hs[1]) ** 2 + (nas[2] - hs[2]) ** 2
#         )
#         keep = distances > 70
#         hs = hs[:, keep]
#
#         # Remove anything outside of rpa
#         keep = hs[0] < rpa[0]
#         hs = hs[:, keep]
#
#         # Remove anything outside of lpa
#         keep = hs[0] > lpa[0]
#         hs = hs[:, keep]
#
#         # Remove headshape points on the neck
#         remove = hs[2] < min(lpa[2], rpa[2]) - 4
#         hs = hs[:, ~remove]
#
#         np.savetxt(fns.head_headshape_file, hs)
#
#
#     def process_session(session, logger):
#         """Coregister a single session."""
#
#         preproc_file = str(output_dir / "preprocessed" / f"{session['id']}_preproc-raw.fif")
#         surfaces_dir = str(output_dir / "anat_surfaces" / session["subject"])
#
#         fns = OSLFilenames(
#             outdir=str(output_dir / "osl"),
#             id=session["id"],
#             preproc_file=preproc_file,
#             surfaces_dir=surfaces_dir,
#         )
#
#         logger.log("Extracting fiducials and headshape...")
#         rhino.extract_fiducials_and_headshape_from_fif(fns)
#
#         logger.log("Fixing headshape points...")
#         fix_headshape_points(fns)
#
#         logger.log("Coregistering MEG to MRI...")
#         rhino.coregister_head_and_mri(fns, use_nose=False)
#
#         logger.log("Done.")
#
#
#     if __name__ == "__main__":
#         parallel.run(
#             process_session,
#             items=sessions,
#             output_dir=output_dir,
#             log_dir=log_dir,
#             plots_dir=plots_dir,
#             n_workers=8,
#         )

#%%
# Step 4: Forward Model and Source Reconstruction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This step computes the forward model and LCMV beamformer weights for
# each session.
#
# .. code-block:: python
#
#     from pathlib import Path
#     import pandas as pd
#     import matplotlib
#     matplotlib.use("Agg")
#     from osl_dynamics.meeg import parallel, rhino, source_recon
#     from osl_dynamics.utils.filenames import OSLFilenames
#
#     output_dir = Path("derivatives")
#     plots_dir = Path("plots")
#     log_dir = Path("logs/4_source_recon")
#
#     sessions = pd.read_csv("sessions.csv").to_dict("records")
#
#
#     def process_session(session, logger):
#         """Source reconstruct a single session."""
#
#         preproc_file = str(output_dir / "preprocessed" / f"{session['id']}_preproc-raw.fif")
#         surfaces_dir = str(output_dir / "anat_surfaces" / session["subject"])
#
#         fns = OSLFilenames(
#             outdir=str(output_dir / "osl"),
#             id=session["id"],
#             preproc_file=preproc_file,
#             surfaces_dir=surfaces_dir,
#         )
#
#         logger.log("Computing forward model...")
#         rhino.forward_model(fns, model="Single Layer", gridstep=8)
#
#         logger.log("Computing LCMV beamformer...")
#         source_recon.lcmv_beamformer(fns, chantypes=["mag", "grad"], rank={"meg": 60})
#
#         logger.log("Done.")
#
#
#     if __name__ == "__main__":
#         parallel.run(
#             process_session,
#             items=sessions,
#             output_dir=output_dir,
#             log_dir=log_dir,
#             plots_dir=plots_dir,
#             n_workers=8,
#         )

#%%
# Step 5: Parcellation
# ^^^^^^^^^^^^^^^^^^^^
#
# The final step applies the beamformer, parcellates the voxel data, and
# saves the output. See the :ref:`parcellations <parcellations>` page for
# the full list of available parcellations.
#
# .. code-block:: python
#
#     from pathlib import Path
#     import pandas as pd
#     import mne
#     import matplotlib
#     matplotlib.use("Agg")
#     from osl_dynamics.meeg import parallel, source_recon, parcellation
#     from osl_dynamics.utils.filenames import OSLFilenames
#
#     output_dir = Path("derivatives")
#     plots_dir = Path("plots")
#     log_dir = Path("logs/5_parc")
#
#     sessions = pd.read_csv("sessions.csv").to_dict("records")
#
#
#     def process_session(session, logger):
#         """Parcellate a single session."""
#
#         preproc_file = str(output_dir / "preprocessed" / f"{session['id']}_preproc-raw.fif")
#         surfaces_dir = str(output_dir / "anat_surfaces" / session["subject"])
#
#         fns = OSLFilenames(
#             outdir=str(output_dir / "osl"),
#             id=session["id"],
#             preproc_file=preproc_file,
#             surfaces_dir=surfaces_dir,
#         )
#
#         logger.log("Applying LCMV beamformer...")
#         voxel_data, voxel_coords = source_recon.apply_lcmv_beamformer(fns)
#
#         parcellation_file = "atlas-DK_nparc-54_space-MNI_res-8x8x8.nii.gz"
#
#         logger.log("Parcellating...")
#         parcel_data = parcellation.parcellate(
#             fns,
#             voxel_data,
#             voxel_coords,
#             method="spatial_basis",
#             orthogonalisation="symmetric",
#             parcellation_file=parcellation_file,
#         )
#
#         logger.log("Saving parcellated data...")
#         raw = mne.io.read_raw_fif(preproc_file, preload=True)
#         parc_fif = str(output_dir / "osl" / session["id"] / "lcmv-parc-raw.fif")
#         parcellation.save_as_fif(
#             parcel_data,
#             raw,
#             extra_chans="stim",
#             filename=parc_fif,
#         )
#
#         logger.log("Saving QC plots...")
#         parcellation.save_qc_plots(parc_fif, parcellation_file)
#
#         logger.log("Done.")
#
#
#     if __name__ == "__main__":
#         parallel.run(
#             process_session,
#             items=sessions,
#             output_dir=output_dir,
#             log_dir=log_dir,
#             plots_dir=plots_dir,
#             n_workers=8,
#         )

#%%
# Summary
# ^^^^^^^
#
# Each step can be run as a standalone script. The ``parallel.run``
# utility handles multiprocessing, logging, and error recovery. Logs
# for each session are written to the ``logs/`` directory — check these
# if a session fails.
#
# The final parcellated data can be loaded for downstream analysis::
#
#     from osl_dynamics.data import Data
#
#     training_data = Data(
#         "derivatives/osl/sub-*/lcmv-parc-raw.fif",
#         picks="misc",
#         reject_by_annotation="omit",
#     )
#
# See the :doc:`documentation <../documentation>` for static and dynamic
# network analysis tutorials.
