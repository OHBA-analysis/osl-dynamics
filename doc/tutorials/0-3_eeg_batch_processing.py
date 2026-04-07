"""
LEMON EEG Batch Processing
===========================

This tutorial shows how to process a large EEG dataset in batch using
osl-dynamics' parallel processing utilities. We use the
`LEMON dataset <https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/>`_
as an example.

The pipeline mirrors the single-session tutorial
(:doc:`MEG Processing <0-1_meg_preprocessing>`) but adapted for EEG data.
Each step is wrapped in a function that is dispatched across sessions
using ``osl_dynamics.meeg.parallel.run``.

Prerequisites
^^^^^^^^^^^^^

- `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki>`_ (needed for surface extraction).
- `osl-dynamics <https://github.com/OHBA-analysis/osl-dynamics>`_.

Input Data
^^^^^^^^^^

The LEMON dataset contains resting-state EEG recordings from ~200
subjects. Each subject also has a structural MRI (MP2RAGE) and an
EEG localizer file (``.mat``) containing digitised electrode positions
and fiducials. The data is organised as::

    lemon/
    ├── sub-*/
    │   ├── RSEEG/sub-*.vhdr
    │   └── ses-01/anat/sub-*_ses-01_inv-2_mp2rage.nii.gz
    └── EEG_MPILMBB_LEMON/EEG_Localizer_BIDS_ID/
        └── sub-*/sub-*.mat

The output of this script is written to ``derivatives/``.
"""

#%%
# Step 0: Find File Paths
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# First, we locate all raw EEG files and their corresponding structural
# MRIs and localizer files, keeping only subjects that have all three.
# The results are saved to a CSV file that acts as the session manifest
# for all subsequent steps.
#
# .. code-block:: python
#
#     from pathlib import Path
#     import pandas as pd
#
#     # Raw data directory
#     raw_dir = "/ohba/pi/mwoolrich/raw_datasets/lemon"
#     localizer_dir = f"{raw_dir}/EEG_MPILMBB_LEMON/EEG_Localizer_BIDS_ID"
#
#     # Get subjects that have localizer data
#     localizer_subjects = set()
#     for loc_dir in sorted(Path(localizer_dir).glob("sub-*")):
#         localizer_subjects.add(loc_dir.name)
#
#     rows = []
#     for sub_dir in sorted(Path(raw_dir).glob("sub-*")):
#         subject = sub_dir.name
#         if subject not in localizer_subjects:
#             continue
#
#         raw_file = sub_dir / "RSEEG" / f"{subject}.vhdr"
#         mri_file = sub_dir / "ses-01" / "anat" / f"{subject}_ses-01_inv-2_mp2rage.nii.gz"
#         loc_file = f"{localizer_dir}/{subject}/{subject}.mat"
#
#         if not raw_file.exists():
#             continue
#         if not mri_file.exists():
#             continue
#
#         rows.append({
#             "id": subject,
#             "subject": subject,
#             "raw_file": str(raw_file),
#             "mri_file": str(mri_file),
#             "localizer_file": loc_file,
#         })
#
#     df = pd.DataFrame(rows)
#     print(f"Found {len(df)} subjects")
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
# Key differences from MEG preprocessing:
#
# - Data is loaded from BrainVision (``.vhdr``) format.
# - A channel montage is set from the localizer ``.mat`` file, which
#   contains digitised electrode positions and fiducials.
# - A synthetic HEOG channel is created from F7 - F8.
# - The first 15 seconds are cropped (equipment settling time).
# - Bad channels are detected, interpolated, and an average reference
#   is applied.
#
# Each session is independently:
#
# - Notch filtered at 50 and 100 Hz.
# - Bandpass filtered (0.5-80 Hz, 5th-order Butterworth).
# - Downsampled to 250 Hz.
# - Checked for bad channels and segments.
# - Bad channels interpolated.
# - Average referenced.
# - QC plots saved.
#
# .. code-block:: python
#
#     from pathlib import Path
#     import numpy as np
#     import pandas as pd
#     import mne
#     from scipy import io
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
#     def set_channel_montage(raw, localizer_file):
#         """Set channel montage from localizer .mat file."""
#         X = io.loadmat(localizer_file, simplify_cells=True)
#         ch_pos = {}
#         for i in range(len(X["Channel"]) - 1):  # final channel is reference
#             key = X["Channel"][i]["Name"].split("_")[2]
#             if key[:2] == "FP":
#                 key = "Fp" + key[2]
#             ch_pos[key] = X["Channel"][i]["Loc"]
#         hp = X["HeadPoints"]["Loc"]
#         nas = np.mean([hp[:, 0], hp[:, 3]], axis=0)
#         lpa = np.mean([hp[:, 1], hp[:, 4]], axis=0)
#         rpa = np.mean([hp[:, 2], hp[:, 5]], axis=0)
#         dig = mne.channels.make_dig_montage(
#             ch_pos=ch_pos, nasion=nas, lpa=lpa, rpa=rpa,
#         )
#         raw.set_montage(dig)
#         return raw
#
#
#     def create_heog(raw):
#         """Create synthetic HEOG channel from F7-F8."""
#         heog = raw.get_data(picks="F7") - raw.get_data(picks="F8")
#         info = mne.create_info(["HEOG"], raw.info["sfreq"], ["eog"])
#         raw.add_channels([mne.io.RawArray(heog, info)], force_update_info=True)
#         return raw
#
#
#     def process_session(session, logger):
#         """Preprocess a single session."""
#
#         logger.log("Loading raw data...")
#         raw = mne.io.read_raw_brainvision(session["raw_file"], preload=True)
#
#         logger.log("Setting channel montage...")
#         raw = set_channel_montage(raw, session["localizer_file"])
#
#         logger.log("Creating HEOG channel...")
#         raw = create_heog(raw)
#         raw.set_channel_types({"VEOG": "eog", "HEOG": "eog"})
#
#         logger.log("Cropping first 15 seconds...")
#         raw = raw.crop(tmin=15)
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
#         logger.log("Detecting bad channels...")
#         raw = preproc.detect_bad_channels(raw, picks="eeg")
#
#         logger.log("Detecting bad segments...")
#         raw = preproc.detect_bad_segments(raw, picks="eeg")
#         raw = preproc.detect_bad_segments(raw, picks="eeg", mode="diff")
#
#         logger.log("Interpolating bad channels...")
#         raw = raw.interpolate_bads()
#
#         logger.log("Setting EEG reference...")
#         raw = raw.set_eeg_reference(projection=True)
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
# The coregistration step includes a dataset-specific function to adjust
# headshape points and fiducials for EEG coregistration. The LEMON
# localizer positions need to be shrunk by 5% and the fiducials shifted
# down and back by 1 cm to account for systematic offsets. You will
# likely need to adapt ``fix_headshape_fiducials`` for your own dataset.
#
# Note for EEG we pass ``include_eeg_as_headshape=True`` when extracting
# fiducials and ``use_headshape=False`` during coregistration.
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
#     def fix_headshape_fiducials(fns, logger):
#         """Shrink headshape and shift fiducials for EEG coregistration."""
#
#         fns = fns.coreg
#
#         hs = np.loadtxt(fns.head_headshape_file)
#         nas = np.loadtxt(fns.head_nasion_file)
#         lpa = np.loadtxt(fns.head_lpa_file)
#         rpa = np.loadtxt(fns.head_rpa_file)
#
#         # Shrink headshape points by 5%
#         hs *= 0.95
#
#         # Move fiducials down 1cm
#         nas[2] -= 10
#         lpa[2] -= 10
#         rpa[2] -= 10
#
#         # Move fiducials back 1cm
#         nas[1] -= 10
#         lpa[1] -= 10
#         rpa[1] -= 10
#
#         logger.log(f"Overwriting {fns.head_headshape_file}")
#         np.savetxt(fns.head_nasion_file, nas)
#         np.savetxt(fns.head_lpa_file, lpa)
#         np.savetxt(fns.head_rpa_file, rpa)
#         np.savetxt(fns.head_headshape_file, hs)
#
#
#     def process_session(session, logger):
#         """Coregister a single session."""
#
#         preproc_file = output_dir / "preprocessed" / f"{session['id']}_preproc-raw.fif"
#         surfaces_dir = str(output_dir / "anat_surfaces" / session["subject"])
#
#         fns = OSLFilenames(
#             outdir=str(output_dir / "osl"),
#             id=session["id"],
#             preproc_file=str(preproc_file),
#             surfaces_dir=surfaces_dir,
#         )
#
#         logger.log("Extracting fiducials and headshape...")
#         rhino.extract_fiducials_and_headshape_from_fif(
#             fns, include_eeg_as_headshape=True,
#         )
#
#         logger.log("Fixing headshape and fiducials...")
#         fix_headshape_fiducials(fns, logger)
#
#         logger.log("Coregistering EEG to MRI...")
#         rhino.coregister_head_and_mri(
#             fns, use_nose=False, use_headshape=False,
#         )
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
# each session. Key differences from MEG:
#
# - A Triple Layer forward model is used (instead of Single Layer).
# - The ``eeg=True`` flag is passed to the forward model.
# - Only the ``eeg`` channel type is used for the beamformer.
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
#         rhino.forward_model(fns, model="Triple Layer", gridstep=8, eeg=True)
#
#         logger.log("Computing LCMV beamformer...")
#         source_recon.lcmv_beamformer(fns, chantypes=["eeg"], rank={"eeg": 54})
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
# for each session are written to the ``logs/`` directory -- check these
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
