"""
MEG: Source Reconstruction of the MNE Sample Dataset
====================================================

This tutorial performs an 'OSL-style' source reconstruction of the publicly
available `MNE sample dataset
<https://mne.tools/stable/documentation/datasets.html#sample>`_, going from the
raw sensor recording to parcellated source-space time courses:

1. Preprocessing — Filter, downsample, detect bad segments.
2. Structural MRI Conversion — Convert the FreeSurfer MRI to NIfTI for FSL.
3. Surface Extraction — Extract skull/scalp surfaces from the structural MRI.
4. Coregistration — Align MEG sensor space to MRI space with RHINO.
5. Forward Model — Compute the lead field matrix.
6. Source Reconstruction — LCMV beamformer to project sensor data to source space.
7. Parcellation — Reduce voxel data to parcel time courses.

The sample dataset is a good way to try the pipeline: it is downloaded with a
single line of code and contains an auditory/visual task (auditory tones
presented to the left or right ear, and left/right visual field stimuli)
recorded on an Elekta Neuromag system, together with the subject's structural
MRI.

This tutorial mirrors the more detailed
:doc:`MEG preprocessing tutorial </tutorials_build/0-1_meg_preprocessing>` —
see that tutorial for a fuller explanation of each step (including ICA
artefact rejection, which we skip here for brevity).

Prerequisites
^^^^^^^^^^^^^

- `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki>`_ (needed for surface
  extraction).
- `osl-dynamics <https://github.com/OHBA-analysis/osl-dynamics>`_. Note,
  TensorFlow is not required for processing M/EEG (osl-dynamics can be
  installed without TensorFlow using the `envs/osld.yml
  <https://github.com/OHBA-analysis/osl-dynamics/blob/main/envs/osld.yml>`_
  environment).
"""

#%%
# Download the dataset
# ^^^^^^^^^^^^^^^^^^^^
# MNE-Python downloads the sample dataset (~1.6 GB) for us. By default it is
# saved to ``~/mne_data``.
#
# .. code-block:: python
#
#     import mne
#
#     data_path = mne.datasets.sample.data_path()
#     raw_file = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
#     t1_mgz = data_path / "subjects" / "sample" / "mri" / "T1.mgz"

#%%
# Setup and Configuration
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: python
#
#     import os
#     from pathlib import Path
#
#     import mne
#     import nibabel as nib
#     %matplotlib inline
#
#     from osl_dynamics.meeg import preproc, rhino, source_recon, parcellation
#     from osl_dynamics.utils.filenames import OSLFilenames
#
#     # Session info
#     id = "sample_audvis"
#
#     # Paths
#     output_dir = Path("derivatives")
#     plots_dir = Path("plots")
#
#     # Preprocessing parameters
#     resample_freq = 250  # Hz
#     bandpass = (1, 45)  # Hz
#     notch_freqs = [60]  # Hz (US mains frequency)
#
#     # Source reconstruction parameters
#     gridstep = 8  # dipole grid resolution in mm
#     chantypes = ["mag", "grad"]  # Elekta has both magnetometers and gradiometers
#
#     # Parcellation
#     parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"

#%%
# Step 1: Preprocessing
# ^^^^^^^^^^^^^^^^^^^^^
#
# We clean the sensor-level MEG data by notch/bandpass filtering, resampling,
# and detecting bad segments. We keep the stimulus ("stim") channels so we can
# epoch the parcellated data later. Note, this recording also contains EEG,
# which we drop here (the pipeline can source reconstruct EEG, but that
# requires a different forward model).
#
# .. code-block:: python
#
#     raw = mne.io.read_raw_fif(raw_file, preload=True)
#     raw = raw.pick(["meg", "stim"])
#
#     # Filter and resample
#     raw = raw.notch_filter(notch_freqs)
#     raw = raw.filter(
#         l_freq=bandpass[0],
#         h_freq=bandpass[1],
#         method="iir",
#         iir_params={"order": 5, "ftype": "butter"},
#     )
#     raw = raw.resample(sfreq=resample_freq)
#
#     # Bad segment detection
#     raw = preproc.detect_bad_segments(raw, picks="mag")
#     raw = preproc.detect_bad_segments(raw, picks="mag", mode="diff")
#     raw = preproc.detect_bad_segments(raw, picks="grad")
#     raw = preproc.detect_bad_segments(raw, picks="grad", mode="diff")
#
#     # Save preprocessed data
#     preproc_file = output_dir / "preprocessed" / f"{id}_preproc-raw.fif"
#     preproc_file.parent.mkdir(parents=True, exist_ok=True)
#     raw.save(preproc_file, overwrite=True)

#%%
# Step 2: Structural MRI Conversion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The sample dataset provides the structural MRI in FreeSurfer's ``.mgz``
# format, but the FSL tools used for surface extraction need NIfTI. We convert
# with nibabel and then fix the sform code: nibabel labels the converted
# file's sform as "aligned to another file" (code 2), whereas RHINO requires
# code 1 (scanner anatomical) or 4 (MNI).
#
# .. code-block:: python
#
#     smri_file = output_dir / "smri" / "sample-T1.nii.gz"
#     smri_file.parent.mkdir(parents=True, exist_ok=True)
#     nib.save(nib.load(t1_mgz), smri_file)
#     os.system(f"fslorient -setsformcode 1 {smri_file}")

#%%
# Step 3: Surface Extraction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We extract the inner skull, outer skull and scalp surfaces from the
# structural MRI using FSL BET. The output plots overlay each extracted
# surface (yellow line) on the structural MRI — check that each surface
# matches the corresponding anatomical boundary.
#
# .. code-block:: python
#
#     surfaces_dir = str(output_dir / "anat_surfaces" / "sample")
#
#     rhino.extract_surfaces(
#         mri_file=str(smri_file),
#         outdir=surfaces_dir,
#         include_nose=False,
#         show=True,
#     )

#%%
# Step 4: Coregistration
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Coregistration aligns the MEG sensor coordinate system ("head" space) to the
# MRI coordinate system using the digitised fiducials and headshape points
# stored in the FIF file. First we create an ``OSLFilenames`` container to
# keep track of all the pipeline output files.
#
# .. code-block:: python
#
#     fns = OSLFilenames(
#         outdir=str(output_dir / "osl"),
#         id=id,
#         preproc_file=str(preproc_file),
#         surfaces_dir=surfaces_dir,
#     )
#
#     rhino.extract_fiducials_and_headshape_from_fif(fns)
#     rhino.coregister_head_and_mri(
#         fns,
#         use_nose=False,
#         allow_mri_scaling=False,
#         show=True,
#     )

#%%
# The coregistration plot is saved to ``fns.coreg_dir/coreg.png``. Check that
# the headshape points (red dots) sit on the scalp surface and the sensors
# surround the head correctly.

#%%
# Step 5: Forward Model
# ^^^^^^^^^^^^^^^^^^^^^
#
# The forward model (lead field matrix) describes how a dipole at each source
# location projects onto the MEG sensors. We use a Single Layer (single shell)
# head model based on the inner skull surface and a volumetric dipole grid
# with 8 mm spacing.
#
# .. code-block:: python
#
#     rhino.forward_model(fns, model="Single Layer", gridstep=gridstep)

#%%
# Step 6: Source Reconstruction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We use an LCMV (Linearly Constrained Minimum Variance) beamformer to project
# the sensor data into source space. First we compute the beamformer weights
# (spatial filters), then we apply them to the sensor data to get voxel time
# courses in MNI space.
#
# .. code-block:: python
#
#     source_recon.lcmv_beamformer(fns, raw, chantypes=chantypes)
#     voxel_data, voxel_coords = source_recon.apply_lcmv_beamformer(fns, raw)
#     print(f"Voxel data shape: {voxel_data.shape} (voxels x time)")

#%%
# .. note::
#
#     A standard LCMV beamformer assumes sources are uncorrelated, and will
#     partially suppress sources that are correlated across hemispheres — such
#     as the bilateral auditory response in this dataset. See the
#     :doc:`bilateral beamformer tutorial
#     </tutorials_build/0-6_bilateral_beamformer>` for how to handle this.

#%%
# Step 7: Parcellation
# ^^^^^^^^^^^^^^^^^^^^
#
# We reduce the voxel data to 52 parcel time courses using the Glasser
# parcellation, and save them as a FIF file. The ``extra_chans="stim"`` option
# carries the stimulus channels over to the parcellated data, so we can epoch
# it later (e.g. to look at task-evoked responses).
#
# .. code-block:: python
#
#     parcel_data = parcellation.parcellate(
#         fns,
#         voxel_data,
#         voxel_coords,
#         method="spatial_basis",
#         orthogonalisation="symmetric",
#         parcellation_file=parcellation_file,
#     )
#     print(f"Parcel data shape: {parcel_data.shape} (parcels x time)")
#
#     parc_fif = str(output_dir / "osl" / id / "lcmv-parc-raw.fif")
#     parcellation.save_as_fif(
#         parcel_data,
#         raw,
#         extra_chans="stim",
#         filename=parc_fif,
#     )

#%%
# As a sanity check, we plot the power spectral density (PSD) of each parcel.
# We expect ~1/f spectra with an alpha (~10 Hz) peak that is strongest in
# posterior parcels.
#
# .. code-block:: python
#
#     parcellation.save_qc_plots(parc_fif, parcellation_file, output_dir=plots_dir / id)

#%%
# Wrap Up
# ^^^^^^^
#
# - We source reconstructed the MNE sample dataset with an 'OSL-style'
#   pipeline: preprocess, extract surfaces, coregister (RHINO), forward model,
#   LCMV beamformer, parcellate.
# - The parcellated data (``lcmv-parc-raw.fif``) is ready for downstream
#   analysis, e.g. epoching around the auditory/visual events, static spectra,
#   or dynamic network modelling (HMM/DyNeMo).
