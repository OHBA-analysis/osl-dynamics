"""
MEG: Bilateral Beamformer for Correlated Sources
================================================

A standard LCMV beamformer assumes sources are temporally uncorrelated. A
source that is correlated with another source is treated as interference to
suppress, so its reconstructed amplitude is underestimated. The canonical
example is an auditory task: left and right auditory cortex respond
simultaneously (and coherently) to a tone, so a standard beamformer partially
cancels the bilateral auditory response.

The fix is a 'bilateral beamformer' (a special case of the multiple
constrained source beamformer of `Brookes et al. (2007)
<https://doi.org/10.1016/j.neuroimage.2006.09.012>`_): instead of scanning one
dipole at a time, we scan *pairs* of bilaterally symmetric dipoles jointly.
The two dipoles' lead fields are concatenated and a joint set of weights is
computed, so neither dipole's correlated twin is suppressed.

In this tutorial we:

1. Source reconstruct the MNE sample dataset (an auditory/visual task) with a
   standard beamformer and with the bilateral beamformer.
2. Epoch the parcellated data around the auditory tones.
3. Compare the evoked responses in the left and right auditory cortex parcels.

Prerequisites
^^^^^^^^^^^^^

Run the :doc:`MNE sample dataset tutorial </tutorials_build/0-5_mne_sample_data>`
first — we reuse its preprocessed data, surfaces, coregistration and forward
model here.
"""

#%%
# Setup
# ^^^^^
# We use the same paths and ``OSLFilenames`` container as the
# :doc:`previous tutorial </tutorials_build/0-5_mne_sample_data>`.
#
# .. code-block:: python
#
#     from pathlib import Path
#
#     import mne
#     import numpy as np
#     import matplotlib.pyplot as plt
#     %matplotlib inline
#
#     from osl_dynamics.meeg import source_recon, parcellation
#     from osl_dynamics.utils.filenames import OSLFilenames
#
#     id = "sample_audvis"
#     output_dir = Path("derivatives")
#     preproc_file = output_dir / "preprocessed" / f"{id}_preproc-raw.fif"
#     surfaces_dir = str(output_dir / "anat_surfaces" / "sample")
#     parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"
#
#     fns = OSLFilenames(
#         outdir=str(output_dir / "osl"),
#         id=id,
#         preproc_file=str(preproc_file),
#         surfaces_dir=surfaces_dir,
#     )
#
#     raw = mne.io.read_raw_fif(preproc_file, preload=True)

#%%
# Standard beamformer
# ^^^^^^^^^^^^^^^^^^^
# First, the standard LCMV pipeline: beamform, apply, parcellate, save.
#
# Note, we use ``orthogonalisation=None`` here (rather than ``"symmetric"``).
# Symmetric orthogonalisation removes all zero-lag correlations between
# parcels — which is exactly the correlated bilateral signal we want to
# study. Leakage correction should not be used when analysing zero-lag
# correlated task responses.
#
# .. code-block:: python
#
#     source_recon.lcmv_beamformer(fns, raw, chantypes=["mag", "grad"])
#     voxel_data, voxel_coords = source_recon.apply_lcmv_beamformer(fns, raw)
#     parcel_data = parcellation.parcellate(
#         fns,
#         voxel_data,
#         voxel_coords,
#         method="spatial_basis",
#         orthogonalisation=None,
#         parcellation_file=parcellation_file,
#     )
#     parc_fif = str(output_dir / "osl" / id / "lcmv-parc-raw.fif")
#     parcellation.save_as_fif(parcel_data, raw, filename=parc_fif, extra_chans="stim")

#%%
# Bilateral beamformer
# ^^^^^^^^^^^^^^^^^^^^
# Now the bilateral beamformer. We simply pass ``use_bilateral_pairs=True``:
#
# - Dipoles are transformed to MNI space, mirrored across the midline
#   (x = 0), and greedily paired with the closest dipole in the opposite
#   hemisphere within ``bilateral_tol`` mm. By default this is set to half
#   the dipole grid spacing (here: 8 mm grid, so 4 mm), which is usually what
#   you want.
# - Dipoles within ``bilateral_tol_midline`` of the midline, and dipoles with
#   no match, are beamformed as usual (defaults to ``bilateral_tol``).
# - Joint weights are computed for each pair by concatenating the two lead
#   fields.
#
# We re-point ``fns.filters`` first so we don't overwrite the standard
# filters.
#
# .. code-block:: python
#
#     fns.filters = f"{fns.src_dir}/filters-lcmv-bilateral.h5"
#
#     source_recon.lcmv_beamformer(
#         fns,
#         raw,
#         chantypes=["mag", "grad"],
#         use_bilateral_pairs=True,
#     )
#     voxel_data, voxel_coords = source_recon.apply_lcmv_beamformer(fns, raw)
#     parcel_data = parcellation.parcellate(
#         fns,
#         voxel_data,
#         voxel_coords,
#         method="spatial_basis",
#         orthogonalisation=None,
#         parcellation_file=parcellation_file,
#     )
#     bilateral_parc_fif = str(output_dir / "osl" / id / "lcmv-bilateral-parc-raw.fif")
#     parcellation.save_as_fif(
#         parcel_data, raw, filename=bilateral_parc_fif, extra_chans="stim"
#     )

#%%
# The beamformer prints how many pairs were found and saves a QC plot of the
# dipole pairing to ``fns.src_dir/bilateral_dipoles.png`` (this also appears
# in the QC report under the "Beamforming" tab). Red lines connect paired
# dipoles, blue dots are midline dipoles, grey dots are unpaired dipoles. You
# can also generate this plot manually with
# ``source_recon.plot_bilateral_pairs(fns, show=True)``.
#
# A few things to be aware of:
#
# - The bilateral beamformer requires a scalar beamformer
#   (``pick_ori='max-power'`` or ``'max-power-pre-weight-norm'``, the
#   default).
# - ``weight_norm='unit-noise-gain-invariant'`` (the default) computes the
#   weights from the lead fields alone, which would discard the joint
#   denominator — so ``weight_norm='unit-noise-gain'`` is used automatically
#   (for a scalar beamformer they give identical weights, up to sign).

#%%
# Epoch the auditory responses
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The stim channels were carried through to the parcellated FIF files, so we
# can epoch the parcel data directly. Auditory tones are event IDs 1 (left
# ear) and 2 (right ear).
#
# .. code-block:: python
#
#     def epoch_parc(parc_fif):
#         parc_raw = mne.io.read_raw_fif(parc_fif, preload=True)
#         events = mne.find_events(parc_raw, min_duration=0.005)
#         epochs = mne.Epochs(
#             parc_raw,
#             events,
#             event_id={"auditory/left": 1, "auditory/right": 2},
#             tmin=-0.2,
#             tmax=0.5,
#             baseline=(None, 0),
#             picks="misc",
#         )
#         return epochs
#
#     epochs_standard = epoch_parc(parc_fif)
#     epochs_bilateral = epoch_parc(bilateral_parc_fif)

#%%
# Compare the evoked responses
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We find the parcels closest to left and right primary auditory cortex
# (MNI approximately (-52, -19, 7) and (52, -19, 7) mm) and compare the
# evoked response from the two beamformers.
#
# Note, beamformer time courses have an arbitrary sign, so we flip each
# evoked response such that its largest deflection in the M100 window is
# positive before plotting.
#
# .. code-block:: python
#
#     # Find the auditory parcels
#     parc = parcellation.Parcellation(parcellation_file)
#     centers = np.array(parc.roi_centers())
#     left_parcel = np.argmin(np.linalg.norm(centers - [-52, -19, 7], axis=1))
#     right_parcel = np.argmin(np.linalg.norm(centers - [52, -19, 7], axis=1))
#
#     def get_evoked(epochs, parcel):
#         evoked = epochs.get_data().mean(axis=0)[parcel]
#         t = epochs.times
#         m100 = (t >= 0.05) & (t <= 0.15)
#         sign = np.sign(evoked[m100][np.argmax(np.abs(evoked[m100]))])
#         return sign * evoked
#
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#     for ax, parcel, title in [
#         (axes[0], left_parcel, "Left auditory cortex"),
#         (axes[1], right_parcel, "Right auditory cortex"),
#     ]:
#         ax.plot(
#             epochs_standard.times,
#             get_evoked(epochs_standard, parcel),
#             label="Standard",
#         )
#         ax.plot(
#             epochs_bilateral.times,
#             get_evoked(epochs_bilateral, parcel),
#             label="Bilateral",
#         )
#         ax.axvline(0, color="k", lw=0.5)
#         ax.set_xlabel("Time (s)")
#         ax.set_title(title)
#         ax.legend()

#%%
# Both hemispheres show a clear evoked response peaking around 100 ms after
# the tone (the M100). Crucially, the bilateral beamformer recovers a larger
# response than the standard beamformer — the amplitude that the standard
# beamformer suppressed because the two hemispheres' responses are
# correlated.

#%%
# Wrap Up
# ^^^^^^^
#
# - A standard LCMV beamformer suppresses correlated sources; for tasks with
#   bilateral responses (auditory in particular) use
#   ``use_bilateral_pairs=True``.
# - The pairing tolerance ``bilateral_tol`` defaults to half the dipole grid
#   spacing, which is usually what you want.
# - Use ``orthogonalisation=None`` when analysing zero-lag correlated
#   responses — symmetric orthogonalisation would remove them.
# - Check the dipole pairing QC plot (``bilateral_dipoles.png``, or the
#   "Beamforming" tab of the QC report).
