M/EEG Processing
================

The models in osl-dynamics are trained on **parcel time courses**: source-space data reduced to one time course per brain region. The :mod:`osl_dynamics.meeg <osl_dynamics.meeg>` subpackage takes raw sensor recordings all the way there. This page describes the steps, what each one does and which software does the work. The :doc:`MEG Preprocessing tutorial <tutorials_build/0-1_meg_preprocessing>` runs the whole pipeline on one session; the :doc:`MEG <tutorials_build/0-3_meg_batch_processing>` and :doc:`EEG <tutorials_build/0-4_eeg_batch_processing>` batch processing tutorials and the `batch scripts <https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples/meg_preproc>`_ run it on many sessions in parallel, and `Canonical-HMM-Networks <https://github.com/OHBA-analysis/Canonical-HMM-Networks>`_ has start-to-end examples for Elekta, CTF, OPM and EEG data.

.. toctree::
   :maxdepth: 1
   :hidden:

   Parcellations <parcellations/index>

Software
--------

osl-dynamics does not reimplement the standard tools. It combines them into one pipeline with consistent file handling and quality-control outputs:

- `MNE-Python <https://mne.tools>`_ reads the raw data (FIF, CTF, EEG formats), and provides the filtering, resampling, ICA, boundary element (BEM) surfaces, forward model and the core of the LCMV beamformer. Every stage works on MNE :class:`~mne.io.Raw` or :class:`~mne.Epochs` objects, so MNE functions can be mixed in at any point.
- `FSL <https://fsl.fmrib.ox.ac.uk>`_ processes the structural MRI: BET extracts the scalp, inner skull and brain surfaces, FLIRT and FNIRT register the MRI to MNI space. FSL is only needed for the surface extraction step; the pipeline looks for it in the :code:`FSLDIR` environment variable.
- `mne-icalabel <https://mne.tools/mne-icalabel>`_ labels ICA components automatically (MEGNet for MEG, ICLabel for EEG).
- Elekta/MEGIN `MaxFilter <https://www.megin.fi>`_ (licensed, optional) for signal space separation of Elekta data, and the adaptive multipole model (AMM, ported from SPM) for interference rejection in OPM data.
- `osl-files <https://github.com/OHBA-analysis/osl-files>`_ holds the parcellations, masks and templates, which are downloaded on first use.

Pipeline
--------

Each step reads the files written by the ones before it and writes its own at paths held by a :class:`~osl_dynamics.meeg.session.Session`, so a step can be rerun on its own and every intermediate file can be inspected. The steps, in order:

.. list-table::
   :header-rows: 1
   :widths: 18 52 15 15

   * - Step
     - What it does
     - Software
     - Functions
   * - Interference rejection (optional)
     - Signal space separation (SSS/tSSS) of Elekta data, or AMM for OPM data, before any other processing.
     - MaxFilter, osl-dynamics
     - :func:`~osl_dynamics.meeg.maxfilter.run_maxfilter`, :func:`~osl_dynamics.meeg.amm.apply_amm`
   * - Preprocessing
     - Sensor-level cleaning: band-pass and notch filtering, resampling, bad segment detection (outlier variance in sliding windows, with the generalised extreme Studentized deviate test), bad channel detection, and ICA artefact rejection of cardiac and ocular components (automatic labelling, or correlation with ECG/EOG channels). Saves QC plots.
     - MNE-Python, mne-icalabel
     - :mod:`~osl_dynamics.meeg.preproc`
   * - Surface extraction
     - From the T1 structural MRI: aligns the MRI with the MNI axes, skull-strips it, extracts the scalp, inner skull and brain surfaces, and registers the MRI to MNI space (affine, then nonlinear; the nonlinear registration is kept only if it improves the fit). Plots of the surfaces and the MNI registration are saved for checking.
     - FSL
     - :func:`~osl_dynamics.meeg.rhino.extract_surfaces`
   * - Coregistration
     - RHINO aligns the MEG head coordinate system to the MRI: first the digitised fiducials (nasion, LPA, RPA) to the fiducials predicted from MNI space, then the digitised headshape points to the scalp surface with an iterative closest point fit. The MRI can be scaled when a template MRI is used. Saves a coregistration plot and the fit error.
     - osl-dynamics
     - :func:`~osl_dynamics.meeg.rhino.coregister_head_and_mri`
   * - Forward model
     - The lead field from each source to each sensor, with a single-layer BEM (MEG) or triple-layer BEM (EEG). Sources are placed on an MNI grid (default 8 mm) transformed into the subject's head, so every subject has sources at the same MNI coordinates, matching the voxels of the parcellation files.
     - MNE-Python
     - :func:`~osl_dynamics.meeg.rhino.forward_model`
   * - Source reconstruction
     - A linearly constrained minimum variance (LCMV) beamformer: a spatial filter per source that passes activity from that location and suppresses the rest. The data covariance is estimated from the preprocessed data (optionally within a frequency band) with its rank reduced to that of the data, and the source orientation is chosen to maximise power. The filters are saved and can be applied to any data from the session, or to a single MNI coordinate (virtual electrode). The bilateral beamformer handles correlated sources in the two hemispheres.
     - MNE-Python, osl-dynamics
     - :func:`~osl_dynamics.meeg.source_recon.lcmv_beamformer`, :func:`~osl_dynamics.meeg.source_recon.apply_lcmv_beamformer`, :func:`~osl_dynamics.meeg.source_recon.virtual_electrodes`
   * - Parcellation
     - Reduces the source space to one time course per parcel of an atlas: the first principal component of the voxels in the parcel, weighted by the parcellation (or the voxel nearest the centroid). Usually done straight from the sensor data without storing voxel time courses. See the :doc:`parcellations page <parcellations/index>` for the atlases available.
     - osl-dynamics
     - :func:`~osl_dynamics.meeg.parcellation.parcellate_lcmv`, :func:`~osl_dynamics.meeg.parcellation.parcellate`
   * - Orthogonalisation
     - Beamformer spatial leakage induces zero-lag correlations between nearby parcels. Symmetric orthogonalisation [1]_ removes them (all parcels at once, keeping the time courses as close as possible to the originals) so that amplitude envelope correlations and other connectivity measures are not dominated by leakage. Optional, :code:`orthogonalisation="symmetric"` in the parcellation step.
     - osl-dynamics
     - :func:`~osl_dynamics.meeg.parcellation.parcellate_lcmv`
   * - Sign flipping
     - The sign of a beamformed parcel time course is arbitrary, and differs between sessions, which scrambles the signs of the cross-parcel covariances that group-level dynamic network models rely on. Each session's parcels are flipped to maximise the correlation of its (time-delay embedded) covariance with a template. Done when the data are loaded for modelling.
     - osl-dynamics
     - :meth:`Data.align_channel_signs <osl_dynamics.data.Data.align_channel_signs>` (:doc:`tutorial <tutorials_build/0-2_sign_flipping>`)
   * - Quality control
     - PSDs and power maps of the parcel time courses, and an HTML report with one row per session (bad segments, MNI registration and coregistration metrics) next to the QC plots of the selected session.
     - osl-dynamics
     - :func:`~osl_dynamics.meeg.parcellation.save_qc_plots`, :func:`~osl_dynamics.meeg.report.generate_report`

The parcel time courses are saved as a FIF file with one channel per parcel (plus any stimulus channels), which :class:`~osl_dynamics.data.Data` loads directly for the :doc:`data preparation <tutorials_build/1-2_data_prepare_meg>` and modelling steps. :func:`~osl_dynamics.meeg.parallel.run` runs any of these steps over many sessions in parallel.

What to check
-------------

Each step saves a plot or a number for exactly this purpose. The ones worth looking at for every session are the bad segment and channel summary and the ICA components (preprocessing), the surfaces and MNI registration overlays (surface extraction), the coregistration plot and fit error (headshape points should sit on the scalp), and the parcel PSDs (an alpha peak strongest in posterior parcels is the usual sanity check). The QC report collects these so that the worst sessions can be sorted to the top.

References
----------

.. [1] Colclough, G. L. et al. (2015). A symmetric multivariate leakage correction for MEG connectomes. NeuroImage, 117, 439-448.
