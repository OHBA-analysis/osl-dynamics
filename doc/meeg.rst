M/EEG Processing
================

The models in osl-dynamics are trained on **parcel time courses**: one time course per brain region, reconstructed from the sensors. :mod:`osl_dynamics.meeg <osl_dynamics.meeg>` takes raw recordings all the way there. This page describes the steps and the software behind them; the :doc:`MEG Preprocessing tutorial <tutorials_build/0-1_meg_preprocessing>` runs them on one session, the :doc:`batch processing tutorials <tutorial_sections/meeg_processing>` on many sessions in parallel.

.. toctree::
   :maxdepth: 1
   :hidden:

   Parcellations <parcellations/index>

Software
--------

osl-dynamics combines the standard tools into one pipeline with consistent file handling and quality-control outputs:

- `MNE-Python <https://mne.tools>`_ reads the raw data and provides filtering, ICA, the BEM surfaces, the forward model and the core of the LCMV beamformer. Every stage works on MNE :class:`~mne.io.Raw` or :class:`~mne.Epochs` objects, so MNE functions can be mixed in anywhere.
- `FSL <https://fsl.fmrib.ox.ac.uk>`_ processes the structural MRI (BET for the surfaces, FLIRT and FNIRT for the MNI registration). Only the surface extraction step needs it, found through :code:`FSLDIR`.
- `mne-icalabel <https://mne.tools/mne-icalabel>`_ labels ICA components (MEGNet for MEG, ICLabel for EEG).
- Elekta/MEGIN MaxFilter (licensed, optional) for Elekta data, and the adaptive multipole model (AMM) for OPM data.
- `osl-files <https://github.com/OHBA-analysis/osl-files>`_ holds the parcellations, masks and templates, downloaded on first use.

Pipeline
--------

Each step reads the files written by the steps before it and writes its own at paths held by a :class:`~osl_dynamics.meeg.session.Session`, so any step can be rerun on its own and every intermediate file inspected.

.. list-table::
   :header-rows: 1
   :widths: 18 52 15 15

   * - Step
     - What it does
     - Software
     - Functions
   * - Interference rejection (optional)
     - Signal space separation of Elekta data, or AMM for OPM data.
     - MaxFilter, osl-dynamics
     - :func:`~osl_dynamics.meeg.maxfilter.run_maxfilter`, :func:`~osl_dynamics.meeg.amm.apply_amm`
   * - Preprocessing
     - Filtering, resampling, bad segment and bad channel detection, and ICA rejection of cardiac and ocular artefacts.
     - MNE-Python, mne-icalabel
     - :mod:`~osl_dynamics.meeg.preproc`
   * - Surface extraction
     - Scalp, inner skull and brain surfaces from the T1 MRI, and the MRI registered to MNI space (affine, then nonlinear if it improves the fit).
     - FSL
     - :func:`~osl_dynamics.meeg.rhino.extract_surfaces`
   * - Coregistration
     - RHINO aligns the MEG head coordinates to the MRI: the fiducials first, then the digitised headshape points to the scalp surface.
     - osl-dynamics
     - :func:`~osl_dynamics.meeg.rhino.coregister_head_and_mri`
   * - Forward model
     - A single-layer (MEG) or triple-layer (EEG) BEM, with sources on an MNI grid so that every subject has sources at the same coordinates as the parcellation voxels.
     - MNE-Python
     - :func:`~osl_dynamics.meeg.rhino.forward_model`
   * - Source reconstruction
     - An LCMV beamformer: a spatial filter per source estimated from the data covariance, with the orientation of maximum power. Filters can be applied to any data from the session or to a single coordinate (virtual electrode).
     - MNE-Python, osl-dynamics
     - :func:`~osl_dynamics.meeg.source_recon.lcmv_beamformer`, :func:`~osl_dynamics.meeg.source_recon.apply_lcmv_beamformer`, :func:`~osl_dynamics.meeg.source_recon.virtual_electrodes`
   * - Parcellation
     - One time course per parcel of an atlas: the first principal component of the parcel's voxels, computed straight from the sensor data. See the :doc:`parcellations page <parcellations/index>`.
     - osl-dynamics
     - :func:`~osl_dynamics.meeg.parcellation.parcellate_lcmv`, :func:`~osl_dynamics.meeg.parcellation.parcellate`
   * - Orthogonalisation (optional)
     - Symmetric orthogonalisation [1]_ removes the zero-lag correlations between parcels caused by spatial leakage, needed for connectivity measures such as amplitude envelope correlation.
     - osl-dynamics
     - :code:`orthogonalisation="symmetric"` in the parcellation step
   * - Sign flipping
     - The sign of a beamformed parcel time course is arbitrary and differs between sessions. Each session's parcels are flipped to match a template covariance, so that sessions can be pooled for group-level models.
     - osl-dynamics
     - :meth:`Data.align_channel_signs <osl_dynamics.data.Data.align_channel_signs>` (:doc:`tutorial <tutorials_build/0-2_sign_flipping>`)
   * - Quality control
     - PSDs and power maps of the parcel time courses, and an HTML report with one row per session next to its QC plots.
     - osl-dynamics
     - :func:`~osl_dynamics.meeg.parcellation.save_qc_plots`, :func:`~osl_dynamics.meeg.report.generate_report`

The output is a FIF file with one channel per parcel, which :class:`~osl_dynamics.data.Data` loads directly. :func:`~osl_dynamics.meeg.parallel.run` runs any step over many sessions in parallel.

References
----------

.. [1] Colclough, G. L. et al. (2015). A symmetric multivariate leakage correction for MEG connectomes. NeuroImage, 117, 439-448.
