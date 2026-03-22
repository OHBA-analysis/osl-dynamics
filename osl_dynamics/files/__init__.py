"""Bundled data files shipped with osl-dynamics.

This subpackage provides brain atlases, MNI surfaces, masks, and other
reference files needed by the processing and analysis pipelines. Each
submodule exposes a ``directory`` attribute pointing to its data directory
so files can be resolved by name at runtime using
:py:func:`~osl_dynamics.files.functions.check_exists`.

Modules
-------
- :py:mod:`~osl_dynamics.files.parcellation` — Volumetric brain
  parcellations (e.g. Glasser, AAL). See :ref:`parcellations` for details.
- :py:mod:`~osl_dynamics.files.mask` — MNI152 brain masks at various
  resolutions and cortical surface meshes for plotting.
- :py:mod:`~osl_dynamics.files.mni152_surfaces` — Pre-extracted MNI152
  skull/scalp surfaces for use with RHINO when no subject MRI is available.
- :py:mod:`~osl_dynamics.files.scanner` — MEG scanner layouts and channel
  name files (CTF-275, Neuromag-306).
- :py:mod:`~osl_dynamics.files.scene` — HCP Workbench scene files for
  cortical surface visualisation.
"""

from osl_dynamics.files import mask, mni152_surfaces, parcellation, scanner, scene
from osl_dynamics.files.functions import *
