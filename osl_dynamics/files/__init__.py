"""Data files used by osl-dynamics.

This subpackage provides brain atlases, MNI surfaces, masks, and other
reference files needed by the processing and analysis pipelines. They are not
shipped with the package: each one is downloaded from
`osl-files <https://github.com/OHBA-analysis/osl-files>`_ on first use and
cached, see :py:mod:`osl_dynamics.files._fetch`. Each submodule exposes a
``directory`` attribute pointing to its data directory so files can be
resolved by name at runtime using
:py:func:`~osl_dynamics.files.functions.check_exists`.

Modules
-------
- :py:mod:`~osl_dynamics.files.parcellation` — Volumetric brain
  parcellations (e.g. Glasser, AAL). See :ref:`parcellations` for details.
- :py:mod:`~osl_dynamics.files.mask` — MNI152 brain masks at various
  resolutions.
- :py:mod:`~osl_dynamics.files.surface` — Cortical surface meshes for
  plotting, at three levels of inflation.
- :py:mod:`~osl_dynamics.files.mni152_surfaces` — Pre-extracted MNI152
  skull/scalp surfaces for use with RHINO when no subject MRI is available.
- :py:mod:`~osl_dynamics.files.scanner` — MEG scanner layouts and channel
  name files (CTF-275, Neuromag-306).
- :py:mod:`~osl_dynamics.files.scene` — HCP Workbench scene files for
  cortical surface visualisation.
"""

from osl_dynamics.files import (
    mask,
    mni152_surfaces,
    parcellation,
    scanner,
    scene,
    surface,
)
from osl_dynamics.files.functions import *
