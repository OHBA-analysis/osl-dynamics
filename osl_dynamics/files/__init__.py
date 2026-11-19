"""Data files used by osl-dynamics.

This subpackage provides brain atlases, MNI surfaces, masks, and other
reference files needed by the processing and analysis pipelines. They are not
shipped with the package: each one is downloaded from
`osl-files <https://github.com/OHBA-analysis/osl-files>`_ on first use and
cached. Each submodule exposes a ``directory`` attribute pointing to its data
directory so files can be resolved by name at runtime using
:py:func:`~osl_dynamics.files.functions.check_exists`.

Any file in osl-files can be downloaded with
:py:func:`~osl_dynamics.files.fetch_file`, which returns the path to the
cached copy::

    from osl_dynamics.files import fetch_file

    path = fetch_file("parcellation/atlas-AAL_nparc-78_space-MNI_res-8x8x8.nii.gz")

:py:func:`~osl_dynamics.files.fetch_directory` downloads a whole directory
and :py:func:`~osl_dynamics.files.fetch_all` downloads everything, e.g. before
working offline. A failed download raises
:py:class:`~osl_dynamics.files.DownloadError`.

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
  cortical surface visualization.
"""

from osl_dynamics.files import (
    mask,
    mni152_surfaces,
    parcellation,
    scanner,
    scene,
    surface,
)
from osl_dynamics.files._fetch import (
    DownloadError,
    fetch_all,
    fetch_directory,
    fetch_file,
)
from osl_dynamics.files.functions import check_exists
