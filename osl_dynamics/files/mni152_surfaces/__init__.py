"""MNI152 standard brain surfaces.

Pre-extracted surfaces from the MNI152 T1 1mm brain using FSL BET. These files
are downloaded on first use, see :py:mod:`osl_dynamics.files._fetch`.

Files
-----
- smri.nii.gz: MNI152 T1 structural MRI.
- inskull_mesh.nii.gz, inskull_mesh.vtk, inskull.png
- outskull_mesh.nii.gz, outskull_mesh.vtk, outskull.png
- outskin_mesh.nii.gz, outskin_mesh.vtk, outskin.png
- mni2mri_flirt_xform.txt: FLIRT transformation matrix (MNI to MRI).
- mni_mri-trans.fif: MNE transformation (MNI to MRI).
"""

from osl_dynamics.files import _fetch

_SUBDIRECTORY = "mni152_surfaces"


def file(name: str) -> str:
    """Path to one surface file, downloading it if needed."""
    return str(_fetch.fetch_file(f"{_SUBDIRECTORY}/{name}"))


def __getattr__(name):
    if name == "path":
        return _fetch.fetch_directory(_SUBDIRECTORY)
    if name == "directory":
        return str(_fetch.fetch_directory(_SUBDIRECTORY))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
