"""MNI152 brain masks and surfaces.

- MNI152_T1_1mm_brain.nii.gz
- MNI152_T1_2mm_brain.nii.gz
- MNI152_T1_5mm_brain.nii.gz
- MNI152_T1_8mm_brain.nii.gz

These files are downloaded on first use, see
:py:mod:`osl_dynamics.files._fetch`.

The masks define the MNI voxel grid :func:`osl_dynamics.meeg.source_recon.apply_lcmv_beamformer` outputs voxel data on. Masks are not needed to plot brain maps, parcel values are plotted on the voxel grid of the parcellation file.
"""

from osl_dynamics.files import _fetch

_SUBDIRECTORY = "mask"


def file(name: str) -> str:
    """Path to one mask file, downloading it if needed."""
    return str(_fetch.fetch_file(f"{_SUBDIRECTORY}/{name}"))


def __getattr__(name):
    if name == "path":
        return _fetch.fetch_directory(_SUBDIRECTORY)
    if name == "directory":
        return str(_fetch.fetch_directory(_SUBDIRECTORY))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
