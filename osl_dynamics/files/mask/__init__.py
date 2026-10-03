"""Mask files.

- MNI152_T1_1mm_brain.nii.gz
- MNI152_T1_2mm_brain.nii.gz
- MNI152_T1_5mm_brain.nii.gz
- MNI152_T1_8mm_brain.nii.gz
- ft_8mm_brain_mask.nii.gz

These files are downloaded on first use, see
:py:mod:`osl_dynamics.files._fetch`.

Mask files can be resampled in FSL, e.g. to resample the 1x1x1 mm grid MNI mask file to a 5x5x5 mm grid:

.. code::

    flirt -in $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -out MNI152_T1_5mm_brain.nii.gz -applyisoxfm 5

When plotting brain maps the resolution of the mask file must match the resolution of the parcellation file. Parcellation files can also be resampled using the code above.
"""

import nibabel as nib
import numpy as np

from osl_dynamics.files import _fetch

_SUBDIRECTORY = "mask"

_SURFACES = {
    "surf_left": "ParcellationPilot.L.midthickness.32k_fs_LR.surf.gii",
    "surf_right": "ParcellationPilot.R.midthickness.32k_fs_LR.surf.gii",
    "surf_left_inf": "ParcellationPilot.L.inflated.32k_fs_LR.surf.gii",
    "surf_right_inf": "ParcellationPilot.R.inflated.32k_fs_LR.surf.gii",
    "surf_left_vinf": "ParcellationPilot.L.very_inflated.32k_fs_LR.surf.gii",
    "surf_right_vinf": "ParcellationPilot.R.very_inflated.32k_fs_LR.surf.gii",
}

_INFLATIONS = {
    0: ["surf_left", "surf_right"],
    1: ["surf_left_inf", "surf_right_inf"],
    2: ["surf_left_vinf", "surf_right_vinf"],
}


def file(name: str) -> str:
    """Path to one mask or surface file, downloading it if needed."""
    return str(_fetch.fetch_file(f"{_SUBDIRECTORY}/{name}"))


def __getattr__(name):
    if name in _SURFACES:
        return file(_SURFACES[name])
    if name == "path":
        return _fetch.fetch_directory(_SUBDIRECTORY)
    if name == "directory":
        return str(_fetch.fetch_directory(_SUBDIRECTORY))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_surf(inflation: int):
    if inflation not in _INFLATIONS:
        raise ValueError(f"inflation must be in {list(_INFLATIONS.keys())}")

    left, right = _INFLATIONS[inflation]
    return combine_surfs(file(_SURFACES[left]), file(_SURFACES[right]))


def combine_surfs(left_file, right_file):
    surf_left = nib.load(left_file)
    surf_right = nib.load(right_file)

    points_left = surf_left.agg_data("pointset")
    points_right = surf_right.agg_data("pointset")

    triangles_left = surf_left.agg_data("triangle")
    triangles_right = surf_right.agg_data("triangle")

    points = np.concatenate([points_left, points_right])
    triangles = np.concatenate([triangles_left, triangles_right + len(points_left)])

    return points, triangles
