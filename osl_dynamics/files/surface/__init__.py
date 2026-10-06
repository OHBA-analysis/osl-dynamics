"""Cortical surface meshes.

Mid-thickness, inflated and very inflated surfaces for the left and right
hemispheres, on the 32k_fs_LR grid. These are used to render brain maps on the
cortical surface, either with HCP Workbench (see
:py:mod:`osl_dynamics.utils.workbench`) or directly from the vertices and
triangles returned by :py:func:`get_surf`.

These files are downloaded on first use, see
:py:mod:`osl_dynamics.files._fetch`.
"""

import nibabel as nib
import numpy as np

from osl_dynamics.files import _fetch

_SUBDIRECTORY = "surface/32k_fs_LR"

_SURFACES = {
    "left": "ParcellationPilot.L.midthickness.32k_fs_LR.surf.gii",
    "right": "ParcellationPilot.R.midthickness.32k_fs_LR.surf.gii",
    "left_inf": "ParcellationPilot.L.inflated.32k_fs_LR.surf.gii",
    "right_inf": "ParcellationPilot.R.inflated.32k_fs_LR.surf.gii",
    "left_vinf": "ParcellationPilot.L.very_inflated.32k_fs_LR.surf.gii",
    "right_vinf": "ParcellationPilot.R.very_inflated.32k_fs_LR.surf.gii",
}

_INFLATIONS = {
    0: ["left", "right"],
    1: ["left_inf", "right_inf"],
    2: ["left_vinf", "right_vinf"],
}


def file(name: str) -> str:
    """Path to one surface file, downloading it if needed."""
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
