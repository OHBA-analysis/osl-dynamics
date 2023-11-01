"""Mask files.

- MNI152_T1_1mm_brain.nii.gz
- MNI152_T1_2mm_brain.nii.gz
- MNI152_T1_8mm_brain.nii.gz
- ft_8mm_brain_mask.nii.gz
"""

from pathlib import Path

import nibabel as nib
import numpy as np

surf_right = str(
    Path(__file__).parent / "ParcellationPilot.R.midthickness.32k_fs_LR.surf.gii"
)
surf_left = str(
    Path(__file__).parent / "ParcellationPilot.L.midthickness.32k_fs_LR.surf.gii"
)
surf_right_inf = str(
    Path(__file__).parent / "ParcellationPilot.R.inflated.32k_fs_LR.surf.gii"
)
surf_left_inf = str(
    Path(__file__).parent / "ParcellationPilot.L.inflated.32k_fs_LR.surf.gii"
)
surf_right_vinf = str(
    Path(__file__).parent / "ParcellationPilot.R.very_inflated.32k_fs_LR.surf.gii"
)
surf_left_vinf = str(
    Path(__file__).parent / "ParcellationPilot.L.very_inflated.32k_fs_LR.surf.gii"
)

path = Path(__file__).parent
directory = str(path)


def get_surf(inflation: int):
    surfs = {
        0: [surf_left, surf_right],
        1: [surf_left_inf, surf_right_inf],
        2: [surf_left_vinf, surf_right_vinf],
    }
    if inflation not in surfs.keys():
        raise ValueError(f"inflation must be in {list(surfs.keys())}")

    return combine_surfs(*surfs[inflation])


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
