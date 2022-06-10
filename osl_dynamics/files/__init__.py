"""Useful files kept in the osl-dynamics package.

This includes parcellation/mask files and scanner layouts.

Available parcellations:

- fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
- fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz

Available masks:

- MNI152_T1_8mm_brain.nii.gz

"""

from osl_dynamics.files import mask, parcellation, scanner, scene
from osl_dynamics.files.functions import *
