"""Parcellations.

- fMRI_parcellation_ds8mm.nii.gz
- fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz (the 'Giles parcellation')
- fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz
- fmri_d100_parcellation_with_PCC_tighterMay15_v2_6mm_exclusive.nii.gz
- fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm.nii.gz
- fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz
- aal_cortical_merged_8mm_stacked.nii.gz
- Glasser50_space-MNI152NLin6_res-8x8x8.nii.gz
- Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
- giles_39_binary.nii.gz
"""

from pathlib import Path

path = Path(__file__).parent
directory = str(path)
