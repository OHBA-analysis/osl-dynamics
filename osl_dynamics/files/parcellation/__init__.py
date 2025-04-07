"""Volumetric parcellations.

- Glasser50_space-MNI152NLin6_res-8x8x8.nii.gz
- Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz (the 'Glasser parcellation')
- fMRI_parcellation_ds8mm.nii.gz
- fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz (the 'Giles parcellation')
- fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz
- fmri_d100_parcellation_with_PCC_tighterMay15_v2_6mm_exclusive.nii.gz
- fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm.nii.gz
- fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz (used in Viduarre, Nat Comms. 2018)
- giles_39_binary.nii.gz
- aal_cortical_merged_8mm_stacked.nii.gz
- Yeo7_8mm.nii.gz
- Yeo17_8mm.nii.gz

The MNI coordinates of each parcel center can be obtained with:

.. code::

    from osl_dynamics.utils.parcellation import Parcellation

    filename 'fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz'
    parc = Parcellation(filename)
    mni_coords = parc.roi_centers().astype(int)

If you use the Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz parcellation, please cite:

    Kohl, O., Woolrich, M., Nobre, A. C., & Quinn, A. (2023). Glasser52: A parcellation for MEG-Analysis [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.10401793

See the link for further info regarding the parcel name/locations.
"""

from pathlib import Path

path = Path(__file__).parent
directory = str(path)
