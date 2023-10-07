"""Plot the Giles parcellation.

fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz is commonly
referred to as the 'Giles parcellation'.

This script can be helpful for understanding what brain regions each parcel
in the Giles parcellation corresponds to.
"""

import os
import numpy as np
from osl_dynamics.analysis import power, connectivity

plot_parcels = True
plot_hemisphere_conn = True

os.makedirs("plots", exist_ok=True)

if plot_parcels:
    # Plot a map with only one parcel activiated

    # Loop through the number of parcels
    for i in range(38):

        # First create a vector of zeros
        p = np.zeros(38)

        # Activate one parcel
        p[i] = 1

        # Plot
        power.save(
            p,
            filename=f"plots/parc_{i}_.png",
            mask_file="MNI152_T1_8mm_brain.nii.gz",
            parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
        )

if plot_hemisphere_conn:
    # Plot connections in the left hemisphere only

    # First assign an edge with value one to each pairwise connection in the
    # left hemisphere
    c = np.zeros([38, 38])
    c[::2, ::2] = 1
    c[::2, 37] = 1
    c[37, ::2] = 1

    # Plot
    connectivity.save(
        c,
        filename="plots/left_hemi_.png",
        parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    )

    # Plot connections in the right hemisphere only

    # First assign an edge with value one to each pairwise connection in the
    # right hemisphere
    c = np.zeros([38, 38])
    c[1::2, 1::2] = 1
    c[1::2, 36] = 1
    c[36, 1::2] = 1

    # Plot
    connectivity.save(
        c,
        filename="plots/right_hemi_.png",
        parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    )
