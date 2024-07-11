"""
HMM: Plotting fMRI Networks
===========================

We normally plot networks from an HMM trained on fMRI directly from the dual-estimated means/covariances. This tutorial covers:

1. Load means/covariances
2. Spatial maps
3. Functional connectivity networks

Note, we assume a group ICA parcellation was used and the HMM was trained on the ICA time courses.
"""

#%%
# Load means/covariances
# ^^^^^^^^^^^^^^^^^^^^^^
# First we need to the dual estimated means and covariances.


import numpy as np

means = np.load("results/dual_estimates/means.npy")
covs = np.load("results/dual_estimates/covs.npy")
print(means.shape)
print(covs.shape)

#%%
# `means` is a (subjects, channels) array and `covariances` is a (subjects, channels, channels) array.
#
# Spatial maps
# ^^^^^^^^^^^^
# The spatial activity maps correspond to the `means`, or if we did not learn a mean, the diagonal of the `covs`. How we plot the spatial maps depends on how the data was preprocessed. If we used a volumetric parcellation, then we can plot the spatial maps with


from osl_dynamics.analysis import power

# Calculate a group average
group_mean = np.mean(means, axis=0)

# Plot
fig, ax = power.save(
    means,
    mask_file="MNI152_T1_2mm_brain.nii.gz",
    parcellation_file="melodic_IC.nii.gz",
    plot_kwargs={"symmetric_cbar": True},
)

#%%
# Alternatively, if we used a surface parcellation, we can use workbench to plot the spatial maps


power.independent_components_to_surface_maps(
    ica_spatial_maps="melodic_IC.dscalar.nii",
    ic_values=group_mean,
    output_file="results/inf_params/means.dscalar.nii",
)

#%%
# You can download workbench `here <https://www.humanconnectome.org/software/get-connectome-workbench>`_.
#
# If you did not learn the mean, then replace `means` above with the diagonal of the covariances.
#
# Functional Connectivity Networks
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The off-diagonal elements in the covariances corresponds to the functional connectivity. In osl-dynamics, we can only plot these if we used a volumetric parcellation with


from osl_dynamics.analysis import connectivity

connectivity.save(
    covs,
    parcellation_file="melodic_IC.nii.gz",
    plot_kwargs={
        "edge_cmap": "Reds",
        "display_mode": "xz",
        "annotate": False,
    },
)

