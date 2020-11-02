"""Functions to generate maps.

"""

import os
import pathlib
import subprocess

import nibabel as nib
import numpy as np
from vrad.analysis import scene, std_masks
from vrad.analysis.functions import validate_array


def state_maps(power_spectra, coherences, components):
    """Calculates spatial maps for each spectral component and state."""

    # Validation
    error_message = (
        "a 3D numpy array (n_channels, n_channels, n_frequency_bins) "
        + "or 4D numpy array (n_states, n_channels, n_channels, "
        + "n_frequency_bins) must be passed for spectra."
    )
    power_spectra = validate_array(
        power_spectra,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )
    coherences = validate_array(
        coherences,
        correct_dimensionality=5,
        allow_dimensions=[3, 4],
        error_message=error_message,
    )

    # Number of subjects, states, channels and frequency bins
    n_subjects, n_states, n_channels, n_channels, n_f = power_spectra.shape

    # Number of components
    n_components = components.shape[0]

    # Remove cross-spectral densities from the power spectra array and concatenate
    # over subjects and states
    psd = power_spectra[:, :, range(n_channels), range(n_channels)].reshape(-1, n_f)

    # PSDs are real valued so we can recast
    psd = psd.real

    # Calculate PSDs for each spectral component
    psd = components @ psd.T
    psd = psd.reshape(n_components, n_states, n_channels)

    # Power map
    p = np.zeros([n_components, n_states, n_channels, n_channels])
    p[:, :, range(n_channels), range(n_channels)] = psd

    # Only keep the upper triangle of the coherences and concatenate over subjects
    # and states
    i, j = np.triu_indices(n_channels, 1)
    coh = coherences[:, :, i, j].reshape(-1, n_f)

    #  Calculate coherences for each spectral component
    coh = components @ coh.T
    coh = coh.reshape(n_components, n_states, n_channels * (n_channels - 1) // 2)

    # Coherence map
    c = np.zeros([n_components, n_states, n_channels, n_channels])
    c[:, :, i, j] = coh
    c[:, :, j, i] = coh
    c[:, :, range(n_channels), range(n_channels)] = 1

    return p, c


def save_nii_file(mask_file, parcellation_file, power_map, filename, component=0):
    """Saves a NITFI file containing a map."""

    # Add extension if it's not already there
    if "nii" not in filename:
        filename += ".nii.gz"

    # Load the mask
    mask = nib.load(mask_file)
    mask_grid = mask.get_data()

    # Flatten the mask
    mask_grid = mask_grid.ravel(order="F")

    # Get indices of non-zero elements, i.e. those which contain the brain
    non_zero_voxels = mask_grid != 0

    # Load the parcellation
    parcellation = nib.load(parcellation_file)
    parcellation_grid = parcellation.get_data()

    # Number of parcels
    n_parcels = parcellation.shape[-1]

    # Make a 1D array of voxel weights for each channel
    voxels = parcellation_grid.reshape(-1, n_parcels, order="F")[non_zero_voxels]

    # Number of voxels
    n_voxels = voxels.shape[0]

    # Normalise the voxels to have comparable weights
    voxels /= voxels.max(axis=0)[np.newaxis, ...]

    # Number of components, states, channels
    n_components, n_states, n_channels, n_channels = power_map.shape

    # Generate spatial map
    spatial_map = np.empty([n_voxels, n_states])
    for i in range(n_states):
        spatial_map[:, i] = voxels @ np.diag(np.squeeze(power_map[component, i]))

    # Subtract mean power across states
    spatial_map -= np.mean(spatial_map, axis=1)[..., np.newaxis]

    # Convert spatial map into a grid
    spatial_map_grid = np.zeros([mask_grid.shape[0], n_states])
    spatial_map_grid[non_zero_voxels] = spatial_map
    spatial_map_grid = spatial_map_grid.reshape(
        mask.shape[0], mask.shape[1], mask.shape[2], n_states, order="F"
    )

    # Save as nii file
    print(f"Saving {filename}")
    nii_file = nib.Nifti1Image(spatial_map_grid, mask.affine, mask.header)
    nib.save(nii_file, filename)


def workbench_render(nii, save_dir=None, interptype="trilinear", visualise=True):
    nii = pathlib.Path(nii)

    if not nii.exists() or ".nii" not in nii.suffixes:
        raise ValueError(f"nii should be a nii or nii.gz file." f"found {nii}.")

    if save_dir is None:
        save_dir = os.getcwd()

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    in_file = pathlib.Path(str(nii).replace(".gz", "").replace(".nii", ""))
    out_file = save_dir / in_file

    # Load surfaces
    surf_right = std_masks.surf_right
    surf_left = std_masks.surf_left
    surf_right_inf = std_masks.surf_right_inf
    surf_left_inf = std_masks.surf_left_inf
    surf_right_vinf = std_masks.surf_right_vinf
    surf_left_vinf = std_masks.surf_left_vinf

    output_right = out_file.parent / (str(out_file) + "_right.func.gii")
    output_left = out_file.parent / (str(out_file) + "_left.func.gii")

    subprocess.run(
        [
            "wb_command",
            "-volume-to-surface-mapping",
            str(nii),
            str(surf_right),
            str(output_right),
            f"-{interptype}",
        ]
    )

    subprocess.run(
        [
            "wb_command",
            "-volume-to-surface-mapping",
            str(nii),
            str(surf_left),
            str(output_left),
            f"-{interptype}",
        ]
    )

    cifti_right = str(output_right).replace(".func.gii", ".dtseries.nii")
    cifti_left = str(output_left).replace(".func.gii", ".dtseries.nii")

    subprocess.run(
        [
            "wb_command",
            "-cifti-create-dense-timeseries",
            cifti_right,
            "-right-metric",
            output_right,
        ]
    )

    subprocess.run(
        [
            "wb_command",
            "-cifti-create-dense-timeseries",
            cifti_left,
            "-left-metric",
            output_left,
        ]
    )

    if visualise:
        subprocess.run(
            [
                "wb_view",
                str(surf_left),
                str(surf_right),
                str(surf_left_inf),
                str(surf_right_inf),
                str(surf_left_vinf),
                str(surf_right_vinf),
                cifti_left,
                cifti_right,
            ]
        )
