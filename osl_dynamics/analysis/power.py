"""Functions to calculate and save network power maps.

"""

import nibabel as nib
import numpy as np
from pathlib import Path
from nilearn import plotting
from tqdm import trange
from osl_dynamics import array_ops, files
from osl_dynamics.analysis.spectral import get_frequency_args_range


def variance_from_spectra(
    frequencies,
    power_spectra,
    components=None,
    frequency_range=None,
):
    """Calculates variance from power spectra.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis of the PSDs. Only used if frequency_range is given.
    power_spectra : np.ndarray
        Power/cross spectra for each channel.
        Can be an (n_channels, n_channels) array or (n_channels,) array.
    components : np.ndarray
        Spectral components. Shape is (n_components, n_f).
    frequency_range : list
        Frequency range to integrate the PSD over (Hz). Default is full range.

    Returns
    -------
    var : np.ndarray
        Variance over a frequency band for each component of each mode.
        Shape is (n_components, n_modes, n_channels, n_channels).
    """

    # Validation
    if power_spectra.ndim == 2:
        # PSDs were passed
        power_spectra = power_spectra[np.newaxis, np.newaxis, ...]
        n_subjects, n_modes, n_channels, n_f = power_spectra.shape

    elif power_spectra.shape[-2] != power_spectra.shape[-3]:
        # PSDs were passed, check dimensionality
        error_message = (
            "A (n_channels, n_f), (n_modes, n_channels, n_f) or "
            + "(n_subjects, n_modes, n_channels, n_f) array must be passed."
        )
        power_spectra = array_ops.validate(
            power_spectra,
            correct_dimensionality=4,
            allow_dimensions=[2, 3],
            error_message=error_message,
        )
        n_subjects, n_modes, n_channels, n_f = power_spectra.shape

    else:
        # Cross spectra were passed, check dimensionality
        error_message = (
            "A (n_channels, n_channels, n_f), "
            + "(n_modes, n_channels, n_channels, n_f) or "
            + "(n_subjects, n_modes, n_channels, n_channels, n_f) "
            + "array must be passed."
        )
        power_spectra = array_ops.validate(
            power_spectra,
            correct_dimensionality=5,
            allow_dimensions=[3, 4],
            error_message=error_message,
        )
        n_subjects, n_modes, n_channels, n_channels, n_f = power_spectra.shape

    if components is not None and frequency_range is not None:
        raise ValueError(
            "Only one of the arguments components or frequency range can be passed."
        )

    if frequency_range is not None and frequencies is None:
        raise ValueError(
            "If frequency_range is passed, frequenices must also be passed."
        )

    # Number of spectral components
    if components is None:
        n_components = 1
    else:
        n_components = components.shape[0]

    # Calculate power maps for each subject
    var = []
    for i in range(n_subjects):

        # Get PSDs
        if power_spectra.shape[-2] == power_spectra.shape[-3]:
            # Cross-spectra densities were passed
            psd = power_spectra[i, :, range(n_channels), range(n_channels)]
            psd = np.swapaxes(psd, 0, 1)
        else:
            # Only the PSDs were passed
            psd = power_spectra[i]

        # Concatenate over modes
        psd = psd.reshape(-1, n_f)
        psd = psd.real

        if components is not None:
            # Calculate PSD for each spectral component
            p = components @ psd.T
            for j in range(n_components):
                p[j] /= np.sum(components[j])

        else:
            # Integrate over the given frequency range
            if frequency_range is None:
                p = np.mean(psd, axis=-1)
            else:
                [f_min_arg, f_max_arg] = get_frequency_args_range(
                    frequencies, frequency_range
                )
                p = np.mean(psd[..., f_min_arg : f_max_arg + 1], axis=-1)

        p = p.reshape(n_components, n_modes, n_channels)

        # Variance
        v = np.zeros([n_components, n_modes, n_channels, n_channels])
        v[:, :, range(n_channels), range(n_channels)] = p
        var.append(v)

    return np.squeeze(var)


def power_map_grid(
    mask_file,
    parcellation_file,
    power_map,
    component=0,
    subtract_mean=False,
    mean_weights=None,
):
    """Takes a power map and returns the power at locations on a spatial grid."""

    # Validation
    error_message = (
        f"Dimensionality of power_map must be less than 4 got ndim={power_map.ndim}."
    )
    power_map = array_ops.validate(
        power_map,
        correct_dimensionality=3,
        allow_dimensions=[1, 2],
        error_message=error_message,
    )

    # Load the mask
    mask = nib.load(mask_file)
    mask_grid = mask.get_data()
    mask_grid = mask_grid.ravel(order="F")

    # Get indices of non-zero elements, i.e. those which contain the brain
    non_zero_voxels = mask_grid != 0

    # Load the parcellation
    parcellation = nib.load(parcellation_file)
    parcellation_grid = parcellation.get_data()

    # Make a 2D array of voxel weights for each parcel
    n_parcels = parcellation.shape[-1]
    voxel_weights = parcellation_grid.reshape(-1, n_parcels, order="F")[non_zero_voxels]

    # Normalise the voxels weights
    voxel_weights /= voxel_weights.max(axis=0)[np.newaxis, ...]

    # Generate a spatial map vector for each mode
    n_voxels = voxel_weights.shape[0]
    n_modes = power_map.shape[1]
    spatial_map_values = np.empty([n_voxels, n_modes])
    for i in range(n_modes):
        spatial_map_values[:, i] = voxel_weights @ power_map[component, i]

    # Subtract weighted mean
    if n_modes == 1:
        subtract_mean = False
    if subtract_mean:
        spatial_map_values -= np.average(
            spatial_map_values,
            axis=1,
            weights=mean_weights,
        )[..., np.newaxis]

    # Final spatial map as a 3D grid for each mode
    spatial_map = np.zeros([mask_grid.shape[0], n_modes])
    spatial_map[non_zero_voxels] = spatial_map_values
    spatial_map = spatial_map.reshape(
        mask.shape[0], mask.shape[1], mask.shape[2], n_modes, order="F"
    )

    return spatial_map


def save(
    power_map,
    filename,
    mask_file,
    parcellation_file,
    component=0,
    subtract_mean=False,
    mean_weights=None,
    **plot_kwargs,
):
    """Saves power maps.

    Parameters
    ----------
    power_map : np.ndarray
        Power map to save. Can be of shape: (n_components, n_modes, n_channels),
        (n_modes, n_channels) or (n_channels,). A (..., n_channels, n_channels)
        array can also be passed. Warning: this function cannot be used if n_modes
        is equal to n_channels.
    filename : str
        Output filename. If extension is .nii.gz the power map is saved as a
        NIFTI file. Or if the extension is png, it is saved as images.
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    component : int
        Spectral component to save.
    subtract_mean : bool
        Should we subtract the mean power across modes?
    mean_weights: np.ndarray
        Numpy array with weightings for each mode to use to calculate the mean.
        Default is equal weighting.
    plot_kwargs : dict
        Keyword arguments to pass to nilearn.plotting.plot_img_on_surf.
    """
    # Validation
    if ".nii.gz" not in filename and ".png" not in filename:
        raise ValueError("filename must have extension .nii.gz or .png.")
    mask_file = files.check_exists(mask_file, files.mask.directory)
    parcellation_file = files.check_exists(
        parcellation_file, files.parcellation.directory
    )

    power_map = np.squeeze(power_map)
    if power_map.ndim > 1:
        if power_map.shape[-1] == power_map.shape[-2]:
            # A n_channels by n_channels array has been passed,
            # extract the diagonal
            power_map = np.diagonal(power_map, axis1=-2, axis2=-1)

    # Calculate power maps
    power_map = power_map_grid(
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        power_map=power_map,
        component=component,
        subtract_mean=subtract_mean,
        mean_weights=mean_weights,
    )

    # Load the mask
    mask = nib.load(mask_file)

    # Save as nii file
    if ".nii.gz" in filename:
        print(f"Saving {filename}")
        nii = nib.Nifti1Image(power_map, mask.affine, mask.header)
        nib.save(nii, filename)

    # Save each map as an image
    elif ".png" in filename:
        n_modes = power_map.shape[-1]
        for i in trange(n_modes, desc="Saving images", ncols=98):
            nii = nib.Nifti1Image(power_map[:, :, :, i], mask.affine, mask.header)
            output_file = "{fn.parent}/{fn.stem}{i:0{w}d}{fn.suffix}".format(
                fn=Path(filename), i=i, w=len(str(n_modes))
            )
            plotting.plot_img_on_surf(
                nii,
                views=["lateral", "medial"],
                hemispheres=["left", "right"],
                colorbar=True,
                output_file=output_file,
                **plot_kwargs,
            )
