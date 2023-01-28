"""Functions to calculate and save network power maps.

"""

from os import makedirs

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
        Spectral components. Shape is (n_components, n_freq).
    frequency_range : list
        Frequency range to integrate the PSD over (Hz). Default is full range.

    Returns
    -------
    var : np.ndarray
        Variance over a frequency band for each component of each mode.
        Shape is (n_components, n_modes, n_channels) or (n_modes, n_channels)
        or (n_channels,).
    """

    # Validation
    if power_spectra.ndim == 2:
        # PSDs were passed
        power_spectra = power_spectra[np.newaxis, np.newaxis, ...]
        n_subjects, n_modes, n_channels, n_freq = power_spectra.shape

    elif power_spectra.shape[-2] != power_spectra.shape[-3]:
        # PSDs were passed, check dimensionality
        error_message = (
            "A (n_channels, n_freq), (n_modes, n_channels, n_freq) or "
            + "(n_subjects, n_modes, n_channels, n_freq) array must be passed."
        )
        power_spectra = array_ops.validate(
            power_spectra,
            correct_dimensionality=4,
            allow_dimensions=[2, 3],
            error_message=error_message,
        )
        n_subjects, n_modes, n_channels, n_freq = power_spectra.shape

    else:
        # Cross spectra were passed, check dimensionality
        error_message = (
            "A (n_channels, n_channels, n_freq), "
            + "(n_modes, n_channels, n_channels, n_freq) or "
            + "(n_subjects, n_modes, n_channels, n_channels, n_freq) "
            + "array must be passed."
        )
        power_spectra = array_ops.validate(
            power_spectra,
            correct_dimensionality=5,
            allow_dimensions=[3, 4],
            error_message=error_message,
        )
        n_subjects, n_modes, n_channels, n_channels, n_freq = power_spectra.shape

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
        psd = psd.reshape(-1, n_freq)
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
        var.append(p)

    return np.squeeze(var)


def power_map_grid(mask_file, parcellation_file, power_map):
    """Takes a power map and returns the power at locations on a spatial grid."""

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

    # check parcellation is compatible:
    if power_map.shape[1] is not n_parcels:
        print(
            "Error: parcellation_file has a different number of parcels to the power_maps"
        )

    voxel_weights = parcellation_grid.reshape(-1, n_parcels, order="F")[non_zero_voxels]

    # Normalise the voxels weights
    voxel_weights /= voxel_weights.max(axis=0)[np.newaxis, ...]

    # Generate a spatial map vector for each mode
    n_voxels = voxel_weights.shape[0]
    n_modes = power_map.shape[0]
    spatial_map_values = np.empty([n_voxels, n_modes])

    for i in range(n_modes):
        spatial_map_values[:, i] = voxel_weights @ power_map[i]

    # Final spatial map as a 3D grid for each mode
    spatial_map = np.zeros([mask_grid.shape[0], n_modes])
    spatial_map[non_zero_voxels] = spatial_map_values
    spatial_map = spatial_map.reshape(
        mask.shape[0], mask.shape[1], mask.shape[2], n_modes, order="F"
    )

    return spatial_map


def save(
    power_map,
    mask_file,
    parcellation_file,
    filename=None,
    component=0,
    subtract_mean=False,
    mean_weights=None,
    plot_kwargs=None,
):
    """Saves power maps.

    Parameters
    ----------
    power_map : np.ndarray
        Power map to save. Can be of shape: (n_components, n_modes, n_channels),
        (n_modes, n_channels) or (n_channels,). A (..., n_channels, n_channels)
        array can also be passed. Warning: this function cannot be used if n_modes
        is equal to n_channels.
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    filename : str
        Output filename. If extension is .nii.gz the power map is saved as a
        NIFTI file. Or if the extension is png/svg/pdf, it is saved as images.
        Optional, if None is passed then the image is shown on screen.
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
    # Create a copy of the power map so we don't modify it
    power_map = np.copy(power_map)

    # Validation
    if filename is not None:
        allowed_extensions = [".nii.gz", ".png", ".svg", ".pdf"]
        if not any([ext in filename for ext in allowed_extensions]):
            raise ValueError(
                "filename must have one of following extensions: "
                + f"{' '.join(allowed_extensions)}."
            )
    mask_file = files.check_exists(mask_file, files.mask.directory)
    parcellation_file = files.check_exists(
        parcellation_file, files.parcellation.directory
    )

    if plot_kwargs is None:
        plot_kwargs = {}

    power_map = np.squeeze(power_map)
    if power_map.ndim > 1:
        if power_map.shape[-1] == power_map.shape[-2]:
            # A n_channels by n_channels array has been passed,
            # extract the diagonal (the np.copy is needed because np.diagonal
            # returns a read-only array (this started in NumPy 1.9))
            power_map = np.copy(np.diagonal(power_map, axis1=-2, axis2=-1))
            if power_map.ndim == 1:
                power_map = power_map[np.newaxis, ...]
    else:
        power_map = power_map[np.newaxis, ...]

    power_map = array_ops.validate(
        power_map,
        correct_dimensionality=3,
        allow_dimensions=[2],
        error_message="power_map.shape is incorrect",
    )

    # Subtract weighted mean
    n_modes = power_map.shape[1]
    if n_modes == 1:
        subtract_mean = False
    if subtract_mean:
        power_map -= np.average(power_map, axis=1, weights=mean_weights)[
            :, np.newaxis, ...
        ]

    # Select the component to plot
    power_map = power_map[component]

    # Calculate power map grid
    power_map = power_map_grid(mask_file, parcellation_file, power_map)

    # Load the mask
    mask = nib.load(mask_file)

    # Save as nii file
    if filename is not None:
        if ".nii.gz" in filename:
            print(f"Saving {filename}")
            nii = nib.Nifti1Image(power_map, mask.affine, mask.header)
            nib.save(nii, filename)

    # Save each map as an image
    else:
        n_modes = power_map.shape[-1]
        for i in trange(n_modes, desc="Saving images", ncols=98):
            nii = nib.Nifti1Image(power_map[:, :, :, i], mask.affine, mask.header)
            if filename is None:
                output_file = None
            else:
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


def multi_save(
    group_power_map,
    subject_power_map,
    mask_file,
    parcellation_file,
    filename=None,
    subjects=None,
    component=0,
    subtract_mean=False,
    mean_weights=None,
    plot_kwargs=None,
):
    """Saves group level and subject level power maps.

    This is a multi-subject wrapper of save.

    Parameters
    ----------
    group_power_map : np.ndarray
        Group level power map to save.
        Can be of shape: (n_components, n_modes, n_channels),
        (n_modes, n_channels) or (n_channels,). A (..., n_channels, n_channels)
        array can also be passed. Warning: this function cannot be used if n_modes
        is equal to n_channels.
    subject_power_map : np.ndarray
        Subject level power maps to save.
        Can be of shape: (n_components, n_subjects, n_modes, n_channels),
        (n_subjects, n_modes, n_channels), (n_modes, n_channels) or (n_channels,).
        A (..., n_channels, n_channels) array can also be passed.
        Warning: this function cannot be used if n_modes = n_channels.
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    filename : str
        Output filename. If extension is .nii.gz the power map is saved as a
        NIFTI file. Or if the extension is png, it is saved as images.
        Optional, if None is passed then the image is shown on screen.
    subjects : list
        List of subject indices to be plot power maps for.
    component : int
        Spectral component to save.
    subtract_mean : bool
        Should we subtract the mean power of the group level power across modes?
    mean_weights: np.ndarray
        Numpy array with weightings for each mode to use to calculate the mean.
        Default is equal weighting.
    plot_kwargs : dict
        Keyword arguments to pass to nilearn.plotting.plot_img_on_surf.
    """
    # Create a copy of the power maps so we don't modify them
    group_power_map = np.copy(group_power_map)
    subject_power_map = np.copy(subject_power_map)

    # Validation
    if group_power_map.ndim > 1:
        if group_power_map.shape[-1] == group_power_map.shape[-2]:
            # np.copy is needed because np.diagonal returns a read only array
            group_power_map = np.copy(np.diagonal(group_power_map, axis1=-2, axis2=-1))
    else:
        group_power_map = group_power_map[np.newaxis, ...]

    if subject_power_map.ndim > 1:
        if subject_power_map.shape[-1] == subject_power_map.shape[-2]:
            # np.copy is needed because np.diagonal returns a read only array
            subject_power_map = np.copy(
                np.diagonal(subject_power_map, axis1=-2, axis2=-1)
            )
    else:
        subject_power_map = subject_power_map[np.newaxis, ...]

    group_power_map = array_ops.validate(
        group_power_map,
        correct_dimensionality=3,
        allow_dimensions=[2],
        error_message="group_power_map.shape is incorrect.",
    )
    subject_power_map = array_ops.validate(
        subject_power_map,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message="subject_power_map.shape is incorrect",
    )

    # Select component to plot
    group_power_map = group_power_map[component]
    subject_power_map = subject_power_map[component]

    if group_power_map.shape[0] != subject_power_map.shape[1]:
        raise ValueError(
            "group and subject level power maps must have the same n_modes."
        )

    # Subtract mean
    n_modes = group_power_map.shape[0]
    if n_modes == 1:
        subtract_mean = False
    if subtract_mean:
        mean_group_power = np.average(group_power_map, axis=0, weights=mean_weights)
        group_power_map -= mean_group_power[np.newaxis, ...]
        subject_power_map -= mean_group_power[np.newaxis, np.newaxis, ...]

    # Save the group power map
    filename = Path(filename)
    group_dir = f"{filename.parent}/group"
    makedirs(group_dir, exist_ok=True)
    group_filename = f"{group_dir}/{filename.stem}{filename.suffix}"

    print("Saving group level power map:")
    save(
        power_map=group_power_map,
        filename=group_filename,
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        plot_kwargs=plot_kwargs,
    )

    # Save the subject lebel power maps
    n_subjects = subject_power_map.shape[0]
    if subjects is None:
        subjects = np.arange(n_subjects)

    for sub in subjects:
        subject_dir = "{fn.parent}/sub_{sub:0{v}d}".format(
            fn=filename, sub=sub, v=len(str(n_subjects))
        )
        makedirs(subject_dir, exist_ok=True)
        subject_filename = f"{subject_dir}/{filename.stem}{filename.suffix}"

        print(f"Saving subject {sub} power map:")
        save(
            power_map=subject_power_map[sub],
            filename=subject_filename,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            plot_kwargs=plot_kwargs,
        )
