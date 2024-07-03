"""Functions to calculate and save network power maps.

"""

import os
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn import image, plotting
from tqdm.auto import trange
from pqdm.threads import pqdm
import matplotlib.pyplot as plt

from osl_dynamics import array_ops, files
from osl_dynamics.analysis.spectral import get_frequency_args_range

_logger = logging.getLogger("osl-dynamics")


def sliding_window_power(
    data, window_length, step_size=None, power_type="mean", concatenate=False, n_jobs=1
):
    """Calculate sliding window power.

    Parameters
    ----------
    data : list or np.ndarray
        Time series data. Shape must be (n_sessions, n_samples, n_channels)
        or (n_samples, n_channels).
    window_length : int
        Window length in samples.
    step_size : int, optional
        Number of samples to slide the window along the time series.
        If :code:`None` is passed, then a 50% overlap is used.
    power_type : str, optional
        Type of power to calculate. Can be :code:`"mean"` or :code:`"var"`.
    concatenate : bool, optional
        Should we concatenate the sliding window power from each array
        into one big time series?
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.

    Returns
    -------
    sliding_window_power : list or np.ndarray
        Time series of power vectors. Shape is (n_sessions, n_windows,
        n_channels) or (n_windows, n_channels).
    """
    # Validation
    if power_type not in ["mean", "var"]:
        raise ValueError(f"power_type must be 'mean' or 'var', not {power_type}")

    if power_type == "var":
        metric = np.var
    else:
        metric = np.mean

    if step_size is None:
        step_size = window_length // 2

    if isinstance(data, np.ndarray):
        if data.ndim != 3:
            data = [data]

    # Helper function to calculate power
    def _swp(x):
        n_samples = x.shape[0]
        n_channels = x.shape[1]
        n_windows = (n_samples - window_length - 1) // step_size + 1

        # Preallocate an array fo hold moving average values
        swp = np.empty([n_windows, n_channels], dtype=np.float32)
        for i in range(n_windows):
            window_ts = x[i * step_size : i * step_size + window_length]
            swp[i] = metric(window_ts, axis=0)

        return swp

    # Setup keyword arguments to pass to the helper function
    kwargs = [{"x": x} for x in data]

    if len(data) == 1:
        _logger.info("Sliding window power")
        results = [_swp(**kwargs[0])]

    elif n_jobs == 1:
        results = []
        for i in trange(len(data), desc="Sliding window power"):
            results.append(_swp(**kwargs[i]))

    else:
        _logger.info(f"Sliding window power")
        results = pqdm(
            kwargs,
            _swp,
            argument_type="kwargs",
            n_jobs=n_jobs,
        )

    if concatenate or len(results) == 1:
        results = np.concatenate(results)

    return results


def variance_from_spectra(
    frequencies,
    power_spectra,
    components=None,
    frequency_range=None,
    method="mean",
):
    """Calculates variance from power spectra.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis of the PSDs. Only used if :code:`frequency_range` is
        given. Shape must be (n_freq,).
    power_spectra : np.ndarray
        Power/cross spectra for each channel.
        Shape must be (n_channels, n_channels) or (n_channels,).
    components : np.ndarray, optional
        Spectral components. Shape must be (n_components, n_freq).
    frequency_range : list, optional
        Frequency range in Hz to integrate the PSD over. Default is full range.
    method : str
        Should take the sum of the PSD over the frequency range
        (:code:`method="sum"`), the integral of the PSD (:code:`"integral"`),
        or take the average value of the PSD (:code:`method="mean"`).

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
        n_sessions, n_modes, n_channels, n_freq = power_spectra.shape

    elif power_spectra.shape[-2] != power_spectra.shape[-3]:
        # PSDs were passed, check dimensionality
        error_message = (
            "A (n_channels, n_freq), (n_modes, n_channels, n_freq) or "
            + "(n_sessions, n_modes, n_channels, n_freq) array must be passed."
        )
        power_spectra = array_ops.validate(
            power_spectra,
            correct_dimensionality=4,
            allow_dimensions=[2, 3],
            error_message=error_message,
        )
        n_sessions, n_modes, n_channels, n_freq = power_spectra.shape

    else:
        # Cross spectra were passed, check dimensionality
        error_message = (
            "A (n_channels, n_channels, n_freq), "
            + "(n_modes, n_channels, n_channels, n_freq) or "
            + "(n_sessions, n_modes, n_channels, n_channels, n_freq) "
            + "array must be passed."
        )
        power_spectra = array_ops.validate(
            power_spectra,
            correct_dimensionality=5,
            allow_dimensions=[3, 4],
            error_message=error_message,
        )
        n_sessions, n_modes, n_channels, n_channels, n_freq = power_spectra.shape

    if components is not None and frequency_range is not None:
        raise ValueError(
            "Only one of the arguments components or frequency range can be passed."
        )

    if frequency_range is not None and frequencies is None:
        raise ValueError(
            "If frequency_range is passed, frequenices must also be passed."
        )

    if method not in ["mean", "sum", "integral"]:
        raise ValueError("method should be 'mean', 'sum' or 'integral'.")

    # Number of spectral components
    if components is None:
        n_components = 1
    else:
        n_components = components.shape[0]

    # Calculate power maps for each array
    var = []
    for i in range(n_sessions):
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
                if method == "sum":
                    p = np.sum(psd, axis=-1)
                elif method == "integral":
                    df = frequencies[1] - frequencies[0]
                    p = np.sum(psd * df, axis=-1)
                else:
                    p = np.mean(psd, axis=-1)
            else:
                [min_arg, max_arg] = get_frequency_args_range(
                    frequencies, frequency_range
                )
                if method == "sum":
                    p = np.sum(psd[..., min_arg:max_arg], axis=-1)
                elif method == "integral":
                    df = frequencies[1] - frequencies[0]
                    p = np.sum(psd[..., min_arg:max_arg] * df, axis=-1)
                else:
                    p = np.mean(psd[..., min_arg:max_arg], axis=-1)

        p = p.reshape(n_components, n_modes, n_channels)
        var.append(p)

    return np.squeeze(var)


def parcel_vector_to_voxel_grid(mask_file, parcellation_file, vector):
    """Takes a vector of parcel values and return a 3D voxel grid.

    Parameters
    ----------
    mask_file : str
        Mask file for the voxel grid. Must be a NIFTI file.
    parcellation_file : str
        Parcellation file. Must be a NIFTI file.
    vector : np.ndarray
        Value at each parcel. Shape must be (n_parcels,).

    Returns
    -------
    voxel_grid : np.ndarray
        Value at each voxel. Shape is (x, y, z), where :code:`x`,
        :code:`y` and :code:`z` correspond to 3D voxel locations.
    """
    # Suppress INFO messages from nibabel
    logging.getLogger("nibabel.global").setLevel(logging.ERROR)

    # Validation
    mask_file = files.check_exists(mask_file, files.mask.directory)
    parcellation_file = files.check_exists(
        parcellation_file, files.parcellation.directory
    )

    # Load the mask
    mask = nib.load(mask_file)
    mask_grid = mask.get_fdata()
    mask_grid = mask_grid.ravel(order="F")

    # Get indices of non-zero elements, i.e. those which contain the brain
    non_zero_voxels = mask_grid != 0

    # Load the parcellation
    parcellation = nib.load(parcellation_file)

    # Make sure parcellation is 4D and contains 1 for voxel assignment
    # to a parcel and 0 otherwise
    parcellation_grid = parcellation.get_fdata()
    if parcellation_grid.ndim == 3:
        unique_values = np.unique(parcellation_grid)[1:]
        parcellation_grid = np.array(
            [(parcellation_grid == value).astype(int) for value in unique_values]
        )
        parcellation_grid = np.rollaxis(parcellation_grid, 0, 4)
        parcellation = nib.Nifti1Image(
            parcellation_grid, parcellation.affine, parcellation.header
        )

    # Make sure the parcellation grid matches the mask file
    parcellation = image.resample_to_img(parcellation, mask)
    parcellation_grid = parcellation.get_fdata()

    # Make a 2D array of voxel weights for each parcel
    n_parcels = parcellation.shape[-1]

    # Check parcellation is compatible
    if vector.shape[0] != n_parcels:
        _logger.error(
            "parcellation_file has a different number of parcels to the vector"
        )

    voxel_weights = parcellation_grid.reshape(-1, n_parcels, order="F")[non_zero_voxels]

    # Normalise the voxels weights
    voxel_weights /= voxel_weights.max(axis=0, keepdims=True)

    # Generate a vector containing value at each voxel
    voxel_values = voxel_weights @ vector

    # Final 3D voxel grid
    voxel_grid = np.zeros(mask_grid.shape[0])
    voxel_grid[non_zero_voxels] = voxel_values
    voxel_grid = voxel_grid.reshape(
        mask.shape[0], mask.shape[1], mask.shape[2], order="F"
    )

    return voxel_grid


def save(
    power_map,
    mask_file,
    parcellation_file,
    filename=None,
    component=0,
    subtract_mean=False,
    mean_weights=None,
    plot_kwargs=None,
    show_plots=True,
    combined=False,
    titles=None,
    n_rows=1,
):
    """Saves power maps.

    This function is a wrapper for `nilearn.plotting.plot_img_on_surf
    <https://nilearn.github.io/stable/modules/generated/nilearn.plotting\
    .plot_img_on_surf.html>`_.

    Parameters
    ----------
    power_map : np.ndarray
        Power map to save. Can be of shape: (n_components, n_modes, n_channels),
        (n_modes, n_channels) or (n_channels,). A (..., n_channels, n_channels)
        array can also be passed. Warning: this function
        cannot be used if :code:`n_modes=n_channels`.
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcellate the training data.
    filename : str, optional
        Output filename. If extension is :code:`.nii.gz` the power map is saved
        as a NIFTI file. Or if the extension is :code:`png/svg/pdf`, it is saved
        as images. If :code:`None` is passed then the image is shown on screen
        and the Matplotlib objects are returned.
    component : int, optional
        Spectral component to save.
    subtract_mean : bool, optional
        Should we subtract the mean power across modes?
    mean_weights: np.ndarray, optional
        Numpy array with weightings for each mode to use to calculate the mean.
        Default is equal weighting.
    plot_kwargs : dict, optional
        Keyword arguments to pass to `nilearn.plotting.plot_img_on_surf
        <https://nilearn.github.io/stable/modules/generated/nilearn.plotting\
        .plot_img_on_surf.html>`_.
    show_plots : bool, optional
        Should the individual plots be shown on screen (for Juptyer notebooks)?
    combined : bool, optional
        Should the individual plots be combined into a single image?
        The combined image is always shown on screen (for Juptyer notebooks).
        Note if :code:`True` is passed, the individual images will be deleted.
    titles : list, optional
        List of titles for each power plot. Only used if :code:`combined=True`.
    n_rows : int, optional
        Number of rows in the combined image. Only used if :code:`combined=True`.

    Returns
    -------
    figures : list of plt.figure
        List of Matplotlib figure object. Only returned if
        :code:`filename=None`.
    axes : list of plt.axis
        List of Matplotlib axis object(s). Only returned if
        :code:`filename=None`.

    Examples
    --------
    Plot power maps with customise display::

        power.save(
            ...,
            plot_kwargs={
                "cmap": "RdBu_r",
                "bg_on_data": 1,
                "darkness": 0.4,
                "alpha": 1,
            },
        )
    """
    # Create a copy of the power map so we don't modify it
    power_map = np.copy(power_map)

    # Validation
    if filename is not None:
        allowed_extensions = [".nii", ".nii.gz", ".png", ".svg", ".pdf"]
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

    # Calculate power map grid for each mode
    power_map = [
        parcel_vector_to_voxel_grid(mask_file, parcellation_file, p) for p in power_map
    ]

    # Make sure n_modes is the last dimension for compatibility with nii files
    # (n_modes, x, y, z) -> (x, y, z, n_modes)
    power_map = np.moveaxis(power_map, 0, -1)

    # Load the mask
    mask = nib.load(mask_file)

    # Just display the power map
    if filename is None:
        figures, axes = [], []
        for i in trange(n_modes, desc="Saving images"):
            nii = nib.Nifti1Image(power_map[:, :, :, i], mask.affine, mask.header)
            fig, ax = plotting.plot_img_on_surf(nii, output_file=None, **plot_kwargs)
            figures.append(fig)
            axes.append(ax)
        return figures, axes

    else:
        # Save as nii file
        if ".nii" in filename:
            _logger.info(f"Saving {filename}")
            nii = nib.Nifti1Image(power_map, mask.affine, mask.header)
            nib.save(nii, filename)

        else:
            # Save each map as an image
            output_files = []
            for i in trange(n_modes, desc="Saving images"):
                nii = nib.Nifti1Image(
                    power_map[:, :, :, i],
                    mask.affine,
                    mask.header,
                )
                fig, ax = plotting.plot_img_on_surf(
                    nii, output_file=None, **plot_kwargs
                )
                output_file = "{fn.parent}/{fn.stem}{i:0{w}d}{fn.suffix}".format(
                    fn=Path(filename), i=i, w=len(str(n_modes))
                )
                output_files.append(output_file)
                fig.savefig(output_file)

                if not show_plots:
                    plt.close(fig)

            if combined:
                n_columns = -(n_modes // -n_rows)

                titles = titles or [None] * n_modes
                # Combine images into a single image
                fig, axes = plt.subplots(
                    n_rows, n_columns, figsize=(n_columns * 5, n_rows * 5)
                )
                for i, ax in enumerate(axes.flatten()):
                    ax.axis("off")
                    if i < n_modes:
                        ax.imshow(plt.imread(output_files[i]))
                        ax.set_title(titles[i], fontsize=20)
                fig.tight_layout()
                fig.savefig(filename)

                # Remove the individual images
                for output_file in output_files:
                    os.remove(output_file)


def multi_save(
    group_power_map,
    session_power_map,
    mask_file,
    parcellation_file,
    filename=None,
    sessions=None,
    subtract_mean=False,
    mean_weights=None,
    plot_kwargs=None,
):
    """Saves group level and array level power maps.

    When training session-specific models we want to plot the group-level map
    and session-specific deviations. This function is a wrapper for `power.save
    <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics\
    /analysis/power/index.html#osl_dynamics.analysis.power.save>`_, which helps
    us plot power maps for session-specific models.

    Parameters
    ----------
    group_power_map : np.ndarray
        Group level power map to save.
        Can be of shape: (n_modes, n_channels) or (n_channels,). A (...,
        n_channels, n_channels) can also be passed. Warning: this function
        cannot be used if :code:`n_modes` is equal to :code:`n_channels`.
    session_power_map : np.ndarray
        Session-level power maps to save.
        Can be of shape: (n_sessions, n_modes, n_channels), (n_modes,
        n_channels) or (n_channels,). A (..., n_channels, n_channels) array can
        also be passed. Warning: this function cannot be used if
        :code:`n_modes=n_channels`.
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcellate the training data.
    filename : str, optional
        Output filename. If extension is :code:`.nii.gz` the power map is saved
        as a NIFTI file. Or if the extension is :code:`png/svg/pdf`, it is saved
        as images. If :code:`None` is passed then the image is shown on screen
        and the Matplotlib objects are returned.
    sessions : list, optional
        List of session indices to be plot power maps for.
    subtract_mean : bool, optional
        Should we subtract the mean power across modes?
    mean_weights: np.ndarray, optional
        Numpy array with weightings for each mode to use to calculate the mean.
        Default is equal weighting.
    plot_kwargs : dict, optional
        Keyword arguments to pass to `nilearn.plotting.plot_img_on_surf
        <https://nilearn.github.io/stable/modules/generated/nilearn.plotting\
        .plot_img_on_surf.html>`_.
    """
    # Create a copy of the power maps so we don't modify them
    group_power_map = np.copy(group_power_map)
    session_power_map = np.copy(session_power_map)

    # Validation
    if group_power_map.ndim > 1:
        if group_power_map.shape[-1] == group_power_map.shape[-2]:
            # np.copy is needed because np.diagonal returns a read only array
            group_power_map = np.copy(np.diagonal(group_power_map, axis1=-2, axis2=-1))
    else:
        group_power_map = group_power_map[np.newaxis, ...]

    if session_power_map.ndim > 1:
        if session_power_map.shape[-1] == session_power_map.shape[-2]:
            # np.copy is needed because np.diagonal returns a read only array
            session_power_map = np.copy(
                np.diagonal(session_power_map, axis1=-2, axis2=-1)
            )
    else:
        session_power_map = session_power_map[np.newaxis, ...]

    group_power_map = array_ops.validate(
        group_power_map,
        correct_dimensionality=2,
        allow_dimensions=[1],
        error_message="group_power_map.shape is incorrect.",
    )
    session_power_map = array_ops.validate(
        session_power_map,
        correct_dimensionality=3,
        allow_dimensions=[1, 2],
        error_message="session_power_map.shape is incorrect",
    )

    if group_power_map.shape[0] != session_power_map.shape[1]:
        raise ValueError("group and array level power maps must have the same n_modes.")

    # Subtract mean
    n_modes = group_power_map.shape[0]
    if n_modes == 1:
        subtract_mean = False
    if subtract_mean:
        mean_group_power = np.average(
            group_power_map,
            axis=0,
            weights=mean_weights,
        )
        group_power_map -= mean_group_power[np.newaxis, ...]
        session_power_map -= mean_group_power[np.newaxis, np.newaxis, ...]

    # Save the group power map
    filename = Path(filename)
    group_dir = f"{filename.parent}/group"
    os.makedirs(group_dir, exist_ok=True)
    group_filename = f"{group_dir}/{filename.stem}{filename.suffix}"

    _logger.info("Saving group level power map:")
    save(
        group_power_map,
        filename=group_filename,
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        plot_kwargs=plot_kwargs,
    )

    # Save the session-level power maps
    n_sessions = session_power_map.shape[0]
    if sessions is None:
        sessions = np.arange(n_sessions)

    for sess in sessions:
        session_dir = "{fn.parent}/sess_{sess:0{v}d}".format(
            fn=filename, sess=sess, v=len(str(n_sessions))
        )
        os.makedirs(session_dir, exist_ok=True)
        session_filename = f"{session_dir}/{filename.stem}{filename.suffix}"

        _logger.info(f"Saving session {sess} power map:")
        save(
            session_power_map[sess],
            filename=session_filename,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            plot_kwargs=plot_kwargs,
        )


def independent_components_to_volumetric_maps(
    ica_spatial_maps,
    ic_values,
    output_file=None,
):
    """Independent components to volumetric spatial map.

    Project the ic_values map to a brain map according to spatial
    maps of group ICA components. For example, ic_values can be the mean
    activation of states, or the rank-one decomposition of FC matrix
    corresponding to different states.

    Parameters
    ----------
    ica_spatial_maps : str or Nifti1Image
        Path to nifti file or nifti object containing the spatial maps
        obtained from group ICA. Should be in (n_x, n_y, n_z, n_voxels)
        format.
    ic_values : np.ndarray
        Independent component values.
        Should be in (n_maps, n_ica_components) format.
    output_file : str, optional
        Path to output file to save volumetric map to.

    Returns
    -------
    vol_map : Nifti1Image
        Volumetric maps. Shape is (n_x, n_y, n_z, n_maps).
    """
    # Load nifti
    if isinstance(ica_spatial_maps, str):
        ica_spatial_maps = nib.load(ica_spatial_maps)

    # Validation
    if not isinstance(ica_spatial_maps, nib.nifti1.Nifti1Image):
        raise ValueError(
            "ica_spatial_maps must be a path to a nifti file or "
            "nib.nifti1.Nifti1Image object."
        )

    # Get data from nifti
    ica_spatial_maps_data = ica_spatial_maps.get_fdata()
    ica_spatial_maps_shape = ica_spatial_maps_data.shape
    n_maps, n_ica_components = ic_values.shape

    # Calculate volumetric maps
    data = np.matmul(ica_spatial_maps_data, ic_values.T)

    # Store map to a Nifti1Image type
    vol_map = nib.nifti1.Nifti1Image(
        data,
        affine=ica_spatial_maps.affine,
        header=ica_spatial_maps.header,
    )

    if output_file is not None:
        # Save
        _logger.info(f"Saving {output_file}")
        nib.save(vol_map, output_file)

    return vol_map


def independent_components_to_surface_maps(
    ica_spatial_maps, ic_values, output_file=None
):
    """Independent components to surface spatial map.

    Project the ic_values map to a surface map according to spatial maps
    of group ICA components. For example, ic_values can be the mean
    activation of states, or the rank-one decomposition of FC matrix
    corresponding to different states.

    Different from independent_components_to_volumetric_maps,
    this function works on CIFTI spatial maps.

    Parameters
    ----------
    ica_spatial_maps : str or Cifti2Image
        Path to cifti file or cifti object containing the spatial maps
        obtained from group ICA. Should be in (n_ica_components, n_voxels)
        format.
    ic_values: np.ndarray
        Independent component values.
        Should be in (n_maps, n_ica_components) format.
    output_file : str, optional
        Path to output file to save surface map to.

    Returns
    -------
    surf_map : Cifti2Image
        Surface map. Shape is (n_maps, n_voxels).
    """
    # Load cifti
    if isinstance(ica_spatial_maps, str):
        ica_spatial_maps = nib.load(ica_spatial_maps)

    # Validation
    if not isinstance(ica_spatial_maps, nib.cifti2.cifti2.Cifti2Image):
        raise ValueError(
            "ica_spatial_maps must be a path to a cifti file or "
            "nib.cifti2.cifti2.Cifti2Image object."
        )

    # Get data from cifti
    ica_spatial_maps_data = ica_spatial_maps.get_fdata()
    header = ica_spatial_maps.header
    n_maps = ic_values.shape[0]

    # Calculate surface maps
    data = np.matmul(ic_values, ica_spatial_maps_data)

    # Define the new axes: the Map i (i = 1 ,..., n_maps)
    axes = nib.cifti2.cifti2_axes.ScalarAxis([f"Map {i + 1}" for i in range(n_maps)])

    # Define the new header using axes + axes[1] from old header
    header = nib.cifti2.cifti2.Cifti2Header.from_axes((axes, header.get_axis(1)))

    # Combine new data and new header to obtain new CIFTI object
    surf_map = nib.cifti2.cifti2.Cifti2Image(data, header)

    if output_file is not None:
        # Save
        _logger.info(f"Saving {output_file}")
        nib.save(surf_map, output_file)

    return surf_map
