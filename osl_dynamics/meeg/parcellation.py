"""Parcellation."""

import logging
import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mne
import scipy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import image, plotting as nilearn_plotting
from fsl import wrappers as fsl_wrappers

from osl_dynamics import files
from osl_dynamics.utils.filenames import OSLFilenames

from . import source_recon

_logger = logging.getLogger("osl-dynamics")


class Parcellation:
    """Class for reading parcellation files.

    Parameters
    ----------
    file : str
        Path to parcellation file.
    """

    def __init__(self, file: Union[str, "Parcellation"]) -> None:
        if isinstance(file, Parcellation):
            self.__dict__.update(file.__dict__)
            return
        self.file = files.check_exists(file, files.parcellation.directory)

        parcellation = nib.load(self.file)

        if parcellation.ndim == 3:
            # Make sure parcellation is 4D and contains 1 for
            # voxel assignment to a parcel and 0 otherwise
            parcellation_grid = parcellation.get_fdata()
            unique_values = np.unique(parcellation_grid)[1:]
            parcellation_grid = np.array(
                [(parcellation_grid == value).astype(int) for value in unique_values]
            )
            parcellation_grid = np.rollaxis(parcellation_grid, 0, 4)
            parcellation = nib.Nifti1Image(
                parcellation_grid, parcellation.affine, parcellation.header
            )

        self.parcellation = parcellation
        self.dims = self.parcellation.shape[:3]
        self.n_parcels = self.parcellation.shape[3]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.file)})"

    def data(self) -> np.ndarray:
        return self.parcellation.get_fdata()

    def nonzero(self) -> list:
        return [np.nonzero(self.data()[..., i]) for i in range(self.n_parcels)]

    def nonzero_coords(self) -> list:
        return [
            nib.affines.apply_affine(
                self.parcellation.affine,
                np.array(nonzero).T,
            )
            for nonzero in self.nonzero()
        ]

    def weights(self) -> list:
        return [
            self.data()[..., i][nonzero] for i, nonzero in enumerate(self.nonzero())
        ]

    def roi_centers(self) -> np.ndarray:
        """Centroid of each parcel."""
        return np.array(
            [
                np.average(c, weights=w, axis=0)
                for c, w in zip(self.nonzero_coords(), self.weights())
            ]
        )

    def plot(self, **kwargs):
        return plot_parcellation(self, **kwargs)

    @staticmethod
    def find_files() -> List[str]:
        paths = Path(files.parcellation.directory).glob("*")
        paths = [path.name for path in paths if not path.name.startswith("__")]
        return sorted(paths)


def plot_parcellation(parcellation: Union[str, "Parcellation"], **kwargs):
    """Plot a parcellation.

    Parameters
    ----------
    parcellation : str or Parcellation
        Parcellation to plot.
    kwargs : keyword arguments, optional
        Keyword arguments to pass to `nilearn.plotting.plot_markers
        <https://nilearn.github.io/stable/modules/generated/nilearn.plotting\
        .plot_markers.html#nilearn.plotting.plot_markers>`_.
    """
    parcellation = Parcellation(parcellation)
    return nilearn_plotting.plot_markers(
        np.zeros(parcellation.n_parcels),
        parcellation.roi_centers(),
        colorbar=False,
        node_cmap="binary_r",
        **kwargs,
    )


def parcel_vector_to_voxel_grid(
    mask_file: str,
    parcellation_file: str,
    vector: np.ndarray,
    remove_subcortical_voxels: bool = False,
) -> np.ndarray:
    """Takes a vector of parcel values and return a 3D voxel grid.

    Parameters
    ----------
    mask_file : str
        Mask file for the voxel grid. Must be a NIFTI file.
    parcellation_file : str
        Parcellation file. Must be a NIFTI file.
    vector : np.ndarray
        Value at each parcel. Shape must be (n_parcels,).
    remove_subcortical_voxels : bool, optional
        Should we set the subcortical voxels to np.nan?

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
    parc = nib.load(parcellation_file)

    # Make sure parcellation is 4D and contains 1 for voxel assignment
    # to a parcel and 0 otherwise
    parcellation_grid = parc.get_fdata()
    if parcellation_grid.ndim == 3:
        unique_values = np.unique(parcellation_grid)[1:]
        parcellation_grid = np.array(
            [(parcellation_grid == value).astype(int) for value in unique_values]
        )
        parcellation_grid = np.rollaxis(parcellation_grid, 0, 4)
        parc = nib.Nifti1Image(parcellation_grid, parc.affine, parc.header)

    # Make sure the parcellation grid matches the mask file
    parc = image.resample_to_img(
        parc,
        mask,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    parcellation_grid = parc.get_fdata()

    # Make a 2D array of voxel weights for each parcel
    n_parcels = parc.shape[-1]

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

    if remove_subcortical_voxels:
        if voxel_grid.shape != (23, 27, 23):
            raise ValueError(
                "remove_subcortical_voxels=True is only compatible with "
                "8x8x8 mm voxel grids."
            )

        # We guess which voxels are subcortical and set them to nan (if zero)
        for xx in range(10, 13):
            for yy in range(12, 19):
                if yy > 15 or yy < 13:
                    for zz in range(10, 11):
                        if voxel_grid[xx, yy, zz] == 0:
                            voxel_grid[xx, yy, zz] = np.nan
                else:
                    for zz in range(7, 12):
                        if voxel_grid[xx, yy, zz] == 0:
                            voxel_grid[xx, yy, zz] = np.nan

        # Suppress warning when plotting
        warnings.filterwarnings("ignore", message="Mean of empty slice")

    return voxel_grid


def parcellate(
    fns: OSLFilenames,
    voxel_data: np.ndarray,
    voxel_coords: np.ndarray,
    method: str,
    parcellation_file: str,
    orthogonalisation: Optional[str] = None,
) -> np.ndarray:
    """Parcellate data.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    voxel_data : np.ndarray
        (nvoxels x n_time) or (nvoxels x n_time x n_trials) and is assumed to be
        on the same grid as parcellation.
    voxel_coords :
        (nvoxels x 3) coordinates in mm in same space as parcellation.
    method : str, optional
        'pca'           - take 1st PC of voxels
        'spatial_basis' - The parcel time-course for each spatial map is the
                          1st PC from all voxels, weighted by the spatial map.
        If the parcellation is unweighted and non-overlapping, 'spatial_basis'
        will give the same result as 'PCA' except with a different normalisation.
    parcellation_file : str
        Path to parcellation file. In same space as voxel_coords.
    orthogonalisation : str, optional
        Method for orthogonalising the data. Can be None or 'symmetric'.

    Returns
    -------
    parcel_data : np.ndarray
        Parcellated data. Shape is (parcels, time) or (parcels, time, epochs).
    """
    print("")
    print("Parcellating data")
    print("-----------------")

    if orthogonalisation not in [None, "symmetric"]:
        raise ValueError("orthogonalisation must be None or 'symmetric'.")

    if method not in ["pca", "spatial_basis"]:
        raise ValueError("method must be 'pca' or 'spatial_basis'.")

    # Get parcellation file
    parcellation_file = files.check_exists(
        parcellation_file, files.parcellation.directory
    )

    # Resample parcellation to match the mask
    parcellation = _resample_parcellation(fns, parcellation_file, voxel_coords)

    # Calculate parcel time courses
    parcel_data, _, _ = _get_parcel_data(voxel_data, parcellation, method=method)

    # Orthogonalisation
    if orthogonalisation == "symmetric":
        parcel_data = _symmetric_orthogonalisation(
            parcel_data, maintain_magnitudes=True
        )

    return parcel_data


def save_as_fif(
    parcel_data: np.ndarray,
    raw: Union[mne.io.Raw, mne.Epochs],
    filename: str,
    extra_chans: Optional[Union[str, List[str]]] = None,
) -> None:
    """Save parcellated data as a fif file.

    Parameters
    ----------
    parcel_data : np.ndarray
        (parcels, time) or (parcels, time, epochs) data.
    raw : mne.Raw or mne.Epochs
        MNE Raw or Epochs objects to get info from.
    filename : str
        Output file path.
    extra_chans : str or list of str
        Extra channels, e.g. 'stim' or 'emg', to include in the parc_raw object.
        Defaults to 'stim'. stim channels are always added to parc_raw if they
        are present in raw.
    """
    print(f"Saving {filename}")

    if isinstance(raw, mne.Epochs):
        # Save as a MNE Epochs object
        parc_epo = _convert2mne_epochs(parcel_data, raw)
        parc_epo.save(filename, overwrite=True)

    else:
        # Save as a MNE Raw object
        if extra_chans is None:
            extra_chans = "stim"
        parc_raw = convert_to_mne_raw(
            parcel_data,
            raw,
            ch_names=[f"parcel_{i}" for i in range(parcel_data.shape[0])],
            extra_chans=extra_chans,
        )
        parc_raw.save(filename, overwrite=True)


def plot_psds(
    parc_fif: str,
    parcellation_file: str,
    fmin: float = 0.5,
    fmax: float = 45,
    filename: Optional[str] = None,
) -> None:
    """Plot PSD of each parcel time course.

    Parameters
    ----------
    parc_fif : mne.Raw or mne.Epochs
        MNE Raw or Epochs object containing the parcel data.
    parcellation_file : str
        Path to parcellation file.
    fmin : float, optional
        Minimum frequency.
    fmax : float, optional
        Maximum frequency.
    filename : str, optional
        Output filename.
    """
    if "epo.fif" in parc_fif:
        raw = mne.Epochs(parc_fif)
    else:
        raw = mne.io.read_raw_fif(parc_fif)

    fs = raw.info["sfreq"]
    parc_ts = raw.get_data(picks="misc", reject_by_annotation="omit")

    if parc_ts.ndim == 3:
        # Calculate PSD for each epoch individually and average
        psd = []
        for i in range(parc_ts.shape[-1]):
            f, p = scipy.signal.welch(parc_ts[..., i], fs=fs, nperseg=fs, nfft=fs * 2)
            psd.append(p)
        psd = np.mean(psd, axis=0)
    else:
        # Calcualte PSD of continuous data
        f, psd = scipy.signal.welch(parc_ts, fs=fs, nperseg=fs, nfft=fs * 2)

    # Plot
    from osl_dynamics.utils.plotting import plot_psd_topo

    plot_psd_topo(
        f,
        psd,
        parcellation_file=parcellation_file,
        frequency_range=[fmin, fmax],
        filename=filename,
    )


def save_qc_plots(
    parc_fif: str,
    parcellation_file: str,
    output_dir: Union[str, Path],
    cmap: str = "hot",
) -> None:
    """Save parcellation QC plots.

    Saves the following files to output_dir:
    - 4_psd_topo.png: PSD topography plot
    - 4_power_delta.png: delta band power map
    - 4_power_theta.png: theta band power map
    - 4_power_alpha.png: alpha band power map
    - 4_power_beta.png: beta band power map
    - 4_power_gamma.png: gamma band power map

    Parameters
    ----------
    parc_fif : str
        Path to parcellated fif file.
    parcellation_file : str
        Parcellation file name.
    output_dir : str or Path
        Directory to save plots to.
    cmap : str, optional
        Colormap for power maps.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PSD topography
    plot_psds(
        parc_fif,
        parcellation_file=parcellation_file,
        filename=str(output_dir / "4_psd_topo.png"),
    )
    plt.close("all")

    # Band power maps
    from osl_dynamics.analysis import power, static
    from osl_dynamics.utils.plotting import plot_brain_surface

    parc_raw = mne.io.read_raw_fif(parc_fif)
    parc_ts = parc_raw.get_data(picks="misc", reject_by_annotation="omit")
    fs = parc_raw.info["sfreq"]
    f, psd = static.welch_spectra(
        parc_ts.T,
        sampling_frequency=fs,
        calc_coh=False,
    )

    mask_file = f"{files.mask.path}/MNI152_T1_8mm_brain.nii.gz"
    bands = {
        "delta": [1, 4],
        "theta": [4, 8],
        "alpha": [8, 13],
        "beta": [13, 30],
        "gamma": [30, 45],
    }
    for band_name, freq_range in bands.items():
        band_power = power.variance_from_spectra(f, psd, frequency_range=freq_range)
        plot_brain_surface(
            band_power,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            title=f"{band_name} ({freq_range[0]}-{freq_range[1]} Hz)",
            cmap=cmap,
            symmetric_cbar=False,
            filename=str(output_dir / f"4_power_{band_name}.png"),
        )
        plt.close("all")


def _resample_parcellation(fns, parcellation_file, voxel_coords):
    """Resample parcellation.

    Resample the parcellation so that the voxel coords correspond (using nearest
    neighbour) to the passed in coords. Passed in voxel_coords and parcellation
    must be in the same space, e.g. MNI.

    Used to make sure that the parcellation's voxel coords are the same as the
    voxel coords for some time series data.

    Parameters
    ----------
    parcellation_file : str
        Path to parcellation file. In same space as voxel_coords.
    voxel_coords :
        (nvoxels x 3) coordinates in mm in same space as parcellation.

    Returns
    -------
    parcellation_asmatrix : np.ndarray
        (nvoxels x n_parcels) resampled parcellation
    """
    gridstep = source_recon._get_gridstep(voxel_coords.T / 1000)
    print(f"gridstep = {gridstep} mm")

    path, name = os.path.split(
        os.path.splitext(os.path.splitext(parcellation_file)[0])[0]
    )

    parcellation_resampled = f"{fns.src_dir}/{name}_{gridstep}mm.nii.gz"

    # Create standard brain of the required resolution
    #
    # Command: flirt -in <parcellation_file> -ref <parcellation_file> \
    #          -out <parcellation_resampled> -applyisoxfm <gridstep>
    #
    # Note, this call raises:
    #
    #   Warning: An input intended to be a single 3D volume has multiple
    #   timepoints. Input will be truncated to first volume, but this
    #   functionality is deprecated and will be removed in a future release.
    #
    # However, it doesn't look like the input be being truncated, the
    # resampled parcellation appears to be a 4D volume.
    fsl_wrappers.flirt(
        parcellation_file,
        parcellation_file,
        out=parcellation_resampled,
        applyisoxfm=gridstep,
    )
    print(f"Resampled parcellation: {parcellation_resampled}")

    n_parcels = nib.load(parcellation_resampled).get_fdata().shape[3]
    n_voxels = voxel_coords.shape[1]

    # parcellation_asmatrix will be the parcels mapped onto the same dipole
    # grid as voxel_coords
    print("Finding nearest neighbour voxel")
    parcellation_asmatrix = np.zeros([n_voxels, n_parcels])
    for i in range(n_parcels):
        coords, vals = source_recon._niimask2mmpointcloud(parcellation_resampled, i)
        kdtree = scipy.spatial.KDTree(coords.T)

        # Find each voxel_coords best matching coords and assign
        # the corresponding parcel value to
        for j in range(n_voxels):
            distance, index = kdtree.query(voxel_coords[:, j])

            # Exclude from parcel any voxel_coords that are further than
            # gridstep away from the best matching coords
            if distance < gridstep:
                parcellation_asmatrix[j, i] = vals[index]

    return parcellation_asmatrix


def _get_parcel_data(voxel_data, parcellation_asmatrix, method="spatial_basis"):
    """Calculate parcel time courses.

    Parameters
    ----------
    voxel_data : np.ndarray
        (nvoxels x n_time) or (nvoxels x n_time x n_trials) and is assumed to be
        on the same grid as parcellation.
    parcellation_asmatrix: np.ndarray
        (nvoxels x n_parcels) and is assumed to be on the same grid as
        voxel_data.
    method : str, optional
        'pca'           - take 1st PC of voxels
        'spatial_basis' - The parcel time-course for each spatial map is the
                          1st PC from all voxels, weighted by the spatial map.
        If the parcellation is unweighted and non-overlapping, 'spatial_basis'
        will give the same result as 'PCA' except with a different normalisation.

    Returns
    -------
    parcel_data : np.ndarray
        n_parcels x n_time, or n_parcels x n_time x n_trials
    voxel_weightings : np.ndarray
        nvoxels x n_parcels
        Voxel weightings for each parcel to compute parcel_data from
        voxel_data
    voxel_assignments : bool np.ndarray
        nvoxels x n_parcels
        Boolean assignments indicating for each voxel the winner takes all
        parcel it belongs to
    """
    print(f"Calculating parcel time courses with {method}")

    if parcellation_asmatrix.shape[0] != voxel_data.shape[0]:
        Exception(
            f"Parcellation has {parcellation_asmatrix.shape[0]} voxels, "
            f"but data has {voxel_data.shape[0]}"
        )

    if len(voxel_data.shape) == 2:
        # Add dim for trials
        voxel_data = np.expand_dims(voxel_data, axis=2)
        added_dim = True
    else:
        added_dim = False

    n_parcels = parcellation_asmatrix.shape[1]
    n_time = voxel_data.shape[1]
    n_trials = voxel_data.shape[2]

    # Combine the trials and time dimensions together, we will
    # re-separate them after the parcel times eries are computed
    voxel_data_reshaped = np.reshape(
        voxel_data, (voxel_data.shape[0], n_time * n_trials)
    )
    parcel_data_reshaped = np.zeros((n_parcels, n_time * n_trials))

    voxel_weightings = np.zeros(parcellation_asmatrix.shape)

    if method == "spatial_basis":
        # estimate temporal-STD of data for normalisation
        temporal_std = np.maximum(
            np.std(voxel_data_reshaped, axis=1), np.finfo(float).eps
        )

        for pp in range(n_parcels):
            # Scale group maps so all have a positive peak of height 1 in case
            # there is a very noisy outlier, choose the sign from the top 5%
            # of magnitudes
            thresh = np.percentile(np.abs(parcellation_asmatrix[:, pp]), 95)
            mapsign = np.sign(
                np.mean(
                    parcellation_asmatrix[parcellation_asmatrix[:, pp] > thresh, pp]
                )
            )
            scaled_parcellation = (
                mapsign
                * parcellation_asmatrix[:, pp]
                / np.max(np.abs(parcellation_asmatrix[:, pp]))
            )

            # Weight all voxels by the spatial map in question.
            # Apply the mask first then weight to reduce memory use
            weighted_ts = voxel_data_reshaped[scaled_parcellation > 0, :]
            weighted_ts = np.multiply(
                weighted_ts,
                np.reshape(scaled_parcellation[scaled_parcellation > 0], [-1, 1]),
            )
            weighted_ts = weighted_ts - np.reshape(
                np.mean(weighted_ts, axis=1), [-1, 1]
            )

            # Perform SVD and take scores of 1st PC as the node time-series
            #
            # U is nVoxels by nComponents - the basis transformation
            # S*V holds nComponents by time sets of PCA scores
            # - the time series data in the new basis
            d, U = scipy.sparse.linalg.eigs(weighted_ts @ weighted_ts.T, k=1)
            U = np.real(U)
            d = np.real(d)
            S = np.sqrt(np.abs(np.real(d)))
            V = weighted_ts.T @ U / S
            pca_scores = S @ V.T

            # 0.5 is a decent arbitrary threshold used in fslnets after
            # playing with various maps
            this_mask = scaled_parcellation[scaled_parcellation > 0] > 0.5

            if np.any(this_mask):  # the mask is non-zero
                # U is the basis by which voxels in the mask are weighted to
                # form the scores of the 1st PC
                relative_weighting = np.abs(U[this_mask]) / np.sum(np.abs(U[this_mask]))
                ts_sign = np.sign(np.mean(U[this_mask]))
                ts_scale = np.dot(
                    np.reshape(relative_weighting, [-1]),
                    temporal_std[scaled_parcellation > 0][this_mask],
                )

                node_ts = (
                    ts_sign
                    * (ts_scale / np.maximum(np.std(pca_scores), np.finfo(float).eps))
                    * pca_scores
                )

                inds = np.where(scaled_parcellation > 0)[0]
                voxel_weightings[inds, pp] = (
                    ts_sign
                    * ts_scale
                    / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                    * (
                        np.reshape(U, [-1])
                        * scaled_parcellation[scaled_parcellation > 0].T
                    )
                )

            else:
                print(
                    f"WARNING: An empty parcel mask was found for parcel {pp} "
                    "when calculating its time-courses\n"
                    "The parcel will have a flat zero time-course.\n"
                    "Check this does not cause further problems with the analysis.\n"
                )

                node_ts = np.zeros(n_time * n_trials)
                inds = np.where(scaled_parcellation > 0)[0]
                voxel_weightings[inds, pp] = 0

            parcel_data_reshaped[pp, :] = node_ts

    elif method == "pca":
        print(
            "PCA assumes a binary parcellation.\n"
            "Parcellation will be binarised if it is not already "
            "(any voxels >0 are set to 1, otherwise voxels are set to 0), "
            "i.e. any weightings will be ignored.\n"
        )

        # Check that each voxel is only a member of one parcel
        if any(np.sum(parcellation_asmatrix, axis=1) > 1):
            print(
                "WARNING: Each voxel is meant to be a member of at most one "
                "parcel, when using the PCA method.\nResults may not be sensible"
            )

        # Estimate temporal-STD of data for normalisation
        temporal_std = np.maximum(
            np.std(voxel_data_reshaped, axis=1), np.finfo(float).eps
        )

        # Perform PCA on each parcel and select 1st PC scores to represent parcel
        for pp in range(n_parcels):
            if any(parcellation_asmatrix[:, pp]):  # non-zero
                parcel_data = voxel_data_reshaped[parcellation_asmatrix[:, pp] > 0, :]
                parcel_data = parcel_data - np.reshape(
                    np.mean(parcel_data, axis=1), [-1, 1]
                )

                # Perform svd and take scores of 1st PC as the node time-series
                #
                # U is nVoxels by nComponents - the basis transformation
                # S*V holds nComponents by time sets of PCA scores
                # - the time series data in the new basis
                d, U = scipy.sparse.linalg.eigs(parcel_data @ parcel_data.T, k=1)
                U = np.real(U)
                d = np.real(d)
                S = np.sqrt(np.abs(np.real(d)))
                V = parcel_data.T @ U / S
                pca_scores = S @ V.T

                # Restore sign and scaling of parcel time-series
                # U indicates the weight with which each voxel in the parcel
                # contributes to the 1st PC
                relative_weighting = np.abs(U) / np.sum(np.abs(U))
                ts_sign = np.sign(np.mean(U))
                ts_scale = np.dot(
                    np.reshape(relative_weighting, [-1]),
                    temporal_std[parcellation_asmatrix[:, pp] > 0],
                )

                node_ts = (
                    ts_sign
                    * ts_scale
                    / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                ) * pca_scores

                inds = np.where(parcellation_asmatrix[:, pp] > 0)[0]
                voxel_weightings[inds, pp] = (
                    ts_sign
                    * ts_scale
                    / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                    * np.reshape(U, [-1])
                )

            else:
                print(
                    f"WARNING: An empty parcel mask was found for parcel {pp} "
                    "when calculating its time-courses\n"
                    "The parcel will have a flat zero time-course.\n"
                    "Check this does not cause further problems with the analysis.\n"
                )

                node_ts = np.zeros(n_time * n_trials)
                inds = np.where(parcellation_asmatrix[:, pp] > 0)[0]
                voxel_weightings[inds, pp] = 0

            parcel_data_reshaped[pp, :] = node_ts

    else:
        Exception("Invalid method specified")

    # Re-separate the trials and time dimensions
    parcel_data = np.reshape(parcel_data_reshaped, (n_parcels, n_time, n_trials))
    if added_dim:
        parcel_data = np.squeeze(parcel_data, axis=2)

    # Compute voxel_assignments using winner takes all
    voxel_assignments = np.zeros(voxel_weightings.shape)
    for ivoxel in range(voxel_weightings.shape[0]):
        winning_parcel = np.argmax(voxel_weightings[ivoxel, :])
        voxel_assignments[ivoxel, winning_parcel] = 1

    return parcel_data, voxel_weightings, voxel_assignments


def _symmetric_orthogonalisation(
    timeseries, maintain_magnitudes=False, compute_weights=False
):
    """Symmetric orthogonalisation.

    Returns orthonormal matrix L which is closest to A, as measured by the
    Frobenius norm of (L-A). The orthogonal matrix is constructed from a
    singular value decomposition of A.

    If maintain_magnitudes is True, returns the orthogonal matrix L, whose
    columns have the same magnitude as the respective columns of A, and which
    is closest to A, as measured by the Frobenius norm of (L-A).

    Parameters
    ----------
    timeseries : numpy.ndarray
        (nparcels x ntpts) or (nparcels x ntpts x ntrials) data to orthoganlise.
        In the latter case, the ntpts and ntrials dimensions are concatenated.
    maintain_magnitudes : bool
    compute_weights : bool

    Returns
    -------
    ortho_timeseries : numpy.ndarray
        (nparcels x ntpts) or (nparcels x ntpts x ntrials) orthoganalised data
    weights : numpy.ndarray
        (optional output depending on compute_weights flag) weighting matrix
        such that, ortho_timeseries = timeseries * weights

    References
    ----------
    Colclough, G. L., Brookes, M., Smith, S. M. and Woolrich, M. W.,
    "A symmetric multivariate leakage correction for MEG connectomes,"
    NeuroImage 117, pp. 439-448 (2015)
    """
    print("Performing symmetric orthogonalisation")

    if len(timeseries.shape) == 2:
        # add dim for trials:
        timeseries = np.expand_dims(timeseries, axis=2)
        added_dim = True
    else:
        added_dim = False

    nparcels = timeseries.shape[0]
    ntpts = timeseries.shape[1]
    ntrials = timeseries.shape[2]
    compute_weights = False

    # combine the trials and time dimensions together,
    # we will re-separate them after the parcel timeseries are computed
    timeseries = np.transpose(np.reshape(timeseries, (nparcels, ntpts * ntrials)))

    if maintain_magnitudes:
        D = np.diag(np.sqrt(np.diag(np.transpose(timeseries) @ timeseries)))
        timeseries = timeseries @ D

    [U, S, V] = np.linalg.svd(timeseries, full_matrices=False)

    # we need to check that we have sufficient rank
    tol = max(timeseries.shape) * S[0] * np.finfo(type(timeseries[0, 0])).eps
    r = sum(S > tol)
    full_rank = r >= timeseries.shape[1]

    if full_rank:
        # polar factors of A
        ortho_timeseries = U @ np.conjugate(V)
    else:
        raise ValueError(
            "Not full rank, rank required is {}, but rank is only {}".format(
                timeseries.shape[1], r
            )
        )

    if compute_weights:
        # weights are a weighting matrix such that,
        # ortho_timeseries = timeseries * weights
        weights = np.transpose(V) @ np.diag(1.0 / S) @ np.conjugate(V)

    if maintain_magnitudes:
        # scale result
        ortho_timeseries = ortho_timeseries @ D

        if compute_weights:
            # weights are a weighting matrix such that,
            # ortho_timeseries = timeseries * weights
            weights = D @ weights @ D

    # Re-separate the trials and time dimensions
    ortho_timeseries = np.reshape(
        np.transpose(ortho_timeseries), (nparcels, ntpts, ntrials)
    )

    if added_dim:
        ortho_timeseries = np.squeeze(ortho_timeseries, axis=2)

    if compute_weights:
        return ortho_timeseries, weights
    else:
        return ortho_timeseries


def convert_to_mne_raw(
    data: np.ndarray,
    raw: mne.io.Raw,
    ch_names: Optional[List[str]] = None,
    extra_chans: Optional[Union[str, List[str]]] = None,
) -> mne.io.Raw:
    """Convert an array to an MNE Raw object, copying metadata from a reference.

    If ``data`` has fewer time points than ``raw``, bad segments are
    re-inserted as zeros so that the output has the same length as ``raw``.

    Parameters
    ----------
    data : np.ndarray
        (n_channels, n_samples) data array.
    raw : mne.io.Raw
        Reference Raw object. Timing, annotations, filter settings,
        description and extra channels are copied from this object.
    ch_names : list of str, optional
        Channel names. Defaults to ``channel_0, ..., channel_{n-1}``.
    extra_chans : str or list of str, optional
        Extra channel types (e.g. ``"stim"``, ``"emg"``) to copy from
        ``raw``. Defaults to ``None`` (no extra channels).

    Returns
    -------
    new_raw : mne.io.Raw
        New Raw object containing ``data`` with metadata from ``raw``.
    """
    if extra_chans is None:
        extra_chans = []
    if isinstance(extra_chans, str):
        extra_chans = [extra_chans]

    # Re-insert bad segments if data is shorter than raw
    if raw.get_data().shape[1] != data.shape[1]:
        _, times = raw.get_data(reject_by_annotation="omit", return_times=True)
        indices = raw.time_as_index(times, use_rounding=True)
        indices = indices[: data.shape[1]]
        full_data = np.zeros(
            [data.shape[0], len(raw.times)],
            dtype=np.float32,
        )
        full_data[:, indices] = data
    else:
        full_data = data

    # Create Info and Raw objects
    if ch_names is None:
        ch_names = [f"channel_{i}" for i in range(full_data.shape[0])]
    new_info = mne.create_info(
        ch_names=ch_names,
        ch_types="misc",
        sfreq=raw.info["sfreq"],
    )
    new_raw = mne.io.RawArray(full_data, new_info)

    # Copy filter info
    with new_raw.info._unlock():
        new_raw.info["highpass"] = float(raw.info["highpass"])
        new_raw.info["lowpass"] = float(raw.info["lowpass"])

    # Copy timing info
    new_raw.set_meas_date(raw.info["meas_date"])
    new_raw.__dict__["_first_samps"] = raw.__dict__["_first_samps"]
    new_raw.__dict__["_last_samps"] = raw.__dict__["_last_samps"]
    new_raw.__dict__["_cropped_samp"] = raw.__dict__["_cropped_samp"]

    # Copy annotations
    new_raw.set_annotations(raw._annotations)

    # Add extra channels
    for extra_chan in extra_chans:
        if extra_chan in raw:
            chan_raw = raw.copy().pick(extra_chan)
            chan_data = chan_raw.get_data()
            chan_info = mne.create_info(
                chan_raw.ch_names,
                raw.info["sfreq"],
                [extra_chan] * chan_data.shape[0],
            )
            chan_raw = mne.io.RawArray(chan_data, chan_info)
            new_raw.add_channels([chan_raw], force_update_info=True)

    # Copy description
    new_raw.info["description"] = raw.info["description"]

    return new_raw


def _convert2mne_epochs(parc_data, epochs, parcel_names=None):
    """Create and returns an MNE Epochs object that contains parcellated data.

    Parameters
    ----------
    parc_data : np.ndarray
        (nparcels x ntpts x epochs) parcel data.
    epochs : mne.Epochs
        mne.io.raw object that produced parc_data via source recon and
        parcellation. Info such as timings and bad segments will be copied
        from this to parc_raw.
    parcel_names : list of str
        List of strings indicating names of parcels. If None then names are
        set to be parcel_0,...,parcel_{n_parcels-1}.

    Returns
    -------
    parc_epo : mne.Epochs
        Generated parcellation in mne.Epochs format.
    """

    # Epochs info
    info = epochs.info

    # Create parc info
    if parcel_names is None:
        parcel_names = [f"parcel_{i}" for i in range(parc_data.shape[0])]

    parc_info = mne.create_info(
        ch_names=parcel_names, ch_types="misc", sfreq=info["sfreq"]
    )
    parc_events = epochs.events

    # Parcellated data Epochs object
    parc_epo = mne.EpochsArray(np.swapaxes(parc_data.T, 1, 2), parc_info, parc_events)

    # Copy the description from the sensor-level Epochs object
    parc_epo.info["description"] = epochs.info["description"]

    return parc_epo
