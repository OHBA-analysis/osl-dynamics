"""Parcellation."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import mne
import scipy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting as nilearn_plotting

from osl_dynamics import files
from osl_dynamics.utils import misc
from osl_dynamics.meeg.session import Session, check_up_to_date

from . import source_recon

_logger = logging.getLogger("osl-dynamics")


class Parcellation:
    """Class for reading parcellation files.

    Parameters
    ----------
    file : str
        Path to parcellation file.
    """

    def __init__(self, file: str | Parcellation) -> None:
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

    def plot(self, **kwargs) -> object:
        return plot_parcellation(self, **kwargs)

    @staticmethod
    def find_files() -> list[str]:
        paths = Path(files.parcellation.directory).glob("*")
        paths = [path.name for path in paths if not path.name.startswith("__")]
        return sorted(paths)


def plot_parcellation(parcellation: str | Parcellation, **kwargs) -> object:
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


def parcel_vector_to_nifti(
    vector: np.ndarray,
    parcellation_file: str,
    remove_subcortical_voxels: bool = False,
) -> nib.Nifti1Image:
    """Takes a vector of parcel values and return a NIFTI image.

    The image is on the voxel grid of the parcellation file. The value at
    each voxel is the sum of the parcel values weighted by the (normalised)
    weight of the voxel for each parcel.

    Parameters
    ----------
    vector : np.ndarray
        Value at each parcel. Shape must be (n_parcels,).
    parcellation_file : str
        Parcellation file. Must be a NIFTI file.
    remove_subcortical_voxels : bool, optional
        Should we set the subcortical voxels to np.nan?

    Returns
    -------
    img : nib.Nifti1Image
        3D image with the value at each voxel.
    """
    # Suppress INFO messages from nibabel
    logging.getLogger("nibabel.global").setLevel(logging.ERROR)

    # Load the parcellation (4D with the weight of each voxel for each parcel)
    parcellation_file = files.check_exists(
        parcellation_file, files.parcellation.directory
    )
    parc = Parcellation(parcellation_file).parcellation
    grid_shape = parc.shape[:3]
    affine = parc.affine
    n_parcels = parc.shape[-1]

    # Check parcellation is compatible
    vector = np.asarray(vector)
    if vector.shape[0] != n_parcels:
        raise ValueError(
            f"parcellation_file has {n_parcels} parcels, "
            f"but vector has {vector.shape[0]} values."
        )

    # 2D array of voxel weights for each parcel
    voxel_weights = parc.get_fdata().reshape(-1, n_parcels)

    # Normalise the voxels weights
    max_weights = voxel_weights.max(axis=0, keepdims=True)
    voxel_weights /= np.where(max_weights > 0, max_weights, 1)

    # Generate a vector containing value at each voxel
    voxel_values = voxel_weights @ vector

    # Final 3D voxel grid
    voxel_grid = voxel_values.reshape(grid_shape)

    if remove_subcortical_voxels:
        # We guess which voxels are subcortical and set them to nan (if zero).
        # The subcortical voxels are defined on the 8 mm MNI152 grid, we find
        # the voxel of this grid that contains each voxel
        ijk = np.indices(grid_shape).reshape(3, -1).T
        coords = nib.affines.apply_affine(affine, ijk)
        xx, yy, zz = np.rint((coords - [90, -126, -72]) / [-8, 8, 8]).astype(int).T
        subcortical = (
            (xx >= 10)
            & (xx <= 12)
            & (yy >= 12)
            & (yy <= 18)
            & np.where((yy > 15) | (yy < 13), zz == 10, (zz >= 7) & (zz <= 11))
        ).reshape(grid_shape)
        voxel_grid[subcortical & (voxel_grid == 0)] = np.nan

        # Suppress warning when plotting
        warnings.filterwarnings("ignore", message="Mean of empty slice")

    return nib.Nifti1Image(voxel_grid, affine)


def parcellate(
    voxel_data: np.ndarray,
    voxel_coords: np.ndarray,
    parcellation_file: str,
    method: str = "pca",
    orthogonalisation: str | None = None,
) -> np.ndarray:
    """Parcellate data.

    See :func:`parcellate_lcmv` to calculate parcel time courses directly from
    the sensor data (without the voxel data).

    Parameters
    ----------
    voxel_data : np.ndarray
        (nvoxels x n_time) or (nvoxels x n_time x n_trials).
    voxel_coords :
        (nvoxels x 3) coordinates in mm in same space as parcellation. Each
        voxel is assigned the parcel weights of the parcellation voxel that
        contains it, so the voxels do not need to be on the same grid as the
        parcellation.
    parcellation_file : str
        Path to parcellation file. In same space as voxel_coords. The weights
        must be non-negative.
    method : str, optional
        'pca'      - The parcel time course is the 1st principal component of
                     the voxels in the parcel, with each voxel weighted by the
                     parcellation (for a binary parcellation all voxels in the
                     parcel have the same weight). It is scaled to the
                     standard deviation of the voxels in the parcel and its
                     sign is chosen so the voxels contribute positively on
                     average.
        'centroid' - Use the time course of the voxel nearest to each parcel
                     centroid.
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

    if method not in ["pca", "centroid"]:
        raise ValueError("method must be 'pca' or 'centroid'.")

    # Get parcellation file
    parcellation_file = files.check_exists(
        parcellation_file, files.parcellation.directory
    )

    if method == "centroid":
        parcel_data = _get_parcel_data_centroid(
            voxel_data, voxel_coords, parcellation_file
        )
    else:
        # Parcel weights at each voxel
        parcellation = _sample_parcellation(parcellation_file, voxel_coords)

        # Calculate parcel time courses
        parcel_data = _get_parcel_data_pca(
            voxel_data, parcellation, parcellation_file, voxel_coords
        )

    # Orthogonalisation
    if orthogonalisation == "symmetric":
        parcel_data = _symmetric_orthogonalisation(
            parcel_data, maintain_magnitudes=True
        )

    return parcel_data


def parcellate_lcmv(
    session: Session,
    parcellation_file: str,
    orthogonalisation: str | None = None,
    raw: mne.io.Raw | mne.Epochs | None = None,
    reject_by_annotation: str | list[str] | None = "omit",
) -> np.ndarray:
    """Calculate parcel time courses from sensor data with the LCMV filters.

    This gives the same result as :func:`source_recon.apply_lcmv_beamformer`
    followed by :func:`parcellate`, but it is much faster and uses much less
    memory because the voxel data is not calculated. The parcel time course
    (the rescaled 1st PC of the dipoles in the parcel) is calculated from the
    covariance of the dipoles (estimated from the sensor data) and applied to
    the sensor data as a spatial filter. See method='pca' in
    :func:`parcellate`.

    Parameters
    ----------
    session : Session
        Files of the session.
    parcellation_file : str
        Path to parcellation file (in MNI space). The weights must be
        non-negative.
    orthogonalisation : str, optional
        Method for orthogonalising the data. Can be None or 'symmetric'.
    raw : mne.io.Raw or mne.Epochs, optional
        The data to calculate parcel time courses for.
        If None, session.preproc_file is used.
    reject_by_annotation : str | list of str | None
        Annotation descriptions to omit when getting the data from a Raw
        object. If None, all time points are used.

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

    parcellation_file = files.check_exists(
        parcellation_file, files.parcellation.directory
    )

    check_up_to_date(session, "LCMV filters")

    # Sensor data after projection/whitening, shape is (channels, samples)
    data, filters, epochs_shape = source_recon._get_filter_input_data(
        session, raw, reject_by_annotation
    )
    if filters["is_free_ori"]:
        raise ValueError(
            "parcellate_lcmv requires a scalar beamformer, "
            "e.g. pick_ori='max-power-pre-weight-norm'."
        )
    # Beamformer weights for each voxel of the MNI grid, shape is
    # (voxels, channels). Voxels without a dipole (outside the inner skull)
    # have zero weights
    fwd = mne.read_forward_solution(session.fwd_model_file, verbose=False)
    voxel_coords = source_recon._get_mni_grid(session, fwd)
    if fwd["nsource"] != filters["weights"].shape[0]:
        raise ValueError(
            f"{session.filters_file} has {filters['weights'].shape[0]} dipoles, but "
            f"{session.fwd_model_file} has {fwd['nsource']}."
        )
    W = np.zeros((len(voxel_coords), filters["weights"].shape[1]))
    W[fwd["src"][0]["vertno"]] = filters["weights"]
    parcellation = _sample_parcellation(parcellation_file, voxel_coords)

    # Covariance of the dipoles is W @ C @ W.T
    data_mean = np.mean(data, axis=1)
    data_cov = np.cov(data, bias=True)

    def voxel_cov(inds):
        return W[inds] @ data_cov @ W[inds].T

    voxel_weightings = _get_parcel_weights(
        voxel_cov, parcellation, parcellation_file, voxel_coords
    )

    # Spatial filter for each parcel, shape is (parcels, channels)
    parcel_filters = voxel_weightings.T @ W
    parcel_data = parcel_filters @ data - (parcel_filters @ data_mean)[:, None]

    if epochs_shape is not None:
        parcel_data = parcel_data.reshape(-1, *epochs_shape)

    # Orthogonalisation
    if orthogonalisation == "symmetric":
        parcel_data = _symmetric_orthogonalisation(
            parcel_data, maintain_magnitudes=True
        )

    return parcel_data


def save_as_fif(
    parcel_data: np.ndarray,
    raw: mne.io.Raw | mne.Epochs,
    filename: str,
    extra_chans: str | list[str] | None = None,
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

    if isinstance(raw, mne.BaseEpochs):
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
    filename: str | None = None,
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
        epochs = mne.read_epochs(parc_fif)
        fs = epochs.info["sfreq"]
        parc_ts = epochs.get_data(picks="misc")
    else:
        raw = mne.io.read_raw_fif(parc_fif)
        fs = raw.info["sfreq"]
        parc_ts = raw.get_data(picks="misc", reject_by_annotation="omit")

    if parc_ts.ndim == 3:
        # Calculate PSD for each epoch individually and average.
        # parc_ts shape is (n_epochs, n_channels, n_samples)
        psd = []
        for i in range(parc_ts.shape[0]):
            f, p = scipy.signal.welch(parc_ts[i], fs=fs, nperseg=fs, nfft=fs * 2)
            psd.append(p)
        psd = np.mean(psd, axis=0)
    else:
        # Calculate PSD of continuous data
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
    output_dir: str | Path | None = None,
    power_maps: bool = False,
    show: bool = False,
    cmap: str = "hot",
) -> None:
    """Save parcellation QC plots.

    Saves the following files to output_dir:
    - psd_topo.png: PSD topography plot
    - power_maps.png: composite band power maps (only if power_maps=True)

    Parameters
    ----------
    parc_fif : str
        Path to parcellated fif file.
    parcellation_file : str
        Parcellation file name.
    output_dir : str or Path, optional
        Directory to save plots to. Defaults to the directory containing
        parc_fif.
    power_maps : bool, optional
        Whether to create band power map plots. Default is False.
    show : bool, optional
        Whether to display the plots interactively. Default is False.
    cmap : str, optional
        Colormap for power maps.
    """
    if output_dir is None:
        output_dir = Path(parc_fif).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from osl_dynamics.analysis import power
    from osl_dynamics.utils.plotting import plot_psd_topo, plot_brain_surface

    # Load data and compute PSD once
    if "epo.fif" in parc_fif:
        parc_raw = mne.read_epochs(parc_fif)
        parc_ts = parc_raw.get_data(picks="misc")
    else:
        parc_raw = mne.io.read_raw_fif(parc_fif)
        parc_ts = parc_raw.get_data(picks="misc", reject_by_annotation="omit")

    fs = parc_raw.info["sfreq"]
    f, psd = scipy.signal.welch(parc_ts, fs=fs, nperseg=fs, nfft=fs * 2)
    if psd.ndim == 3:
        # Average over epochs
        psd = psd.mean(axis=0)

    # PSD topography
    plot_psd_topo(
        f,
        psd,
        parcellation_file=parcellation_file,
        frequency_range=[1, 45],
        filename=str(output_dir / "psd_topo.png"),
    )
    if not show:
        plt.close("all")

    if not power_maps:
        return

    # Band power maps — render each band and composite into a single image
    bands = {
        "delta": [1, 4],
        "theta": [4, 8],
        "alpha": [8, 13],
        "beta": [13, 30],
        "gamma": [30, 45],
    }
    band_images = []
    for band_name, freq_range in bands.items():
        band_power = power.variance_from_spectra(f, psd, frequency_range=freq_range)
        fig, ax = plot_brain_surface(
            band_power,
            parcellation_file=parcellation_file,
            title=f"{band_name} ({freq_range[0]}-{freq_range[1]} Hz)",
            cmap=cmap,
            symmetric_cbar=False,
        )
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        band_images.append(img)
        plt.close(fig)

    composite_fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    for ax, img in zip(axes, band_images):
        ax.imshow(img)
        ax.axis("off")
    composite_fig.tight_layout()
    composite_fig.savefig(
        str(output_dir / "power_maps.png"), dpi=150, bbox_inches="tight"
    )
    if not show:
        plt.close(composite_fig)


def _sample_parcellation(parcellation_file: str, coords: np.ndarray) -> np.ndarray:
    """Sample a parcellation at a set of coordinates.

    Each coordinate is given the parcel weights of the parcellation voxel that
    contains it (nearest neighbour sampling). The coordinates do not need to be
    on the same grid as the parcellation.

    Parameters
    ----------
    parcellation_file : str
        Path to parcellation file. In same space as coords.
    coords : np.ndarray
        (n_coords, 3) coordinates in mm in the same space as the parcellation.

    Returns
    -------
    parcellation_asmatrix : np.ndarray
        (n_coords, n_parcels) parcel weights at each coordinate. Coordinates
        outside the parcellation's field of view are given zero weights.
    """
    parcellation = Parcellation(parcellation_file)
    img = parcellation.parcellation
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(img.affine), coords)).astype(
        int
    )
    inside = np.all((ijk >= 0) & (ijk < img.shape[:3]), axis=1)
    parcellation_asmatrix = np.zeros([len(coords), parcellation.n_parcels])
    parcellation_asmatrix[inside] = np.asarray(img.dataobj)[tuple(ijk[inside].T)]
    return parcellation_asmatrix


def _get_parcel_weights(
    voxel_cov: callable,
    parcellation_asmatrix: np.ndarray,
    parcellation_file: str,
    voxel_coords: np.ndarray,
) -> np.ndarray:
    """Calculate the voxel weights that give each parcel time course.

    The parcel time course is the (rescaled) 1st PC of the voxels in the
    parcel, weighted by the parcellation. This only depends on the covariance
    of the voxels in each parcel, so the voxel time courses are not needed.

    Parameters
    ----------
    voxel_cov : callable
        Function that takes an array of voxel indices and returns the
        (len(inds), len(inds)) covariance (normalised by n_samples) of these
        voxels.
    parcellation_asmatrix: np.ndarray
        (nvoxels x n_parcels) parcel weights for each voxel.
    parcellation_file : str
        Parcellation file, used for the error message if a parcel does not
        contain any dipoles.
    voxel_coords : np.ndarray
        (nvoxels, 3) MNI coordinates in mm, used for the error message if a
        parcel does not contain any dipoles.

    Returns
    -------
    voxel_weightings : np.ndarray
        (nvoxels x n_parcels) such that the parcel time courses are
        voxel_weightings.T @ (voxel_data - voxel_data.mean(axis=1)).
    """
    print("Calculating parcel time courses")

    if np.any(parcellation_asmatrix < 0):
        raise ValueError(f"The weights in {parcellation_file} must be non-negative.")

    voxel_weightings = np.zeros(parcellation_asmatrix.shape)
    empty_parcels = []
    for pp in range(parcellation_asmatrix.shape[1]):
        # Voxels in the parcel and their weights (scaled to a peak of 1)
        weights = parcellation_asmatrix[:, pp]
        inds = np.flatnonzero(weights > 0)
        if len(inds) == 0:
            empty_parcels.append(pp)
            continue
        spatial_map = weights[inds] / weights[inds].max()

        # Covariance of the voxels in the parcel. Voxels without a dipole
        # have no data
        cov = voxel_cov(inds)
        temporal_std = np.sqrt(np.diag(cov))

        # The sign and scale of the parcel time course is taken from the
        # voxels with a weight greater than 0.5 (the threshold used in
        # fslnets)
        this_mask = spatial_map > 0.5
        if not np.any(this_mask & (temporal_std > 0)):
            empty_parcels.append(pp)
            continue

        # 1st PC of the weighted voxels. The PCA scores are U.T @ weighted_ts
        # and their standard deviation is the square root of the eigenvalue
        weighted_cov = spatial_map[:, None] * cov * spatial_map[None, :]
        d, U = misc.top_eig(weighted_cov, k=1)
        U = U[:, 0]
        pca_std = np.maximum(np.sqrt(np.abs(d[0])), np.finfo(float).eps)

        # Restore the sign and scaling of the parcel time course. U indicates
        # the weight with which each voxel in the parcel contributes to the
        # 1st PC
        relative_weighting = np.abs(U[this_mask]) / np.sum(np.abs(U[this_mask]))
        ts_sign = np.sign(np.mean(U[this_mask]))
        ts_scale = np.dot(relative_weighting, temporal_std[this_mask])

        voxel_weightings[inds, pp] = ts_sign * ts_scale / pca_std * U * spatial_map

    if empty_parcels:
        raise ValueError(
            _empty_parcels_message(empty_parcels, parcellation_file, voxel_coords)
        )

    return voxel_weightings


def _empty_parcels_message(
    empty_parcels: list[int], parcellation_file: str, voxel_coords: np.ndarray
) -> str:
    """Error message for parcels that do not contain any dipoles."""
    msg = f"{len(empty_parcels)} parcel(s) do not contain any dipoles: {empty_parcels}."
    parcellation = Parcellation(parcellation_file)
    centres = np.round(parcellation.roi_centers()[empty_parcels]).astype(int)
    msg += f" MNI coordinates (mm) of the parcel centres: {centres.tolist()}.\n\n"
    msg += "This can happen if:\n"
    parcellation_res = float(parcellation.parcellation.header.get_zooms()[0])
    gridstep = source_recon._get_gridstep(voxel_coords / 1000)
    if gridstep > parcellation_res + 0.5:
        msg += (
            f"- The dipole grid ({gridstep} mm) is coarser than the "
            f"parcellation ({parcellation_res:g} mm), so small parcels can "
            "fall between dipoles. Use rhino.forward_model with "
            f"gridstep={parcellation_res:g}, or a parcellation with larger "
            "parcels.\n"
        )
    msg += (
        "- The parcel is outside the subject's inner skull surface (dipoles "
        "outside it, or closer than mindist to it, are removed by "
        "rhino.forward_model). Check the surface extraction and "
        "coregistration plots, or use a smaller mindist in rhino.forward_model."
    )
    return msg


def _get_parcel_data_pca(
    voxel_data: np.ndarray,
    parcellation_asmatrix: np.ndarray,
    parcellation_file: str,
    voxel_coords: np.ndarray,
) -> np.ndarray:
    """Calculate parcel time courses using PCA over the voxels in each parcel.

    Parameters
    ----------
    voxel_data : np.ndarray
        (nvoxels x n_time) or (nvoxels x n_time x n_trials).
    parcellation_asmatrix: np.ndarray
        (nvoxels x n_parcels) parcel weights for each voxel.
    parcellation_file : str
        Parcellation file, used for the error message if a parcel does not
        contain any dipoles.
    voxel_coords : np.ndarray
        (nvoxels, 3) MNI coordinates in mm, used for the error message if a
        parcel does not contain any dipoles.

    Returns
    -------
    parcel_data : np.ndarray
        n_parcels x n_time, or n_parcels x n_time x n_trials
    """
    if parcellation_asmatrix.shape[0] != voxel_data.shape[0]:
        raise ValueError(
            f"Parcellation has {parcellation_asmatrix.shape[0]} voxels, "
            f"but data has {voxel_data.shape[0]}"
        )

    # Combine the trials and time dimensions together, we will
    # re-separate them after the parcel time series are computed
    voxel_data_reshaped = np.reshape(voxel_data, (voxel_data.shape[0], -1))
    voxel_mean = np.mean(voxel_data_reshaped, axis=1)

    def voxel_cov(inds):
        x = voxel_data_reshaped[inds] - voxel_mean[inds, None]
        return x @ x.T / x.shape[1]

    voxel_weightings = _get_parcel_weights(
        voxel_cov, parcellation_asmatrix, parcellation_file, voxel_coords
    )

    parcel_data = (
        voxel_weightings.T @ voxel_data_reshaped
        - (voxel_weightings.T @ voxel_mean)[:, None]
    )

    # Re-separate the trials and time dimensions
    return np.reshape(parcel_data, (-1,) + voxel_data.shape[1:])


def _get_parcel_data_centroid(
    voxel_data: np.ndarray, voxel_coords: np.ndarray, parcellation_file: str
) -> np.ndarray:
    """Calculate parcel time courses using the voxel nearest to each parcel
    centroid.

    Parameters
    ----------
    voxel_data : np.ndarray
        (n_voxels, n_time) or (n_voxels, n_time, n_trials) and is assumed to be
        on the same grid as voxel_coords.
    voxel_coords : np.ndarray
        (n_voxels, 3) voxel coordinates in mm in the same space as the
        parcellation.
    parcellation_file : str
        Path to parcellation file.

    Returns
    -------
    parcel_data : np.ndarray
        (n_parcels, n_time) or (n_parcels, n_time, n_trials).
    """
    print("Calculating parcel time courses with centroid")

    parcellation = Parcellation(parcellation_file)
    centers = parcellation.roi_centers()  # (n_parcels, 3) in mm

    gridstep = source_recon._get_gridstep(voxel_coords / 1000)

    kdtree = scipy.spatial.KDTree(voxel_coords)
    distances, indices = kdtree.query(centers)

    far = distances > gridstep
    if np.any(far):
        _logger.warning(
            f"{int(far.sum())} parcel centroid(s) are further than "
            f"{gridstep} mm from the nearest voxel."
        )
    if len(np.unique(indices)) < len(indices):
        _logger.warning(
            "Multiple parcels map to the same voxel under method='centroid'. "
            "Consider a finer voxel grid or a different method."
        )

    return voxel_data[indices]


def _symmetric_orthogonalisation(
    timeseries: np.ndarray,
    maintain_magnitudes: bool = False,
    compute_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
    ch_names: list[str] | None = None,
    extra_chans: str | list[str] | None = None,
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
        full_data = np.zeros([data.shape[0], len(raw.times)], dtype=np.float32)
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

    # Copy annotations
    annotations = raw.annotations.copy()
    if annotations.orig_time is None:
        annotations.onset -= raw.first_time
    for i, names in enumerate(annotations.ch_names):
        if not set(names).issubset(new_raw.ch_names):
            annotations.ch_names[i] = ()
    new_raw.set_annotations(annotations)

    # Copy description
    new_raw.info["description"] = raw.info["description"]

    return new_raw


def _convert2mne_epochs(
    parc_data: np.ndarray, epochs: mne.Epochs, parcel_names: list[str] | None = None
) -> mne.Epochs:
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
    parc_epo = mne.EpochsArray(
        np.swapaxes(parc_data.T, 1, 2),
        parc_info,
        events=parc_events,
        tmin=epochs.tmin,
        event_id=epochs.event_id,
    )

    # Copy the description from the sensor-level Epochs object
    parc_epo.info["description"] = epochs.info["description"]

    return parc_epo
