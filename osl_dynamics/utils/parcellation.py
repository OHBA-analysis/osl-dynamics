"""Parcellation related classes and functions."""

import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import image, plotting

from osl_dynamics import files

_logger = logging.getLogger("osl-dynamics")


class Parcellation:
    """Class for reading parcellation files.

    Parameters
    ----------
    file : str
        Path to parcellation file.
    """

    def __init__(self, file):
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

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.file)})"

    def data(self):
        return self.parcellation.get_fdata()

    def nonzero(self):
        return [np.nonzero(self.data()[..., i]) for i in range(self.n_parcels)]

    def nonzero_coords(self):
        return [
            nib.affines.apply_affine(
                self.parcellation.affine,
                np.array(nonzero).T,
            )
            for nonzero in self.nonzero()
        ]

    def weights(self):
        return [
            self.data()[..., i][nonzero] for i, nonzero in enumerate(self.nonzero())
        ]

    def roi_centers(self):
        """Centroid of each parcel in MNI coordinates."""
        return np.array(
            [
                np.average(c, weights=w, axis=0)
                for c, w in zip(self.nonzero_coords(), self.weights())
            ]
        )

    def plot(self, **kwargs):
        return plot_parcellation(self, **kwargs)

    @staticmethod
    def find_files():
        paths = Path(files.parcellation.directory).glob("*")
        paths = [path.name for path in paths if not path.name.startswith("__")]
        return sorted(paths)


def plot_parcellation(parcellation, **kwargs):
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
    return plotting.plot_markers(
        np.zeros(parcellation.n_parcels),
        parcellation.roi_centers(),
        colorbar=False,
        node_cmap="binary_r",
        **kwargs,
    )


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
    parcellation = image.resample_to_img(
        parcellation,
        mask,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
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
