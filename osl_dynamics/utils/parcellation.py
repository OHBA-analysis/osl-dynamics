"""Parcellation related classes and functions.

"""

import nibabel as nib
import numpy as np
from pathlib import Path
from nilearn.plotting import plot_markers

from osl_dynamics import files


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
    return plot_markers(
        np.zeros(parcellation.n_parcels),
        parcellation.roi_centers(),
        colorbar=False,
        node_cmap="binary_r",
        **kwargs,
    )
