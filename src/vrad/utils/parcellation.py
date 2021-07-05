from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
from vrad import files


class Parcellation:
    def __init__(self, file: Union[str, Path]):
        self.file = files.check_exists(file, files.parcellation.directory)
        self.parcellation = nib.load(self.file)
        self.dims = self.parcellation.shape[:3]
        self.n_parcels = self.parcellation.shape[3]

    def __repr__(self):
        return f'{self.__class__.__name__}("{str(self.file)}")'

    def data(self):
        return self.parcellation.get_fdata()

    def nonzero(self):
        return [np.nonzero(self.data()[..., i]) for i in range(self.n_parcels)]

    def nonzero_coords(self):
        return [
            nib.affines.apply_affine(self.parcellation.affine, np.array(nonzero).T)
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

    @staticmethod
    def find_files():
        paths = Path(files.parcellation.directory).glob("*")
        paths = [path.name for path in paths if not path.name.startswith("__")]
        return sorted(paths)
