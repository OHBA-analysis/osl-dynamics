import logging
from typing import Union

import mat73
import numpy as np
from vrad.data.io import load_data
from vrad.data.manipulation import scale, time_embed
from vrad.utils import plotting
from vrad.utils.decorators import auto_repr

_logger = logging.getLogger("VRAD")


class Subject:
    """A class for single-subject data."""

    @auto_repr
    def __init__(
        self,
        time_series: Union[str, np.ndarray, list],
        _id: int = None,
        sampling_frequency: float = 1.0,
    ):
        # Raw data
        self.time_series, self.sampling_frequency = load_data(
            time_series=time_series, sampling_frequency=sampling_frequency,
        )

        # Fields
        self._from_file = time_series if isinstance(time_series, str) else False
        self._id = _id
        self._original_shape = self.time_series.shape
        self.prepared = False

        # Time axis
        self.t = np.arange(self.time_series.shape[0]) / self.sampling_frequency

    def __str__(self):
        return_string = [
            f"{self.__class__.__name__} {self._id}:",
            f"from_file: {self._from_file}",
            f"original_shape: {self._original_shape}",
            f"current_shape: {self.time_series.shape}",
            f"prepared: {self.prepared}",
        ]
        return "\n  ".join(return_string)

    def __getitem__(self, val):
        return self.time_series[val]

    def __getattr__(self, attr):
        if attr[:2] == "__":
            raise AttributeError(f"No attribute called {attr}.")
        return getattr(self.time_series, attr)

    def __array__(self, *args, **kwargs):
        return np.asarray(self.time_series, *args, **kwargs)

    def scale(self):
        self.time_series = scale(self.time_series)

    def time_embed(self, n_embeddings: int, random_seed: int = None):
        logging.info(f"Applying time embedding with n_embeddings={n_embeddings}")
        self.time_series = time_embed(
            self.time_series, n_embeddings, random_seed=random_seed
        )

    def plot(self, n_samples: int = 10000, filename: str = None):
        """Plot data.

        """
        plotting.plot_time_series(
            self.time_series, n_samples=n_samples, filename=filename
        )

    def savemat(self, filename: str, field_name: str = "X"):
        """Save data to a .mat file.

        Parameters
        ----------
        filename: str
            The file to save to (with or without .mat extension).
        field_name: str
            The dictionary key (MATLAB object field) which references the data.
        """
        mat73.savemat(filename, {field_name: self.time_series})

    def save(self, filename: str):
        """Save data to a numpy (.npy) file.

        Parameters
        ----------
        filename: str
            The file to save to (with or without .npy extension).
        """
        np.save(filename, self.time_series)
