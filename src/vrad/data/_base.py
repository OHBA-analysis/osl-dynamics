import logging
from typing import Any, Union

import mat73
import numpy as np
from vrad import array_ops
from vrad.data.io import load_data
from vrad.data.manipulation import (
    covariance,
    eigen_decomposition,
    multiply_by_eigenvectors,
    pca,
    scale,
    standardize,
    time_embed,
    whiten_eigenvectors,
)
from vrad.utils import plotting
from vrad.utils.decorators import auto_repr

_logger = logging.getLogger("VRAD")


class Data:
    """An object for storing time series data with various methods to act on it.

    Data is designed to standardize the workflow required to work with inputs with
    the format numpy.ndarray, numpy files, MAT (MATLAB) files, MATv7.3 files and
    SPM MEEG objects (also from MATLAB).

    If the input provided is a numpy.ndarray, it is taken as is. If the input is a
    string, VRAD will check the file extension to see if it is .npy (read by
    numpy.load) or .mat. If a .mat file is found, it is first opened using
    mat73.loadmat and if that fails, mat73.loadmat. Any input other than these is
    considered valid if it can be converted to a numpy array using numpy.array.

    When importing from MAT files, the values in the class variable ignored_keys are
    ignored if found in the dictionary created by loadmat. If the key 'D' is found, the
    file will be treated as an SPM MEEG object and the data extracted from the .dat
    file defined within the dictionary.

    Once instantiated, any property or function which has not been specified for
    Data is provided by the internal numpy.ndarray, time_series. The array can be
    accessed using slice notation on the Data object (e.g. meg_data[:1000, 2:5]
    would return the first 1000 samples and channels 2, 3 and 4. The time axis of the
    array can also be reduced using the data_limits method. This creates an pair of
    internal variables which reduce the length of the data which can be extracted from
    the object without modifying the underlying array.

    A variety of methods are provided for preparing data for analysis. These are
    detailed below.

    Parameters
    ----------
    time_series: numpy.ndarray or str or array-like
        Either an array, array-like object or a string specifying the location of a
        NumPy or MATLAB file.
    sampling_frequency: float
        The sampling frequency of the time_series. The default of 1 means that each
        sample is considered to be a time point (i.e. 1Hz).

    Methods
    -------


    """

    @auto_repr
    def __init__(
        self, time_series: Union[np.ndarray, str, Any], sampling_frequency: float = 1,
    ):
        # Raw data
        self.time_series, self.sampling_frequency = load_data(
            time_series=time_series, sampling_frequency=sampling_frequency,
        )

        self._from_file = time_series if isinstance(time_series, str) else False
        self._n_total_samples = len(
            self.time_series.reshape(-1, self.time_series.shape[-1])
        )
        self._original_shape = self.time_series.shape

        # Flags for data manipulation
        self.prepared = False

        # Time axis
        self.t = np.arange(self.time_series.shape[0]) / self.sampling_frequency

    def __str__(self):
        return_string = [
            f"{self.__class__.__name__}:",
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

    def standardize(
        self,
        n_components: Union[float, int] = 0.9,
        pre_scale: bool = True,
        do_pca: Union[bool, str] = True,
        post_scale: bool = True,
    ):
        self.time_series = standardize(
            time_series=self.time_series,
            n_components=n_components,
            pre_scale=pre_scale,
            do_pca=do_pca,
            post_scale=post_scale,
        )

    def scale(self):
        self.time_series = scale(self.time_series)

    def pca(
        self,
        n_components: Union[int, float] = 1,
        whiten: bool = True,
        random_state: int = None,
    ):
        logging.info(f"Applying PCA with n_components={n_components}")
        self.time_series = pca(self.time_series, n_components, whiten, random_state)

    def time_embed(self, n_embeddings: int, random_seed: int = None):
        logging.info(f"Applying time embedding with n_embeddings={n_embeddings}")
        self.time_series = time_embed(
            self.time_series, n_embeddings, random_seed=random_seed
        )

    def eigen_decomposition_dimensionality_reduction(
        self, n_components: int, whiten: bool = True
    ):
        logging.info(
            f"Calculating eigen decomposition and keeping {n_components} components"
        )

        # Calculate the weighted covariance matrix for each session and average
        sigma = covariance(self.time_series)

        # Calculate the eigen decomposition for each session and keep the top
        # n_components
        eigenvalues, eigenvectors = eigen_decomposition(sigma, n_components)

        # Whiten the eigenvectors
        if whiten:
            eigenvectors = whiten_eigenvectors(eigenvalues, eigenvectors)

        # Apply dimensionality reduction
        self.time_series = multiply_by_eigenvectors(self.time_series, eigenvectors)

    def prepare(
        self,
        n_embeddings: int,
        n_pca_components: Union[int, float],
        whiten: bool,
        random_seed: int = None,
    ):
        """This method reproduces the data preparation performed in
           teh_groupinference_parcels.m
        """
        if not self.prepared:
            self.time_embed(n_embeddings, random_seed=random_seed)
            self.scale()
            self.eigen_decomposition_dimensionality_reduction(
                n_pca_components, whiten=whiten
            )
            self.prepared = True

        else:
            logging.warning("Data has already been prepared. No changes made.")

    def plot(self, n_samples: int = 10000, filename: str = None):
        """Plot time_series.

        """
        plotting.plot_time_series(
            self.time_series, n_samples=n_samples, filename=filename
        )

    def savemat(self, filename: str, field_name: str = "X"):
        """Save time_series to a .mat file.

        Parameters
        ----------
        filename: str
            The file to save to (with or without .mat extension).
        field_name: str
            The dictionary key (MATLAB object field) which references the data.
        """
        mat73.savemat(filename, {field_name: self.time_series})

    def save(self, filename: str):
        """Save time_series to a numpy (.npy) file.

        Parameters
        ----------
        filename: str
            The file to save to (with or without .npy extension).
        """
        np.save(filename, self.time_series)


# noinspection PyPep8Naming
class OSL_HMM:
    """Import and encapsulate OSL HMMs"""

    def __init__(self, filename):
        self.filename = filename
        self.hmm = mat73.loadmat(filename)["hmm"]

        self.state = self.hmm.state
        self.k = int(self.hmm.K)
        self.p = self.hmm.P
        self.dir_2d_alpha = self.hmm.Dir2d_alpha
        self.pi = self.hmm.Pi
        self.dir_alpha = self.hmm.Dir_alpha
        self.prior = self.hmm.prior
        self.train = self.hmm.train

        self.data_files = self.hmm.data_files
        self.epoched_state_path_sub = self.hmm.epoched_statepath_sub
        self.filenames = self.hmm.filenames
        self.f_sample = self.hmm.fsample
        self.gamma = self.hmm.gamma
        self.is_epoched = self.hmm.is_epoched
        self.options = self.hmm.options
        self.state_map_parcel_vectors = self.hmm.statemap_parcel_vectors
        self.subject_state_map_parcel_vectors = self.hmm.statemap_parcel_vectors_persubj
        self.state_maps = self.hmm.statemaps
        self.state_path = self.hmm.statepath.astype(np.int)
        self.subject_indices = self.hmm.subj_inds

        # Aliases
        self.covariances = np.array([state["Gam_rate"] for state in self.state.Omega])
        self.state_time_course = self.gamma
        self.viterbi_path = array_ops.get_one_hot(self.state_path)

    def plot_covariances(self, *args, **kwargs):
        """Wraps plotting.plot_matrices for self.covariances."""
        plotting.plot_matrices(self.covariances, *args, **kwargs)

    def plot_states(self, *args, **kwargs):
        """Wraps plotting.highlight_states for self.viterbi_path."""

        plotting.highlight_states(self.viterbi_path, *args, **kwargs)

    def __str__(self):
        return f"OSL HMM object from file {self.filename}"
