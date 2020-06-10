import logging
from datetime import datetime
from typing import Any, Tuple, Union

import mat73
import numpy as np
import scipy.io
from sklearn.decomposition import PCA
from taser import plotting
from taser.decorators import auto_repr
from taser.helpers.misc import time_axis_first


class MEGData:
    """An object for storing time series data with various methods to act on it.

    MEGData is designed to standardize the workflow required to work with inputs with
    the format numpy.ndarray, numpy files, MAT (MATLAB) files, MATv7.3 files and
    SPM MEEG objects (also from MATLAB).

    If the input provided is a numpy.ndarray, it is taken as is. If the input is a
    string, TASER will check the file extension to see if it is .npy (read by
    numpy.load) or .mat. If a .mat file is found, it is first opened using
    scipy.io.loadmat and if that fails, mat73.loadmat. Any input other than these is
    considered valid if it can be converted to a numpy array using numpy.array.

    When importing from MAT files, the values in the class variable ignored_keys are
    ignored if found in the dictionary created by loadmat. If the key 'D' is found, the
    file will be treated as an SPM MEEG object and the data extracted from the .dat
    file defined within the dictionary.

    If multiple time series are found in the file, the user can specify multi_sequence.
    With its default value 'all', the time series will be concatenated along the time
    axis. Otherwise and integer specifies which time series to read.

    Once instantiated, any property or function which has not been specified for
    MEGData is provided by the internal numpy.ndarray, time_series. The array can be
    accessed using slice notation on the MEGData object (e.g. meg_data[:1000, 2:5]
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
    multi_sequence: str or int
        If the time_series provided contains multiple time series, "all" will
        concatenate them while providing an int will specify the corresponding array.

    Methods
    -------


    """

    ignored_keys = [
        "__globals__",
        "__header__",
        "__version__",
        "save_time",
        "pca_applied",
        "T",
    ]

    @auto_repr
    def __init__(
        self,
        time_series: Union[np.ndarray, str, Any],
        sampling_frequency: float = 1,
        multi_sequence: Union[str, int] = "all",
    ):
        self.from_file = time_series if isinstance(time_series, str) else False

        if isinstance(time_series, str):
            if time_series[-4:] == ".npy":
                time_series = np.load(time_series)
            elif time_series[-4:] == ".mat":
                try:
                    mat = scipy.io.loadmat(time_series)
                except NotImplementedError:
                    logging.info(
                        f"{time_series} is a MAT v7.3 file so importing will"
                        f" be handled by `mat73`."
                    )
                    mat = mat73.loadmat(time_series)
                finally:
                    if "D" in mat:
                        logging.info(
                            "Assuming that key 'D' corresponds to an "
                            "SPM MEEG object."
                        )
                        time_series, sampling_frequency = load_spm(time_series)
                    else:
                        for key in mat:
                            if key not in MEGData.ignored_keys:
                                time_series = mat[key]

        if isinstance(time_series, list):
            if multi_sequence == "all":
                logging.warning(
                    f"{len(time_series)} sequences detected. "
                    f"Concatenating along first axis."
                )
                time_series = np.concatenate(time_series)
            if isinstance(multi_sequence, int):
                logging.warning(
                    f"{len(time_series)} sequences detected. "
                    f"Using sequence with index {multi_sequence}."
                )
                time_series = time_series[multi_sequence]

        self.sampling_frequency = sampling_frequency
        # TODO: Make raw_data read only using @property and self._raw_data
        self.raw_data = np.array(time_series)
        self.time_series = self.raw_data.copy()
        self.pca_applied = False

        self.t = None
        if time_series.ndim == 2:
            self.t = np.arange(self.time_series.shape[0]) / self.sampling_frequency
            self.time_axis_first()

        self.n_min, self.n_max = None, None

    def __str__(self):
        return_string = [
            f"{self.__class__.__name__}:",
            f"from_file: {self.from_file}",
            f"n_channels: {self.time_series.shape[1]}",
            f"n_time_points: {self.time_series.shape[0]}",
            f"pca_applied: {self.pca_applied}",
            f"data_limits: {self.n_min}, {self.n_max}",
            f"original_shape: {self.raw_data.shape}",
            f"current_shape: {self[:].shape}",
        ]
        return "\n  ".join(return_string)

    def __getitem__(self, val):
        return self.time_series[self.n_min : self.n_max][val]

    def __getattr__(self, attr):
        if attr[:2] == "__":
            raise AttributeError(f"No attribute called {attr}.")
        return getattr(self[:], attr)

    def __array__(self, *args, **kwargs):
        return np.asarray(self[:], *args, **kwargs)

    def data_limits(self, n_min: int = None, n_max: int = None):
        """Set the maximum and minimum sample numbers for the object.

        For a time_series of length 1000, meg_data.data_limits(5, 100) would mean that
        calls using numpy.array(meg_data) or meg_data[:] would only be viewing
        only the 5th to the 100th sample. Any new indexing is performed relative to
        the values n_min and n_max. The underlying time_series remains unchanged.

        Parameters
        ----------
        n_min: int
            The index of the first sample to be in the view.
        n_max: int
            The index of the final sample to be in the view.
        """
        self.n_min = n_min
        self.n_max = n_max

    @property
    def shape(self):
        """Get the shape of time_series with any modifications from data_limits.

        Returns
        -------
        shape: tuple of int
            The shape of the view of time_series provided by the object.

        """
        return self[:].shape

    @property
    def data_shape(self):
        """Get the shape of time_series without any modifications from data_limits.

        Returns
        -------
        shape: tuple of int
            The shape of time_series, unmodified by data_limits.

        """
        return self.time_series.shape

    def time_axis_first(self):
        """Forces the longer axis of the data to be the first indexed axis.

        If time_series is provided with dimensions (channel x time) where time is
        assumed to be the longer of the two axes, time_series will be transposed. This
        affects the internal variable but does not cause any change to raw_data which
        is reserved to be a copy of the data provided.

        """
        self.time_series, transposed = time_axis_first(self.time_series)
        if transposed:
            logging.warning("Assuming time to be the longer axis and transposing.")

    def trim_trials(
        self, trial_start: int = None, trial_cutoff: int = None, trial_skip: int = None
    ):
        """Remove trials from input data.

        If given as a three dimensional input with axes (channels x trials x time),
        remove trials by slicing and stepping.

        Parameters
        ----------
        trial_start: int
            The first trial to keep.
        trial_cutoff: int
            The last trial to keep.
        trial_skip: int
            How many steps to take between selected trials.

        """
        if self.ndim == 3:
            self.time_series = self.time_series[
                :, trial_start:trial_cutoff, ::trial_skip
            ]
        else:
            logging.warning(f"Array is not 3D (ndim = {self.ndim}). Can't trim trials.")

    def make_continuous(self):
        """Given trial data, return a continuous time series.

        With data input in the form (channels x trials x time), reshape the array to
        create a (time x channels) array. Wraps trials_to_continuous.

        """
        self.time_series = trials_to_continuous(self.time_series)
        self.t = np.arange(self.time_series.shape[0]) / self.sampling_frequency

    def standardize(
        self,
        n_components: Union[float, int] = 0.9,
        pre_scale: bool = True,
        do_pca: Union[bool, str] = True,
        post_scale: bool = True,
    ):
        """Function for scaling and performing PCA on time_series.

        Wraps MEGData.scale and MEGData.pca.

        Parameters
        ----------
        n_components: int or float
            If >1, number of components to be kept in PCA. If <1, the amount of
            variance to be explained by PCA. Passed to MEGData.pca.
        pre_scale: bool
            If True (default) scale data to have mean of zero and standard
            deviation of one before PCA is applied.
        do_pca: bool or str
            If True perform PCA on time_series. n_components = 1 is equivalent to
            False. If PCA has already been performed on time_series, set to "force".
            This is a safety check to make sure PCA isn't accidentally run twice.
        post_scale: bool
            If True (default) scale data to have mean of zero and standard
            deviation of one after PCA is applied.
        """
        force_pca = False
        if self.pca_applied and do_pca == "force":
            force_pca = True
        if pre_scale:
            self.scale()
        if do_pca:
            self.pca(n_components, force=force_pca)
        if post_scale:
            self.scale()

    def scale(self):
        """Scale time_series to have mean zero and standard deviation 1.

        """
        self.time_series = (
            self.time_series - self.time_series.mean(axis=0)
        ) / self.time_series.std(axis=0)

    def pca(self, n_components: Union[int, float] = 1, force=False):
        """Perform PCA on time_series.

        Wrapper for sklearn.decomposition.PCA.

        Parameters
        ----------
        n_components: float or int
            If >1, number of components to be kept in PCA. If <1, the amount of
            variance to be explained by PCA. If equal to 1, no PCA applied.
        force: bool
            If True, apply PCA even if it has already been applied.

        Returns
        -------

        """
        if self.pca_applied and not force:
            logging.warning("PCA already performed. Skipping.")
            return

        if self.ndim != 2:
            raise ValueError("time_series must be a 2D array")

        if n_components == 1:
            logging.info("n_components of 1 was passed. Skipping PCA.")

        else:
            pca_from_variance = PCA(n_components=n_components)
            self.time_series = pca_from_variance.fit_transform(self.time_series)
            if 0 < n_components < 1:
                print(
                    f"{pca_from_variance.n_components_} components are required to "
                    f"explain {n_components * 100}% of the variance "
                )
        self.pca_applied = True

    def plot(self, n_time_points: int = 10000):
        """Plot time_series.

        Plot n_time_points samples of time_series. Limits set in data_limits are
        ignored.

        Parameters
        ----------
        n_time_points: int
            Number of time points (samples) to plot.
        """
        plotting.plot_time_series(self.time_series, n_time_points=n_time_points)

    def savemat(self, filename: str, field_name: str = "x"):
        """Save time_series to a .mat file.

        Save time_series to a MATLAB .mat file. A time stamp and whether PCA has been
        applied to the data are included in the dictionary of values.

        If data limits have been set, they will be observed.

        Parameters
        ----------
        filename: str
            The file to save to (with or without .mat extension).
        field_name: str
            The dictionary key (MATLAB object field) which references the data.
        """
        scipy.io.savemat(
            filename,
            {
                field_name: self[:],
                "save_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "pca_applied": self.pca_applied,
            },
        )

    def save(self, filename: str):
        """Save time_series to a numpy (.npy) file.

        If data limits have been set, they will be observed.

        Parameters
        ----------
        filename: str
            The file to save to (with or without .npy extension).
        """
        np.save(filename, self[:])


def get_alpha_order(real_alpha, est_alpha):
    """Correlate covariance matrices to match known and inferred states.

    Parameters
    ----------
    real_alpha : array-like
    est_alpha : array-like

    Returns
    -------
    alpha_res_order : numpy.array
        The order of inferred states as determined by known states.

    """
    # establish ordering of factors so that they match real alphas
    ccs = np.zeros((real_alpha.shape[1], real_alpha.shape[1]))
    alpha_res_order = np.ones((real_alpha.shape[1]), int)

    for kk in range(real_alpha.shape[1]):
        for jj in range(real_alpha.shape[1]):
            if jj is not kk:
                cc = np.corrcoef(real_alpha[:, kk], est_alpha[:, jj])
                ccs[kk, jj] = cc[0, 1]
        alpha_res_order[kk] = int(np.argmax(ccs[kk, :]))

    return alpha_res_order


def trials_to_continuous(trials_time_course: np.ndarray) -> np.ndarray:
    """Given trial data, return a continuous time series.

    With data input in the form (channels x trials x time), reshape the array to
    create a (time x channels) array.

    Parameters
    ----------
    trials_time_course: numpy.ndarray
        A (channels x trials x time) time course.

    Returns
    -------
    concatenated: numpy.ndarray
        The (time x channels) array created by concatenating trial data.

    """
    if trials_time_course.ndim == 2:
        logging.warning(
            "A 2D time series was passed. Assuming it doesn't need to "
            "be concatenated."
        )
        if trials_time_course.shape[1] > trials_time_course.shape[0]:
            trials_time_course = trials_time_course.T
        return trials_time_course

    if trials_time_course.ndim != 3:
        raise ValueError(
            f"trials_time_course has {trials_time_course.ndim}"
            f" dimensions. It should have 3."
        )
    concatenated = np.concatenate(
        np.transpose(trials_time_course, axes=[2, 0, 1]), axis=1
    )
    if concatenated.shape[1] > concatenated.shape[0]:
        concatenated = concatenated.T
        logging.warning(
            "Assuming longer axis to be time and transposing. Check your inputs to be "
            "sure."
        )

    return concatenated


def load_spm(file_name: str) -> Tuple[np.ndarray, float]:
    """Load an SPM MEEG object.

    Highly untested function for reading SPM MEEG objects from MATLAB.

    Parameters
    ----------
    file_name: str
        Filename of an SPM MEEG object.

    Returns
    -------
    data: numpy.ndarray
        The time series referenced in the SPM MEEG object.
    sampling_frequency: float
        The sampling frequency listed in the SPM MEEG object.

    """
    spm = scipy.io.loadmat(file_name)
    data_file = spm["D"][0][0][6][0][0][0][0]
    n_channels = spm["D"][0][0][6][0][0][1][0][0]
    n_time_points = spm["D"][0][0][6][0][0][1][0][1]
    sampling_frequency = spm["D"][0][0][2][0][0]
    try:
        data = np.fromfile(data_file, dtype=np.float64).reshape(
            n_time_points, n_channels
        )
    except ValueError:
        data = np.fromfile(data_file, dtype=np.float32).reshape(
            n_time_points, n_channels
        )
    return data, sampling_frequency
