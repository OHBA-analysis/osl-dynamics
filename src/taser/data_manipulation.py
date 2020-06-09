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
        self.n_min = n_min
        self.n_max = n_max

    @property
    def shape(self):
        return self[:].shape

    @property
    def data_shape(self):
        return self.time_series.shape

    def time_axis_first(self):
        self.time_series, transposed = time_axis_first(self.time_series)
        if transposed:
            logging.warning("Assuming time to be the longer axis and transposing.")

    def trim_trials(self, trial_start=None, trial_cutoff=None, trial_skip=None):
        if self.ndim == 3:
            self.time_series = self.time_series[
                :, trial_start:trial_cutoff, ::trial_skip
            ]
        else:
            logging.warning(f"Array is not 3D (ndim = {self.ndim}). Can't trim trials.")

    def make_continuous(self):
        self.time_series = trials_to_continuous(self.time_series)
        self.t = np.arange(self.time_series.shape[0]) / self.sampling_frequency

    def standardize(
        self,
        n_components: Union[float, int] = 0.9,
        pre_scale: bool = True,
        do_pca: Union[bool, str] = True,
        post_scale: bool = True,
    ):
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
        self.time_series = (
            self.time_series - self.time_series.mean(axis=0)
        ) / self.time_series.std(axis=0)

    def pca(self, n_components: Union[int, float] = 1, force=False):

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

    def plot(self, n_time_points=10000):
        plotting.plot_time_series(self.time_series, n_time_points=n_time_points)

    def savemat(self, filename: str, field_name: str = "x"):
        scipy.io.savemat(
            filename,
            {
                field_name: self[:],
                "save_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "pca_applied": self.pca_applied,
            },
        )

    def save(self, filename: str):
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
