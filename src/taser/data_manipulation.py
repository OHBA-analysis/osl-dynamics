import logging
from datetime import datetime
from typing import List, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import scipy.io
import mat73

from taser.helpers.decorators import transpose
from taser import plotting


class MEGData:

    ignored_keys = [
        "__globals__",
        "__header__",
        "__version__",
        "save_time",
        "pca_applied",
    ]

    def __init__(
        self, time_series: Union[np.ndarray, str], sampling_frequency: float = 1
    ):
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
                    for key in mat:
                        if key not in MEGData.ignored_keys:
                            time_series = mat[key]

        self.raw_data = time_series
        self.time_series = time_series.copy()
        self.pca_applied = False

        self.sampling_frequency = sampling_frequency
        self.t = None
        if time_series.ndim == 2:
            self.t = np.arange(self.time_series.shape[0]) / self.sampling_frequency

        self.n_min, self.n_max = None, None

    def __getitem__(self, val):
        return self.time_series[self.n_min : self.n_max][val]

    def data_limits(self, n_min=None, n_max=None):
        self.n_min = n_min
        self.n_max = n_max

    def trim_trials(self, trial_start=None, trial_cutoff=None, trial_skip=None):
        try:
            self.time_series = trim_trial_list(
                self.time_series,
                trial_start=trial_start,
                trial_cutoff=trial_cutoff,
                trial_skip=trial_skip,
            )
        except ValueError:
            logging.warning(
                "self.time_series is not a 3D array. "
                "Has it already been made continuous?"
            )

    def make_continuous(self):
        self.time_series = trials_to_continuous(self.time_series)
        self.t = np.arange(self.time_series.shape[0]) / self.sampling_frequency

    def standardize(
        self, n_components=0.9, pre_scale=True, do_pca=True, post_scale=True
    ):
        if do_pca and self.pca_applied:
            logging.warning("PCA already performed. Skipping.")
            return
        self.time_series = standardize(
            self.time_series,
            n_components=n_components,
            pre_scale=pre_scale,
            do_pca=do_pca,
            post_scale=post_scale,
        )
        self.pca_applied = True

    def plot(self, n_time_points=10000):
        plotting.plot_time_series(self.time_series, n_time_points=n_time_points)

    def mean(self, axis=None, **kwargs):
        return np.mean(self.time_series, axis=axis, **kwargs)

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


@transpose
def pca(time_series: np.ndarray, n_components: Union[int, float] = None,) -> np.ndarray:

    if time_series.ndim != 2:
        raise ValueError("time_series must be a 2D array")

    standard_scaler = StandardScaler()
    data_std = standard_scaler.fit_transform(time_series)

    pca_from_variance = PCA(n_components=n_components)
    data_pca = pca_from_variance.fit_transform(data_std)
    if 0 < n_components < 1:
        print(
            f"{pca_from_variance.n_components_} components are required to "
            f"explain {n_components * 100}% of the variance "
        )

    return data_pca


@transpose(0, "time_series")
def scale(time_series: np.ndarray) -> np.ndarray:
    scaled = StandardScaler().fit_transform(time_series)
    return scaled


def scale_pca(time_series: np.ndarray, n_components: Union[int, float]):
    return scale(pca(time_series=time_series, n_components=n_components))


def standardize(
    time_series: np.ndarray,
    n_components: Union[int, float] = 0.9,
    pre_scale: bool = True,
    do_pca: bool = True,
    post_scale: bool = True,
):

    if pre_scale:
        time_series = scale(time_series)
    if do_pca:
        time_series = pca(time_series, n_components)
    if post_scale:
        time_series = scale(time_series)

    return time_series


def load_file_list(file_list: List[str]) -> List[np.ndarray]:
    if isinstance(file_list, str):
        file_list = [file_list]
    return [
        np.load(file).astype(np.float32)
        for file in tqdm(file_list, desc="Loading files")
    ]


def trim_trial_list(
    trials_time_course: np.ndarray,
    trial_start: int = None,
    trial_cutoff: int = None,
    trial_skip: int = None,
):
    if trials_time_course.ndim != 3:
        raise ValueError("trials_time_course should be a 3D array.")
    return trials_time_course[:, trial_start:trial_cutoff, ::trial_skip]


def count_trials(trials_time_course: List[np.ndarray]):
    if isinstance(trials_time_course, np.ndarray):
        trials_time_course = [trials_time_course]
    return np.sum([time_course.shape[2] for time_course in trials_time_course])


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
            f"Assuming longer axis to be time and transposing. Check your inputs to be "
            f"sure."
        )

    return concatenated
