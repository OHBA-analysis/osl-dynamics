import logging
from typing import Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from taser.helpers.decorators import transpose


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


def pca(time_series: np.ndarray, n_components: Union[int, float] = None,) -> np.ndarray:

    if time_series.ndim == 3:
        logging.warning("Assuming 3D array is [channels x time x trials]")
        time_series = trials_to_continuous(time_series)
    if time_series.ndim != 2:
        raise ValueError("time_series must be a 2D array")
    if time_series.shape[0] < time_series.shape[1]:
        logging.warning("Assuming longer axis to be time and transposing.")
        time_series = time_series.T

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


def scale_pca_scale(time_series: np.ndarray, n_components: Union[int, float]):
    return scale(pca(scale(time_series), n_components=n_components))


def process_data(dataset_parameters):
    raw_data = np.load(dataset_parameters["input_data"]).astype(np.float32)

    retrialed_data = raw_data[
        :, : dataset_parameters["trial_cutoff"], :: dataset_parameters["trial_skip"]
    ]
    concatenated_data = trials_to_continuous(retrialed_data)

    events = concatenated_data[dataset_parameters["event_channel"]]
    input_data = concatenated_data[dataset_parameters["data_start"] :]

    if dataset_parameters["standardize"]:
        input_data = scale(input_data)
    if dataset_parameters["pca"]:
        input_data = pca(input_data, n_components=dataset_parameters["n_pcs"])
    if dataset_parameters["standardize_pcs"]:
        input_data = scale(input_data)

    return input_data, events


def trials_to_continuous(trials_time_course: np.ndarray):
    if trials_time_course.ndim == 2:
        logging.warning(
            "A 2D time series was passed. Assuming it doesn't need to"
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
    return np.concatenate(np.transpose(trials_time_course, axes=[2, 0, 1]), axis=1)