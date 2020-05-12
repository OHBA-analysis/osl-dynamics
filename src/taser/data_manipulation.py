import logging
from typing import Union, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from taser.helpers.decorators import transpose
from tqdm import tqdm


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
    trials_time_course_list: List[np.ndarray], trial_cutoff: int, trial_skip: int
):
    if isinstance(trials_time_course_list, np.ndarray):
        trials_time_course_list = [trials_time_course_list]

    if trial_cutoff is None:
        trial_cutoff = trials_time_course_list[0].shape[1]
    return [
        trials_time_course[:, :trial_cutoff, ::trial_skip]
        for trials_time_course in trials_time_course_list
    ]


def count_trials(trials_time_course: List[np.ndarray]):
    if isinstance(trials_time_course, np.ndarray):
        trials_time_course = [trials_time_course]
    return np.sum([time_course.shape[2] for time_course in trials_time_course])


def subjects_to_time_course(
    file_list: List[str],
    trial_cutoff: int = None,
    trial_skip: int = 1,
    event_channel: int = None,
    data_start: int = 0,
    **kwargs: dict,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    raw_data_list = load_file_list(file_list)

    trimmed_trial_list = trim_trial_list(raw_data_list, trial_cutoff, trial_skip)

    concatenated_data_list = [
        trials_to_continuous(trimmed) for trimmed in trimmed_trial_list
    ]

    input_data_list = [
        concatenated_data[data_start:] for concatenated_data in concatenated_data_list
    ]

    scaled_data_list = [scale(input_data) for input_data in input_data_list]

    data_shapes = [arr.shape[1] for arr in scaled_data_list]
    max_data_shape = max(data_shapes)
    shape_difference = np.array([max_data_shape - shape for shape in data_shapes])
    needs_padding = np.argwhere(shape_difference != 0)
    if needs_padding.size > 0:
        logging.warning(
            f"Padding required for inputs:\n"
            f"\t   file: {', '.join([str(i[0]) for i in needs_padding])}\n"
            f"\tpadding: {', '.join(str(i[0]) for i in shape_difference[needs_padding])}"
        )

    padded_data_list = [
        np.pad(arr, [[0, 0], [0, diff]])
        for arr, diff in zip(scaled_data_list, shape_difference)
    ]

    all_concatenated = np.concatenate(padded_data_list, axis=0)

    if event_channel is None:
        return all_concatenated
    else:
        events = np.concatenate(
            [data[event_channel] for data in concatenated_data_list]
        )
        return all_concatenated, events


def trials_to_continuous(trials_time_course: np.ndarray):
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
    return np.concatenate(np.transpose(trials_time_course, axes=[2, 0, 1]), axis=1)
