"""Functions to fit a Gaussian Mixture Model to data.

"""

import logging
import warnings
from typing import Union

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from tqdm import trange
from vrad.data import Data
from vrad.inference.functions import cholesky_factor
from vrad.simulation import BasicHMMSimulation, SequenceHMMSimulation
from vrad.utils.misc import override_dict_defaults, time_axis_first

_logger = logging.getLogger("VRAD")
_rng = np.random.default_rng()


def wishart_covariances(time_series, n_states, covariance_prior, mean_prior):
    reg_covar = 1e-6
    n_components = n_states

    n_samples, dof_prior = time_series.shape

    resp = _rng.random((n_samples, n_components))

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    xk = np.dot(resp.T, time_series) / nk[:, None]

    mean_precision = nk + 1
    means = (mean_prior + nk[:, None] * xk) / mean_precision[:, None]

    n_features = xk.shape[1]
    dof = n_features + nk

    full_diff = time_series - means[:, None, :]
    sk = (
        np.matmul(resp.T[:, None, :] * full_diff.transpose(0, 2, 1), full_diff)
        / nk[:, None, None]
        + np.eye(n_features, n_features) * reg_covar
    )
    full_diff = xk - mean_prior
    covariances = (
        covariance_prior
        + nk[:, None, None] * sk
        + (nk / mean_precision)[:, None, None]
        * full_diff[..., None]
        @ full_diff[:, None, :]
    )

    covariances /= dof[:, None, None]

    return covariances


def initial_covariances(data, n_states):
    gmm = BayesianGaussianMixture(n_components=n_states, max_iter=1)
    warnings.filterwarnings("ignore")
    gmm.fit(data)
    covariances = gmm.covariances_.astype(np.float32)  # for tensorflow
    return covariances


def final_means_covariances(
    data: Union[np.ndarray, Data],
    n_states: int,
    learn_means: bool = False,
    covariance_type: str = "full",
    retry_attempts: int = 5,
    gmm_kwargs: dict = None,
    take_random_sample: int = None,
    return_gmm: bool = False,
    random_seed: int = None,
):
    """Find a mixture of Gaussian distributions which characterises a dataset.

    Using scikit-learn's `BayesianGaussianMixture` class, find a set of Gaussian
    distributions which when linearly combined can explain a dataset.

    Parameters
    ----------
    data : numpy.ndarray
        Input data of dimensions [n_channels x n_time_points]
    n_states : int
        Number of states (Gaussian distributions) to try to find.
    learn_means : bool
        If False (default), means will be assumed to be zero and given a strong
        weight as a prior.
    retry_attempts : int
        Number of times to retry fitting if `.fit()` doesn't converge.
    gmm_kwargs : dict
        Keyword arguments for the `BayesianGaussianMixture`
    take_random_sample : int
        Number of time points to sample.


    Returns
    -------
    means : numpy.ndarray
        Means of the states (Gaussian distributions) found with dimensions [n_states x
        n_channels]
    covariances : numpy.ndarray
        Covariances of the states (Gaussian distributions) found with dimensions
        [n_states x n_channels x n_channels]
    """
    if retry_attempts < 1:
        raise ValueError("retry_attempts cannot be less than 1")

    default_gmm_kwargs = {"verbose": 2, "n_init": 1}
    gmm_kwargs = override_dict_defaults(default_gmm_kwargs, gmm_kwargs)

    data = time_axis_first(data)

    if take_random_sample:
        data = np.random.permutation(data)[:take_random_sample]

    n_channels = data.shape[1]

    if learn_means:
        # use sklearn learn to do GMM
        gmm = BayesianGaussianMixture(
            n_components=n_states,
            covariance_type=covariance_type,
            **gmm_kwargs,
            random_state=random_seed,
        )
    else:
        # make sure we force means to be zero:
        gmm = BayesianGaussianMixture(
            n_components=n_states,
            covariance_type=covariance_type,
            mean_prior=np.zeros(n_channels),
            mean_precision_prior=1e12,
            random_state=random_seed,
            **gmm_kwargs,
        )
    for attempt in range(retry_attempts):
        warnings.filterwarnings("ignore")
        gmm.fit(data)
        if gmm.converged_:
            print(f"Converged on iteration {attempt}")
            break
        print(f"Failed to converge on iteration {attempt}")

    # Cast means and covariances to float32 for tensorflow
    means = gmm.means_.astype(np.float32)
    if covariance_type == "diag":
        covariances = np.empty([n_states, n_channels, n_channels], dtype=np.float32)
        for i in range(gmm.covariances_.shape[0]):
            covariances[i] = np.diag(gmm.covariances_[i])
    else:
        covariances = gmm.covariances_.astype(np.float32)

    if return_gmm:
        return gmm, means, covariances

    return means, covariances
