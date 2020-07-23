"""Different priors for the means/covariances of the generative model.

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


def wishart(data, n_states):
    gmm = BayesianGaussianMixture(n_components=n_states, max_iter=1)
    warnings.filterwarnings("ignore")
    gmm.fit(data)

    # Cast to float32 for tensorflow
    covariances = gmm.covariances_.astype(np.float32)

    return covariances


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


def gaussian_mixture_model(
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
    covariances : numpy.ndarray
        Covariances of the states (Gaussian distributions) found with dimensions
        [n_states x n_channels x n_channels]
    means : numpy.ndarray
        Means of the states (Gaussian distributions) found with dimensions [n_states x
        n_channels]
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


def hmm(
    data: Union[np.ndarray, Data],
    n_states: int,
    stay_prob: float = 0.9,
    learn_means: bool = False,
    n_initialisations: int = 10,
    simulation: str = "sequence",
    random: bool = False,
    random_seed: int = None,
):
    """Estimates means and covariances by sampling an HMM.
    
    """
    if isinstance(data, Data):
        data = data.time_series

    n_samples = data.shape[0]
    n_channels = data.shape[1]

    # Priors for the means and covariances
    means = np.empty([n_initialisations, n_states, n_channels], dtype=np.float32)
    covariances = np.empty(
        [n_initialisations, n_states, n_channels, n_channels], dtype=np.float32
    )

    # Log likelihood
    ll = np.zeros(n_initialisations)

    # Initialise n_initialisation-times
    for n in range(n_initialisations):

        # Simulate the state time courses
        if simulation == "basic":
            sim = BasicHMMSimulation(
                n_samples=n_samples,
                n_channels=n_channels,
                n_states=n_states,
                stay_prob=stay_prob,
                simulate=False,
            )
        else:
            if simulation != "sequence":
                _logger.warning(
                    f"simulation={simulation} unknown. "
                    + "SequenceHMMSimulation will be used."
                )
            sim = SequenceHMMSimulation(
                n_samples=n_samples,
                n_channels=n_channels,
                n_states=n_states,
                stay_prob=stay_prob,
                simulate=False,
            )
        stc = sim.state_time_course

        # Calculate the mean and covariance for each state
        if random:
            # Randomly sample the elements of the mean vector and covariance matrix
            rng = np.random.default_rng(random_seed)
            for i in range(n_states):
                if learn_means:
                    means[n, i] = rng.normal(0, 1, n_channels)
                else:
                    means[n, i] = np.zeros(n_channels)
                covariances[n, i] = np.diag(abs(rng.normal(0, 1, n_channels)))
        else:
            # Compute the mean and covariance when the state is on
            for i in range(n_states):
                time_series = data[stc[:, i] == 1]
                means[n, i] = np.mean(time_series, axis=0)
                covariances[n, i] = np.cov(time_series, rowvar=False)

                if not learn_means:
                    # Absorb means into the covariances
                    covariances[n, i] += np.outer(means[n, i], means[n, i])
                    means[n, i] = np.zeros(n_channels, dtype=np.float32)

        # Calculate the log likelihood:
        # ll = c - 0.5 * log(det(sigma)) - 0.5 * [(x - mu)^T sigma^-1 (x - mu)]
        # where:
        # - x are the observations
        # - mu is the mean vector
        # - sigma is the covariance matrix
        # - c is a constant (which we ignore)
        log_det_sigma = np.empty(n_states)
        inv_sigma = np.empty([n_states, n_channels, n_channels])
        for i in range(n_states):
            _, log_det_sigma[i] = np.linalg.slogdet(covariances[n, i])
            inv_sigma[i] = np.linalg.inv(covariances[n, i] + 1e-8 * np.eye(n_channels))

        for i in range(n_samples):
            state = stc[i].argmax()
            x_minus_mu = data[i] - means[n, state]
            second_term = -0.5 * log_det_sigma[state]
            third_term = -0.5 * x_minus_mu.T @ inv_sigma[state] @ x_minus_mu
            ll[n] += second_term + third_term

        # Print the negative log likelihood
        print(f"Initialization {n}: nll={-ll[n]}")

    # Find the initialisation which has the maximum likelihood
    argmax_ll = np.argmax(ll)
    print(f"Using initialization {argmax_ll}")
    means = means[argmax_ll]
    covariances = covariances[argmax_ll]

    return means, covariances


def new_hmm(
    data: Union[np.ndarray, Data],
    n_states: int,
    stay_prob: float = 0.9,
    learn_means: bool = False,
    n_initialisations: int = 10,
    simulation: str = "sequence",
    random: bool = False,
    random_seed: int = None,
):
    rng = np.random.default_rng(random_seed)

    if isinstance(data, Data):
        data = data.time_series
    n_samples = data.shape[0]
    n_channels = data.shape[1]

    stcs = [
        BasicHMMSimulation(
            n_samples=n_samples,
            n_channels=n_channels,
            n_states=n_states,
            stay_prob=stay_prob,
            simulate=False,
        ).state_time_course.argmax(axis=-1)
        for initialisation in trange(n_initialisations, desc="Simulating stcs")
    ]

    if random:
        if learn_means:
            means = rng.normal(size=(n_initialisations, n_states, n_channels))
        else:
            means = np.zeros((n_initialisations, n_states, n_channels))

        # covariances = np.abs(
        #     rng.normal(size=(n_initialisations, n_states, n_channels, n_channels))
        #     * np.eye(n_channels)
        # )
        covariances = np.array(
            [
                wishart_covariances(data, n_states, np.cov(data.T), data.mean(axis=0))
                for init in trange(n_initialisations, desc="Generating covariances")
            ]
        )

    else:
        covariances = np.array(
            [
                [np.cov(data[stc == state], rowvar=False) for state in range(n_states)]
                for stc in stcs
            ]
        )
        means = np.array(
            [
                [np.mean(data[stc == state], axis=0) for state in range(n_states)]
                for stc in stcs
            ]
        )

        if not learn_means:
            covariances = np.array(
                [
                    [
                        covariances[init, state]
                        + np.outer(means[init, state], means[init, state])
                        for state in range(n_states)
                    ]
                    for init in range(n_initialisations)
                ]
            )
            means = np.zeros((n_initialisations, n_states, n_channels))

    _, log_det_sigma = np.linalg.slogdet(covariances)
    inv_sigma = np.linalg.inv(covariances) + 1e-8 * np.eye(covariances.shape[-1])

    ll = np.zeros(n_initialisations)
    for init in trange(n_initialisations, desc="Calculating LL"):
        sep_states = [data[stcs[init] == i] for i in range(n_states)]
        num_state_samples = np.array([arr.shape[0] for arr in sep_states])
        max_samples = num_state_samples.max()
        state_sorted = np.zeros([n_states, max_samples, n_channels])
        for i, state in enumerate(sep_states):
            state_sorted[i, : state.shape[0], :n_channels] = state
        xmm = state_sorted - means[init, :, None, :]
        ll[init] = -0.5 * np.tensordot(
            np.matmul(xmm, inv_sigma[init]), xmm, axes=3
        ) - 0.5 * np.sum(num_state_samples * log_det_sigma[init])

    argmax_ll = np.argmax(ll)
    print(f"Using initialization {argmax_ll}\nFE: {-ll[argmax_ll]}")
    means = means[argmax_ll]
    covariances = covariances[argmax_ll]

    return means, covariances
