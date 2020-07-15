"""Different priors for the means/covariances of the generative model.

"""

import logging
import warnings
from typing import Union

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from vrad.data import Data
from vrad.simulation import BasicHMMSimulation, SequenceHMMSimulation
from vrad.inference.functions import cholesky_factor, full_covariances
from vrad.utils.misc import override_dict_defaults, time_axis_first

_logger = logging.getLogger("VRAD")


def wishart(data, n_states):
    gmm = BayesianGaussianMixture(n_components=n_states, max_iter=1)
    warnings.filterwarnings("ignore")
    gmm.fit(data)

    # Cast to float32 for tensorflow
    covariances = gmm.covariances_.astype(np.float32)

    return covariances


def gaussian_mixture_model(
    data: Union[np.ndarray, Data],
    n_states: int,
    learn_means: bool = False,
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
            covariance_type="full",
            **gmm_kwargs,
            random_state=random_seed,
        )
    else:
        # make sure we force means to be zero:
        gmm = BayesianGaussianMixture(
            n_components=n_states,
            covariance_type="full",
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
    covariances = gmm.covariances_.astype(np.float32)

    if return_gmm:
        return gmm, means, covariances

    return means, covariances


def hmm(
    data: Union[np.ndarray, Data],
    n_states: int,
    stay_prob: float = 0.9,
    learn_means: bool = False,
    n_initialisations: int = 5,
    simulation: str = "sequence",
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
            )
        stc = sim.state_time_course

        # Calculate the mean and covariance for each state
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
