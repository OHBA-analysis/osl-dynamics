"""Multivariate normal observation model.

"""

from typing import Union
import numpy as np


class MultivariateNormal:
    """Class that generates data from a multivariate normal distribution.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    n_states : int
        Number of states.
    means : np.ndarray or str
        Mean vector for each state, shape should be (n_states, n_channels).
        Either a numpy array or 'zero' or 'random'.
    covariances : np.ndarray or str
        Covariance matrix for each state, shape should be (n_states,
        n_channels, n_channels). Either a numpy array or 'random'.
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for the random number generator.
    """

    def __init__(
        self,
        means: Union[np.ndarray, str],
        covariances: Union[np.ndarray, str],
        n_states: int = None,
        n_channels: int = None,
        observation_error: float = None,
        random_seed: int = None,
    ):
        self._rng = np.random.default_rng(random_seed)
        self.observation_error = observation_error

        # Both the means and covariances were passed as numpy arrays
        if isinstance(means, np.ndarray) and isinstance(covariances, np.ndarray):
            if means.shape[0] != covariances.shape[0]:
                raise ValueError(
                    "means and covariances have a different number of states."
                )
            if means.shape[1] != covariances.shape[1]:
                raise ValueError(
                    "means and covariances have a different number of channels."
                )
            self.n_states = means.shape[0]
            self.n_channels = means.shape[1]
            self.means = means
            self.covariances = covariances

        # Only the means were passed as a numpy array
        elif isinstance(means, np.ndarray) and not isinstance(covariances, np.ndarray):
            self.n_states = means.shape[0]
            self.n_channels = means.shape[1]
            self.means = means
            self.covariances = self.create_covariances(covariances)

        # Only the covariances were passed a numpy array
        elif not isinstance(means, np.ndarray) and isinstance(covariances, np.ndarray):
            self.n_states = covariances.shape[0]
            self.n_channels = covariances.shape[1]
            self.means = self.create_means(means)
            self.covariances = covariances

        # Neither means or covariances were passed as numpy arrays
        elif not isinstance(means, np.ndarray) and not isinstance(
            covariances, np.ndarray
        ):
            if n_states is None or n_channels is None:
                raise ValueError(
                    "If we are generating and means and covariances, "
                    + "n_states and n_channels must be passed."
                )
            self.n_states = n_states
            self.n_channels = n_channels
            self.means = self.create_means(means)
            self.covariances = self.create_covariances(covariances)

        else:
            raise ValueError("means and covariance arugments not passed correctly.")

    def create_means(self, option):

        if option == "zero":
            means = np.zeros([self.n_states, self.n_channels])

        elif option == "random":
            means = self._rng.normal(size=[self.n_states, self.n_channels])

        else:
            raise ValueError("means must be a np.array or 'zero' or 'random'.")

        return means

    def create_covariances(self, option, eps=1e-6):

        if option == "random":
            # Randomly select the elements of W from a normal distribution
            W = self._rng.normal(size=[self.n_states, self.n_channels, self.n_channels])

            # A small value to add to the diagonal to ensure the covariances are
            # invertible
            eps = np.tile(np.eye(self.n_channels), [self.n_states, 1, 1]) * eps

            # Calculate the covariances
            covariances = W @ W.transpose([0, 2, 1]) + eps

            # Trace normalisation
            covariances /= np.trace(covariances, axis1=1, axis2=2)[
                ..., np.newaxis, np.newaxis
            ]

        else:
            raise ValueError("covariances must be a np.array or 'random'.")

        return covariances
