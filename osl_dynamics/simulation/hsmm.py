"""Classes for simulating Hidden Semi-Markov Models (HSMMs).

"""

import numpy as np

from osl_dynamics import array_ops
from osl_dynamics.simulation.mvn import MVN
from osl_dynamics.simulation.base import Simulation


class HSMM:
    """HSMM base class.

    Contains the probability distribution function for sampling state lifetimes.
    Uses a Gamma distribution for the probability distribution function.

    Parameters
    ----------
    gamma_shape : float
        Shape parameter for the Gamma distribution of state lifetimes.
    gamma_scale : float
        Scale parameter for the Gamma distribution of state lifetimes.
    off_diagonal_trans_prob : np.ndarray, optional
        Transition probabilities for out of state transitions.
    full_trans_prob : np.ndarray, optional
        A transition probability matrix, the diagonal of which will be ignored.
    n_states : int, optional
        Number of states.
    state_vectors : np.ndarray, optional
        Mode vectors define the activation of each components for a state.
        E.g. :code:`state_vectors=[[1,0,0],[0,1,0],[0,0,1]]` are mutually
        exclusive states. :code:`state_vector.shape[0]` must be more than
        :code:`n_states`.
    """

    def __init__(
        self,
        gamma_shape,
        gamma_scale,
        off_diagonal_trans_prob=None,
        full_trans_prob=None,
        state_vectors=None,
        n_states=None,
    ):
        # Validation
        if off_diagonal_trans_prob is not None and full_trans_prob is not None:
            raise ValueError(
                "Exactly one of off_diagonal_trans_prob and full_trans_prob "
                "must be specified."
            )

        # Get the number of states from trans_prob
        if off_diagonal_trans_prob is not None:
            self.n_states = off_diagonal_trans_prob.shape[0]
        elif full_trans_prob is not None:
            self.n_states = full_trans_prob.shape[0]

        # Both off_diagonal_trans_prob and full_trans_prob are None
        elif n_states is None:
            raise ValueError(
                "If off_diagonal_trans_prob and full_trans_prob are not given, "
                "n_states must be passed."
            )
        else:
            self.n_states = n_states

        self.off_diagonal_trans_prob = off_diagonal_trans_prob
        self.full_trans_prob = full_trans_prob

        self.construct_off_diagonal_trans_prob()

        # Define state vectors
        if state_vectors is None:
            self.state_vectors = np.eye(self.n_states)
        elif state_vectors.shape[0] < self.n_states:
            raise ValueError(
                "Less state vectors than the number of states were provided."
            )
        else:
            self.state_vectors = state_vectors

        # Parameters of the lifetime distribution
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale

    def construct_off_diagonal_trans_prob(self):
        if (self.off_diagonal_trans_prob is None) and (self.full_trans_prob is None):
            self.off_diagonal_trans_prob = np.ones([self.n_states, self.n_states])

        if self.full_trans_prob is not None:
            self.off_diagonal_trans_prob = (
                self.full_trans_prob / self.full_trans_prob.sum(axis=1)[:, None]
            )

        np.fill_diagonal(self.off_diagonal_trans_prob, 0)
        self.off_diagonal_trans_prob = (
            self.off_diagonal_trans_prob
            / self.off_diagonal_trans_prob.sum(axis=1)[:, None]
        )

    def generate_states(self, n_samples):
        cumsum_off_diagonal_trans_prob = np.cumsum(
            self.off_diagonal_trans_prob,
            axis=1,
        )
        alpha = np.zeros([n_samples, self.state_vectors.shape[1]])

        current_state = np.random.randint(0, self.n_states)
        current_position = 0

        while current_position < len(alpha):
            state_lifetime = np.round(
                np.random.gamma(shape=self.gamma_shape, scale=self.gamma_scale)
            ).astype(int)

            alpha[current_position : current_position + state_lifetime] = (
                self.state_vectors[current_state]
            )

            current_state = np.argmin(
                cumsum_off_diagonal_trans_prob[current_state] < np.random.uniform()
            )
            current_position += state_lifetime

        return alpha.astype(int)


class HSMM_MVN(Simulation):
    """Hidden Semi-Markov Model Simulation.

    We sample the state using a transition probability matrix with zero
    probability for self-transitions. The lifetime of each state is sampled
    from a Gamma distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    gamma_shape : float
        Shape parameter for the Gamma distribution of state lifetimes.
    gamma_scale : float
        Scale parameter for the Gamma distribution of state lifetimes.
    off_diagonal_trans_prob : np.ndarray, optional
        Transition probabilities for out of state transitions.
    full_trans_prob : np.ndarray, optional
        A transition probability matrix, the diagonal of which will be ignored.
    means : np.ndarray or str, optional
        Mean vector for each state, shape should be (n_states, n_channels).
        Or :code:`'zero'` or :code:`'random'`.
    covariances : numpy.ndarray or str, optional
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels). Or :code:`'random'`.
    n_states : int, optional
        Number of states. Can pass this argument with keyword :code:`n_modes`
        instead.
    n_channels : int, optional
        Number of channels in the observation model.
    observation_error : float, optional
        Standard deviation of random noise to be added to the observations.
    """

    def __init__(
        self,
        n_samples,
        gamma_shape,
        gamma_scale,
        off_diagonal_trans_prob=None,
        full_trans_prob=None,
        means=None,
        covariances=None,
        n_states=None,
        n_modes=None,
        n_channels=None,
        observation_error=0.0,
    ):
        if n_states is None:
            n_states = n_modes

        # Observation model object
        self.obs_mod = MVN(
            means=means,
            covariances=covariances,
            n_modes=n_states,
            n_channels=n_channels,
            observation_error=observation_error,
        )

        self.n_states = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        # HSMM object
        # N.b. we use a different random seed to the observation model
        self.hsmm = HSMM(
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            off_diagonal_trans_prob=off_diagonal_trans_prob,
            full_trans_prob=full_trans_prob,
            n_states=self.n_states,
        )

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        self.state_time_course = self.hsmm.generate_states(self.n_samples)
        self.time_series = self.obs_mod.simulate_data(self.state_time_course)

    @property
    def n_modes(self):
        return self.n_states

    @property
    def mode_time_course(self):
        return self.state_time_course

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        elif attr in dir(self.hsmm):
            return getattr(self.hsmm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def standardize(self):
        sigma = np.std(self.time_series, axis=0)
        super().standardize()
        self.obs_mod.covariances /= np.outer(sigma, sigma)[np.newaxis, ...]


class MixedHSMM_MVN(Simulation):
    """Hidden Semi-Markov Model Simulation with a mixture of states at each
    time point.

    Each mixture of states has it's own row/column in the transition
    probability matrix. The lifetime of each state mixture is sampled from
    a Gamma distribution.

    state_mixing_vectors is a 2D numpy array containing mixtures of the
    the states that can be simulated, e.g. with :code:`n_states=3` we could have
    :code:`state_mixing_vectors=[[0.5, 0.5, 0], [0.1, 0, 0.9]]`.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    gamma_shape : float
        Shape parameter for the Gamma distribution of state lifetimes.
    gamma_scale : float
        Scale parameter for the Gamma distribution of state lifetimes.
    mixed_state_vectors : np.ndarray, optional
        Vectors containing mixing factors for mixed states.
    mixed_mode_vectors : np.ndarray, optional
        Vectors containing mixing factors for mixed states.
    off_diagonal_trans_prob : np.ndarray, optional
        Transition probabilities for out of state transitions.
    full_trans_prob : np.ndarray, optional
        A transition probability matrix, the diagonal of which will be ignored.
    means : np.ndarray or str, optional
        Mean vector for each state, shape should be (n_states, n_channels).
        Or :code:`'zero'` or :code:`'random'`.
    covariances : numpy.ndarray or str, optional
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels). Or :code:`'random'`.
    n_channels : int, optional
        Number of channels in the observation model.
    observation_error : float, optional
        Standard deviation of random noise to be added to the observations.
    """

    def __init__(
        self,
        n_samples,
        gamma_shape,
        gamma_scale,
        mixed_state_vectors=None,
        mixed_mode_vectors=None,
        off_diagonal_trans_prob=None,
        full_trans_prob=None,
        means=None,
        covariances=None,
        n_channels=None,
        observation_error=0.0,
    ):
        if mixed_state_vectors is None:
            mixed_state_vectors = mixed_mode_vectors

        # Get the number of single activation states and mixed states
        self.n_states = mixed_state_vectors.shape[1]
        self.n_mixed_states = mixed_state_vectors.shape[0]

        # Mode vectors of mixed states
        self.mixed_state_vectors = mixed_state_vectors

        # Assign self.state_vectors
        self.construct_state_vectors(self.n_states)

        # Observation model object
        self.obs_mod = MVN(
            means=means,
            covariances=covariances,
            n_modes=self.n_states,
            n_channels=n_channels,
            observation_error=observation_error,
        )
        self.n_channels = self.obs_mod.n_channels

        # HSMM object
        # - hsmm.n_states is the total of n_states + n_mixed_states because
        #   we pretend each mixed state is a state in its own right in the
        #   transition probability matrix.
        # - we use a different random seed to the observation model
        self.hsmm = HSMM(
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            off_diagonal_trans_prob=off_diagonal_trans_prob,
            full_trans_prob=full_trans_prob,
            state_vectors=self.state_vectors,
            n_states=self.n_states + self.n_mixed_states,
        )

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        self.state_time_course = self.hsmm.generate_states(self.n_samples)
        self.time_series = self.obs_mod.simulate_data(self.state_time_course)

    @property
    def n_modes(self):
        return self.n_states

    @property
    def mode_time_course(self):
        return self.state_time_course

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        elif attr in dir(self.hsmm):
            return getattr(self.hsmm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def construct_state_vectors(self, n_states):
        non_mixed_state_vectors = array_ops.get_one_hot(np.arange(n_states))
        self.state_vectors = np.append(
            non_mixed_state_vectors, self.mixed_state_vectors, axis=0
        )

    def standardize(self):
        sigma = np.std(self.time_series, axis=0)
        super().standardize()
        self.obs_mod.covariances /= np.outer(sigma, sigma)[np.newaxis, ...]
