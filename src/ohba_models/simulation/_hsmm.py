"""Classes for simulating Hidden Semi-Markov Models (HSMMs).

"""

import logging

import numpy as np
from ohba_models.array_ops import get_one_hot, cov2corr
from ohba_models.simulation import MVN, Simulation

_logger = logging.getLogger("OHBA-Models")


class HSMM:
    """HSMM base class.

    Contains the probability distribution function for sampling mode lifetimes.
    Uses a gamma distribution for the probability distribution function.

    Parameters
    ----------
    gamma_shape : float
        Shape parameter for the gamma distribution of mode lifetimes.
    gamma_scale : float
        Scale parameter for the gamma distribution of mode lifetimes.
    off_diagonal_trans_prob : np.ndarray
        Transition probabilities for out of mode transitions.
    full_trans_prob : np.ndarray
        A transition probability matrix, the diagonal of which will be ignored.
    n_modes : int
        Number of modes.
    mode_vectors : np.ndarray
        Mode vectors define the activation of each components for a mode.
        E.g. mode_vectors=[[1,0,0],[0,1,0],[0,0,1]] are mutually exclusive
        modes. mode_vector.shape[0] must be more than n_modes.
    random_seed : int
        Seed for random number generator.
    """

    def __init__(
        self,
        gamma_shape: float,
        gamma_scale: float,
        off_diagonal_trans_prob: np.ndarray = None,
        full_trans_prob: np.ndarray = None,
        mode_vectors: np.ndarray = None,
        n_modes: int = None,
        random_seed: int = None,
    ):
        # Validation
        if off_diagonal_trans_prob is not None and full_trans_prob is not None:
            raise ValueError(
                "Exactly one of off_diagonal_trans_prob and full_trans_prob "
                "must be specified."
            )

        # Get the number of modes from trans_prob
        if off_diagonal_trans_prob is not None:
            self.n_modes = off_diagonal_trans_prob.shape[0]
        elif full_trans_prob is not None:
            self.n_modes = full_trans_prob.shape[0]

        # Both off_diagonal_trans_prob and full_trans_prob are None
        elif n_modes is None:
            raise ValueError(
                "If off_diagonal_trans_prob and full_trans_prob are not given, "
                + "n_modes must be passed."
            )
        else:
            self.n_modes = n_modes

        self.off_diagonal_trans_prob = off_diagonal_trans_prob
        self.full_trans_prob = full_trans_prob

        self.construct_off_diagonal_trans_prob()

        # Define mode vectors
        if mode_vectors is None:
            self.mode_vectors = np.eye(self.n_modes)
        elif mode_vectors.shape[0] < self.n_modes:
            raise ValueError(
                "Less mode vectors than the number of modes were provided."
            )
        else:
            self.mode_vectors = mode_vectors

        # Parameters of the lifetime distribution
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale

        # Setup random number generator
        self._rng = np.random.default_rng(random_seed)

    def construct_off_diagonal_trans_prob(self):
        if (self.off_diagonal_trans_prob is None) and (self.full_trans_prob is None):
            self.off_diagonal_trans_prob = np.ones([self.n_modes, self.n_modes])

        if self.full_trans_prob is not None:
            self.off_diagonal_trans_prob = (
                self.full_trans_prob / self.full_trans_prob.sum(axis=1)[:, None]
            )

        np.fill_diagonal(self.off_diagonal_trans_prob, 0)
        self.off_diagonal_trans_prob = (
            self.off_diagonal_trans_prob
            / self.off_diagonal_trans_prob.sum(axis=1)[:, None]
        )

        with np.printoptions(linewidth=np.nan):
            _logger.info(
                f"off_diagonal_trans_prob is:\n{str(self.off_diagonal_trans_prob)}"
            )

    def generate_modes(self, n_samples):
        cumsum_off_diagonal_trans_prob = np.cumsum(self.off_diagonal_trans_prob, axis=1)
        alpha = np.zeros([n_samples, self.mode_vectors.shape[1]])

        gamma_sample = self._rng.gamma
        random_sample = self._rng.uniform
        current_mode = self._rng.integers(0, self.n_modes)
        current_position = 0

        while current_position < len(alpha):
            mode_lifetime = np.round(
                gamma_sample(shape=self.gamma_shape, scale=self.gamma_scale)
            ).astype(np.int)

            alpha[
                current_position : current_position + mode_lifetime
            ] = self.mode_vectors[current_mode]

            rand = random_sample()
            current_mode = np.argmin(
                cumsum_off_diagonal_trans_prob[current_mode] < rand
            )
            current_position += mode_lifetime

        return alpha


class HSMM_MVN(Simulation):
    """Hidden Semi-Markov Model Simulation.

    We sample the mode using a transition probability matrix with zero
    probability for self-transitions. The lifetime of each mode is sampled
    from a Gamma distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    gamma_shape : float
        Shape parameter for the gamma distribution of mode lifetimes.
    gamma_scale : float
        Scale parameter for the gamma distribution of mode lifetimes.
    off_diagonal_trans_prob : np.ndarray
        Transition probabilities for out of mode transitions.
    full_trans_prob : np.ndarray
        A transition probability matrix, the diagonal of which will be ignored.
    means : np.ndarray or str
        Mean vector for each mode, shape should be (n_modes, n_channels).
        Or 'zero' or 'random'.
    covariances : numpy.ndarray or str
        Covariance matrix for each mode, shape should be (n_modes, n_channels,
        n_channels). Or 'random'.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels in the observation model.
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    simulate : bool
        Should data be simulated? Can be called using .simulate later.
    random_seed : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int,
        gamma_shape: float,
        gamma_scale: float,
        off_diagonal_trans_prob: np.ndarray = None,
        full_trans_prob: np.ndarray = None,
        means: np.ndarray = None,
        covariances: np.ndarray = None,
        n_modes: int = None,
        n_channels: int = None,
        observation_error: float = 0.0,
        random_seed: int = None,
    ):
        # Observation model object
        self.obs_mod = MVN(
            means=means,
            covariances=covariances,
            n_modes=n_modes,
            n_channels=n_channels,
            observation_error=observation_error,
            random_seed=random_seed,
        )

        self.n_modes = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        # HSMM object
        # N.b. we use a different random seed to the observation model
        self.hsmm = HSMM(
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            off_diagonal_trans_prob=off_diagonal_trans_prob,
            full_trans_prob=full_trans_prob,
            n_modes=self.n_modes,
            random_seed=random_seed if random_seed is None else random_seed + 1,
        )

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        self.mode_time_course = self.hsmm.generate_modes(self.n_samples)
        self.time_series = self.obs_mod.simulate_data(self.mode_time_course)

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        elif attr in dir(self.hsmm):
            return getattr(self.hsmm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def standardize(self):
        super().standardize()
        self.obs_mod.covariances = cov2corr(self.obs_mod.covariances)


class MixedHSMM_MVN(Simulation):
    """Hidden Semi-Markov Model Simulation with a mixture of modes at each time point.

    Each mixture of modes has it's own row/column in the transition probability matrix.
    The lifetime of each mode mixture is sampled from a Gamma distribution.

    mode_mixing_vectors is a 2D numpy array containing mixtures of the
    the modes that can be simulated, e.g. with n_modes=3 we could have
    mode_mixing_vectors=[[0.5, 0.5, 0], [0.1, 0, 0.9]]

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    mixed_mode_vectors : np.ndarray
        Vectors containing mixing factors for mixed modes.
    gamma_shape : float
        Shape parameter for the gamma distribution of mode lifetimes.
    gamma_scale : float
        Scale parameter for the gamma distribution of mode lifetimes.
    off_diagonal_trans_prob : np.ndarray
        Transition probabilities for out of mode transitions.
    full_trans_prob : np.ndarray
        A transition probability matrix, the diagonal of which will be ignored.
    means : np.ndarray or str
        Mean vector for each mode, shape should be (n_modes, n_channels).
        Or 'zero' or 'random'.
    covariances : numpy.ndarray or str
        Covariance matrix for each mode, shape should be (n_modes, n_channels,
        n_channels). Or 'random'.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels in the observation model.
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    simulate : bool
        Should data be simulated? Can be called using .simulate later.
    random_seed : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int,
        mixed_mode_vectors: np.ndarray,
        gamma_shape: float,
        gamma_scale: float,
        off_diagonal_trans_prob: np.ndarray = None,
        full_trans_prob: np.ndarray = None,
        means: np.ndarray = None,
        covariances: np.ndarray = None,
        n_channels: int = None,
        observation_error: float = 0.0,
        random_seed: int = None,
    ):
        # Get the number of single activation modes and mixed modes
        self.n_modes = mixed_mode_vectors.shape[1]
        self.n_mixed_modes = mixed_mode_vectors.shape[0]

        # Mode vectors of mixed modes
        self.mixed_mode_vectors = mixed_mode_vectors

        # Assign self.mode_vectors
        self.construct_mode_vectors(self.n_modes)

        # Observation model object
        self.obs_mod = MVN(
            means=means,
            covariances=covariances,
            n_modes=self.n_modes,
            n_channels=n_channels,
            observation_error=observation_error,
            random_seed=random_seed,
        )
        self.n_channels = self.obs_mod.n_channels

        # HSMM object
        # - hsmm.n_modes is the total of n_modes + n_mixed_modes because
        #   we pretend each mixed mode is a mode in its own right in the
        #   transition probability matrix.
        # - we use a different random seed to the observation model
        self.hsmm = HSMM(
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            off_diagonal_trans_prob=off_diagonal_trans_prob,
            full_trans_prob=full_trans_prob,
            mode_vectors=self.mode_vectors,
            n_modes=self.n_modes + self.n_mixed_modes,
            random_seed=random_seed if random_seed is None else random_seed + 1,
        )

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        self.mode_time_course = self.hsmm.generate_modes(self.n_samples)
        self.time_series = self.obs_mod.simulate_data(self.mode_time_course)

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        elif attr in dir(self.hsmm):
            return getattr(self.hsmm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def construct_mode_vectors(self, n_modes):
        non_mixed_mode_vectors = get_one_hot(np.arange(n_modes))
        self.mode_vectors = np.append(
            non_mixed_mode_vectors, self.mixed_mode_vectors, axis=0
        )

    def standardize(self):
        super().standardize()
        self.obs_mod.covariances = cov2corr(self.obs_mod.covariances)
