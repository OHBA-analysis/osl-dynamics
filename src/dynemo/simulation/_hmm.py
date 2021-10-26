"""Classes for simulating Hidden Markov Models (HMMs).

"""

import logging
from typing import Union

import numpy as np
from dynemo.array_ops import get_one_hot
from dynemo.simulation import MAR, MVN, SingleSine, Simulation

_logger = logging.getLogger("DyNeMo")


class HMM:
    """HMM base class.

    Contains the transition probability matrix.

    Parameters
    ----------
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a str ('sequence',
        'uniform') to generate a transition probability matrix.
    stay_prob : float
        Used to generate the transition probability matrix is trans_prob is a str.
        Optional.
    n_modes : int
        Number of modes. Needed when trans_prob is a str to construct the
        transition probability matrix. Optional.
    random_seed : int
        Seed for random number generator. Optional.
    """

    def __init__(
        self,
        trans_prob: Union[np.ndarray, str],
        stay_prob: float = None,
        n_modes: int = None,
        random_seed: int = None,
    ):

        if isinstance(trans_prob, list):
            trans_prob = np.ndarray(trans_prob)

        if isinstance(trans_prob, np.ndarray):
            # Don't need to generate the transition probability matrix

            if trans_prob.ndim != 2:
                raise ValueError("trans_prob must be a 2D array.")

            if trans_prob.shape[0] != trans_prob.shape[1]:
                raise ValueError("trans_prob must be a square matrix.")

            # Check the rows of the transition probability matrix sum to one
            # We allow a small error (1e-12) because of rounding errors
            row_sums = trans_prob.sum(axis=1)
            col_sums = trans_prob.sum(axis=0)
            ones = np.ones(trans_prob.shape[0])
            if np.any(abs(row_sums - ones) > 1e-12):
                if np.all(abs(col_sums - ones) < 1e-12):
                    trans_prob = trans_prob.T
                    _logger.warning(
                        "Rows of trans_prob matrix must sum to 1. Transpose taken."
                    )
                else:
                    raise ValueError("Rows of trans_prob must sum to 1.")

            self.trans_prob = trans_prob

        elif isinstance(trans_prob, str):
            # We generate the transition probability matrix

            # Validation
            if trans_prob not in ["sequence", "uniform"]:
                raise ValueError(
                    "trans_prob must be a np.array, 'sequence' or 'uniform'."
                )

            # Special case of there being only one mode
            if n_modes == 1:
                self.trans_prob = np.ones([1, 1])

            # Sequential transition probability matrix
            elif trans_prob == "sequence":
                if stay_prob is None or n_modes is None:
                    raise ValueError(
                        "If trans_prob is 'sequence', stay_prob and n_modes "
                        + "must be passed."
                    )
                self.trans_prob = self.construct_sequence_trans_prob(stay_prob, n_modes)

            # Uniform transition probability matrix
            elif trans_prob == "uniform":
                if n_modes is None:
                    raise ValueError(
                        "If trans_prob is 'uniform', n_modes must be passed."
                    )
                if stay_prob is None:
                    stay_prob = 1.0 / n_modes
                self.trans_prob = self.construct_uniform_trans_prob(stay_prob, n_modes)

        elif trans_prob is None and n_modes == 1:
            self.trans_prob = np.ones([1, 1])

        # Infer number of modes from the transition probability matrix
        self.n_modes = self.trans_prob.shape[0]

        # Setup random number generator
        self._rng = np.random.default_rng(random_seed)

    @staticmethod
    def construct_sequence_trans_prob(stay_prob, n_modes):
        trans_prob = np.zeros([n_modes, n_modes])
        np.fill_diagonal(trans_prob, stay_prob)
        np.fill_diagonal(trans_prob[:, 1:], 1 - stay_prob)
        trans_prob[-1, 0] = 1 - stay_prob
        return trans_prob

    @staticmethod
    def construct_uniform_trans_prob(stay_prob, n_modes):
        single_trans_prob = (1 - stay_prob) / (n_modes - 1)
        trans_prob = np.ones((n_modes, n_modes)) * single_trans_prob
        trans_prob[np.diag_indices(n_modes)] = stay_prob
        return trans_prob

    def generate_modes(self, n_samples):
        # Here the time course always start from mode 0
        rands = [
            iter(self._rng.choice(self.n_modes, size=n_samples, p=self.trans_prob[i]))
            for i in range(self.n_modes)
        ]
        modes = np.zeros(n_samples, int)
        for sample in range(1, n_samples):
            modes[sample] = next(rands[modes[sample - 1]])
        return get_one_hot(modes, n_modes=self.n_modes)


class HMM_MAR(Simulation):
    """Simulate an HMM with a multivariate autoregressive observation model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a str ('sequence',
        'uniform') to generate a transition probability matrix.
    coeffs : np.ndarray
        Array of MAR coefficients. Shape must be (n_modes, n_lags, n_channels,
        n_channels).
    covs : np.ndarray
        Variance of eps_t. Shape must be (n_modes, n_channels).
    stay_prob : float
        Used to generate the transition probability matrix is trans_prob is a str.
        Optional.
    random_seed : int
        Seed for random number generator. Optional.
    """

    def __init__(
        self,
        n_samples: int,
        trans_prob: Union[np.ndarray, str, None],
        coeffs: np.ndarray,
        covs: np.ndarray,
        stay_prob: float = None,
        random_seed: int = None,
    ):
        # Observation model
        self.obs_mod = MAR(
            coeffs=coeffs,
            covs=covs,
            random_seed=random_seed,
        )

        self.n_modes = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        # HMM object
        # N.b. we use a different random seed to the observation model
        self.hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_modes=self.n_modes,
            random_seed=random_seed if random_seed is None else random_seed + 1,
        )

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        self.mode_time_course = self.hmm.generate_modes(self.n_samples)
        self.time_series = self.obs_mod.simulate_data(self.mode_time_course)

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        elif attr in dir(self.hmm):
            return getattr(self.hmm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")


class HMM_MVN(Simulation):
    """Simulate an HMM with a mulitvariate normal observation model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a str ('sequence',
        'uniform') to generate a transition probability matrix.
    means : np.ndarray or str
        Mean vector for each mode, shape should be (n_modes, n_channels).
        Either a numpy array or 'zero' or 'random'.
    covariances : np.ndarray or str
        Covariance matrix for each mode, shape should be (n_modes,
        n_channels, n_channels). Either a numpy array or 'random'.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    stay_prob : float
        Used to generate the transition probability matrix is trans_prob is a str.
        Optional.
    observation_error : float
        Standard deviation of the error added to the generated data.
    multiple_scale: bool
        Do we want to simulate different time courses for mean, variance, and fc?
    random_seed : int
        Seed for random number generator. Optional.
    """

    def __init__(
        self,
        n_samples: int,
        trans_prob: Union[np.ndarray, str, None],
        means: Union[np.ndarray, str],
        covariances: Union[np.ndarray, str],
        n_modes: int = None,
        n_channels: int = None,
        stay_prob: float = None,
        observation_error: float = 0.0,
        multiple_scale: bool = False,
        random_seed: int = None,
        fix_variance: bool = False,
    ):
        # Observation model
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

        # HMM object
        # N.b. we use a different random seed to the observation model
        self.hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_modes=self.n_modes,
            random_seed=random_seed if random_seed is None else random_seed + 1,
        )
        # placeholder for standard deviation for the simulated data
        self.standard_deviations = None

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        if not multiple_scale:
            self.mode_time_course = self.hmm.generate_modes(self.n_samples)
            self.time_series = self.obs_mod.simulate_data(self.mode_time_course)
        else:
            self.mode_time_course = np.zeros([self.n_samples, self.n_modes, 3], int)
            for i in range(3):
                self.mode_time_course[:, :, i] = self.hmm.generate_modes(self.n_samples)
            if fix_variance:
                # if fix_variance, then generate the variance time course with identity transition
                # probability matrix.
                self.hmm_fix_variance = HMM(
                    trans_prob=np.eye(self.n_modes),
                    stay_prob=stay_prob,
                    n_modes=self.n_modes,
                    random_seed=random_seed if random_seed is None else random_seed + 1,
                )
                self.mode_time_course[:, :, 1] = self.hmm_fix_variance.generate_modes(
                    self.n_samples
                )
            self.time_series = self.obs_mod.simulate_data_multiple_scales(
                self.mode_time_course
            )

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        elif attr in dir(self.hmm):
            return getattr(self.hmm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def standardize(self):
        self.standard_deviations = np.std(self.time_series, axis=0)
        super().standardize()
        self.obs_mod.covariances /= np.outer(
            self.standard_deviations, self.standard_deviations
        )[np.newaxis, ...]


class HierarchicalHMM_MVN(Simulation):
    """Hierarchical two-level HMM simulation.

    The bottom level HMMs share the same modes, i.e. have the same
    observation model. Only the transition probability matrix changes.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    top_level_trans_prob : np.ndarray
        Transition probability matrix of the top level HMM, which
        selects the bottom level HMM at each time point.
    bottom_level_trans_prob : list of np.ndarray
        Transitions probability matrices for the bottom level HMMs,
        which generate the observed data.
    means : np.ndarray or str
        Mean vector for each mode, shape should be (n_modes, n_channels).
        Either a numpy array or 'zero' or 'random'.
    covariances : np.ndarray or str
        Covariance matrix for each mode, shape should be (n_modes,
        n_channels, n_channels). Either a numpy array or 'random'.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    top_level_random_seed : int
        Random seed for generating the mode time course of the top level HMM.
    bottom_level_random_seeds : list of int
        Random seeds for the bottom level HMMs.
    data_random_seed : int
        Random seed for generating the observed data.
    """

    def __init__(
        self,
        n_samples: int,
        top_level_trans_prob: np.ndarray,
        bottom_level_trans_probs: list,
        means: np.ndarray = None,
        covariances: np.ndarray = None,
        n_modes: int = None,
        n_channels: int = None,
        observation_error: float = 0.0,
        top_level_random_seed: int = None,
        bottom_level_random_seeds: list = None,
        data_random_seed: int = None,
    ):
        # Observation model
        self.obs_mod = MVN(
            means=means,
            covariances=covariances,
            n_modes=n_modes,
            n_channels=n_channels,
            observation_error=observation_error,
            random_seed=data_random_seed,
        )

        self.n_modes = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        if bottom_level_random_seeds is None:
            bottom_level_random_seeds = [None] * len(bottom_level_trans_probs)

        # Top level HMM
        # This will select the bottom level HMM at each time point
        self.top_level_hmm = HMM(
            trans_prob=top_level_trans_prob,
            random_seed=top_level_random_seed,
        )

        # The bottom level HMMs
        # These will generate the data
        self.n_bottom_level_hmms = len(bottom_level_trans_probs)
        self.bottom_level_hmms = [
            HMM(
                trans_prob=bottom_level_trans_probs[i],
                random_seed=bottom_level_random_seeds[i],
            )
            for i in range(self.n_bottom_level_hmms)
        ]

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        self.mode_time_course = self.generate_modes(self.n_samples)
        self.time_series = self.obs_mod.simulate_data(self.mode_time_course)

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        elif attr in dir(self.hmm):
            return getattr(self.hmm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def generate_modes(self, n_samples):
        stc = np.empty([n_samples, self.n_modes])

        # Top level HMM to select the bottom level HMM at each time point
        top_level_stc = self.top_level_hmm.generate_modes(n_samples)

        # Generate mode time courses when each bottom level HMM is activate
        for i in range(self.n_bottom_level_hmms):
            time_points_active = np.argwhere(top_level_stc[:, i] == 1)[:, 0]
            stc[time_points_active] = self.bottom_level_hmms[i].generate_modes(
                n_samples
            )[: len(time_points_active)]

        return stc

    def standardize(self):
        standard_deviations = np.std(self.time_series, axis=0)
        super().standardize()
        self.obs_mod.covariances /= np.outer(standard_deviations, standard_deviations)[
            np.newaxis, ...
        ]


class HMM_Sine(Simulation):
    """Simulate an HMM with sine waves as the observation model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a str ('sequence',
        'uniform') to generate a transition probability matrix.
    amplitudes : np.ndarray
        Amplitude for the sine wave for each mode and channel.
        Shape must be (n_modes, n_channels).
    frequenices : np.ndarray
        Frequency for the sine wave for each mode and channel.
        Shape must be (n_modes, n_channels).
    sampling_frequency : float
        Sampling frequency in Hertz.
    covariances : np.ndarray
        Covariances for each mode. Shape must be (n_modes, n_channels)
        or (n_modes, n_channels, n_channels).
    stay_prob : float
        Used to generate the transition probability matrix is trans_prob is a str.
        Optional.
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for random number generator. Optional.
    """

    def __init__(
        self,
        n_samples: int,
        trans_prob: Union[np.ndarray, str, None],
        amplitudes: np.ndarray,
        frequencies: np.ndarray,
        sampling_frequency: float,
        covariances: np.ndarray = None,
        stay_prob: float = None,
        observation_error: float = 0.0,
        random_seed: int = None,
    ):
        # Observation model
        self.obs_mod = SingleSine(
            amplitudes=amplitudes,
            frequencies=frequencies,
            sampling_frequency=sampling_frequency,
            covariances=covariances,
            observation_error=observation_error,
            random_seed=random_seed,
        )

        self.n_modes = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        # HMM object
        # N.b. we use a different random seed to the observation model
        self.hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_modes=self.n_modes,
            random_seed=random_seed if random_seed is None else random_seed + 1,
        )

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        self.mode_time_course = self.hmm.generate_modes(self.n_samples)
        self.time_series = self.obs_mod.simulate_data(self.mode_time_course)

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        elif attr in dir(self.hmm):
            return getattr(self.hmm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")
