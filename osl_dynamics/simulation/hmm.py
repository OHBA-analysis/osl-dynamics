"""Classes for simulating Hidden Markov Models (HMMs).

"""

import warnings
import numpy as np

from osl_dynamics import array_ops
from osl_dynamics.simulation.mar import MAR
from osl_dynamics.simulation.mvn import MVN, MDyn_MVN, MSubj_MVN
from osl_dynamics.simulation.sin import SingleSine
from osl_dynamics.simulation.hsmm import HSMM
from osl_dynamics.simulation.base import Simulation


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
    n_states : int
        Number of states. Needed when trans_prob is a str to construct the
        transition probability matrix.
    random_seed : int
        Seed for random number generator.
    """

    def __init__(
        self,
        trans_prob,
        stay_prob=None,
        n_states=None,
        random_seed=None,
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
                    warnings.warn(
                        "Rows of trans_prob matrix must sum to 1. Transpose taken.",
                        RuntimeWarning,
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

            # Special case of there being only one state
            if n_states == 1:
                self.trans_prob = np.ones([1, 1])

            # Sequential transition probability matrix
            elif trans_prob == "sequence":
                if stay_prob is None or n_states is None:
                    raise ValueError(
                        "If trans_prob is 'sequence', stay_prob and n_states "
                        + "must be passed."
                    )
                self.trans_prob = self.construct_sequence_trans_prob(
                    stay_prob, n_states
                )

            # Uniform transition probability matrix
            elif trans_prob == "uniform":
                if n_states is None:
                    raise ValueError(
                        "If trans_prob is 'uniform', n_states must be passed."
                    )
                if stay_prob is None:
                    stay_prob = 1.0 / n_states
                self.trans_prob = self.construct_uniform_trans_prob(stay_prob, n_states)

        elif trans_prob is None and n_states == 1:
            self.trans_prob = np.ones([1, 1])

        # Infer number of states from the transition probability matrix
        self.n_states = self.trans_prob.shape[0]

        # Setup random number generator
        self._rng = np.random.default_rng(random_seed)

    @staticmethod
    def construct_sequence_trans_prob(stay_prob, n_states):
        trans_prob = np.zeros([n_states, n_states])
        np.fill_diagonal(trans_prob, stay_prob)
        np.fill_diagonal(trans_prob[:, 1:], 1 - stay_prob)
        trans_prob[-1, 0] = 1 - stay_prob
        return trans_prob

    @staticmethod
    def construct_uniform_trans_prob(stay_prob, n_states):
        single_trans_prob = (1 - stay_prob) / (n_states - 1)
        trans_prob = np.ones((n_states, n_states)) * single_trans_prob
        trans_prob[np.diag_indices(n_states)] = stay_prob
        return trans_prob

    def generate_states(self, n_samples):
        # Here the time course always start from state 0
        rands = [
            iter(self._rng.choice(self.n_states, size=n_samples, p=self.trans_prob[i]))
            for i in range(self.n_states)
        ]
        states = np.zeros(n_samples, int)
        for sample in range(1, n_samples):
            states[sample] = next(rands[states[sample - 1]])
        return array_ops.get_one_hot(states, n_states=self.n_states)


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
        Array of MAR coefficients. Shape must be (n_states, n_lags, n_channels,
        n_channels).
    covs : np.ndarray
        Variance of eps_t. Shape must be (n_states, n_channels).
    stay_prob : float
        Used to generate the transition probability matrix is trans_prob is a str.
    random_seed : int
        Seed for random number generator.
    """

    def __init__(
        self,
        n_samples,
        trans_prob,
        coeffs,
        covs,
        stay_prob=None,
        random_seed=None,
    ):
        # Observation model
        self.obs_mod = MAR(
            coeffs=coeffs,
            covs=covs,
            random_seed=random_seed,
        )

        self.n_states = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        # HMM object
        # N.b. we use a different random seed to the observation model
        self.hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_states=self.n_states,
            random_seed=random_seed if random_seed is None else random_seed + 1,
        )

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        self.state_time_course = self.hmm.generate_states(self.n_samples)
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
        Mean vector for each state, shape should be (n_states, n_channels).
        Either a numpy array or 'zero' or 'random'.
    covariances : np.ndarray or str
        Covariance matrix for each state, shape should be (n_states,
        n_channels, n_channels). Either a numpy array or 'random'.
    n_states : int
        Number of states. Can pass this argument with keyword n_modes instead.
    n_channels : int
        Number of channels.
    n_covariances_act : int
        Number of iterations to add activations to covariance matrices.
    stay_prob : float
        Used to generate the transition probability matrix is trans_prob is a str.
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for random number generator.
    """

    def __init__(
        self,
        n_samples,
        trans_prob,
        means,
        covariances,
        n_states=None,
        n_modes=None,
        n_channels=None,
        n_covariances_act=1,
        stay_prob=None,
        observation_error=0.0,
        random_seed=None,
    ):
        if n_states is None:
            n_states = n_modes

        # Observation model
        self.obs_mod = MVN(
            means=means,
            covariances=covariances,
            n_modes=n_states,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            observation_error=observation_error,
            random_seed=random_seed,
        )

        self.n_states = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        # HMM object
        # N.b. we use a different random seed to the observation model
        self.hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_states=self.n_states,
            random_seed=random_seed if random_seed is None else random_seed + 1,
        )

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        self.state_time_course = self.hmm.generate_states(self.n_samples)
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
        elif attr in dir(self.hmm):
            return getattr(self.hmm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def standardize(self):
        means = np.mean(self.time_series, axis=0).astype(np.float64)
        standard_deviations = np.std(self.time_series, axis=0).astype(np.float64)
        super().standardize()
        self.obs_mod.means = (
            self.obs_mod.means - means[None, ...]
        ) / standard_deviations[None, ...]
        self.obs_mod.covariances /= np.outer(standard_deviations, standard_deviations)[
            np.newaxis, ...
        ]
        self.obs_mod.instantaneous_covs /= np.outer(
            standard_deviations, standard_deviations
        )[np.newaxis, ...]


class MDyn_HMM_MVN(Simulation):
    """Simulate an HMM with a mulitvariate normal observation model.

    Multi-time-scale version of HMM_MVN.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a str ('sequence',
        'uniform') to generate a transition probability matrix.
    means : np.ndarray or str
        Mean vector for each state, shape should be (n_states, n_channels).
        Either a numpy array or 'zero' or 'random'.
    covariances : np.ndarray or str
        Covariance matrix for each state, shape should be (n_states,
        n_channels, n_channels). Either a numpy array or 'random'.
    n_states : int
        Number of states. Can pass this argument with keyword n_modes instead.
    n_channels : int
        Number of channels.
    n_covariances_act : int
        Number of iterations to add activations to covariance matrices.
    stay_prob : float
        Used to generate the transition probability matrix is trans_prob is a str.
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for random number generator.
    """

    def __init__(
        self,
        n_samples,
        trans_prob,
        means,
        covariances,
        n_states=None,
        n_modes=None,
        n_channels=None,
        n_covariances_act=1,
        stay_prob=None,
        observation_error=0.0,
        random_seed=None,
    ):
        if n_states is None:
            n_states = n_modes

        # Observation model
        self.obs_mod = MDyn_MVN(
            means=means,
            covariances=covariances,
            n_modes=n_states,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            observation_error=observation_error,
            random_seed=random_seed,
        )

        self.n_states = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        # HMM objects for sampling state time courses
        # N.b. we use a different random seed to the observation model
        self.alpha_hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_states=self.n_states,
            random_seed=random_seed if random_seed is None else random_seed + 1,
        )
        self.gamma_hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_states=self.n_states,
            random_seed=random_seed if random_seed is None else random_seed + 2,
        )

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate state time courses
        alpha = self.alpha_hmm.generate_states(self.n_samples)
        gamma = self.gamma_hmm.generate_states(self.n_samples)

        self.state_time_course = np.array([alpha, gamma])

        # Simulate data
        self.time_series = self.obs_mod.simulate_data(self.state_time_course)
        self.instantaneous_covs = self.obs_mod.instantaneous_covs

    @property
    def n_modes(self):
        return self.n_states

    @property
    def mode_time_course(self):
        return self.state_time_course

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def standardize(self):
        means = np.mean(self.time_series, axis=0).astype(np.float64)
        standard_deviations = np.std(self.time_series, axis=0).astype(np.float64)
        super().standardize()
        self.obs_mod.means = (
            self.obs_mod.means - means[None, ...]
        ) / standard_deviations[None, ...]
        self.obs_mod.stds /= standard_deviations[None, ...]
        self.obs_mod.instantaneous_covs /= np.outer(
            standard_deviations, standard_deviations
        )[np.newaxis, ...]


class MSubj_HMM_MVN(Simulation):
    """Simulate an HMM with multivariate normal observation model for each subject.

    Parameters
    ----------
    n_samples : int
        Number of samples per subject to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a str ('sequence',
        'uniform') to generate a transition probability matrix.
    subject_means : np.ndarray or str
        Subject mean vector for each state, shape should be (n_subjects, n_states, n_channels).
        Either a numpy array or 'zero' or 'random'.
    subject_covariances : np.ndarray or str
        Subject covariance matrix for each state, shape should be
        (n_subjects, n_states, n_channels, n_channels). Either a numpy array or 'random'.
    n_states : int
        Number of states. Can pass this argument with keyword n_modes instead.
    n_modes : int
        Number of modes.
    n_channels : int
        Number of channels.
    n_covariances_act : int
        Number of iterations to add activations to covariance matrices.
    n_subjects : int
        Number of subjects.
    n_groups : int
        Number of groups of subjects when subject means or covariances are 'random'.
    stay_prob : float
        Used to generate the transition probability matrix is trans_prob is a str.
    subject_tc_std : float
        Standard deviation when generating subject specific stay probability.
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for random number generator.
    """

    def __init__(
        self,
        n_samples,
        trans_prob,
        subject_means,
        subject_covariances,
        n_states=None,
        n_modes=None,
        n_channels=None,
        n_covariances_act=1,
        n_subjects=None,
        n_subject_embedding_dim=None,
        n_mode_embedding_dim=None,
        subject_embedding_scale=None,
        n_groups=None,
        between_group_scale=None,
        subject_tc_std=0.0,
        stay_prob=None,
        observation_error=0.0,
        random_seed=None,
    ):
        if n_states is None:
            n_states = n_modes

        # Construct trans_prob for each subject
        if isinstance(trans_prob, str) or trans_prob is None:
            trans_prob = [trans_prob] * n_subjects

        # Observation model
        self.obs_mod = MSubj_MVN(
            subject_means=subject_means,
            subject_covariances=subject_covariances,
            n_modes=n_states,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            n_subjects=n_subjects,
            n_subject_embedding_dim=n_subject_embedding_dim,
            n_mode_embedding_dim=n_mode_embedding_dim,
            subject_embedding_scale=subject_embedding_scale,
            n_groups=n_groups,
            between_group_scale=between_group_scale,
            observation_error=observation_error,
            random_seed=random_seed,
        )

        self.n_states = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels
        self.n_subjects = self.obs_mod.n_subjects

        # vary the stay probability for each subject
        if stay_prob is not None:
            subject_stay_prob = self.obs_mod._rng.normal(
                loc=stay_prob,
                scale=subject_tc_std,
                size=self.n_subjects,
            )
            # truncate stay_prob at 0 and 1
            subject_stay_prob = np.minimum(subject_stay_prob, 1)
            subject_stay_prob = np.maximum(subject_stay_prob, 0)
        else:
            subject_stay_prob = [stay_prob] * n_subjects

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate state time courses all subjects
        self.state_time_course = []
        self.hmm = []
        for subject in range(self.n_subjects):
            # Build HMM object with the subject's stay probalibity with different seeds
            hmm = HMM(
                trans_prob=trans_prob[subject],
                stay_prob=subject_stay_prob[subject],
                n_states=self.n_states,
                random_seed=random_seed
                if random_seed is None
                else random_seed + 1 + subject,
            )
            self.hmm.append(hmm)
            self.state_time_course.append(hmm.generate_states(self.n_samples))
        self.state_time_course = np.array(self.state_time_course)

        # Simulate data
        self.time_series = self.obs_mod.simulate_multi_subject_data(
            self.state_time_course
        )

    @property
    def n_modes(self):
        return self.n_states

    @property
    def mode_time_course(self):
        return self.state_time_course

    def __getattr__(self, attr):
        if attr in dir(self.obs_mod):
            return getattr(self.obs_mod, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def standardize(self):
        means = np.mean(self.time_series, axis=1).astype(np.float64)
        standard_deviations = np.std(self.time_series, axis=1).astype(np.float64)
        super().standardize(axis=1)
        self.obs_mod.subject_means = (
            self.obs_mod.subject_means - means[:, None, :]
        ) / standard_deviations[:, None, :]
        self.obs_mod.subject_covariances /= np.expand_dims(
            standard_deviations[:, :, None] @ standard_deviations[:, None, :], 1
        )
        self.obs_mod.instantaneous_covs /= np.expand_dims(
            standard_deviations[:, :, None] @ standard_deviations[:, None, :], 1
        )


class HierarchicalHMM_MVN(Simulation):
    """Hierarchical two-level HMM simulation.

    The bottom level HMMs share the same states, i.e. have the same
    observation model. Only the transition probability matrix changes.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    top_level_trans_prob : np.ndarray or str
        Transition probability matrix of the top level HMM, which
        selects the bottom level HMM at each time point. Used when
        top_level_hmm_type = 'hmm'
    bottom_level_trans_prob : list of np.ndarray or str
        Transitions probability matrices for the bottom level HMMs,
        which generate the observed data.
    means : np.ndarray or str
        Mean vector for each state, shape should be (n_states, n_channels).
        Either a numpy array or 'zero' or 'random'.
    covariances : np.ndarray or str
        Covariance matrix for each state, shape should be (n_states,
        n_channels, n_channels). Either a numpy array or 'random'.
    n_states : int
        Number of states. Can pass this argument with keyword n_modes instead.
    n_channels : int
        Number of channels.
    n_covariances_act : int
        Number of iterations to add activations to covariance matrices.
    observation_error : float
        Standard deviation of random noise to be added to the observations.
    top_level_random_seed : int
        Random seed for generating the state time course of the top level HMM.
    bottom_level_random_seeds : list of int
        Random seeds for the bottom level HMMs.
    data_random_seed : int
        Random seed for generating the observed data.
    top_level_stay_prob : float
        The stay_prob for the top level HMM. Used if top_level_trans_prob is
        a str. Used when top_level_hmm_type = 'hmm'.
    bottom_level_stay_probs : list of float
        The list of stay_prob values for the bottom level HMMs. Used when the
        correspondining entry in bottom_level_trans_prob is a str.
    top_level_hmm_type: str
        The type of HMM to use at the top level -- either hmm or hsmm.
    top_level_gamma_shape: float
        The shape parameter for the gamma distribution used by
        the top level hmm when top_level_hmm_type = 'hsmm'.
    top_level_gamma_scale: float = None
        The scale parameter for the gamma distribution used by
        the top level hmm when top_level_hmm_type = 'hsmm'.
    """

    def __init__(
        self,
        n_samples,
        top_level_trans_prob,
        bottom_level_trans_probs,
        means=None,
        covariances=None,
        n_states=None,
        n_modes=None,
        n_channels=None,
        n_covariances_act=1,
        observation_error=0.0,
        top_level_random_seed=None,
        bottom_level_random_seeds=None,
        data_random_seed=None,
        top_level_stay_prob=None,
        bottom_level_stay_probs=None,
        top_level_hmm_type="hmm",
        top_level_gamma_shape=None,
        top_level_gamma_scale=None,
    ):
        if n_states is None:
            n_states = n_modes

        # Observation model
        self.obs_mod = MVN(
            means=means,
            covariances=covariances,
            n_modes=n_states,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            observation_error=observation_error,
            random_seed=data_random_seed,
        )

        self.n_states = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        if bottom_level_stay_probs is None:
            bottom_level_stay_probs = [None] * len(bottom_level_trans_probs)

        if bottom_level_random_seeds is None:
            bottom_level_random_seeds = [None] * len(bottom_level_trans_probs)

        # Top level HMM
        # This will select the bottom level HMM at each time point
        if top_level_hmm_type.lower() == "hmm":
            self.top_level_hmm = HMM(
                trans_prob=top_level_trans_prob,
                random_seed=top_level_random_seed,
                stay_prob=top_level_stay_prob,
                n_states=len(bottom_level_trans_probs),
            )
        elif top_level_hmm_type.lower() == "hsmm":
            self.top_level_hmm = HSMM(
                gamma_shape=top_level_gamma_shape,
                gamma_scale=top_level_gamma_scale,
                n_states=len(bottom_level_trans_probs),
                random_seed=top_level_random_seed,
            )
        else:
            raise ValueError(f"Unsupported top_level_hmm_type: {top_level_hmm_type}")

        # The bottom level HMMs
        # These will generate the data
        self.n_bottom_level_hmms = len(bottom_level_trans_probs)
        self.bottom_level_hmms = [
            HMM(
                trans_prob=bottom_level_trans_probs[i],
                random_seed=bottom_level_random_seeds[i],
                stay_prob=bottom_level_stay_probs[i],
                n_states=n_states,
            )
            for i in range(self.n_bottom_level_hmms)
        ]

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        self.state_time_course = self.generate_states(self.n_samples)
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
        elif attr in dir(self.hmm):
            return getattr(self.hmm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def generate_states(self, n_samples):
        stc = np.empty([n_samples, self.n_states])

        # Top level HMM to select the bottom level HMM at each time point
        self.top_level_stc = self.top_level_hmm.generate_states(n_samples)

        # Generate state time courses when each bottom level HMM is activate
        for i in range(self.n_bottom_level_hmms):
            time_points_active = np.argwhere(self.top_level_stc[:, i] == 1)[:, 0]
            stc[time_points_active] = self.bottom_level_hmms[i].generate_states(
                n_samples
            )[: len(time_points_active)]

        return stc

    def standardize(self):
        standard_deviations = np.std(self.time_series, axis=0).astype(np.float64)
        super().standardize()
        self.obs_mod.covariances /= np.outer(standard_deviations, standard_deviations)[
            np.newaxis, ...
        ]
        self.obs_mod.instantaneous_covs /= np.outer(
            standard_deviations, standard_deviations
        )[np.newaxis, ...]


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
        Amplitude for the sine wave for each state and channel.
        Shape must be (n_states, n_channels).
    frequenices : np.ndarray
        Frequency for the sine wave for each state and channel.
        Shape must be (n_states, n_channels).
    sampling_frequency : float
        Sampling frequency in Hertz.
    covariances : np.ndarray
        Covariances for each state. Shape must be (n_states, n_channels)
        or (n_states, n_channels, n_channels).
    stay_prob : float
        Used to generate the transition probability matrix is trans_prob is a str.
    observation_error : float
        Standard deviation of the error added to the generated data.
    random_seed : int
        Seed for random number generator.
    """

    def __init__(
        self,
        n_samples,
        trans_prob,
        amplitudes,
        frequencies,
        sampling_frequency,
        covariances=None,
        stay_prob=None,
        observation_error=0.0,
        random_seed=None,
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

        self.n_states = self.obs_mod.n_states
        self.n_channels = self.obs_mod.n_channels

        # HMM object
        # N.b. we use a different random seed to the observation model
        self.hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_states=self.n_states,
            random_seed=random_seed if random_seed is None else random_seed + 1,
        )

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate data
        self.state_time_course = self.hmm.generate_states(self.n_samples)
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
        elif attr in dir(self.hmm):
            return getattr(self.hmm, attr)
        else:
            raise AttributeError(f"No attribute called {attr}.")
