"""Classes for simulating Hidden Markov Models (HMMs).

"""

import warnings
import numpy as np

from osl_dynamics import array_ops
from osl_dynamics.simulation.mar import MAR
from osl_dynamics.simulation.mvn import MVN, MDyn_MVN, MSess_MVN
from osl_dynamics.simulation.hsmm import HSMM
from osl_dynamics.simulation.base import Simulation
from osl_dynamics.simulation.poi import Poisson


class HMM:
    """HMM base class.

    Contains the transition probability matrix.

    Parameters
    ----------
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a :code:`str`
        (:code:`'sequence'` or :code:`'uniform'`) to generate a transition
        probability matrix.
    stay_prob : float, optional
        Used to generate the transition probability matrix is :code:`trans_prob`
        is a :code:`str`. Must be between 0 and 1.
    n_states : int, optional
        Number of states. Needed when :code:`trans_prob` is a :code:`str` to
        construct the transition probability matrix.
    """

    def __init__(
        self,
        trans_prob,
        stay_prob=None,
        n_states=None,
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
            if not all(np.isclose(row_sums, 1)):
                if all(np.isclose(col_sums, 1)):
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
                        "must be passed."
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
                self.trans_prob = self.construct_uniform_trans_prob(
                    stay_prob,
                    n_states,
                )

        elif trans_prob is None and n_states == 1:
            self.trans_prob = np.ones([1, 1])

        # Infer number of states from the transition probability matrix
        self.n_states = self.trans_prob.shape[0]

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
            iter(np.random.choice(self.n_states, size=n_samples, p=self.trans_prob[i]))
            for i in range(self.n_states)
        ]
        states = np.zeros(n_samples, int)
        for sample in range(1, n_samples):
            states[sample] = next(rands[states[sample - 1]])
        return array_ops.get_one_hot(states, n_states=self.n_states)


class HMM_MAR(Simulation):
    """Simulate an HMM with a multivariate autoregressive (MAR) observation
    model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a :code:`str`
        (:code:`'sequence'` or :code:`'uniform'`) to generate a transition
        probability matrix.
    coeffs : np.ndarray
        Array of MAR coefficients. Shape must be (n_states, n_lags, n_channels,
        n_channels).
    covs : np.ndarray
        Variance of :math:`\epsilon_t`. See `simulation.MAR \
        <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/\
        simulation/mar/index.html#osl_dynamics.simulation.mar.MAR>`_ for further
        details. Shape must be (n_states, n_channels).
    stay_prob : float, optional
        Used to generate the transition probability matrix is
        :code:`trans_prob` is a :code:`str`. Must be between 0 and 1.
    """

    def __init__(
        self,
        n_samples,
        trans_prob,
        coeffs,
        covs,
        stay_prob=None,
    ):
        # Observation model
        self.obs_mod = MAR(coeffs=coeffs, covs=covs)

        self.n_states = self.obs_mod.n_states
        self.n_channels = self.obs_mod.n_channels

        # HMM object
        # N.b. we use a different random seed to the observation model
        self.hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_states=self.n_states,
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
        Transition probability matrix as a numpy array or a :code:`str`
        (:code:`'sequence'` or :code:`'uniform'`) to generate a transition
        probability matrix.
    means : np.ndarray or str
        Mean vector for each state, shape should be (n_states, n_channels).
        Either a numpy array or :code:`'zero'` or :code:`'random'`.
    covariances : np.ndarray or str
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels). Either a numpy array or :code:`'random'`.
    n_states : int, optional
        Number of states. Can pass this argument with keyword :code:`n_modes`
        instead.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    stay_prob : float, optional
        Used to generate the transition probability matrix is :code:`trans_prob`
        is a :code:`str`. Must be between 0 and 1.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
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
        )

        self.n_states = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        # HMM object
        # N.b. we use a different random seed to the observation model
        self.hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_states=self.n_states,
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

    def get_instantaneous_covariances(self):
        """Get the ground truth covariances at each time point.

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances.
            Shape is (n_samples, n_channels, n_channels).
        """
        return self.obs_mod.get_instantaneous_covariances(self.state_time_course)

    def standardize(self):
        mu = np.mean(self.time_series, axis=0).astype(np.float64)
        sigma = np.std(self.time_series, axis=0).astype(np.float64)
        super().standardize()
        self.obs_mod.means = (self.obs_mod.means - mu[np.newaxis, ...]) / sigma[
            np.newaxis, ...
        ]
        self.obs_mod.covariances /= np.outer(sigma, sigma)[np.newaxis, ...]


class MDyn_HMM_MVN(Simulation):
    """Simulate an HMM with a mulitvariate normal observation model.

    Multi-time-scale version of :code:`HMM_MVN`.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a :code:`str`
        (:code:`'sequence'` or :code:`'uniform'`) to generate a transition
        probability matrix.
    means : np.ndarray or str
        Mean vector for each state, shape should be (n_states, n_channels).
        Either a numpy array or :code:`'zero'` or :code:`'random'`.
    covariances : np.ndarray or str
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels). Either a numpy array or :code:`'random'`.
    n_states : int, optional
        Number of states. Can pass this argument with keyword :code:`n_modes`
        instead.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    stay_prob : float, optional
        Used to generate the transition probability matrix is :code:`trans_prob`
        is a :code:`str`. Must be between 0 and 1.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
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
        )

        self.n_states = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        # HMM objects for sampling state time courses
        # N.b. we use a different random seed to the observation model
        self.alpha_hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_states=self.n_states,
        )
        self.beta_hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_states=self.n_states,
        )

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate state time courses
        alpha = self.alpha_hmm.generate_states(self.n_samples)
        beta = self.beta_hmm.generate_states(self.n_samples)

        self.state_time_course = np.array([alpha, beta])

        # Simulate data
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
        else:
            raise AttributeError(f"No attribute called {attr}.")

    def get_instantaneous_covariances(self):
        """Get the ground truth covariances at each time point.

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances.
            Shape is (n_samples, n_channels, n_channels).
        """
        return self.obs_mod.get_instantaneous_covariances(self.state_time_course)

    def standardize(self):
        mu = np.mean(self.time_series, axis=0).astype(np.float64)
        sigma = np.std(self.time_series, axis=0).astype(np.float64)
        super().standardize()
        self.obs_mod.means = (self.obs_mod.means - mu[np.newaxis, ...]) / sigma[
            np.newaxis, ...
        ]
        self.obs_mod.stds /= sigma[np.newaxis, ...]


class HMM_Poi(Simulation):
    """Simulate an HMM with Poisson distribution as the observation model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a str ('sequence',
        'uniform') to generate a transition probability matrix.
    rates : np.ndarray
        Amplitude for the sine wave for each state and channel.
        Shape must be (n_states, n_channels).
    stay_prob : float
        Used to generate the transition probability matrix is trans_prob is a str.
    """

    def __init__(
        self,
        n_samples,
        trans_prob,
        rates,
        n_states=None,
        n_channels=None,
        stay_prob=None,
    ):
        # Observation model
        self.obs_mod = Poisson(
            rates=rates,
            n_states=n_states,
            n_channels=n_channels,
        )

        self.n_states = self.obs_mod.n_states
        self.n_channels = self.obs_mod.n_channels

        # HMM object
        # N.b. we use a different random seed to the observation model
        self.hmm = HMM(
            trans_prob=trans_prob,
            stay_prob=stay_prob,
            n_states=self.n_states,
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


class MSess_HMM_MVN(Simulation):
    """Simulate an HMM with multivariate normal observation model for each
    session.

    Parameters
    ----------
    n_samples : int
        Number of samples per session to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a :code:`str`
        (:code:`'sequence'` or :code:`'uniform'`) to generate a transition
        probability matrix.
    session_means : np.ndarray or str
        Session mean vector for each state, shape should be
        (n_sessions, n_states, n_channels).
        Either a numpy array or :code:`'zero'` or :code:`'random'`.
    session_covariances : np.ndarray or str
        Session covariance matrix for each state, shape should be
        (n_sessions, n_states, n_channels, n_channels).
        Either a numpy array or :code:`'random'`.
    n_states : int, optional
        Number of states. Can pass this argument with keyword :code:`n_modes`
        instead.
    n_modes : int, optional
        Number of modes.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    embedding_vectors : np.ndarray, optional
        Embedding vectors for each state, shape should be
        (n_states, embeddings_dim).
    n_sessions : int, optional
        Number of sessions.
    embeddings_dim : int
        Dimension of the embedding vectors.
    spatial_embeddings_dim : int
        Dimension of the spatial embedding vectors.
    embeddings_scale : float
        Scale of the embedding vectors.
    n_groups : int, optional
        Number of groups when session means or covariances are
        :code:`'random'`.
    between_group_scale : float, optional
        Scale of variability between session observation parameters.
    stay_prob : float, optional
        Used to generate the transition probability matrix is :code:`trans_prob`
        is a :code:`str`. Must be between 0 and 1.
    tc_std : float, optional
        Standard deviation when generating session-specific stay probability.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
    """

    def __init__(
        self,
        n_samples,
        trans_prob,
        session_means,
        session_covariances,
        n_states=None,
        n_modes=None,
        n_channels=None,
        n_covariances_act=1,
        embedding_vectors=None,
        n_sessions=None,
        embeddings_dim=None,
        spatial_embeddings_dim=None,
        embeddings_scale=None,
        n_groups=None,
        between_group_scale=None,
        tc_std=0.0,
        stay_prob=None,
        observation_error=0.0,
    ):
        if n_states is None:
            n_states = n_modes

        # Observation model
        self.obs_mod = MSess_MVN(
            session_means=session_means,
            session_covariances=session_covariances,
            n_modes=n_states,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            embedding_vectors=embedding_vectors,
            n_sessions=n_sessions,
            embeddings_dim=embeddings_dim,
            spatial_embeddings_dim=spatial_embeddings_dim,
            embeddings_scale=embeddings_scale,
            n_groups=n_groups,
            between_group_scale=between_group_scale,
            observation_error=observation_error,
        )

        self.n_states = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels
        self.n_sessions = self.obs_mod.n_sessions

        # Construct trans_prob for each session
        if isinstance(trans_prob, str) or trans_prob is None:
            trans_prob = [trans_prob] * self.n_sessions

        # Vary the stay probability for each session
        if stay_prob is not None:
            session_stay_prob = np.random.normal(
                loc=stay_prob,
                scale=tc_std,
                size=self.n_sessions,
            )
            # truncate stay_prob at 0 and 1
            session_stay_prob = np.minimum(session_stay_prob, 1)
            session_stay_prob = np.maximum(session_stay_prob, 0)
        else:
            session_stay_prob = [stay_prob] * self.n_sessions

        # Initialise base class
        super().__init__(n_samples=n_samples)

        # Simulate state time courses for all sessions
        self.state_time_course = []
        self.hmm = []
        for i in range(self.n_sessions):
            # Build HMM object with the session's stay probalibity with
            # different seeds
            hmm = HMM(
                trans_prob=trans_prob[i],
                stay_prob=session_stay_prob[i],
                n_states=self.n_states,
            )
            self.hmm.append(hmm)
            self.state_time_course.append(hmm.generate_states(self.n_samples))
        self.state_time_course = np.array(self.state_time_course)

        # Simulate data
        self.time_series = self.obs_mod.simulate_multi_session_data(
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

    def get_instantaneous_covariances(self):
        """Get the ground truth covariances at each time point.

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances.
            Shape is (n_samples, n_channels, n_channels).
        """
        return self.obs_mod.get_instantaneous_covariances(self.state_time_course)

    def standardize(self):
        mu = np.mean(self.time_series, axis=1).astype(np.float64)
        sigma = np.std(self.time_series, axis=1).astype(np.float64)
        super().standardize(axis=1)
        self.obs_mod.session_means = (
            self.obs_mod.session_means - mu[:, np.newaxis, :]
        ) / sigma[:, np.newaxis, :]
        self.obs_mod.session_covariances /= np.expand_dims(
            sigma[:, :, np.newaxis] @ sigma[:, np.newaxis, :], 1
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
        :code:`top_level_hmm_type='hmm'`.
    bottom_level_trans_prob : list of np.ndarray or str
        Transitions probability matrices for the bottom level HMMs,
        which generate the observed data.
    means : np.ndarray or str, optional
        Mean vector for each state, shape should be (n_states, n_channels).
        Either a numpy array or :code:`'zero'` or :code:`'random'`.
    covariances : np.ndarray or str, optional
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels). Either a numpy array or :code:`'random'`.
    n_states : int, optional
        Number of states. Can pass this argument with keyword :code:`n_modes`
        instead.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    observation_error : float, optional
        Standard deviation of random noise to be added to the observations.
    top_level_stay_prob : float, optional
        The stay_prob for the top level HMM. Used if
        :code:`top_level_trans_prob` is a :code:`str`.
        Used when :code:`top_level_hmm_type='hmm'`.
    bottom_level_stay_probs : list of float, optional
        The list of :code:`stay_prob` values for the bottom level HMMs.
        Used when the correspondining entry in :code:`bottom_level_trans_prob`
        is a :code:`str`.
    top_level_hmm_type: str, optional
        The type of HMM to use at the top level, either :code:`'hmm'` or
        :code:`'hsmm'`.
    top_level_gamma_shape: float, optional
        The shape parameter for the Gamma distribution used by
        the top level hmm when :code:`top_level_hmm_type='hsmm'`.
    top_level_gamma_scale: float, optional
        The scale parameter for the Gamma distribution used by
        the top level hmm when :code:`top_level_hmm_type='hsmm'`.
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
        )

        self.n_states = self.obs_mod.n_modes
        self.n_channels = self.obs_mod.n_channels

        if bottom_level_stay_probs is None:
            bottom_level_stay_probs = [None] * len(bottom_level_trans_probs)

        # Top level HMM
        # This will select the bottom level HMM at each time point
        if top_level_hmm_type.lower() == "hmm":
            self.top_level_hmm = HMM(
                trans_prob=top_level_trans_prob,
                stay_prob=top_level_stay_prob,
                n_states=len(bottom_level_trans_probs),
            )
        elif top_level_hmm_type.lower() == "hsmm":
            self.top_level_hmm = HSMM(
                gamma_shape=top_level_gamma_shape,
                gamma_scale=top_level_gamma_scale,
                n_states=len(bottom_level_trans_probs),
            )
        else:
            raise ValueError(f"Unsupported top_level_hmm_type: {top_level_hmm_type}")

        # The bottom level HMMs
        # These will generate the data
        self.n_bottom_level_hmms = len(bottom_level_trans_probs)
        self.bottom_level_hmms = [
            HMM(
                trans_prob=bottom_level_trans_probs[i],
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
        sigma = np.std(self.time_series, axis=0).astype(np.float64)
        super().standardize()
        self.obs_mod.covariances /= np.outer(sigma, sigma)[np.newaxis, ...]
