"""Hidden Markov Models (HMMs)."""

import warnings
from typing import List, Optional, Union

import numpy as np

from osl_dynamics.simulation.base import Simulation
from osl_dynamics.simulation.hsmm import HSMM
from osl_dynamics.simulation.obs_mod import (
    MAR,
    MVN,
    MDyn_MVN,
    MSess_MVN,
    OscillatoryBursts,
    OscillatoryMVN,
    Poisson,
    Dirichlet,
    TDECovs,
)
from osl_dynamics.utils import array_ops


class HMM:
    """Hidden Markov Model (HMM) state model.

    Parameters
    ----------
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a :code:`str`
        (:code:`'sequence'` or :code:`'uniform'`) to generate a transition
        probability matrix.
    stay_prob : float, optional
        Used to generate the transition probability matrix if :code:`trans_prob`
        is a :code:`str`. Must be between 0 and 1.
    n_states : int, optional
        Number of states. Needed when :code:`trans_prob` is a :code:`str` to
        construct the transition probability matrix.
    """

    def __init__(
        self,
        trans_prob: Union[np.ndarray, str, None],
        stay_prob: Optional[float] = None,
        n_states: Optional[int] = None,
    ) -> None:
        if isinstance(trans_prob, list):
            trans_prob = np.array(trans_prob)

        if isinstance(trans_prob, np.ndarray):
            # Don't need to generate the transition probability matrix

            if trans_prob.ndim != 2:
                raise ValueError("trans_prob must be a 2D array.")

            if trans_prob.shape[0] != trans_prob.shape[1]:
                raise ValueError("trans_prob must be a square matrix.")

            # Check the rows of the transition probability matrix sum to one
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
    def construct_sequence_trans_prob(stay_prob: float, n_states: int) -> np.ndarray:
        trans_prob = np.zeros([n_states, n_states])
        np.fill_diagonal(trans_prob, stay_prob)
        np.fill_diagonal(trans_prob[:, 1:], 1 - stay_prob)
        trans_prob[-1, 0] = 1 - stay_prob
        return trans_prob

    @staticmethod
    def construct_uniform_trans_prob(stay_prob: float, n_states: int) -> np.ndarray:
        single_trans_prob = (1 - stay_prob) / (n_states - 1)
        trans_prob = np.ones((n_states, n_states)) * single_trans_prob
        trans_prob[np.diag_indices(n_states)] = stay_prob
        return trans_prob

    def generate_states(self, n_samples: int) -> np.ndarray:
        """Generate a state time course.

        The time course always starts in the first state.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        state_time_course : np.ndarray
            One-hot state time course. Shape is (n_samples, n_states).
        """
        rands = [
            iter(np.random.choice(self.n_states, size=n_samples, p=self.trans_prob[i]))
            for i in range(self.n_states)
        ]
        states = np.zeros(n_samples, int)
        for sample in range(1, n_samples):
            states[sample] = next(rands[states[sample - 1]])
        return array_ops.get_one_hot(states, n_states=self.n_states)


class MDynHMM:
    """Two independent HMMs for multi-time-scale dynamics.

    The first HMM (:code:`alpha_hmm`) gives the time course of the means and
    standard deviations, the second (:code:`beta_hmm`) gives the time course
    of the correlations. See :class:`~osl_dynamics.simulation.MDyn_MVN`.

    Parameters
    ----------
    trans_prob : np.ndarray or str
        Transition probability matrix of both HMMs. See :class:`HMM`.
    stay_prob : float, optional
        Used to generate the transition probability matrix if
        :code:`trans_prob` is a :code:`str`.
    n_states : int, optional
        Number of states.
    """

    def __init__(
        self,
        trans_prob: Union[np.ndarray, str],
        stay_prob: Optional[float] = None,
        n_states: Optional[int] = None,
    ) -> None:
        self.alpha_hmm = HMM(
            trans_prob=trans_prob, stay_prob=stay_prob, n_states=n_states
        )
        self.beta_hmm = HMM(
            trans_prob=trans_prob, stay_prob=stay_prob, n_states=n_states
        )
        self.n_states = self.alpha_hmm.n_states

    def generate_states(self, n_samples: int) -> np.ndarray:
        """Generate the two state time courses.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        state_time_courses : np.ndarray
            Shape is (2, n_samples, n_states).
        """
        alpha = self.alpha_hmm.generate_states(n_samples)
        beta = self.beta_hmm.generate_states(n_samples)
        return np.array([alpha, beta])


class HierarchicalHMM:
    """Two-level hierarchical HMM.

    A top-level HMM (or HSMM) selects which bottom-level HMM generates the
    state at each time point. The bottom-level HMMs share the same states and
    only differ in their transition probability matrix.

    Parameters
    ----------
    top_level_trans_prob : np.ndarray or str
        Transition probability matrix of the top-level HMM. Used when
        :code:`top_level_hmm_type='hmm'`.
    bottom_level_trans_probs : list of np.ndarray or str
        Transition probability matrices for the bottom-level HMMs.
    n_states : int, optional
        Number of states.
    top_level_stay_prob : float, optional
        The :code:`stay_prob` for the top-level HMM. Used if
        :code:`top_level_trans_prob` is a :code:`str`.
    bottom_level_stay_probs : list of float, optional
        The :code:`stay_prob` for each bottom-level HMM. Used when the
        corresponding entry in :code:`bottom_level_trans_probs` is a
        :code:`str`.
    top_level_hmm_type : str, optional
        Either :code:`'hmm'` or :code:`'hsmm'`.
    top_level_gamma_shape : float, optional
        Shape parameter for the Gamma distribution of the top-level state
        lifetimes when :code:`top_level_hmm_type='hsmm'`.
    top_level_gamma_scale : float, optional
        Scale parameter for the Gamma distribution of the top-level state
        lifetimes when :code:`top_level_hmm_type='hsmm'`.
    """

    def __init__(
        self,
        top_level_trans_prob: Union[np.ndarray, str, None],
        bottom_level_trans_probs: List[Union[np.ndarray, str]],
        n_states: Optional[int] = None,
        top_level_stay_prob: Optional[float] = None,
        bottom_level_stay_probs: Optional[List[float]] = None,
        top_level_hmm_type: str = "hmm",
        top_level_gamma_shape: Optional[float] = None,
        top_level_gamma_scale: Optional[float] = None,
    ) -> None:
        if bottom_level_stay_probs is None:
            bottom_level_stay_probs = [None] * len(bottom_level_trans_probs)

        # Top-level HMM to select the bottom-level HMM at each time point
        self.n_bottom_level_hmms = len(bottom_level_trans_probs)
        if top_level_hmm_type.lower() == "hmm":
            self.top_level_hmm = HMM(
                trans_prob=top_level_trans_prob,
                stay_prob=top_level_stay_prob,
                n_states=self.n_bottom_level_hmms,
            )
        elif top_level_hmm_type.lower() == "hsmm":
            self.top_level_hmm = HSMM(
                gamma_shape=top_level_gamma_shape,
                gamma_scale=top_level_gamma_scale,
                n_states=self.n_bottom_level_hmms,
            )
        else:
            raise ValueError(f"Unsupported top_level_hmm_type: {top_level_hmm_type}")

        # Bottom-level HMMs to generate the states
        self.bottom_level_hmms = [
            HMM(
                trans_prob=bottom_level_trans_probs[i],
                stay_prob=bottom_level_stay_probs[i],
                n_states=n_states,
            )
            for i in range(self.n_bottom_level_hmms)
        ]
        self.n_states = self.bottom_level_hmms[0].n_states

    def generate_states(self, n_samples: int) -> np.ndarray:
        """Generate a state time course.

        The top-level state time course is stored in :code:`top_level_stc`.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        state_time_course : np.ndarray
            One-hot state time course. Shape is (n_samples, n_states).
        """
        stc = np.empty([n_samples, self.n_states], dtype=int)
        self.top_level_stc = self.top_level_hmm.generate_states(n_samples)
        for i in range(self.n_bottom_level_hmms):
            time_points_active = np.argwhere(self.top_level_stc[:, i] == 1)[:, 0]
            stc[time_points_active] = self.bottom_level_hmms[i].generate_states(
                n_samples
            )[: len(time_points_active)]
        return stc


class HMM_MVN(Simulation):
    """HMM with a multivariate normal observation model.

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
        Number of states.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    stay_prob : float, optional
        Used to generate the transition probability matrix if :code:`trans_prob`
        is a :code:`str`. Must be between 0 and 1.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
    """

    def __init__(
        self,
        n_samples: int,
        trans_prob: Union[np.ndarray, str],
        means: Union[np.ndarray, str],
        covariances: Union[np.ndarray, str],
        n_states: Optional[int] = None,
        n_channels: Optional[int] = None,
        n_covariances_act: int = 1,
        stay_prob: Optional[float] = None,
        observation_error: float = 0.0,
    ) -> None:
        obs_model = MVN(
            means=means,
            covariances=covariances,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            observation_error=observation_error,
        )
        obs_model.initialize(n_states)
        self.hmm = HMM(
            trans_prob=trans_prob, stay_prob=stay_prob, n_states=obs_model._n
        )
        super().__init__(self.hmm, obs_model, n_samples)


class HMM_OscillatoryMVN(Simulation):
    """HMM with an oscillatory multivariate normal observation model.

    Each state has a covariance and a spectral peak at its own frequency. See
    :class:`~osl_dynamics.simulation.OscillatoryMVN`.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a :code:`str`
        (:code:`'sequence'` or :code:`'uniform'`) to generate a transition
        probability matrix.
    frequencies : np.ndarray or list
        Peak frequency (Hz) of each state, shape (n_states,), or of each state
        and channel, shape (n_states, n_channels).
    sampling_frequency : float
        Sampling frequency in Hz.
    covariances : np.ndarray or str
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels). Either a numpy array or :code:`'random'`.
    means : np.ndarray or str, optional
        Mean vector for each state, shape should be (n_states, n_channels).
        Either a numpy array or :code:`'zero'` or :code:`'random'`.
    peak_width : float or np.ndarray, optional
        Width (Hz) of the spectral peaks. A float, or one value per state.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    stay_prob : float, optional
        Used to generate the transition probability matrix if :code:`trans_prob`
        is a :code:`str`. Must be between 0 and 1.
    observation_error : float, optional
        Standard deviation of the white noise added to the generated data.
    aperiodic_covariance : np.ndarray, optional
        Covariance of the aperiodic (:math:`1/f`) background activity, shape
        (n_channels, n_channels). :code:`None` for no background.
    aperiodic_exponent : float, optional
        Exponent of the aperiodic power spectrum.
    aperiodic_knee : float, optional
        Knee frequency (Hz) of the aperiodic power spectrum.
    """

    def __init__(
        self,
        n_samples: int,
        trans_prob: Union[np.ndarray, str],
        frequencies: Union[np.ndarray, List[float]],
        sampling_frequency: float,
        covariances: Union[np.ndarray, str],
        means: Union[np.ndarray, str] = "zero",
        peak_width: Union[float, np.ndarray] = 2.0,
        n_channels: Optional[int] = None,
        n_covariances_act: int = 1,
        stay_prob: Optional[float] = None,
        observation_error: float = 0.0,
        aperiodic_covariance: Optional[np.ndarray] = None,
        aperiodic_exponent: float = 1.0,
        aperiodic_knee: float = 1.0,
    ) -> None:
        obs_model = OscillatoryMVN(
            frequencies=frequencies,
            sampling_frequency=sampling_frequency,
            covariances=covariances,
            means=means,
            peak_width=peak_width,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            observation_error=observation_error,
            aperiodic_covariance=aperiodic_covariance,
            aperiodic_exponent=aperiodic_exponent,
            aperiodic_knee=aperiodic_knee,
        )
        self.hmm = HMM(
            trans_prob=trans_prob, stay_prob=stay_prob, n_states=obs_model._n
        )
        super().__init__(self.hmm, obs_model, n_samples)


class HMM_MAR(Simulation):
    r"""HMM with a multivariate autoregressive (MAR) observation model.

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
        Variance of :math:`\epsilon_t`. See
        :class:`~osl_dynamics.simulation.MAR` for further details. Shape must
        be (n_states, n_channels).
    stay_prob : float, optional
        Used to generate the transition probability matrix if
        :code:`trans_prob` is a :code:`str`. Must be between 0 and 1.
    """

    def __init__(
        self,
        n_samples: int,
        trans_prob: Union[np.ndarray, str],
        coeffs: np.ndarray,
        covs: np.ndarray,
        stay_prob: Optional[float] = None,
    ) -> None:
        obs_model = MAR(coeffs=coeffs, covs=covs)
        self.hmm = HMM(
            trans_prob=trans_prob, stay_prob=stay_prob, n_states=obs_model._n
        )
        super().__init__(self.hmm, obs_model, n_samples)


class MDyn_HMM_MVN(Simulation):
    """HMM with a multivariate normal observation model and multi-time-scale
    dynamics.

    Two independent HMMs give the time course of the means and standard
    deviations, and of the correlations. The state time course has shape
    (2, n_samples, n_states).

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
        Number of states.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    stay_prob : float, optional
        Used to generate the transition probability matrix if :code:`trans_prob`
        is a :code:`str`. Must be between 0 and 1.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
    """

    def __init__(
        self,
        n_samples: int,
        trans_prob: Union[np.ndarray, str],
        means: Union[np.ndarray, str],
        covariances: Union[np.ndarray, str],
        n_states: Optional[int] = None,
        n_channels: Optional[int] = None,
        n_covariances_act: int = 1,
        stay_prob: Optional[float] = None,
        observation_error: float = 0.0,
    ) -> None:
        obs_model = MDyn_MVN(
            means=means,
            covariances=covariances,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            observation_error=observation_error,
        )
        obs_model.initialize(n_states)
        state_model = MDynHMM(
            trans_prob=trans_prob, stay_prob=stay_prob, n_states=obs_model._n
        )
        super().__init__(state_model, obs_model, n_samples)


class HMM_Poi(Simulation):
    """HMM with a Poisson observation model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a str ('sequence',
        'uniform') to generate a transition probability matrix.
    rates : np.ndarray or str
        Poisson rate for each state and channel, shape (n_states, n_channels),
        or :code:`'random'`.
    n_states : int, optional
        Number of states. Needed if :code:`rates='random'`.
    n_channels : int, optional
        Number of channels. Needed if :code:`rates='random'`.
    stay_prob : float, optional
        Used to generate the transition probability matrix if trans_prob is a
        str.
    """

    def __init__(
        self,
        n_samples: int,
        trans_prob: Union[np.ndarray, str],
        rates: Union[np.ndarray, str],
        n_states: Optional[int] = None,
        n_channels: Optional[int] = None,
        stay_prob: Optional[float] = None,
    ) -> None:
        obs_model = Poisson(rates=rates, n_channels=n_channels)
        obs_model.initialize(n_states)
        self.hmm = HMM(
            trans_prob=trans_prob, stay_prob=stay_prob, n_states=obs_model._n
        )
        super().__init__(self.hmm, obs_model, n_samples)


class HMM_Dirichlet(Simulation):
    """HMM with a Dirichlet observation model.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw. Must be divisible by :code:`window_size`.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array or a str ('sequence',
        'uniform') to generate a transition probability matrix.
    concentration : np.ndarray or str
        Concentration vector for each state, shape (n_states, n_channels),
        or :code:`'random'`.
    n_states : int, optional
        Number of states. Needed if :code:`concentration='random'`.
    n_channels : int, optional
        Number of channels. Needed if :code:`concentration='random'`.
    stay_prob : float, optional
        Used to generate the transition probability matrix if trans_prob is a
        str.
    window_size : int, optional
        Samples per hidden-state step. State transitions occur between
        windows, and the samples within a window share a state.
    """

    def __init__(
        self,
        n_samples: int,
        trans_prob: Union[np.ndarray, str],
        concentration: Union[np.ndarray, str],
        n_states: Optional[int] = None,
        n_channels: Optional[int] = None,
        stay_prob: Optional[float] = None,
        window_size: int = 1,
    ) -> None:
        if not isinstance(window_size, (int, np.integer)) or window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if n_samples % window_size:
            raise ValueError("n_samples must be divisible by window_size.")
        self.window_size = window_size

        obs_model = Dirichlet(concentration=concentration, n_channels=n_channels)
        obs_model.initialize(n_states)
        self.hmm = HMM(
            trans_prob=trans_prob, stay_prob=stay_prob, n_states=obs_model._n
        )
        super().__init__(self.hmm, obs_model, n_samples)

    def _generate_time_course(self, state_model) -> np.ndarray:
        states = state_model.generate_states(self.n_samples // self.window_size)
        return np.repeat(states, self.window_size, axis=0)


class MSess_HMM_MVN(Simulation):
    """HMM with session-specific multivariate normal observation models.

    Parameters
    ----------
    n_samples : int
        Number of samples per session to draw from the model.
    trans_prob : np.ndarray or str
        Transition probability matrix as a numpy array, shape
        (n_states, n_states) or (n_sessions, n_states, n_states), or a
        :code:`str` (:code:`'sequence'` or :code:`'uniform'`) to generate a
        transition probability matrix.
    session_means : np.ndarray or str
        Session mean vector for each state, shape should be
        (n_sessions, n_states, n_channels).
        Either a numpy array or :code:`'zero'` or :code:`'random'`.
    session_covariances : np.ndarray or str
        Session covariance matrix for each state, shape should be
        (n_sessions, n_states, n_channels, n_channels).
        Either a numpy array or :code:`'random'`.
    n_states : int, optional
        Number of states.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    embedding_vectors : np.ndarray, optional
        Embedding vectors for each session, shape should be
        (n_sessions, embeddings_dim).
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
        Used to generate the transition probability matrix if :code:`trans_prob`
        is a :code:`str`. Must be between 0 and 1.
    tc_std : float, optional
        Standard deviation when generating session-specific stay probability.
    observation_error : float, optional
        Standard deviation of the error added to the generated data.
    """

    _time_courses_first = True

    def __init__(
        self,
        n_samples: int,
        trans_prob: Union[np.ndarray, str, None],
        session_means: Union[np.ndarray, str],
        session_covariances: Union[np.ndarray, str],
        n_states: Optional[int] = None,
        n_channels: Optional[int] = None,
        n_covariances_act: int = 1,
        embedding_vectors: Optional[np.ndarray] = None,
        n_sessions: Optional[int] = None,
        embeddings_dim: Optional[int] = None,
        spatial_embeddings_dim: Optional[int] = None,
        embeddings_scale: Optional[float] = None,
        n_groups: Optional[int] = None,
        between_group_scale: Optional[float] = None,
        tc_std: float = 0.0,
        stay_prob: Optional[float] = None,
        observation_error: float = 0.0,
    ) -> None:
        obs_model = MSess_MVN(
            session_means=session_means,
            session_covariances=session_covariances,
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
        obs_model.initialize(n_states)
        n_sessions = obs_model.n_sessions

        # Construct trans_prob for each session
        if isinstance(trans_prob, np.ndarray):
            if trans_prob.ndim == 2:
                # Same transition probability matrix for each session
                trans_prob = [trans_prob] * n_sessions
            else:
                # A (n_sessions, n_states, n_states) array was passed
                trans_prob = list(trans_prob)
        elif isinstance(trans_prob, str) or trans_prob is None:
            trans_prob = [trans_prob] * n_sessions

        # Vary the stay probability for each session
        if stay_prob is not None:
            session_stay_prob = np.random.normal(
                loc=stay_prob,
                scale=tc_std,
                size=n_sessions,
            )
            # Truncate stay_prob at 0 and 1
            session_stay_prob = np.minimum(session_stay_prob, 1)
            session_stay_prob = np.maximum(session_stay_prob, 0)
        else:
            session_stay_prob = [stay_prob] * n_sessions

        self.hmm = [
            HMM(
                trans_prob=trans_prob[i],
                stay_prob=session_stay_prob[i],
                n_states=obs_model._n,
            )
            for i in range(n_sessions)
        ]
        super().__init__(self.hmm, obs_model, n_samples)


class HierarchicalHMM_MVN(Simulation):
    """Hierarchical two-level HMM with a multivariate normal observation model.

    See :class:`HierarchicalHMM`. The top-level state time course is
    :code:`top_level_stc`.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw from the model.
    top_level_trans_prob : np.ndarray or str
        Transition probability matrix of the top-level HMM, which
        selects the bottom-level HMM at each time point. Used when
        :code:`top_level_hmm_type='hmm'`.
    bottom_level_trans_probs : list of np.ndarray or str
        Transition probability matrices for the bottom-level HMMs,
        which generate the states.
    means : np.ndarray or str, optional
        Mean vector for each state, shape should be (n_states, n_channels).
        Either a numpy array or :code:`'zero'` or :code:`'random'`.
    covariances : np.ndarray or str, optional
        Covariance matrix for each state, shape should be (n_states, n_channels,
        n_channels). Either a numpy array or :code:`'random'`.
    n_states : int, optional
        Number of states.
    n_channels : int, optional
        Number of channels.
    n_covariances_act : int, optional
        Number of iterations to add activations to covariance matrices.
    observation_error : float, optional
        Standard deviation of random noise to be added to the observations.
    top_level_stay_prob : float, optional
        The stay_prob for the top-level HMM. Used if
        :code:`top_level_trans_prob` is a :code:`str`.
    bottom_level_stay_probs : list of float, optional
        The list of :code:`stay_prob` values for the bottom-level HMMs.
        Used when the corresponding entry in :code:`bottom_level_trans_probs`
        is a :code:`str`.
    top_level_hmm_type : str, optional
        The type of HMM to use at the top level, either :code:`'hmm'` or
        :code:`'hsmm'`.
    top_level_gamma_shape : float, optional
        The shape parameter for the Gamma distribution used by
        the top-level HMM when :code:`top_level_hmm_type='hsmm'`.
    top_level_gamma_scale : float, optional
        The scale parameter for the Gamma distribution used by
        the top-level HMM when :code:`top_level_hmm_type='hsmm'`.
    """

    def __init__(
        self,
        n_samples: int,
        top_level_trans_prob: Union[np.ndarray, str, None],
        bottom_level_trans_probs: List[Union[np.ndarray, str]],
        means: Optional[Union[np.ndarray, str]] = None,
        covariances: Optional[Union[np.ndarray, str]] = None,
        n_states: Optional[int] = None,
        n_channels: Optional[int] = None,
        n_covariances_act: int = 1,
        observation_error: float = 0.0,
        top_level_stay_prob: Optional[float] = None,
        bottom_level_stay_probs: Optional[List[float]] = None,
        top_level_hmm_type: str = "hmm",
        top_level_gamma_shape: Optional[float] = None,
        top_level_gamma_scale: Optional[float] = None,
    ) -> None:
        obs_model = MVN(
            means=means,
            covariances=covariances,
            n_channels=n_channels,
            n_covariances_act=n_covariances_act,
            observation_error=observation_error,
        )
        obs_model.initialize(n_states)
        state_model = HierarchicalHMM(
            top_level_trans_prob=top_level_trans_prob,
            bottom_level_trans_probs=bottom_level_trans_probs,
            n_states=obs_model._n,
            top_level_stay_prob=top_level_stay_prob,
            bottom_level_stay_probs=bottom_level_stay_probs,
            top_level_hmm_type=top_level_hmm_type,
            top_level_gamma_shape=top_level_gamma_shape,
            top_level_gamma_scale=top_level_gamma_scale,
        )
        super().__init__(state_model, obs_model, n_samples)


class HMM_OscillatoryBursts(Simulation):
    """HMM with an oscillatory burst observation model.

    While a state is active, its channels produce sinusoids at the state's
    frequency. See :class:`~osl_dynamics.simulation.OscillatoryBursts`. The
    noise-free signal is :code:`true_signal`.

    Parameters
    ----------
    n_samples : int
        Number of samples per session.
    n_sessions : int, optional
        Number of sessions.
    n_groups : int, optional
        Number of groups of sessions with different frequencies.
        :code:`n_sessions` must be divisible by :code:`n_groups`.
    frequencies : np.ndarray or list, optional
        Frequency (Hz) of each oscillatory state. Shape: (n_states,). If not
        given, frequencies are linearly spaced between 8 and 25 Hz.
    group_freq_shift : float, optional
        Frequency shift between groups in Hz.
    channel_activity : np.ndarray, optional
        Binary matrix indicating the active channels of each oscillatory
        state. Shape: (n_states, n_channels). If not given, each state has
        :code:`n_channels_per_state` channels.
    n_states : int, optional
        Number of oscillatory states. Ignored if :code:`channel_activity` or
        :code:`frequencies` is given.
    n_channels_per_state : int, optional
        Channels per state when generating :code:`channel_activity`.
    sampling_frequency : float, optional
        Sampling frequency in Hz.
    snr : float, optional
        Signal-to-noise ratio.
    stay_prob : float, optional
        HMM state stay probability.
    background_state : bool, optional
        Whether to add a state with no oscillations (noise only). It is the
        last state.
    """

    def __init__(
        self,
        n_samples: int,
        n_sessions: Optional[int] = None,
        n_groups: int = 1,
        frequencies: Optional[Union[np.ndarray, List[float]]] = None,
        group_freq_shift: float = 0.5,
        channel_activity: Optional[np.ndarray] = None,
        n_states: int = 4,
        n_channels_per_state: int = 8,
        sampling_frequency: float = 100,
        snr: float = 4,
        stay_prob: float = 0.98,
        background_state: bool = True,
    ) -> None:
        if channel_activity is None:
            if frequencies is not None:
                n_states = len(frequencies)
            channel_activity = np.tile(np.eye(n_states), n_channels_per_state)
        channel_activity = np.asarray(channel_activity)
        n_oscillatory_states, n_channels = channel_activity.shape

        if frequencies is None:
            frequencies = np.linspace(8, 25, n_oscillatory_states)
        frequencies = np.asarray(frequencies, dtype=float)

        # Shift the frequencies of each group of sessions
        if n_groups > 1:
            if n_sessions is None or n_sessions % n_groups:
                raise ValueError("n_sessions must be divisible by n_groups.")
            shifts = (np.arange(n_groups) - (n_groups - 1) / 2) * group_freq_shift
            shifts = np.repeat(shifts, n_sessions // n_groups)
            frequencies = frequencies[np.newaxis] + shifts[:, np.newaxis]

        if background_state:
            channel_activity = np.vstack([channel_activity, np.zeros(n_channels)])
            nan = np.full(frequencies.shape[:-1] + (1,), np.nan)
            frequencies = np.concatenate([frequencies, nan], axis=-1)

        obs_model = OscillatoryBursts(
            frequencies=frequencies,
            channel_activity=channel_activity,
            sampling_frequency=sampling_frequency,
            snr=snr,
        )
        self.hmm = HMM(trans_prob="uniform", stay_prob=stay_prob, n_states=obs_model._n)
        super().__init__(self.hmm, obs_model, n_samples, n_sessions=n_sessions)

        if self.n_sessions is None:
            self.true_signal = obs_model.get_true_signal(self.state_time_course)
        else:
            self.true_signal = np.array(
                [
                    obs_model.get_true_signal(
                        stc, session if obs_model.n_sessions is not None else None
                    )
                    for session, stc in enumerate(self.state_time_course)
                ]
            )


class HMM_TDECovs(Simulation):
    """HMM with a TDE covariance observation model.

    Each state has a time-delay embedded covariance matrix, which encodes
    both its spectral and cross-channel structure. See
    :class:`~osl_dynamics.simulation.TDECovs`.

    Parameters
    ----------
    n_samples : int
        Number of samples per session.
    tde_covariances : list of np.ndarray
        TDE covariance matrix of each state, each of shape
        ``(n_channels * n_embeddings, n_channels * n_embeddings)``.
    n_sessions : int, optional
        Number of sessions.
    n_embeddings : int, optional
        Number of time-delay embeddings.
    stay_prob : float, optional
        HMM state stay probability.
    rho : float, optional
        Regularization parameter for inverting the covariance.
    """

    def __init__(
        self,
        n_samples: int,
        tde_covariances: List[np.ndarray],
        n_sessions: Optional[int] = None,
        n_embeddings: int = 1,
        stay_prob: float = 0.98,
        rho: float = 0.1,
    ) -> None:
        obs_model = TDECovs(
            tde_covariances=tde_covariances,
            n_embeddings=n_embeddings,
            rho=rho,
        )
        self.hmm = HMM(trans_prob="uniform", stay_prob=stay_prob, n_states=obs_model._n)
        super().__init__(self.hmm, obs_model, n_samples, n_sessions=n_sessions)
