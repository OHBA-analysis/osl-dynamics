"""Simulation from a state model and an observation model."""

from typing import Optional

import numpy as np

_COUNT_AND_TIME_COURSE = {
    "n_states",
    "n_modes",
    "state_time_course",
    "mode_time_course",
}


def _outputs_states(state_model) -> bool:
    """Whether a state model generates states (rather than modes)."""
    has_states = hasattr(state_model, "generate_states")
    has_modes = hasattr(state_model, "generate_modes")
    if has_states == has_modes:
        raise TypeError(
            f"{type(state_model).__name__} must have exactly one of "
            "generate_states or generate_modes."
        )
    return has_states


class Simulation:
    """Simulate data from a state model and an observation model.

    The state model generates a time course and the observation model
    generates data from it. A state model either generates mutually exclusive
    states (it has :code:`n_states` and :code:`generate_states`, e.g.
    :class:`~osl_dynamics.simulation.HMM`), giving a simulation with
    :code:`n_states` and :code:`state_time_course`, or a mixture of modes (it
    has :code:`n_modes` and :code:`generate_modes`, e.g.
    :class:`~osl_dynamics.simulation.MixedSine`), giving a simulation with
    :code:`n_modes` and :code:`mode_time_course`.

    Other attributes are looked up in the observation model and then the state
    model, e.g. :code:`sim.covariances` or :code:`sim.trans_prob`.

    Parameters
    ----------
    state_model : state model or list of state models
        Generates the state or mode time course. Pass a list for a different
        state model in each session.
    obs_model : observation model
        Generates the data given the time course. Random parameters are
        generated with the number of states or modes of the state model.
    n_samples : int
        Number of samples (per session) to simulate.
    n_sessions : int, optional
        Number of sessions. Each session has its own time course and data. If
        the observation model has session-specific parameters, the number of
        sessions is taken from it.
    """

    # Generate the time courses for all sessions before the data
    _time_courses_first = False

    def __init__(
        self,
        state_model,
        obs_model,
        n_samples: int,
        n_sessions: Optional[int] = None,
    ) -> None:
        self.state_model = state_model
        self.obs_model = obs_model
        self.n_samples = n_samples

        models = (
            state_model if isinstance(state_model, (list, tuple)) else [state_model]
        )
        kinds = {_outputs_states(model) for model in models}
        if len(kinds) != 1:
            raise ValueError("All state models must generate either states or modes.")
        self._generates_states = kinds.pop()

        counts = {
            model.n_states if self._generates_states else model.n_modes
            for model in models
        }
        if len(counts) != 1:
            raise ValueError(
                "All state models must have the same number of states or modes."
            )
        self._n = counts.pop()
        obs_model.initialize(self._n)

        obs_sessions = getattr(obs_model, "n_sessions", None)
        self._session_specific = obs_sessions is not None
        if self._session_specific:
            if n_sessions is not None and n_sessions != obs_sessions:
                raise ValueError(
                    f"n_sessions={n_sessions} but the observation model has "
                    f"parameters for {obs_sessions} sessions."
                )
            n_sessions = obs_sessions
        if isinstance(state_model, (list, tuple)) and len(state_model) != n_sessions:
            raise ValueError("Pass one state model per session.")
        self.n_sessions = n_sessions

        self._time_course, self.time_series = self._simulate()

    def _session_state_models(self) -> list:
        if isinstance(self.state_model, (list, tuple)):
            return list(self.state_model)
        return [self.state_model] * self.n_sessions

    def _generate_time_course(self, state_model) -> np.ndarray:
        if self._generates_states:
            return state_model.generate_states(self.n_samples)
        return state_model.generate_modes(self.n_samples)

    def _simulate_session(self, session: int, time_course: np.ndarray) -> np.ndarray:
        if self._session_specific:
            return self.obs_model.simulate_session_data(session, time_course)
        return self.obs_model.simulate_data(time_course)

    def _simulate(self) -> tuple:
        if self.n_sessions is None:
            time_course = self._generate_time_course(self.state_model)
            return time_course, self.obs_model.simulate_data(time_course)

        models = self._session_state_models()
        if self._time_courses_first:
            time_courses = [self._generate_time_course(model) for model in models]
            data = [
                self._simulate_session(session, time_course)
                for session, time_course in enumerate(time_courses)
            ]
        else:
            time_courses, data = [], []
            for session, model in enumerate(models):
                time_courses.append(self._generate_time_course(model))
                data.append(self._simulate_session(session, time_courses[-1]))
        return np.array(time_courses), np.array(data)

    @property
    def n_states(self) -> int:
        """Number of states (simulations of mutually exclusive states only)."""
        if not self._generates_states:
            raise AttributeError
        return self._n

    @property
    def n_modes(self) -> int:
        """Number of modes (simulations of modes only)."""
        if self._generates_states:
            raise AttributeError
        return self._n

    @property
    def state_time_course(self) -> np.ndarray:
        """State time course (simulations of mutually exclusive states only)."""
        if not self._generates_states:
            raise AttributeError
        return self._time_course

    @property
    def mode_time_course(self) -> np.ndarray:
        """Mode time course (simulations of modes only)."""
        if self._generates_states:
            raise AttributeError
        return self._time_course

    def __getattr__(self, attr: str):
        if attr.startswith("_") or attr in ("state_model", "obs_model"):
            raise AttributeError(f"No attribute called {attr}.")
        if attr in _COUNT_AND_TIME_COURSE:
            kind = "states" if self.__dict__.get("_generates_states") else "modes"
            raise AttributeError(
                f"No attribute called {attr}: this simulation generates {kind}."
            )
        for model in (self.__dict__.get("obs_model"), self.__dict__.get("state_model")):
            if model is not None and not isinstance(model, (list, tuple)):
                if hasattr(model, attr):
                    return getattr(model, attr)
        raise AttributeError(f"No attribute called {attr}.")

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.asarray(self.time_series, dtype=dtype)

    def __iter__(self):
        if self.n_sessions is None:
            return iter([self.time_series])
        return iter(list(self.time_series))

    def __len__(self) -> int:
        return 1 if self.n_sessions is None else self.n_sessions

    def standardize(self) -> None:
        """Standardize the data and update the ground truth parameters.

        Each channel is standardized over time. With session-specific
        parameters each session is standardized separately, otherwise the
        sessions are standardized together.
        """
        if self.n_sessions is None:
            axis = 0
        elif self._session_specific:
            axis = 1
        else:
            axis = (0, 1)
        mu = np.mean(self.time_series, axis=axis, keepdims=True)
        sigma = np.std(self.time_series, axis=axis, keepdims=True)
        self.time_series = (self.time_series - mu) / sigma
        self.obs_model.rescale(
            np.squeeze(mu, axis=axis).astype(np.float64),
            np.squeeze(sigma, axis=axis).astype(np.float64),
        )

    def get_instantaneous_covariances(self) -> np.ndarray:
        """Get the ground truth covariance at each time point.

        Returns
        -------
        inst_covs : np.ndarray
            Instantaneous covariances. Shape is
            (n_samples, n_channels, n_channels), with a leading session
            dimension if there are multiple sessions.
        """
        if not hasattr(self.obs_model, "get_instantaneous_covariances"):
            raise AttributeError(
                f"{type(self.obs_model).__name__} has no instantaneous covariances."
            )
        if self.n_sessions is None or self._session_specific:
            return self.obs_model.get_instantaneous_covariances(self._time_course)
        return np.array(
            [
                self.obs_model.get_instantaneous_covariances(time_course)
                for time_course in self._time_course
            ]
        )
