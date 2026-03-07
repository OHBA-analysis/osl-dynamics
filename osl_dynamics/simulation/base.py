"""Base simulation class."""

from typing import Optional

import numpy as np


class Simulation:
    """Simulation base class.

    Parameters
    ----------
    n_samples : int
        Number of time points to generate.
    """

    def __init__(self, n_samples: int) -> None:
        self.n_samples = n_samples
        self.time_series = None

    def __array__(self) -> np.ndarray:
        return self.time_series

    def __iter__(self):
        return iter([self.time_series])

    def __getattr__(self, attr: str):
        if attr == "time_series":
            raise NameError("time_series has not yet been created.")
        if attr[:2] == "__":
            raise AttributeError(f"No attribute called {attr}.")
        return getattr(self.time_series, attr)

    def __len__(self) -> int:
        return 1

    def standardize(self, axis: int = 0) -> None:
        mu = np.mean(self.time_series, axis=axis, keepdims=True)
        sigma = np.std(self.time_series, axis=axis, keepdims=True)
        self.time_series = (self.time_series - mu) / sigma
