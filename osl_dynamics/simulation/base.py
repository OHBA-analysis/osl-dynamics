"""Base simulation class.

"""

from osl_dynamics.data.processing import standardize


class Simulation:
    """Simulation base class.

    Parameters
    ----------
    n_samples : int
        Number of time points to generate.
    """

    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.time_series = None

    def __array__(self):
        return self.time_series

    def __iter__(self):
        return iter([self.time_series])

    def __getattr__(self, attr):
        if attr == "time_series":
            raise NameError("time_series has not yet been created.")
        if attr[:2] == "__":
            raise AttributeError(f"No attribute called {attr}.")
        return getattr(self.time_series, attr)

    def __len__(self):
        return 1

    def standardize(self, axis=0):
        self.time_series = standardize(self.time_series, axis=axis)
