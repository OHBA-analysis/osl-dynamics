import mat73
import numpy as np
from vrad import array_ops
from vrad.utils import plotting


class OSL_HMM:
    """Imports and encapsulates OSL HMMs as python objects.

    Parameters
    ----------
    filename : str
        The location of the OSL HMM saved as a mat7.3 file.
    """

    def __init__(self, filename):
        self.filename = filename
        self.hmm = mat73.loadmat(filename)["hmm"]

        self.state = self.hmm.state
        self.k = int(self.hmm.K)
        self.p = self.hmm.P
        self.dir_2d_alpha = self.hmm.Dir2d_alpha
        self.pi = self.hmm.Pi
        self.dir_alpha = self.hmm.Dir_alpha
        self.prior = self.hmm.prior
        self.train = self.hmm.train

        hmm_fields = dir(self.hmm)

        # State probabilities
        if "gamma" in hmm_fields:
            self.gamma = self.hmm.gamma
        elif "Gamma" in hmm_fields:
            self.gamma = self.hmm.Gamma
        else:
            self.gamma = None

        # State time course
        if self.gamma is not None:
            stc = self.gamma.argmax(axis=1)
            self.state_time_course = array_ops.get_one_hot(stc)

        # State covariances
        self.covariances = np.array(
            [
                state.Omega.Gam_rate
                / (state.Omega.Gam_shape - len(state.Omega.Gam_rate) - 1)
                for state in self.state
            ]
        )

    def plot_covariances(self, *args, **kwargs):
        """Wraps plotting.plot_matrices for self.covariances."""
        plotting.plot_matrices(self.covariances, *args, **kwargs)

    def plot_states(self, *args, **kwargs):
        """Wraps plotting.highlight_states for self.state_time_course."""
        plotting.state_barcode(self.state_time_course, *args, **kwargs)

    def __str__(self):
        return f"OSL HMM object from file {self.filename}"
