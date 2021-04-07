from typing import Union

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

        # State probabilities
        if "gamma" in self.hmm:
            self.gamma = self.hmm.gamma.astype(np.float32)
        elif "Gamma" in self.hmm:
            self.gamma = self.hmm.Gamma.astype(np.float32)
        else:
            raise AttributeError("OSL HMM object does not contain gamma.")

        # State time course
        if self.gamma is not None:
            stc = self.gamma.argmax(axis=1)
            self.stc = array_ops.get_one_hot(stc).astype(np.float32)

        # State covariances
        self.covariances = np.array(
            [
                state.Omega.Gam_rate
                / (state.Omega.Gam_shape - len(state.Omega.Gam_rate) - 1)
                for state in self.state
            ]
        )

        # Discontinuities in the training data which indicate the number of data
        # points for different subjects
        if "T" in self.hmm:
            self.discontinuities = [np.squeeze(T).astype(int) for T in self.hmm.T]
        else:
            # Assume gamma has no discontinuities
            self.discontinuities = [self.gamma.shape[0]]

    def __str__(self):
        return f"OSL HMM object from file {self.filename}"

    def alpha(
        self, concatenate: bool = False, pad_n_embeddings: int = None
    ) -> Union[list, np.ndarray]:
        """Alpha for each subject.

        Alpha is equivalent to gamma in OSL HMM.

        Parameters
        ----------
        concatenate : bool
            Should we concatenate the alphas for each subejcts? Optional, default is
            False.
        pad_n_embeddings : int
            Pad the alpha for each subject with zeros to replace the data points lost
            by performing n_embeddings. Optional, default is no padding.
        """
        if pad_n_embeddings is None:
            if concatenate:
                return self.gamma
            else:
                return np.split(self.gamma, np.cumsum(self.discontinuities[:-1]))
        else:
            return [
                np.pad(alpha, [[pad_n_embeddings, pad_n_embeddings], [0, 0]])
                for alpha in np.split(self.gamma, np.cumsum(self.discontinuities[:-1]))
            ]

    def state_time_course(
        self, concatenate: bool = False, pad_n_embeddings: int = None
    ) -> Union[list, np.ndarray]:
        """State time course for each subject.

        Parameters
        ----------
        concatenate : bool
            Should we concatenate the state time course for each subjects? Optional,
            default is False.
        pad_n_embeddings : int
            Pad the state time course for each subject with zeros to replace the data
            points lost by performing n_embeddings. Optional, default is no padding.
        """
        if pad_n_embeddings is None:
            if concatenate:
                return self.stc
            else:
                return np.split(self.stc, np.cumsum(self.discontinuities[:-1]))
        else:
            return [
                np.pad(stc, [[pad_n_embeddings, pad_n_embeddings], [0, 0]])
                for stc in np.split(self.stc, np.cumsum(self.discontinuities[:-1]))
            ]

    def plot_covariances(self, *args, **kwargs):
        """Wraps plotting.plot_matrices for self.covariances."""
        plotting.plot_matrices(self.covariances, *args, **kwargs)

    def plot_states(self, *args, **kwargs):
        """Wraps plotting.highlight_states for self.state_time_course."""
        plotting.state_barcode(self.state_time_course, *args, **kwargs)

    def covariance_weights(self) -> np.ndarray:
        """Calculate covariance weightings based on variance (trace).

        Method to wrap `array_ops.trace_weights`.

        Returns
        -------
        weights: np.ndarray
            Statewise weights.

        """
        return array_ops.trace_weights(self.covariances)
