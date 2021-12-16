from typing import Union

import mat73
import numpy as np
from dynemo import array_ops
from dynemo.inference import modes


class OSL_HMM:
    """Imports and encapsulates OSL HMMs as python objects.

    Parameters
    ----------
    filename : str
        The location of the OSL HMM saved as a mat7.3 file.
    """

    def __init__(self, filename):
        self.filename = filename
        self.hmm = mat73.loadmat(filename, use_attrdict=True)["hmm"]

        self.mode = self.hmm.state
        self.k = int(self.hmm.K)
        self.p = self.hmm.P
        self.dir_2d_alpha = self.hmm.Dir2d_alpha
        self.pi = self.hmm.Pi
        self.dir_alpha = self.hmm.Dir_alpha
        self.prior = self.hmm.prior
        self.train = self.hmm.train

        # Mode probabilities
        if "gamma" in self.hmm:
            self.gamma = self.hmm.gamma.astype(np.float32)
        elif "Gamma" in self.hmm:
            self.gamma = self.hmm.Gamma.astype(np.float32)
        else:
            self.gamma = None

        # Mode time course
        if self.gamma is not None:
            vpath = self.gamma.argmax(axis=1)
            self.vpath = array_ops.get_one_hot(vpath).astype(np.float32)
        else:
            self.vpath = None

        # Mode means
        self.means = np.array([mode.W.Mu_W for mode in self.mode])

        # Mode covariances
        self.covariances = np.array(
            [
                mode.Omega.Gam_rate
                / (mode.Omega.Gam_shape - len(mode.Omega.Gam_rate) - 1)
                for mode in self.mode
            ]
        )

        # Transition probability matrix
        self.trans_prob = self.p

        # Discontinuities in the training data which indicate the number of data
        # points for different subjects
        if "T" in self.hmm:
            self.discontinuities = [np.squeeze(T).astype(int) for T in self.hmm.T]
        elif self.gamma is not None:
            # Assume gamma has no discontinuities
            self.discontinuities = [self.gamma.shape[0]]
        else:
            self.discontinuities = None

    def __str__(self):
        return f"OSL HMM object from file {self.filename}"

    def alpha(
        self, concatenate: bool = False, pad: int = None
    ) -> Union[list, np.ndarray]:
        """Alpha for each subject.

        Alpha is equivalent to gamma in OSL HMM.

        Parameters
        ----------
        concatenate : bool
            Should we concatenate the alphas for each subejcts? Optional, default is
            False.
        pad : int
            Pad the alpha for each subject with zeros to replace the data points lost
            by performing n_embeddings. Optional, default is no padding.
        """
        if self.gamma is None:
            return None

        if pad is None:
            if concatenate or len(self.discontinuities) == 1:
                return self.gamma
            else:
                return np.split(self.gamma, np.cumsum(self.discontinuities[:-1]))
        else:
            padded_alpha = [
                np.pad(alpha, [[pad, pad], [0, 0]])
                for alpha in np.split(self.gamma, np.cumsum(self.discontinuities[:-1]))
            ]
            if concatenate:
                return np.concatenate(padded_alpha)
            else:
                return padded_alpha

    def fractional_occupancies(self) -> np.ndarray:
        """Fractional Occupancy of each mode.

        Returns
        -------
        np.ndarray
            Fractional occupancies.
        """
        stc = self.mode_time_course(concatenate=True)
        return modes.fractional_occupancies(stc)

    def mode_time_course(
        self, concatenate: bool = False, pad: int = None
    ) -> Union[list, np.ndarray]:
        """Mode time course for each subject.

        Parameters
        ----------
        concatenate : bool
            Should we concatenate the mode time course for each subjects? Optional,
            default is False.
        pad : int
            Pad the mode time course for each subject with zeros to replace the data
            points lost by performing n_embeddings. Optional, default is no padding.
        """
        if self.vpath is None:
            return None

        if pad is None:
            if concatenate or len(self.discontinuities) == 1:
                return self.vpath
            else:
                return np.split(self.vpath, np.cumsum(self.discontinuities[:-1]))
        else:
            padded_stc = [
                np.pad(self.vpath, [[pad, pad], [0, 0]])
                for stc in np.split(self.vpath, np.cumsum(self.discontinuities[:-1]))
            ]
            if concatenate:
                return np.concatenate(padded_stc)
            else:
                return padded_stc

    def covariance_weights(self) -> np.ndarray:
        """Calculate covariance weightings based on variance (trace).

        Method to wrap `array_ops.trace_weights`.

        Returns
        -------
        weights: np.ndarray
            Modewise weights.

        """
        return array_ops.trace_weights(self.covariances)
