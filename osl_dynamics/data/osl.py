"""Class to read HMM-MAR objects from MATLAB.

"""

import numpy as np
from osl_dynamics import array_ops
from osl_dynamics.data.processing import trim_time_series
from osl_dynamics.data.rw import loadmat


def OSL_HMM(filename):
    """Wrapper for osl_dynamics.data.osl.HMM_MAR.

    Parameters
    ----------
    filename : str
        The location of the HMM-MAR object saved as a mat7.3 file.
    """
    return HMM_MAR(filename)


class HMM_MAR:
    """Imports and encapsulates HMM-MAR objects as a python class.

    Parameters
    ----------
    filename : str
        The location of the HMM-MAR object saved as a mat7.3 file.
    """

    def __init__(self, filename):
        self.filename = filename
        hmm = loadmat(filename)
        self.hmm = hmm["hmm"] if "hmm" in hmm else hmm

        self.state = self.hmm["state"]
        self.mode = self.hmm["state"]
        self.k = int(self.hmm["K"])
        self.p = self.hmm["P"]
        self.dir_2d_alpha = self.hmm["Dir2d_alpha"]
        self.pi = self.hmm["Pi"]
        self.dir_alpha = self.hmm["Dir_alpha"]
        self.prior = self.hmm["prior"]
        self.train = self.hmm["train"]

        # State probabilities
        self.Gamma = None
        for gamma in ["gamma", "Gamma"]:
            if gamma in self.hmm:
                self.Gamma = self.hmm[gamma].astype(np.float32)
            elif gamma in hmm:
                self.Gamma = hmm[gamma].astype(np.float32)

        # State time course
        if self.Gamma is not None:
            vpath = self.Gamma.argmax(axis=1)
            self.vpath = array_ops.get_one_hot(vpath).astype(np.float32)
        else:
            self.vpath = None

        # State means
        self.means = np.array([state["W"]["Mu_W"] for state in self.state])

        # State covariances
        self.covariances = np.array(
            [
                state["Omega"]["Gam_rate"]
                / (state["Omega"]["Gam_shape"] - len(state["Omega"]["Gam_rate"]) - 1)
                for state in self.state
            ]
        )

        # Transition probability matrix
        self.trans_prob = self.p

        # Discontinuities in the training data which indicate the number of data
        # points for different subjects
        if "T" in self.hmm:
            self.discontinuities = [np.squeeze(T).astype(int) for T in self.hmm["T"]]
        elif self.Gamma is not None:
            # Assume gamma has no discontinuities
            self.discontinuities = [self.Gamma.shape[0]]
        else:
            self.discontinuities = None

    def __str__(self):
        return f"OSL HMM object from file {self.filename}"

    def gamma(self, concatenate=False, pad=None):
        """State probabilities for each subject.

        Parameters
        ----------
        concatenate : bool
            Should we concatenate the gammas for each subjects?
        pad : int
            Pad the gamma for each subject with zeros to replace the data points lost
            by performing n_embeddings. Default is no padding.

        Returns
        -------
        gamma : np.ndarray or list
            State probabilities.
        """
        if self.Gamma is None:
            return None

        if pad is None:
            if concatenate or len(self.discontinuities) == 1:
                return self.Gamma
            else:
                return np.split(self.Gamma, np.cumsum(self.discontinuities[:-1]))
        else:
            padded_gamma = [
                np.pad(gamma, [[pad, pad], [0, 0]])
                for gamma in np.split(self.Gamma, np.cumsum(self.discontinuities[:-1]))
            ]
            if concatenate:
                return np.concatenate(padded_gamma)
            else:
                return padded_gamma

    def trimmed_gamma(self, sequence_length, concatenate=False):
        """Trimmed state probabilities for each subject.

        Data points that would be lost due to separating the time series into
        sequences are removed from each subject.

        Parameters
        ----------
        sequence_length : int
            Sequence length.
        concatenate : bool
            Should we concatenate the gammas for each subject?

        Returns
        -------
        gamma : np.ndarray or list
            State probabilities.
        """
        return trim_time_series(
            self.gamma(), sequence_length=sequence_length, concatenate=concatenate
        )

    def state_time_course(self, concatenate=False, pad=None):
        """State time course for each subject.

        Parameters
        ----------
        concatenate : bool
            Should we concatenate the state time course for each subjects?
        pad : int
            Pad the state time course for each subject with zeros to replace the data
            points lost by performing n_embeddings. Default is no padding.

        Returns
        -------
        stc : np.ndarray or list
            State time course.
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

    def mode_time_course(self, *args, **kwargs):
        """Wrapper for the state_time_course method.

        Returns
        -------
        mtc : np.ndarray or list
            Mode time course.
        """
        return self.state_time_course(*args, **kwargs)
