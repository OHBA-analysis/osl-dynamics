import mat73
import numpy as np
from vrad import array_ops
from vrad.utils import plotting


class OSL_HMM:
    """Import and encapsulate OSL HMMs"""

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
                state.Gam_rate / (state.Gam_shape - len(state.Gam_rate) - 1)
                for state in self.state.Omega
            ]
        )

    def get_state_time_course(
        self, discontinuities, n_embeddings, sequence_length,
    ):
        # Separate the state time course in subject state time courses
        subject_data_lengths = [sum(d) for d in discontinuities]
        hmm_stc = []
        for i in range(len(subject_data_lengths)):
            start = sum(subject_data_lengths[:i])
            end = sum(subject_data_lengths[:i+1])
            hmm_stc.append(self.state_time_course[start:end])

        # Remove data points lost to separating into sequences
        for i in range(len(hmm_stc)):
            n_sequences = hmm_stc[i].shape[0] // sequence_length
            hmm_stc[i] = hmm_stc[i][: n_sequences * sequence_length]

        return hmm_stc

    def plot_covariances(self, *args, **kwargs):
        """Wraps plotting.plot_matrices for self.covariances."""
        plotting.plot_matrices(self.covariances, *args, **kwargs)

    def plot_states(self, *args, **kwargs):
        """Wraps plotting.highlight_states for self.state_time_course."""

        plotting.state_barcode(self.state_time_course, *args, **kwargs)

    def __str__(self):
        return f"OSL HMM object from file {self.filename}"
