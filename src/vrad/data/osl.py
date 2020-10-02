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

        self.data_files = self.hmm.data_files if "data_files" in hmm_fields else None
        self.epoched_state_path_sub = (
            self.hmm.epoched_statepath_sub
            if "epoched_state_path_sub" in hmm_fields
            else None
        )
        self.filenames = self.hmm.filenames if "filenames" in hmm_fields else None
        self.f_sample = self.hmm.fsample if "fsample" in hmm_fields else None
        self.gamma = self.hmm.gamma if "gamma" in hmm_fields else None
        self.is_epoched = self.hmm.is_epoched if "is_epoched" in hmm_fields else None
        self.options = self.hmm.options if "options" in hmm_fields else None
        self.state_map_parcel_vectors = (
            self.hmm.statemap_parcel_vectors
            if "statemap_parcel_vectors" in hmm_fields
            else None
        )
        self.subject_state_map_parcel_vectors = (
            self.hmm.statemap_parcel_vectors_persubj
            if "statemap_parcel_vectors_persubj" in hmm_fields
            else None
        )
        self.state_maps = self.hmm.statemaps if "statemaps" in hmm_fields else None
        self.state_path = (
            self.hmm.statepath.astype(np.int) if "statepath" in hmm_fields else None
        )
        self.subject_indices = self.hmm.subj_inds if "subj_inds" in hmm_fields else None

        # Aliases
        self.covariances = np.array(
            [
                state.Gam_rate / (state.Gam_shape - len(state.Gam_rate) - 1)
                for state in self.state.Omega
            ]
        )
        self.alpha = self.gamma
        self.state_time_course = (
            array_ops.get_one_hot(self.state_path - 1)
            if self.state_path is not None
            else None
        )

        if self.gamma is not None and self.state_time_course is None:
            stc = self.gamma.argmax(axis=1)
            self.state_time_course = array_ops.get_one_hot(stc)

    def plot_covariances(self, *args, **kwargs):
        """Wraps plotting.plot_matrices for self.covariances."""
        plotting.plot_matrices(self.covariances, *args, **kwargs)

    def plot_states(self, *args, **kwargs):
        """Wraps plotting.highlight_states for self.state_time_course."""

        plotting.state_barcode(self.state_time_course, *args, **kwargs)

    def __str__(self):
        return f"OSL HMM object from file {self.filename}"
