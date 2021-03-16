import numpy as np
from typing import Union

from vrad.analysis import spectral


class Analysis:
    """Class for analysing data after fitting."""

    def autocorrelation_functions(
        self,
        covariances: Union[list, np.ndarray],
    ) -> np.ndarray:
        """Calculates the autocorrelation function the state covariance matrices.

        An autocorrelation function is calculated for each state for each subject.

        Parameters
        ----------
        covariances : np.ndarray
            State covariance matrices.
            Shape is (n_subjects, n_states, n_channels, n_channels).
            These must be subject specific covariances.

        Returns
        -------
        np.ndarray
            Autocorrelation function.
            Shape is (n_subjects, n_states, n_channels, n_channels, n_acf)
            or (n_states, n_channels, n_channels, n_acf).
        """
        # Get covariance of time embedded data
        te_covs = self.reverse_pca(covariances)

        # Take elements from the time embedded covariances that
        # correspond to the autocorrelation function
        autocorrelation_functions = []
        for n in range(len(te_covs)):
            autocorrelation_functions.append(
                spectral.autocorrelation_function(
                    te_covs[n], self.n_embeddings, self.n_raw_data_channels
                )
            )

        return np.squeeze(autocorrelation_functions)

    def raw_covariances(
        self,
        state_covariances: Union[list, np.ndarray],
    ) -> np.ndarray:
        """Covariance matrix of the raw channels.

        PCA and standardization is reversed to give you to the covariance
        matrix for the raw channels.

        Parameters
        ----------
        state_covariances : np.ndarray
            State covariance matrices.
            Shape is (n_subjects, n_states, n_channels, n_channels).
            These must be subject specific covariances.

        Returns
        -------
        np.ndarray
            The variance for each channel, state and subject.
            Shape is (n_subjects, n_states, n_channels, n_channels) or
            (n_states, n_channels, n_channels).
        """
        # Get covariance of time embedded data
        te_covs = self.reverse_pca(state_covariances)

        # Take elements from the time embedded covariances that
        # correspond to the raw channel covariances
        raw_covariances = []
        for n in range(len(te_covs)):
            raw_covariances.append(
                te_covs[n][
                    :,
                    self.n_embeddings // 2 :: self.n_embeddings,
                    self.n_embeddings // 2 :: self.n_embeddings,
                ]
            )

        return np.squeeze(raw_covariances)

    def reverse_pca(self, covariances: Union[list, np.ndarray]):
        """Reverses the effect of PCA on a covariance matrix.

        Parameters
        ----------
        covariances : np.ndarray
            State covariance matrices.
            Shape is (n_subjects, n_states, n_channels, n_channels).
            These must be subject specific covariances.
        """

        # Validation
        if self.pca_weights is None:
            raise ValueError("PCA has not been applied.")

        if isinstance(covariances, np.ndarray):
            if covariances.ndim != 3:
                raise ValueError(
                    "covariances must be shape (n_states, n_channels, n_channels) or"
                    + " (n_subjects, n_states, n_channels, n_channels)."
                )
            covariances = [covariances]

        if not isinstance(covariances, list):
            raise ValueError(
                "covariances must be a list of numpy arrays or a numpy array."
            )

        n_subjects = len(covariances)
        n_states = covariances[0].shape[0]

        # Reverse the PCA
        te_covs = []
        for n in range(n_subjects):
            te_cov = np.array(
                [
                    self.pca_weights @ covariances[n][i] @ self.pca_weights.T
                    for i in range(n_states)
                ]
            )
            te_covs.append(te_cov)

        return te_covs
