import numpy as np

from vrad.analysis import spectral


class Analysis:
    """Class for analysing data after fitting."""

    def _reverse_std_pca(self, covariances, reverse_standardization, subject_index):

        # Validation
        if not self.prepared:
            raise ValueError(
                "Data must have been prepared in VRAD if this method is called."
            )

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

        te_covs = []
        for n in range(n_subjects):
            if reverse_standardization:
                for i in range(n_states):
                    # Get the standard deviation of the prepared data
                    if subject_index is None:
                        prepared_data_std = self.prepared_data_std[n]
                    else:
                        prepared_data_std = self.prepared_data_std[subject_index]

                    # Reverse the standardisation
                    covariances[n][i] = (
                        np.diag(prepared_data_std)
                        @ covariances[n][i]
                        @ np.diag(prepared_data_std)
                    )

            # Reverse the PCA
            te_cov = []
            for i in range(n_states):
                te_cov.append(self.pca_weights @ covariances[n][i] @ self.pca_weights.T)
            te_cov = np.array(te_cov)

            if reverse_standardization:
                for i in range(n_states):
                    # Get the standard deviation of the raw data
                    if subject_index is None:
                        raw_data_std = self.raw_data_std[n]
                    else:
                        raw_data_std = self.raw_data_std[subject_index]

                    # Reverse the standardisation
                    te_cov[i] = (
                        np.diag(np.repeat(raw_data_std, self.n_embeddings))
                        @ te_cov[i]
                        @ np.diag(np.repeat(raw_data_std, self.n_embeddings))
                    )

            te_covs.append(te_cov)

        return te_covs

    def autocorrelation_functions(
        self,
        covariances: Union[list, np.ndarray],
        reverse_standardization: bool = False,
        subject_index: int = None,
    ) -> np.ndarray:
        """Calculates the autocorrelation function the state covariance matrices.

        An autocorrelation function is calculated for each state for each subject.

        Parameters
        ----------
        covariances : np.ndarray
            State covariance matrices.
            Shape is (n_subjects, n_states, n_channels, n_channels).
            These must be subject specific covariances.
        reverse_standardization : bool
            Should we reverse the standardization performed on the dataset?
            Optional, the default is False.
        subject_index : int
            Index for the subject if the covariances corresponds to a single
            subject. Optional. Only used if reverse_standardization is True.

        Returns
        -------
        np.ndarray
            Autocorrelation function.
            Shape is (n_subjects, n_states, n_channels, n_channels, n_acf)
            or (n_states, n_channels, n_channels, n_acf).
        """
        # Get covariance of time embedded data
        te_covs = self._reverse_std_pca(
            covariances, reverse_standardization, subject_index
        )

        # Take elements from the time embedded covariances that
        # correspond to the autocorrelation function
        autocorrelation_functions = []
        for n in range(len(te_covs)):
            autocorrelation_function.append(
                spectral.autocorrelation_function(
                    te_covs[n], self.n_embeddings, self.n_raw_data_channels
                )
            )

        return np.squeeze(autocorrelation_function)

    def raw_covariances(
        self,
        state_covariances: Union[list, np.ndarray],
        reverse_standardization: bool = False,
        subject_index: int = None,
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
        reverse_standardization : bool
            Should we reverse the standardization performed on the dataset?
            Optional, the default is False.
        subject_index : int
            Index for the subject if the covariances corresponds to a single
            subject. Optional. Only used if reverse_standardization is True.

        Returns
        -------
        np.ndarray
            The variance for each channel, state and subject.
            Shape is (n_subjects, n_states, n_channels, n_channels) or
            (n_states, n_channels, n_channels).
        """
        # Get covariance of time embedded data
        te_covs = self._reverse_std_pca(
            state_covariances, reverse_standardization, subject_index
        )

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
