from typing import Union

import numpy as np
import yaml
from tqdm import tqdm
from vrad.analysis import spectral
from vrad.data import Data, manipulation
from vrad.utils.misc import MockArray


class PreprocessedData(Data):
    """Class for loading preprocessed data.

    Contains methods which can be used to prepare the data for training a model.
    This includes methods to perform time embedding and PCA.

    Parameters
    ----------
    inputs : list of str or str
        Filenames to be read.
    sampling_frequency : float
        Sampling frequency of the data in Hz. Optional, default is 1.0.
    store_dir : str
        Directory to save results and intermediate steps to. Optional, default
        is /tmp.
    prepared_data_file : str
        Filename to save memory map to. Optional.
    """

    def __init__(
        self,
        inputs: list,
        sampling_frequency: float = 1.0,
        store_dir: str = "tmp",
        prepared_data_file: str = None,
    ):
        super().__init__(inputs, sampling_frequency, store_dir)
        if prepared_data_file is None:
            self.prepared_data_file = f"dataset_{self._identifier}.npy"

    def prepare_memmap_filenames(self):
        self.te_pattern = "te_data_{{i:0{width}d}}_{identifier}.npy".format(
            width=len(str(len(self.inputs))), identifier=self._identifier
        )
        self.prepared_data_pattern = (
            "prepared_data_"
            "{{i:0{width}d}}_"
            "{identifier}.npy".format(
                width=len(str(len(self.inputs))), identifier=self._identifier
            )
        )

        # Prepared data memory maps (time embedded and pca'ed)
        self.prepared_data_memmaps = []
        self.prepared_data_filenames = [
            str(self.store_dir / self.prepared_data_pattern.format(i=i))
            for i, _ in enumerate(self.inputs)
        ]
        self.prepared_data_mean = []
        self.prepared_data_std = []

    def prepare(
        self,
        n_embeddings: int = 1,
        n_pca_components: int = None,
        whiten: bool = False,
    ):
        """Prepares data to train the model with.

        Performs standardization, time embedding and principle component analysis.

        Parameters
        ----------
        n_embeddings : int
            Number of data points to embed the data. Optional, default is 1.
        n_pca_components : int
            Number of PCA components to keep. Optional, default is no PCA.
        whiten : bool
            Should we whiten the PCA'ed data? Optional, default is False.
        """
        # Class attributes related to data preparation
        self.n_embeddings = n_embeddings
        self.n_te_channels = self.n_raw_data_channels * n_embeddings
        self.n_pca_components = n_pca_components
        self.whiten = whiten

        # Create filenames for memmaps (i.e. self.prepared_data_filenames)
        self.prepare_memmap_filenames()

        # Principle component analysis (PCA)
        # NOTE: the approach used here only works for zero mean data
        if n_pca_components is not None:

            # Calculate the PCA components by performing SVD on the covariance
            # of the data
            covariance = np.zeros([self.n_te_channels, self.n_te_channels])
            for raw_data_memmap in tqdm(
                self.raw_data_memmaps, desc="Calculating PCA components", ncols=98
            ):
                # Standardise the data and time embed
                std_data = manipulation.standardize(raw_data_memmap)
                te_data = manipulation.time_embed(std_data, n_embeddings)

                # Calculate the covariance of the entire dataset
                covariance += np.transpose(te_data) @ te_data

                # Clear intermediate data
                del std_data, te_data

            # Use SVD to calculate PCA components
            u, s, vh = np.linalg.svd(covariance)
            u = u[:, :n_pca_components].astype(np.float32)
            s = s[:n_pca_components].astype(np.float32)
            if whiten:
                u = u @ np.diag(1.0 / np.sqrt(s))
            self.pca_weights = u
        else:
            self.pca_weights = None

        # Prepare the data
        for raw_data_memmap, prepared_data_file in zip(
            tqdm(self.raw_data_memmaps, desc="Preparing data", ncols=98),
            self.prepared_data_filenames,
        ):
            # Standardise the data and time embed
            std_data = manipulation.standardize(raw_data_memmap)
            te_data = manipulation.time_embed(std_data, n_embeddings)

            # Apply PCA to get the prepared data
            if self.pca_weights is not None:
                prepared_data = te_data @ self.pca_weights

            # Otherwise, the time embedded data is the prepared data
            else:
                prepared_data = te_data

            # Create a memory map for the prepared data
            prepared_data_memmap = MockArray.get_memmap(
                prepared_data_file, prepared_data.shape, dtype=np.float32
            )

            # Record the mean and standard deviation of the prepared
            # data and standardise to get the final data
            self.prepared_data_mean.append(np.mean(prepared_data, axis=0))
            self.prepared_data_std.append(np.std(prepared_data, axis=0))
            prepared_data_memmap = manipulation.standardize(
                prepared_data, create_copy=False
            )
            self.prepared_data_memmaps.append(prepared_data_memmap)

            # Clear intermediate data
            del std_data, te_data, prepared_data

        # Update subjects to return the prepared data
        self.subjects = self.prepared_data_memmaps

    def trim_raw_time_series(
        self, n_embeddings: int = None, sequence_length: int = None
    ) -> np.ndarray:
        """Trims the raw preprocessed data time series.

        Removes the data points that are removed when the data is prepared,
        i.e. due to time embedding and separating into sequences, but does not
        perform time embedding or batching into sequences on the time series.

        Parameters
        ----------
        n_embeddings : int
            Number of data points to embed the data.
        sequence_length : int
            Length of the segement of data to feed into the model.

        Returns
        -------
        np.ndarray
            Trimed time series.
        """
        trimmed_raw_time_series = []
        for memmap in self.raw_data_memmaps:
            if n_embeddings is not None:
                # Remove data points which are removed due to time embedding
                memmap = memmap[n_embeddings // 2 : -n_embeddings // 2]
            if sequence_length is not None:
                # Remove data points which are removed due to separating into sequences
                n_sequences = memmap.shape[0] // sequence_length
                memmap = memmap[: n_sequences * sequence_length]
            trimmed_raw_time_series.append(memmap)
        return trimmed_raw_time_series

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
        # Validation
        if self.n_embeddings <= 1:
            raise ValueError(
                "To calculate an autocorrelation function we have to train on time "
                + "embedded data."
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

        autocorrelation_function = []
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

            # Calculate the autocorrelation function
            autocorrelation_function.append(
                spectral.autocorrelation_function(
                    te_cov, self.n_embeddings, self.n_raw_data_channels
                )
            )

        return np.squeeze(autocorrelation_function)

    def raw_covariances(
        self,
        state_covariances: Union[list, np.ndarray],
        reverse_standardization: bool = False,
        subject_index: int = None,
    ) -> np.ndarray:
        """Variance of each channel based on the inferred state covariances.

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
        # Validation
        if self.n_embeddings <= 1:
            raise ValueError(
                "To calculate an autocorrelation function we have to train on time "
                + "embedded data."
            )

        if isinstance(state_covariances, np.ndarray):
            if state_covariances.ndim != 3:
                raise ValueError(
                    "state_covariances must be shape (n_states, n_channels, n_channels) or"
                    + " (n_subjects, n_states, n_channels, n_channels)."
                )
            state_covariances = [state_covariances]

        if not isinstance(state_covariances, list):
            raise ValueError(
                "state_covariances must be a list of numpy arrays or a numpy array."
            )

        n_subjects = len(state_covariances)
        n_states = state_covariances[0].shape[0]

        raw_covariances = []
        for n in range(n_subjects):
            if reverse_standardization:
                for i in range(n_states):
                    # Get the standard deviation of the prepared data
                    if subject_index is None:
                        prepared_data_std = self.prepared_data_std[n]
                    else:
                        prepared_data_std = self.prepared_data_std[subject_index]

                    # Reverse the standardisation
                    state_covariances[n][i] = (
                        np.diag(prepared_data_std)
                        @ state_covariances[n][i]
                        @ np.diag(prepared_data_std)
                    )

            # Reverse the PCA
            te_cov = []
            for i in range(n_states):
                te_cov.append(
                    self.pca_weights @ state_covariances[n][i] @ self.pca_weights.T
                )
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

            # Get the raw data covariance
            raw_covariances.append(
                te_cov[
                    :,
                    self.n_embeddings // 2 :: self.n_embeddings,
                    self.n_embeddings // 2 :: self.n_embeddings,
                ]
            )

        return np.squeeze(raw_covariances)

    def _process_from_yaml(self, file, **kwargs):
        with open(file) as f:
            settings = yaml.load(f, Loader=yaml.Loader)

        prep_settings = settings.get("prepare", {})
        if prep_settings.get("do", False):
            self.prepare(
                n_embeddings=prep_settings.get("n_embeddings"),
                n_pca_components=prep_settings.get("n_pca_components", None),
                whiten=prep_settings.get("whiten", False),
            )
